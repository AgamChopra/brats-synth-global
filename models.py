# -*- coding: utf-8 -*-
"""
Created on June 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
    - PyTorch 2.0 stable documentation @ https://pytorch.org/docs/stable/
"""
import torch
import torch.nn as nn
from math import ceil


def pad3d(input_, target, device='cpu'):
    delta = [target.shape[2+i] - input_.shape[2+i] for i in range(3)]
    return nn.functional.pad(input=input_, pad=(ceil(delta[2]/2), delta[2] - ceil(delta[2]/2),
                                                ceil(
                                                    delta[1]/2), delta[1] - ceil(delta[1]/2),
                                                ceil(delta[0]/2), delta[0] - ceil(delta[0]/2)),
                             mode='constant', value=0).to(dtype=torch.float, device=device)


class attention_grid(nn.Module):
    def __init__(self, x_c, g_c, i_c, stride=2, mode='trilinear'):
        super(attention_grid, self).__init__()
        self.input_filter = nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride, bias=False)
        self.gate_filter = nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1, bias=True)
        self.psi = nn.Conv3d(in_channels=i_c, out_channels=1,
                             kernel_size=1, stride=1, bias=True)
        self.bnorm = nn.InstanceNorm3d(i_c)
        self.mode = mode

    def forward(self, x, g, device):
        x_shape = x.shape

        a = self.input_filter(x)
        b = self.gate_filter(g)

        if a.shape[-1] < b.shape[-1]:
            a = pad3d(a, b, device)

        elif a.shape[-1] > b.shape[-1]:
            b = pad3d(b, a, device)

        w = torch.sigmoid(self.psi(nn.functional.relu(a + b)))
        w = nn.functional.interpolate(w, size=x_shape[2:], mode=self.mode)

        y = x * w
        y = self.bnorm(y)
        return y, w


class Block(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None, final_layer=False):
        super(Block, self).__init__()
        if hid_c is None:
            self.mlp = nn.Linear(embd_dim, out_c)

            self.layer = nn.Sequential(nn.Conv3d(
                in_channels=in_c, out_channels=out_c, kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(out_c))

            self.out_block = nn.Sequential(nn.Conv3d(
                in_channels=out_c, out_channels=out_c, kernel_size=2), nn.ReLU(), nn.InstanceNorm3d(out_c))

            if final_layer:
                self.pool = False

            else:
                self.pool = True
                self.pool_block = nn.Sequential(nn.Conv3d(
                    in_channels=out_c, out_channels=out_c, kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm3d(out_c))

        else:
            self.pool = False

            self.mlp = nn.Linear(embd_dim, hid_c)

            self.layer = nn.Sequential(nn.Conv3d(
                in_channels=in_c, out_channels=hid_c, kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(hid_c))

            self.out_block = nn.Sequential(nn.Conv3d(in_channels=hid_c, out_channels=hid_c, kernel_size=2),
                                           nn.ReLU(), nn.InstanceNorm3d(hid_c),
                                           nn.ConvTranspose3d(
                                               in_channels=hid_c, out_channels=out_c, kernel_size=2, stride=2),
                                           nn.ReLU(), nn.InstanceNorm3d(out_c))

    def forward(self, x, embds):
        embds = self.mlp(embds)
        embds = embds[(..., ) + (None, ) * 3]
        y = self.layer(x)
        y = y + embds
        y = self.out_block(y)

        if self.pool:
            y_ = self.pool_block(y)
            return y, y_
        else:
            return y


class Attention_UNetT(nn.Module):
    def __init__(self, in_c, out_c, embd_dim, n=1, mode='trilinear'):
        super(Attention_UNetT, self).__init__()
        self.mode = mode
        self.out_c = out_c

        self.layer1 = Block(in_c=in_c,
                            embd_dim=embd_dim, out_c=int(64/n))

        self.layer2 = Block(in_c=int(64/n),
                            embd_dim=embd_dim, out_c=int(128/n))

        self.layer3 = Block(in_c=int(128/n),
                            embd_dim=embd_dim, out_c=int(256/n))

        self.layer4 = Block(in_c=int(256/n),
                            embd_dim=embd_dim, out_c=int(512/n))

        self.layer5 = Block(in_c=int(512/n), embd_dim=embd_dim,
                            out_c=int(512/n), hid_c=int(1024/n))

        self.layer6 = Block(in_c=int(1024/n), embd_dim=embd_dim,
                            out_c=int(256/n), hid_c=int(512/n))

        self.layer7 = Block(in_c=int(512/n), embd_dim=embd_dim,
                            out_c=int(128/n), hid_c=int(256/n))

        self.layer8 = Block(in_c=int(256/n), embd_dim=embd_dim,
                            out_c=int(64/n), hid_c=int(128/n))

        self.layer9 = Block(in_c=int(128/n), embd_dim=embd_dim,
                            out_c=int(64/n), final_layer=True)

        self.out = nn.Conv3d(in_channels=int(
            64/n), out_channels=out_c, kernel_size=1)

        self.skip1 = attention_grid(int(64/n), int(64/n), int(64/n))

        self.skip2 = attention_grid(int(128/n), int(128/n), int(128/n))

        self.skip3 = attention_grid(int(256/n), int(256/n), int(256/n))

        self.skip4 = attention_grid(int(512/n), int(512/n), int(512/n))

    def forward(self, x, embds, out_att=False):
        assert x.device == embds.device, "inputs 'x' and 'embds' are expected to be on the same device, but found at least two devices, '%s' and '%s'!" % (
            x.device, embds.device)

        device = x.device

        x = x.to(dtype=torch.float, device=device)

        y, y1 = self.layer1(x, embds)

        y, y2 = self.layer2(y, embds)

        y, y3 = self.layer3(y, embds)

        y, y4 = self.layer4(y, embds)

        y = self.layer5(y, embds)
        y4, a4 = self.skip4(y4, y, device=device)

        y = torch.cat((y4, pad3d(y, y4, device=device)), dim=1)
        y = self.layer6(y, embds)
        y3, a3 = self.skip3(y3, y, device=device)

        y = torch.cat((y3, pad3d(y, y3, device=device)), dim=1)
        y = self.layer7(y, embds)
        y2, a2 = self.skip2(y2, y, device=device)

        y = torch.cat((y2, pad3d(y, y2, device=device)), dim=1)
        y = self.layer8(y, embds)
        y1, a1 = self.skip1(y1, y, device=device)

        y = torch.cat((y1, pad3d(y, y1, device=device)), dim=1)
        y = self.layer9(y, embds)

        y = self.out(y)

        y = nn.functional.interpolate(y, size=x.shape[2:], mode=self.mode)

        if out_att:
            return y, (a1, a2, a3, a4)
        else:
            return y


def test_model(device='cpu', N=1):
    a = torch.ones((N, 3, 120, 120, 120), device=device)
    embds = torch.ones((N, 4), device=device)

    model = Attention_UNetT(in_c=3, out_c=1, embd_dim=4, n=64).to(device)

    b = model(a, embds)

    print(model)
    print(a.shape)
    print(b.shape)


if __name__ == '__main__':
    test_model('cuda', 2)
