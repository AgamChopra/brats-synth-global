# -*- coding: utf-8 -*-
"""
Created on Jan 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
"""
import os
from random import randint
import numpy as np
from numpy.random import choice
import torch
import torch.nn as nn
from tqdm import trange
from matplotlib import pyplot as plt
from pytorch_msssim import SSIM
import nibabel as nib

import dataloader
import Disc_PET_model as models

torch.set_printoptions(precision=10)
torch.set_float32_matmul_precision('high')


def np_mse(y, yp):
    error = np.mean((y - yp)**2)
    return error


def one_hot_generator(T):  # (T, T)
    one_hot = torch.zeros((T, T))
    for i in range(T):
        one_hot[i, i] = 1
    return one_hot


def get_random_batch_modalities(data, one_hot):  # input: (Batch, 4, ...), (N,N); Output: Normalized (Batch, 3, ...), Normalized (Batch, 1, ...), (Batch, 4)
    out_idx = [randint(0, one_hot.shape[0] - 1) for _ in range(data.shape[0])]
    


class PSNR():
    def __init__(self, epsilon=1E-9):
        self.name = "PSNR"
        self.epsilon = epsilon

    def __call__(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(torch.max(x)) - 10 * torch.log10(mse)
        loss = (psnr + self.epsilon) ** -1
        return loss


class ssim_loss(nn.Module):
    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(ssim_loss, self).__init__()
        self.ssim = SSIM(channel=channel, spatial_dims=spatial_dims,
                         win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        loss = 1 - self.ssim(x, y)
        return loss


def norm(x):
    EPSILON = 1E-9
    try:
        return (x - torch.min(x)) / ((torch.max(x) - torch.min(x)) + EPSILON)
    except:
        try:
            return (x - np.min(x)) / ((np.max(x) - np.min(x)) + EPSILON)
        except Exception:
            print('WARNING: Input could not be normalized!')


def train(checkpoint_path, epochs=100, lr=1E-4, batch=32,
          device='cpu', model_path=None, lambdas=[0.15, 0.15, 0.60, 0.20],
          loss_functions=[nn.MSELoss(), nn.L1Loss(),
                          ssim_loss(win_size=3, win_sigma=0.1), PSNR()], T=4,
          iter_val=7):
    print(device)
    # load the model
    neural_network = models.Attention_UNetTT(2, 1, T, 8).to(device)
    if model_path is not None:
        neural_network.load_state_dict(
            torch.load(model_path, map_location=device))
    neural_network = torch.compile(neural_network)

    # load the optimizer and criterion
    optimizer = torch.optim.Adam(neural_network.parameters(), lr)
    criterion = loss_functions

    # load dataloader
    data = dataloader.train_dataloader(augment=True, batch=batch)
    data_val = dataloader.valset_dataloader(batch=1)
    print('%d training samples, %d validation samples' %
          (data.max_id, len(data_val.pid)))

    # (T, T) one hot encoded...
    one_hot_encodes = one_hot_generator(T).to(device=device)

    # how many times to iterate each epoch
    iterations = 1 * (int(data.max_id / batch) + (data.max_id % batch > 0))
    iterations_val = iter_val  # !!! TO BE SET MANUALLY

    # store training loss for visualization
    losses = []
    losses_train = []
    losses_temp = []
    losses_val = []

    # optimization loop
    for eps in range(epochs):
        print('Epoch %d:' % (eps))

        neural_network.train()
        print(next(neural_network.parameters()).is_cuda)

        for i in trange(iterations):
            optimizer.zero_grad()

            a = data.load_batch().to(dtype=torch.float, device=device)

            input_modalities, output_modality, encodes = get_random_batch_modalities(
                a, one_hot_encodes)  # Normalized (Batch, 3, ...), (Batch, 1, ...), (Batch, 4)

            synth_output_modality = neural_network(input_modalities, encodes)

            error = sum([lambdas[i] * criterion[i](output_modality,
                        synth_output_modality) for i in range(len(criterion))])
            error.backward()
            optimizer.step()

            losses.append(error.item())

        # losses_train.append(sum(losses[-iterations:])/iterations)
        losses_train.append(sum(losses)/iterations)
        losses = []

        neural_network.eval()
        # validation loop (after each epoch)
        for i in trange(iterations_val):
            with torch.no_grad():
                a = data_val.load_batch().to(dtype=torch.float, device=device)
                a = data.load_batch().to(dtype=torch.float, device=device)

                input_modalities, output_modality, encodes = get_random_batch_modalities(
                    a, one_hot_encodes)  # Normalized (Batch, 3, ...), (Batch, 1, ...), (Batch, 4)

                synth_output_modality = neural_network(
                    input_modalities, encodes)

                error = sum([lambdas[i] * criterion[i](output_modality,
                            synth_output_modality) for i in range(len(criterion))])

                losses_temp.append(error.item())

                plot_idx = 0
                plots = [norm(torch.squeeze(a[plot_idx, 0]).detach().cpu().numpy()),
                         norm(torch.squeeze(a[plot_idx, 1]
                                            ).detach().cpu().numpy()),
                         norm(torch.squeeze(a[plot_idx, 2]
                                            ).detach().cpu().numpy()),
                         norm(torch.squeeze(
                             synth_output_modality[plot_idx]).detach().cpu().numpy()),
                         norm(torch.squeeze(
                             output_modality[plot_idx]).detach().cpu().numpy()),
                         norm(np.abs(np.squeeze(np.squeeze(output_modality[plot_idx].detach().cpu().numpy()) - np.squeeze(synth_output_modality[plot_idx].detach().cpu().numpy()))))]

                dataloader.plot_scans(plots, (3, 6), 150)

        losses_val.append(sum(losses_temp)/iterations_val)
        losses_temp = []

        if (eps % 2 == 0 or eps == epochs - 1) and eps != 0:
            plt.plot(losses_train, label='Training Loss')
            plt.plot(losses_val, label='Validation Loss')
            plt.title('Compound Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        if eps % 5 == 0 and eps != 0:
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'checkpoint_%d_epochs.pt' % (eps))

        print(' Average Train Loss: %.4f, Validation Loss: %.4f' %
              (losses_train[-1], losses_val[-1]))

    if model_path is None:
        torch.save(neural_network.cpu().state_dict(), checkpoint_path +
                   'trained_paramaters_%d_epochs.pt' % (epochs))
    else:
        torch.save(neural_network.cpu().state_dict(), checkpoint_path +
                   'fine_trained_paramaters_%d_epochs.pt' % (epochs))

    return losses_train, losses_val


def trn(checkpoint_path, epochs=500, lr=1E-4, batch=32,
        device='cpu', params=None, N=100, T=128):
    losst, lossv = train(checkpoint_path, epochs,
                         lr, batch, device, params, N=N, T=T)
    plt.plot(losst, label='Training Loss')
    plt.plot(lossv, label='Validation Loss')
    plt.title('Compound Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    checkpoint_path = '/home/agam/Documents/pet-project/ADNI_Disc-PET_v5_fulldata/'
    epochs = 1000
    N = 50
    T = 64
    lr = 1E-5
    batch = 6
    device = 'cuda'
    params = '/home/agam/Documents/pet-project/ADNI_Disc-PET_v5/' + \
        'checkpoint_175_epochs.pt'  # full path

    trn(checkpoint_path, epochs, lr, batch, device, params, N=N, T=T)
