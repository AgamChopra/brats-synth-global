# -*- coding: utf-8 -*-
"""
Created on June 2023
@author: Agamdeep Chopra, Tianyi
@email: achopra4@uw.edu, tr1@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
"""
import os
import numpy as np
import torch
from torchio.transforms import RandomFlip, RandomAffine, RandomElasticDeformation
import random
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import pickle


def norm(A):
    return (A - torch.min(A))/(torch.max(A) - torch.min(A))


def rand_augment(x):
    flip = RandomFlip()
    affine = RandomAffine(image_interpolation='nearest',
                          degrees=45, translation=5)
    elastic = RandomElasticDeformation(image_interpolation='trilinear',
                                       max_displacement=(4, 4, 4),
                                       num_control_points=random.randint(5, 7))
    x = flip(norm(affine(elastic(x))))
    return x


def plot_scans(scans=[], figsize=(15, 15), dpi=180, title=None):
    c = len(scans[0].shape)
    r = len(scans)
    i = 0

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for scan in scans:
        scan = zoom(scan, [1, 1.2, 1], order=0)

        if i < 6:
            cmap = 'gray'
        elif i < 12:
            cmap = 'gray'  # 'jet'#'gray'
        else:
            cmap = 'viridis'  # 'gist_ncar'#'jet'

        a = scan[int(scan.shape[0]/2)]
        a[0, 0] = 0
        a[0, 1] = 1
        fig.add_subplot(r, c, i+1)
        plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
        plt.subplots_adjust(wspace=0.01, hspace=.05)
        plt.axis('off')

        a = scan[:, int(scan.shape[1]/2)]
        a[0, 0] = 0
        a[0, 1] = 1
        fig.add_subplot(r, c, i+2)
        plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
        plt.subplots_adjust(wspace=0.01, hspace=.05)
        plt.axis('off')

        a = scan[:, :, int(2 * scan.shape[2]/3)]
        a[0, 0] = 0
        a[0, 1] = 1
        fig.add_subplot(r, c, i+3)
        plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
        plt.subplots_adjust(wspace=0.01, hspace=.05)
        plt.axis('off')

        # plt.colorbar(fraction=0.1, pad=0.05)

        i += 3

    if title is not None:
        plt.suptitle(title)
    plt.show()


def augment_batch(x):
    for i in range(x.shape[0]):
        x[i] = rand_augment(x[i])
    return x


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def prs_dta(x):
    x = torch.from_numpy(np.ascontiguousarray(x[None, ...]))
    return x


def load_patient(path, idx):
    x1, x2, x3, x4, y1 = pkload(os.path.join(
        path+'BraTS2021_'+f'{idx:05}'+'.pkl'))
    x1, x2, x3, x4, y1 = prs_dta(x1), prs_dta(
        x2), prs_dta(x3), prs_dta(x4), prs_dta(y1)
    data = torch.cat(x1, x2, x3, x4, y1)
    return data


def load_batch_dataset(path, idx_list):
    data = [load_patient(path, idx) for idx in idx_list]
    return torch.cat(data, dim=0)


class train_dataloader():
    def __init__(self, path='/Volumes/Kurtlab/Brats2022/Brats_SEG/Training_1/',
                 batch=1, max_id=100, post=False, augment=True):
        self.augment = augment
        self.max_id = max_id  # last patient to load from 0 to max_id
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        self.path = path

    def randomize(self):
        sample_len = self.max_id - 1
        self.idx = random.sample(range(1, self.max_id + 1), sample_len)

    def load_batch(self, post=False):
        if self.Flag:  # only runs the first time
            self.randomize()
            self.Flag = False

        max_id = self.max_id

        if self.id + self.batch > max_id:
            if self.id < max_id:
                batch_raw = load_batch_dataset(self.path, self.idx[self.id:])
            elif self.id == max_id:
                batch_raw = load_batch_dataset(
                    self.path, self.idx[self.id:self.id + 1])
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...')
        else:
            batch_raw = load_batch_dataset(
                self.path, self.idx[self.id:self.id + self.batch])
            self.id += self.batch

        if self.augment:
            batch_raw = augment_batch(batch_raw)

        return batch_raw


class val_dataloader():
    def __init__(self, path='/home/agam/Desktop/PET_PROCESSED/val',
                 pid=[0, 1, 351, 352, 353, 354, 355], batch=1):
        self.path = path
        self.pid = pid
        self.id = 0
        self.max_id = len(pid)
        self.batch = 1

    def load_batch(self):
        if self.id >= self.max_id:
            self.id = 0

        ids = [idx for idx in self.pid[self.id:self.id + self.batch]]
        batch_raw = load_batch_dataset(self.path, ids)

        self.id += self.batch

        return batch_raw
