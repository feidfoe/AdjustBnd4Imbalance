# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

import pickle

import torch.utils.data as data
import torchvision.datasets as datasets


class CIFARLoader(Dataset):
    def __init__(self, root='./data', train=True, 
                       imbalance=1, transform=None):
        self.T = transform
        dataset = root.split('/')[-1]

        if train:
            if dataset == 'cifar10':
                data_list = ['data_batch_%d'%(i+1) for i in range(5)]
            elif dataset == 'cifar100':
                data_list = ['train']
        else:
            if dataset == 'cifar10':
                data_list = ['test_batch']
            elif dataset == 'cifar100':
                data_list = ['test']



        self.data = []
        self.label = []
        for filename in data_list:
            filepath = os.path.join(os.path.join(root,filename))
            with open(filepath, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.label.extend(entry['labels'])
                else:
                    self.label.extend(entry['fine_labels'])

        data = np.vstack(self.data).reshape(-1,3,32,32)
        data = data.transpose((0,2,3,1)) #NHWC
        labels = np.array(self.label)

        n_class = np.max(labels) + 1
        img_max = data.shape[0] // n_class
        imb_factor = 1. / imbalance
        img_list, lbl_list = [], []
        for i in range(n_class):
            idx = np.squeeze(np.argwhere(labels == i))
            img = data[idx]
            lbl = labels[idx]
            num_sample = int(img_max * (imb_factor**(i/(n_class - 1))))
            img_list.append(img[:num_sample])
            lbl_list.append(lbl[:num_sample])
        
        self.images = np.concatenate(img_list)
        self.labels = np.concatenate(lbl_list)


    def __getitem__(self,index):
        image = self.images[index]
        label = self.labels[index]

        return self.T(image), label.astype(np.long)


    def __len__(self):
        return self.images.shape[0]
