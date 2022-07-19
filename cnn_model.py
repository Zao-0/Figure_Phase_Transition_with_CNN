# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:28:00 2022

@author: Zao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from os import listdir
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class cnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.conv_list = []
        for i in range(n-3):
            self.conv_list.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2))
        self.conv_list = nn.ModuleList(self.conv_list)
        '''
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        #self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(2**6, 2**4)
        self.fc2 = nn.Linear(2**4, 1)
    
    def forward(self, x):
        '''
        for i,layer in enumerate(self.conv_list):
            #print(x.size())
            x = F.relu(layer(x))
        '''
        x = F.relu(self.conv(x))
        x = F.relu(self.conv(x))
        x = F.relu(self.conv(x))
        x = F.relu(self.conv(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
class MyTrainDataset(Dataset):
    def __init__(self, n, tag):
        assert n in [4,5,6,7,8]
        assert tag in [0,1]
        self.n = n
        self.tag = tag
        self.flist = listdir('D:\\pic\\train\\n_{}\\tag_{}'.format(n,tag))
        
    def __len__(self):
        return len(self.flist)
    def __getitem__(self, index):
        X = torch.load('D:\\pic\\train\\n_{}\\tag_{}\\'.format(self.n,self.tag)+self.flist[index]).float()
        X = X[None,:]
        y = self.tag
        return X, y

class MyTestDataset(Dataset):
    def __init__(self, n, w):
        assert n in [4,5,6,7,8]
        assert w > 0.5 and w < 8
        self.n = n
        self.w = w
        self.flist = listdir('D:\\pic\\test\\n_{}\\W_{}'.format(n,w))
    def __len__(self):
        return len(self.flist)
    def __getitem__(self, idx):
        X = torch.load('D:\\pic\\test\\n_{}\\W_{}\\'.format(self.n, self.w)+self.flist[idx]).float()
        X = X[None,:]
        y = 0
        return X, y, self.w

def get_dataloaders(n):
    train_set = ConcatDataset([MyTrainDataset(n, 0),MyTrainDataset(n, 1)])
    valid_len = int(train_set.__len__()*0.1)
    valid_set, train_set = random_split(train_set, [valid_len, train_set.__len__()-valid_len], generator = torch.Generator().manual_seed(2264))
    test_set = MyTestDataset(n, 1)
    for i in range(2,8):
        test_set = ConcatDataset([test_set, MyTestDataset(n,i)])
    train_loader = DataLoader(train_set, batch_size = 20, shuffle = True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size = 20, shuffle = True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size = 20, shuffle = True, pin_memory = True)
    return {'train':train_loader, 'valid':valid_loader, 'test':test_loader}