# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:42:36 2022

@author: Zao
"""

import torch
import numpy as np
import os

def greater_than1 (X):
    n,m = X.shape
    for i in range(n):
        for j in range(m):
            if abs(X[i,j])>1:
                print('illegal value detected')
                return

n_list = list(range(4,8))
for n in n_list:
    for tag in [0,1]:
        file_dir = 'D:\\pic\\train\\n_{}\\tag_{}'.format(n,tag)
        file_list = os.listdir(file_dir)
        #print(file_list)
        for f in file_list:
            X = torch.load(file_dir+'\\'+f).numpy()
            greater_than1(X)

w_list = list(range(1,8))
for n in n_list:
    for w in w_list:
        file_dir = 'D:\\pic\\test\\n_{}\\W_{}'.format(n,w)
        file_list = os.listdir(file_dir)
        for f in file_list:
            X = torch.load(file_dir+'\\'+f).numpy()
            greater_than1(X)