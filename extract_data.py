# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:42:13 2022

@author: Zao
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

num_spin = 6
record = {}
with open('record_{}.pkl'.format(num_spin), 'rb') as f:
    record = pickle.load(f)

w_list = list(range(1,8))
pred = []
pred_sd = []
for i in range(1,8):
    ans = record[i]
    pred.append(np.mean(ans))
    pred_sd.append(np.std(ans))


plt.errorbar(w_list, pred, yerr = pred_sd)