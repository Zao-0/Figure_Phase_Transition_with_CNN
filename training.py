# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:24:58 2022

@author: Zao
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from cnn_model import cnn_model, get_dataloaders
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

def TRAIN(model, train_dataloader, valid_dataloader, num_epochs, criterion, optimizer, device):
    train_loss_record=[]
    valid_loss_record=[]
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_step = 0
        print('epoch:{}/{}, Start Training!!!'.format(epoch, num_epochs))
        for dm, labels in tqdm(train_dataloader):
            dm = dm.to(device)
            labels = torch.tensor(labels, dtype=torch.float, device = device)
            labels = labels.view(-1,1)
            outputs = model(dm)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step+=1
            if train_step==10:
                train_loss_record.append(loss.detach().numpy())
                train_step=0
                with torch.no_grad():
                    model.eval()
                    val_running_loss = 0.0
                    val_running_corrects = 0
                    for vdm, vlabels in (valid_dataloader):
                        vdm = vdm.to(device)
                        vlabels = torch.tensor(vlabels, dtype=torch.float, device=device)
                        vlabels = vlabels.view(-1,1)
                        voutputs = model(vdm)
                        #print(voutputs)
                        vloss = criterion(voutputs, vlabels)
                        val_running_loss += vloss.item()
                        _, vpreds = torch.max(voutputs.data, 1)
                        val_running_corrects += torch.sum(vpreds == vlabels.data)
                    valid_loss = val_running_loss / len(valid_dataloader)
                    valid_acc = val_running_corrects / float(len(valid_dataloader.dataset))
                    valid_loss_record.append(valid_loss)
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        train_loss = running_loss / len(train_dataloader)
        train_acc = running_corrects / float(len(train_dataloader.dataset))
        
        print('epoch:{}/{}, Start Evaluation!!!'.format(epoch, num_epochs))
        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            for vdm, labels in tqdm(valid_dataloader):
                vdm = vdm.to(device)
                labels = torch.tensor(labels, dtype=torch.float, device=device)
                labels = labels.view(-1,1)
                outputs = model(vdm)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data)
            valid_loss = running_loss / len(valid_dataloader)
            valid_acc = running_corrects / float(len(valid_dataloader.dataset))
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f},  Valid Acc: {:.4f}'
              .format(epoch, num_epochs, train_loss, train_acc, valid_loss, valid_acc))
    
    print('Finished Training')
    plt.figure()
    #print(train_loss_record)
    plt.plot(train_loss_record, 'ro--',label='train_loss', linewidth=1, markersize=2)
    plt.plot(valid_loss_record, 'bo--',label='valid_loss', linewidth=1, markersize=2)
    plt.xlabel('step x10')
    plt.ylabel('LOSS')
    plt.legend(loc='upper right')
    plt.savefig('fig1.png')
    plt.clf()
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

num_spin = 7
num_epochs = 50
model = cnn_model()

model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dtloaders = get_dataloaders(num_spin)


TRAIN(model=model, train_dataloader=dtloaders['train'], valid_dataloader=dtloaders['valid'],
      num_epochs=num_epochs, criterion=criterion, optimizer=optimizer, device=device)

from sklearn.metrics import accuracy_score,f1_score, roc_curve, roc_auc_score, auc

val_label=[]
val_pred=[]
print('ROC Curve Evaluate')
for vdm, vlabel in tqdm(dtloaders['valid']):
    vdm = vdm.to(device)
    vlabel = vlabel.clone().detach().to(device)
    vlabel = vlabel.view(-1,1)
    outputs = model(vdm)
    val_label.extend(vlabel.detach().cpu().numpy()) 
    val_pred.extend(outputs.detach().cpu().numpy())
fpr, tpr, threshold = roc_curve(val_label, val_pred)
ra=auc(fpr,tpr)
lb1 = 'roc({})'.format('%.2f',ra)
plt.plot(fpr, tpr, 'b-', label=lb1, linewidth=1)
plt.plot([0,1], [0,1], 'r--', label='exp', linewidth=1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim([0,1.0])
plt.ylim([0,1.0])
plt.legend('lower right')
plt.savefig('roc_{}.png'.format(num_spin))

record = {1:[],
          2:[],
          3:[],
          4:[],
          5:[],
          6:[],
          7:[]
    }
print('Testing')
for dm, _, w in tqdm(dtloaders['test']):
    dm = dm.to(device)
    outputs = model(dm)
    w = list(w)
    num = len(w)
    outputs = outputs.detach().cpu().numpy()
    outputs = np.reshape(outputs, num)
    for i in range(num):
        record[int(w[i].item())].append(outputs[i])

import pickle
dic_file = open('record_{}.pkl'.format(num_spin),'wb')
pickle.dump(record, dic_file)
dic_file.close()