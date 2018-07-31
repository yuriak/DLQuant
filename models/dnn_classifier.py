# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
import os


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, class_number):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_out = nn.Linear(hidden_size // 2, class_number)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.fc_out(output)
        output = self.softmax(output)
        return output


class DNNClassifier(object):
    def __init__(self, input_size, hidden_size, class_number):
        super(DNNClassifier, self).__init__()
        self.dnn = DNN(input_size, hidden_size, class_number=class_number)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.dnn.parameters(), lr=1e-3)
    
    def train(self, X, y):
        batch_x = torch.tensor(X)
        batch_y = torch.tensor(y)
        instance_num = batch_y.shape[0]
        if instance_num > 1:
            batch_x = (batch_x - batch_x.mean(dim=1, keepdim=True)) / batch_x.std(dim=1, keepdim=True)
        self.optimizer.zero_grad()
        y_hat = self.dnn(batch_x)
        loss = self.loss_func(y_hat, batch_y)
        loss.backward()
        clip_grad_norm_(self.dnn.parameters(), 1)
        self.optimizer.step()
        topv, topi = y_hat.topk(1)
        incorrect = float(torch.sign((topi.view(-1) - batch_y).abs()).sum().item())
        acc = (instance_num - incorrect) / instance_num
        return loss.item(), acc
    
    def test(self, X, y):
        batch_x = torch.tensor(X)
        batch_y = torch.tensor(y)
        instance_num = batch_y.shape[0]
        if instance_num > 1:
            batch_x = (batch_x - batch_x.mean(dim=1, keepdim=True)) / batch_x.std(dim=1, keepdim=True)
        y_hat = self.dnn(batch_x).detach()
        loss = self.loss_func(y_hat, batch_y)
        topv, topi = y_hat.topk(1)
        incorrect = float(torch.sign((topi.view(-1) - batch_y).abs()).sum().item())
        acc = (instance_num - incorrect) / instance_num
        return loss.item(), acc
