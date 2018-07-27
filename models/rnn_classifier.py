# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class RNN(nn.Module):
    def __init__(self, input_shape, hidden_size, class_number):
        super(RNN, self).__init__()
        self.gru = nn.GRU(input_shape, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_out = nn.Linear(hidden_size // 2, class_number)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, hidden):
        _, hidden = self.gru(x, hidden)
        output = self.relu(self.fc1(hidden))
        output = self.relu(self.fc_out(output))
        output = self.softmax(output)
        return output


class RNNClassifier(object):
    def __init__(self, input_shape, hidden_size, class_number):
        super(RNNClassifier, self).__init__()
        self.rnn = RNN(input_shape, hidden_size, class_number=class_number)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=1e-3)
    
    def train(self, X, y):
        batch_x = torch.tensor(X)
        batch_y = torch.tensor(y)
        instance_num = batch_y.shape[0]
        self.optimizer.zero_grad()
        y_hat = self.rnn(batch_x, hidden=None).squeeze(0)
        loss = self.loss_func(y_hat, batch_y)
        print('train loss:', loss)
        loss.backward()
        topv, topi = y_hat.topk(1)
        acc = (instance_num - float((topi.view(-1) - batch_y).abs().sum().item())) / instance_num
        print('train acc', acc)
        self.optimizer.step()
    
    def test(self, X, y):
        batch_x = torch.tensor(X)
        batch_y = torch.tensor(y)
        instance_num = batch_y.shape[0]
        y_hat = self.rnn(batch_x, hidden=None).squeeze(0).detach()
        
        loss = self.loss_func(y_hat, batch_y)
        print('test loss', loss)
        
        topv, topi = y_hat.topk(1)
        acc = (instance_num - float((topi.view(-1) - batch_y).abs().sum().item())) / instance_num
        print('test acc', acc)
