# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import sklearn
import numpy as np
import os


class RNN(nn.Module):
    def __init__(self, d_x, d_h, d_o, cell=nn.GRU, rnn_layers=2, ffn_layers=5, dp=0.5):
        super(RNN, self).__init__()
        self.dropout = nn.Dropout(p=dp)
        self.rnn_layers = rnn_layers
        self.rnn = cell(input_size=d_h, hidden_size=d_h, num_layers=rnn_layers, batch_first=True, dropout=dp)
        self.hiddens = nn.ModuleList([nn.Linear(in_features=d_h, out_features=d_h) for _ in range(ffn_layers)])
        self.lms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(ffn_layers)])
        self.f_out = nn.Linear(in_features=d_h, out_features=d_o)
        for p in self.rnn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, hidden):
        x_in, hidden = self.rnn(x, hidden)
        output = x_in
        for hidden, lm in zip(self.hiddens, self.lms):
            output = self.dropout(lm(F.leaky_relu(hidden(output))))
        output = self.f_out(x_in + output)
        return output


class RNNClassifier(object):
    def __init__(self, d_x, d_h, d_o, rnn_layers=2, ffn_layers=5, lr=1e-3, cell=nn.GRU, dp=0.5):
        super(RNNClassifier, self).__init__()
        self.d_h = d_h
        self.rnn_layers = rnn_layers
        self.cell = cell
        self.rnn = RNN(d_x=d_x, d_h=d_h, d_o=d_o, rnn_layers=rnn_layers, ffn_layers=ffn_layers, cell=cell, dp=dp)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=lr)
        if self.cell == nn.GRU:
            self.tmp_hidden = torch.zeros(self.rnn_layers, 1, d_h)
        elif self.cell == nn.LSTM:
            self.tmp_hidden = (torch.zeros(self.rnn_layers, 1, d_h), torch.zeros(self.rnn_layers, 1, d_h))
    
    def reset_model(self):
        if self.cell == nn.GRU:
            self.tmp_hidden = torch.zeros(self.rnn_layers, 1, self.d_h)
        elif self.cell == nn.LSTM:
            self.tmp_hidden = (torch.zeros(self.rnn_layers, 1, self.d_h), torch.zeros(self.rnn_layers, 1, self.d_h))
    
    def train(self, X, y):
        x = torch.tensor(X[None, :, :], dtype=torch.float32)
        y_true = torch.tensor(y, dtype=torch.long)
        self.rnn.train(True)
        self.optimizer.zero_grad()
        y_hat, self.tmp_hidden = self.rnn(x, hidden=self.tmp_hidden)
        loss = self.loss_func(y_hat, y_true)
        loss.backward()
        clip_grad_norm_(self.rnn.parameters(), 1)
        self.optimizer.step()
        topv, topi = y_hat.topk(1)
        y_hat = topi.view(-1).detach()
        self.tmp_hidden = self.tmp_hidden.detach()
        acc = sklearn.metrics.accuracy_score(y_true=y.flatten(), y_pred=y_hat)
        mcc = sklearn.metrics.matthews_corrcoef(y_true=y.flatten(), y_pred=y_hat)
        return loss.item(), acc, mcc, y_hat.numpy().flatten()
    
    def test(self, X, y):
        self.rnn.eval()
        with torch.no_grad():
            x = torch.tensor(X[None, :, :], dtype=torch.float32)
            y_true = torch.tensor(y, dtype=torch.long)
            y_hat, self.tmp_hidden = self.rnn(x, hidden=self.tmp_hidden)
            loss = self.loss_func(y_hat, y_true)
            topv, topi = y_hat.topk(1)
            y_hat = topi.view(-1)
            acc = sklearn.metrics.accuracy_score(y_true=y.flatten(), y_pred=y_hat)
            mcc = sklearn.metrics.matthews_corrcoef(y_true=y.flatten(), y_pred=y_hat)
            return loss.item(), acc, mcc, y_hat.numpy().flatten()
    
    def inference(self, X):
        self.rnn.eval()
        with torch.no_grad():
            x = torch.tensor(X[None, :, :], dtype=torch.float32)
            y_hat, self.tmp_hidden = self.rnn(x, hidden=self.tmp_hidden)
            y_hat = y_hat.topk(1)[1].view(-1)
            return y_hat.numpy().flatten()
