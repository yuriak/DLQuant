# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
import os


class FFN(nn.Module):
    def __init__(self, d_x, d_h, d_o, ffn_layers=5, dp=0.5):
        super(FFN, self).__init__()
        self.f_in = nn.Linear(in_features=d_x, out_features=d_h)
        self.hiddens = nn.ModuleList([nn.Linear(in_features=d_x, out_features=d_h) for _ in range(ffn_layers)])
        self.lms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(ffn_layers)])
        self.f_out = nn.Linear(in_features=d_h, out_features=d_o)
        self.dropout = nn.Dropout(p=dp)
    
    def forward(self, x):
        x_in = self.dropout(self.f_in(x))
        output = x_in
        for hidden, lm in zip(self.hiddens, self.lms):
            output = self.dropout(lm(F.leaky_relu(hidden(output))))
        output = self.f_out(x_in + output)
        return output


class FFNClassifier(object):
    def __init__(self, d_x, d_h, d_o, lr=1e-3, dp=0.5, ffn_layers=5):
        super(FFNClassifier, self).__init__()
        self.ffn = FFN(d_x, d_h, d_o=d_o, dp=dp, ffn_layers=ffn_layers)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.ffn.parameters(), lr=lr)
        for p in self.ffn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def train(self, X, y):
        x = torch.tensor(X, dtype=torch.float32)
        y_true = torch.tensor(y, dtype=torch.long)
        self.optimizer.zero_grad()
        y_hat = self.ffn(x)
        loss = self.loss_func(y_hat, y_true).mean()
        loss.backward()
        clip_grad_norm_(self.ffn.parameters(), 1)
        self.optimizer.step()
        topv, topi = y_hat.topk(1)
        y_hat = topi.view(-1).detach()
        acc = accuracy_score(y_true=y.flatten(), y_pred=y_hat)
        mcc = matthews_corrcoef(y_true=y.flatten(), y_pred=y_hat)
        return loss.item(), acc, mcc, y_hat
    
    def test(self, X, y):
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            y_true = torch.tensor(y, dtype=torch.long)
            y_hat = self.ffn(x)
            loss = self.loss_func(y_hat, y_true).mean()
            topv, topi = y_hat.topk(1)
            y_hat = topi.view(-1).detach()
            acc = accuracy_score(y_true=y.flatten(), y_pred=y_hat)
            mcc = matthews_corrcoef(y_true=y.flatten(), y_pred=y_hat)
            return loss.item(), acc, mcc, y_hat
    
    def inference(self, X):
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            y_hat = self.ffn(x)
            topv, topi = y_hat.topk(1)
            y_hat = topi.view(-1).detach()
            return y_hat

    def load_model(self, model_path='./FFNModel'):
        self.ffn = torch.load(model_path + '/model.pkl')

    def save_model(self, model_path='./FFNModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.ffn, model_path + '/model.pkl')