# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef
import numpy as np
import os

lmap = lambda func, it: list(map(lambda x: func(x), it))


class Attention(nn.Module):
    def __init__(self, d_x, d_h, temporal=False, dp=0.5):
        super(Attention, self).__init__()
        self.temporal = temporal
        self.w = nn.Linear(d_x, d_h)
        self.dropout = nn.Dropout(p=dp)
        if self.temporal:
            self.u = nn.Linear(d_h, d_h, bias=False)
    
    def forward(self, x_t, hidden=None):
        if self.temporal and hidden is not None:
            v_t = self.dropout(F.relu(self.w(x_t) + self.u(hidden)))
        else:
            v_t = self.dropout(F.relu(self.w(x_t)))
        e_t = torch.matmul(v_t, torch.transpose(v_t, 2, 1))
        e_t = e_t.sum(dim=1, keepdim=True)
        e_t = F.softmax(e_t, dim=-1)
        a_t = torch.matmul(e_t, x_t)
        return a_t, e_t


class AttnRNN(nn.Module):
    def __init__(self, d_x, d_h, d_o, rnn_layers=2, ffn_layers=5, cell=nn.GRU, temporal=False, dp=0.5):
        super(AttnRNN, self).__init__()
        self.temporal = temporal
        self.dropout = nn.Dropout(p=dp)
        self.f_in = nn.Linear(in_features=d_x,out_features=d_h)
        self.attn = Attention(d_h, d_h, temporal=self.temporal)
        self.rnn = cell(input_size=d_h, hidden_size=d_h, num_layers=rnn_layers, batch_first=True, dropout=dp)
        self.hls = nn.ModuleList([nn.Linear(in_features=d_h, out_features=d_h) for _ in range(ffn_layers)])
        self.lms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(ffn_layers)])
        self.f_out = nn.Linear(in_features=d_h, out_features=d_o)
        for p in self.attn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x_t, hidden):
        x_t=self.f_in(x_t)
        if self.temporal and hidden is not None:
            if type(hidden) == tuple:
                h_t_1 = hidden[0]
            elif type(hidden) == torch.Tensor:
                h_t_1 = hidden
            else:
                raise RuntimeError("h_t_1 is None")
            a_t, e_t = self.attn(x_t, h_t_1[-1, None, :, :])
        else:
            a_t, e_t = self.attn(x_t, hidden=None)
        o_t, hidden = self.rnn(a_t, hidden)
        o_t_initial = o_t
        for hl, lm in zip(self.hls, self.lms):
            o_t = self.dropout(lm(F.leaky_relu(hl(o_t))))
        o_t = self.f_out(o_t + o_t_initial)
        return o_t, hidden, e_t


class AttnFFN(nn.Module):
    def __init__(self, d_x, d_h, d_o, ffn_layers=5, dp=0.5):
        super(AttnFFN, self).__init__()
        self.f_in = nn.Linear(in_features=d_x,out_features=d_h)
        self.attn = Attention(d_h, d_h, temporal=False)
        self.hiddens = nn.ModuleList([nn.Linear(in_features=d_h, out_features=d_h) for _ in range(ffn_layers)])
        self.lms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(ffn_layers)])
        self.f_out = nn.Linear(in_features=d_h, out_features=d_o)
        self.dropout = nn.Dropout(p=dp)
    
    def forward(self, x_t, hidden=None):
        # use hidden here for compatible
        o_t=self.f_in(x_t)
        o_t, e_t = self.attn(o_t)
        o_t_initial = o_t
        for hl, lm in zip(self.hiddens, self.lms):
            o_t = self.dropout(lm(F.leaky_relu(hl(o_t))))
        o_t = self.f_out(o_t + o_t_initial)
        return o_t, hidden, e_t


class AttentionClassifier(object):
    def __init__(self, d_x, d_h, d_o, attn_model, lr=1e-3):
        super(AttentionClassifier, self).__init__()
        self.d_h = d_h
        self.attn_model = attn_model
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.attn_model.parameters()), lr=lr)
        self.tmp_hidden = None
        for p in self.attn_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def train(self, X, y):
        self.attn_model.train(True)
        self.attn_model.attn.train(True)
        self.optimizer.zero_grad()
        loss = 0
        y_hats = []
        attns = []
        for i, x in enumerate(X):
            y_hat_t, self.tmp_hidden, e_t = self.attn_model(
                torch.tensor(x[None, :, :], dtype=torch.float32),
                self.tmp_hidden
            )
            y_hat_t = y_hat_t.squeeze(0)
            y_true_t = torch.tensor([y[i]], dtype=torch.long)
            loss += self.loss_func(y_hat_t, y_true_t)
            topv, topi = y_hat_t.topk(1)
            y_hat = topi.detach().item()
            y_hats.append(y_hat)
            attns.append(e_t.view(-1).detach().numpy())
        loss = loss / len(X)
        loss.backward()
        clip_grad_norm_(self.attn_model.parameters(), 1)
        self.optimizer.step()
        y_hat = np.array(y_hats)
        if type(self.tmp_hidden) == tuple:
            self.tmp_hidden = tuple([h.detach() for h in self.tmp_hidden])
        elif type(self.tmp_hidden) == torch.Tensor:
            self.tmp_hidden = self.tmp_hidden.detach()
        else:
            self.tmp_hidden = None
        acc = accuracy_score(y_true=y.flatten(), y_pred=y_hat)
        mcc = matthews_corrcoef(y_true=y.flatten(), y_pred=y_hat)
        return loss.item(), acc, mcc, y_hat, attns
    
    def test(self, X, y):
        self.attn_model.eval()
        self.attn_model.attn.eval()
        with torch.no_grad():
            loss = 0
            y_hats = []
            attns = []
            for i, x in enumerate(X):
                y_hat_t, self.tmp_hidden, e_t = self.attn_model(torch.tensor(x[None, :, :], dtype=torch.float32), self.tmp_hidden)
                y_hat_t = y_hat_t.squeeze(0)
                y_t = torch.tensor([y[i]], dtype=torch.long)
                loss += self.loss_func(y_hat_t, y_t)
                topv, topi = y_hat_t.topk(1)
                y_hat = topi.detach().item()
                y_hats.append(y_hat)
                attns.append(e_t.view(-1).detach().numpy())
            loss = loss / len(X)
            y_hat = np.array(y_hats)
            acc = accuracy_score(y_true=y.flatten(), y_pred=y_hat)
            mcc = matthews_corrcoef(y_true=y.flatten(), y_pred=y_hat)
            return loss.item(), acc, mcc, y_hat, attns
    
    def inference(self, X):
        self.attn_model.eval()
        self.attn_model.attn.eval()
        with torch.no_grad():
            y_hats = []
            attns = []
            for i, x in enumerate(X):
                y_hat_t, self.tmp_hidden, e_t = self.attn_model(torch.tensor(x[None, :, :], dtype=torch.float32), self.tmp_hidden)
                y_hat_t = y_hat_t.squeeze(0)
                topv, topi = y_hat_t.topk(1)
                y_hat = topi.detach().item()
                y_hats.append(y_hat)
                attns.append(e_t.view(-1).detach().numpy())
            y_hat = np.array(y_hats)
            return y_hat, attns
    
    def reset_model(self):
        self.tmp_hidden = None
    
    def load_model(self, model_path='./AttnModel'):
        self.attn_model = torch.load(model_path + '/model.pkl')
    
    def save_model(self, model_path='./AttnModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.attn_model, model_path + '/model.pkl')
