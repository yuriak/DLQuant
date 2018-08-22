# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import os
import pickle

writer = SummaryWriter(log_dir='./logs/ELMO')


class CharacterEmbedding(nn.Module):
    def __init__(self, weight_matrix, dropout, filters=[32, 32, 32, 32], kernel_sizes=[2, 3, 4, 5]):
        super(CharacterEmbedding, self).__init__()
        input_size = weight_matrix.shape[1]
        self.output_size = sum(filters)
        self.embedding = nn.Embedding(input_size, input_size, _weight=weight_matrix)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_size, out_channels=filters[i], kernel_size=(1, kernel_sizes[i])) for i in range(len(filters))])
        self.highway_h = nn.Linear(sum(filters), sum(filters))
        self.highway_t = nn.Linear(sum(filters), sum(filters))
        self.dropout = dropout
    
    def conv_and_pool(self, x, conv):
        b, w, c, e = tuple(x.shape)
        x_out = conv(x.view(b, e, w, c)).max(dim=-1)[0]
        return x_out.view(b, w, -1)
    
    def highway(self, x):
        x_h = F.relu(self.highway_h(x))
        x_t = F.sigmoid(self.highway_t(x))
        x_out = x_h * x_t + x * (1 - x_t)
        return x_out
    
    def forward(self, x, train=False):
        x = self.embedding(x)
        results = list(map(lambda conv: self.conv_and_pool(x, conv), self.convs))
        results = torch.cat(results, dim=-1)
        results = self.highway(results)
        if train:
            results = self.dropout(results)
        return results


class BiLM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, word_number, dropout):
        super(BiLM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fw_fc = nn.Linear(hidden_size, word_number)
        self.bw_fc = nn.Linear(hidden_size, word_number)
        self.dropout = dropout
    
    def forward(self, x, hidden=None, train=False):
        out = x
        if train:
            out = self.dropout(out)
        out, (h_n, h_c) = self.lstm(out, hidden)
        out_fw = out[:, :, :self.hidden_size]
        out_bw = out[:, :, self.hidden_size:]
        out_fw = F.log_softmax(F.relu(self.fw_fc(out_fw)), dim=-1)
        out_bw = F.log_softmax(F.relu(self.bw_fc(out_bw)), dim=-1)
        return out_fw, out_bw, (h_n, h_c)


class Elmo(object):
    def __init__(self, weight_matrix, word_number, hidden_size=64, num_layers=2, learning_rate=1e-3, dp=0.2):
        super(Elmo, self).__init__()
        self.dropout = nn.Dropout(p=dp)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.char_emb = CharacterEmbedding(weight_matrix=torch.tensor(weight_matrix, dtype=torch.float32), dropout=self.dropout)
        self.bilm = BiLM(input_size=self.char_emb.output_size, hidden_size=self.hidden_size, num_layers=self.num_layers, word_number=word_number, dropout=self.dropout)
        self.optimizer = optim.Adam(list(self.char_emb.parameters()) + list(self.bilm.parameters()), lr=learning_rate)
        self.criterion = nn.NLLLoss()
    
    def _encode(self, x, gamma=1):
        # weight vector of every layer not implemented!
        max_word_length = (((x > 0).sum(axis=-1)) > 0).sum(axis=1).max()
        x_w = x[:, :max_word_length, :]
        x_w = self.char_emb(torch.tensor(x_w), train=False)
        _, _, (hn, hc) = self.bilm(torch.tensor(x_w), train=False)
        hns = []
        hcs = []
        for i in range(0, self.num_layers * 2, 2):
            hns.append(torch.cat([hn[i, :, :], hn[i + 1, :, :]], dim=-1))
            hcs.append(torch.cat([hc[i, :, :], hc[i + 1, :, :]], dim=-1))
        # same weight for each layer
        hns = torch.stack(hns).mean(0)
        hcs = torch.stack(hcs).mean(0)
        
        encode_result = torch.cat([hns, hcs], dim=-1)
        return encode_result.detach().numpy()
    
    def encode(self, X, batch_size=64, gamma=1):
        pointer = 0
        results_ = np.zeros((1, self.hidden_size * 4))
        while pointer < X.shape[0]:
            batch_x = X[pointer:(pointer + batch_size)]
            result_batch = self._encode(batch_x, gamma=gamma)
            results_ = np.concatenate((results_, result_batch))
            pointer += batch_size
        return results_[1:]
    
    def _train(self, x, y):
        max_word_length = (y > 0).sum(axis=1).max()
        batch_x = x[:, :max_word_length, :]
        batch_y = y[:, :max_word_length]
        self.optimizer.zero_grad()
        y_f = batch_y[:, 2:]
        y_b = batch_y[:, :-2]
        x_w = self.char_emb(torch.tensor(batch_x), train=True)
        y_hat_f, y_hat_b, (_, _) = self.bilm(x_w, train=True)
        y_hat_f = y_hat_f[:, :-2]
        y_hat_b = y_hat_b[:, :-2]
        loss_f = self.criterion(y_hat_f.transpose(1, -1), torch.tensor(y_f))
        loss_b = self.criterion(y_hat_b.transpose(1, -1), torch.tensor(y_b))
        loss = (loss_f + loss_b) / 2
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, X, y, batch_size=64, epoch=2):
        global_step = 0
        for e in range(epoch):
            pointer = 0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            while pointer < X.shape[0]:
                batch_x = X_shuffled[pointer:(pointer + batch_size)]
                batch_y = y_shuffled[pointer:(pointer + batch_size)]
                mean_loss = self._train(batch_x, batch_y)
                print(mean_loss, 'batch%:', round((pointer / X.shape[0]) * 100, 4), 'epoch:', e)
                pointer += batch_size
                writer.add_scalar(tag='loss', scalar_value=mean_loss, global_step=global_step)
                global_step += 1
            self.save_model()
    
    def save_model(self, model_path='./ELMO'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        print("saving models")
        torch.save(self.char_emb, model_path + '/char_emb.pkl')
        torch.save(self.bilm, model_path + '/bilm.pkl')
    
    def load_model(self, model_path='./ELMO'):
        print("loading models")
        self.char_emb = torch.load(model_path + '/char_emb.pkl')
        self.bilm = torch.load(model_path + '/bilm.pkl')


if __name__ == '__main__':
    with open('./data/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    with open('./data/characters.pkl', 'rb') as f:
        characters = pickle.load(f)
    corpus_tokens = np.load('./data/corpus_tokens.npy')
    corpus_chars = np.load('./data/corpus_chars.npy')
    assert corpus_chars.shape[0] == corpus_tokens.shape[0]
    onehot = np.eye(len(characters))
    onehot[0, 0] = 0
    elmo = Elmo(weight_matrix=onehot, word_number=len(vocabulary), learning_rate=1e-3)
    elmo.train(X=corpus_chars, y=corpus_tokens, batch_size=128, epoch=2)
    results = elmo.encode(X=corpus_chars, batch_size=64, gamma=1)
    np.save('data/encoded_title_elmo', results)
