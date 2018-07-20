# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

weight_matrix = torch.tensor(np.load('./data/title_weight_matrix.npy'), dtype=torch.float32)
sequence = torch.tensor(np.load('./data/title_sequence.npy'), dtype=torch.int64)


class Encoder(nn.Module):
    def __init__(self, dict_size, emb_size, hidden_size, emb_matrix):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, emb_size, _weight=emb_matrix)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, emb_size, dict_size, emb_matrix):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, emb_size, _weight=emb_matrix)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden):
        output = self.embedding(x)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden


class RNNAE(object):
    def __init__(self, weight_matrix, hidden_size):
        self.hidden_size = hidden_size
        self.encoder = Encoder(
            dict_size=weight_matrix.shape[0],
            emb_size=weight_matrix.shape[1],
            hidden_size=hidden_size,
            emb_matrix=weight_matrix)
        self.decoder = Decoder(
            dict_size=weight_matrix.shape[0],
            emb_size=weight_matrix.shape[1],
            hidden_size=hidden_size,
            emb_matrix=torch.tensor(weight_matrix, dtype=torch.float32))
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=1e-3)
        self.criterion = nn.NLLLoss()

    def _train(self, batch_x, sentence_length):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0
        encoder_output, encoder_hidden = self.encoder(batch_x, hidden=None)
        decoder_input = torch.ones(batch_x.shape[0], 1, dtype=torch.int64)
        decoder_hidden = encoder_hidden
        for i in range(sentence_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1, dim=-1)
            decoder_input = topi.squeeze(-1).detach()
            loss += self.criterion(decoder_output.squeeze(1), batch_x[:, i])
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / sentence_length

    def train(self, X, batch_size=64, epoch=10):
        for e in range(epoch):
            pointer = 0
            while pointer < X.shape[0]:
                batch_x = X[pointer:(pointer + batch_size)]
                mean_loss = self._train(batch_x, sentence_length=X.shape[1])
                print(mean_loss, round(pointer / X.shape[0], 2))
                pointer += batch_size

    def encode(self, X, batch_size=64):
        pointer = 0
        results = torch.zeros(1, self.hidden_size)
        while pointer < X.shape[0]:
            batch_x = X[pointer:(pointer + batch_size)]
            out, hidden = self.encoder(batch_x, hidden=None)
            results = torch.cat((results, hidden.squeeze(0)))
            pointer += batch_size
        return results[1:]

if __name__ == '__main__':
    rnnae = RNNAE(weight_matrix=weight_matrix, hidden_size=50)
    rnnae.train(X=sequence, batch_size=128, epoch=5)
    results = rnnae.encode(sequence, batch_size=256)
    np.save('encoded_title', results.detach().numpy())