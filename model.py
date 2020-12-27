__author__ = 'YaelSegal'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from utils import LambdaLayer



class CnnLstmRaw(nn.Module):
    def __init__(self, ninput=63, channels=256, nhid=100, nlayers=1, dropout=0.5, num_classes=3, bidirectional=False):
        super(CnnLstmRaw, self).__init__()

        self.enc = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=512, stride=4, padding=254, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=16, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=4, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, ninput, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(ninput),
            nn.LeakyReLU(),
            LambdaLayer(lambda x: x.transpose(1,2)),
        )

        
        self.drop = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.rnn = nn.LSTM(input_size=ninput, hidden_size=nhid, num_layers=nlayers, bidirectional=bidirectional,
                           dropout=dropout, batch_first=True)

        self.hidden_size = nhid
        self.num_layers = nlayers
        self.linear_hid = nn.Linear(nhid * self.directions, nhid)

        self.linear_hid_activ = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.linear_hid2 = nn.Linear(nhid, num_classes)
        self.linear_hid_activ2 = nn.Softmax(dim=1)

        self.seq = nn.Sequential(self.drop2, self.linear_hid2, self.linear_hid_activ2)


    def forward(self, raw_input, hidden, seq_len_list):

        # z = raw_input.unsqueeze(1)
        # for m in self.enc.children():
        #     z = m(z)
        enc_input = self.enc(raw_input.unsqueeze(1))
        input_packed = torch.nn.utils.rnn.pack_padded_sequence(enc_input, seq_len_list, batch_first=True, enforce_sorted=False)
        output_pack, hidden = self.rnn(input_packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_pack, batch_first=True)
        batch_size, seq_len, features = output.size()
        final_out = torch.FloatTensor(batch_size, max(seq_len_list), self.num_classes).to(output.device)
        for batch_idx, batch_output in enumerate(output):
            # h2 = output.contiguous().view(seq_len * batch_size, -1)  # reshape
            cur_seq_len = seq_len_list[batch_idx]
            h2 = batch_output[:cur_seq_len]
            h2 = self.drop(h2)

            h3 = self.linear_hid(h2)  # linear transform
            h3 = self.linear_hid_activ(h3)
            final = self.seq(h3)
            final_out[batch_idx,:cur_seq_len] = final

        return final_out, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.FloatTensor(self.num_layers * self.directions, batch_size, self.hidden_size)
        c0 = torch.FloatTensor(self.num_layers * self.directions, batch_size, self.hidden_size)
        # nn.init.xavier_uniform_(h0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(c0, gain=np.sqrt(2))
        nn.init.uniform_(h0, -0.05, 0.05)
        nn.init.uniform_(c0, -0.05, 0.05)

        h0 = h0.to(device)
        c0 = c0.to(device)
        return (h0, c0)


def load_model(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    params = checkpoint['params']

    model = CnnLstmRaw(ninput=params["input_size"], channels=params["channels"], nhid=params["hidden_size"], nlayers=params["num_layers"], dropout=0, bidirectional=params["biLSTM"], num_classes=params["class_num"])
    normalize = params["normalize"]
    model.load_state_dict(checkpoint["net"])

    return model, normalize