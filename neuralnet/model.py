import torch
import torch.nn as nn
from torch.nn import functional as F


class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,  # output_class
        "n_feats": 128,     # n_mels
        "dropout": 0.2,
        "hidden_size": 768,
        "num_layers": 2     # RNN Layers
    }

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, kernel_size=5, stride=1, padding=5//2),
            ActDropNormCNN1D(n_feats, dropout, keep_shape=True),
            nn.Conv1d(n_feats, n_feats, kernel_size=5, stride=2, padding=5//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)         # x2 for bidirectionality
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size * 2, num_classes)  # x2 for bidirectionality

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs)                # x2 for bidirectional

    def forward(self, x, hidden):
        x = x.squeeze(1)                                  # batch, feature, time
        x = self.cnn(x)                                   # batch, time, feature
        x = self.dense(x)                                 # batch, time, feature
        x = x.transpose(0, 1)                             # time, batch, feature
        out, hn = self.gru(x, hidden)
        out = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        out = self.final_fc(out)
        return out, hn
