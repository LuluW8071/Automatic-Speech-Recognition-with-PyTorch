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
    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 3, 2, padding=3//2),
            ActDropNormCNN1D(n_feats, dropout)
        )
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            bidirectional=True,
            batch_first=False
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size * 2, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs)  # *2 for bidirectional

    def forward(self, x, hidden):
        # x shape: (batch, channel, feature, time)
        batch_size = x.size(0)
        x = x.squeeze(1)                                            # (batch, feature, time)
        
        x = self.cnn(x)                                             # CNN: (batch, time, feature)

        x = self.dense(x)                                           # Dense layers: (batch, time, 128)
        x = x.transpose(0, 1)                                       # (time, batch, 128)

        gru_out, hidden = self.gru(x, hidden)                       # BiGRU: (time, batch, hidden*2)
        
        output = self.dropout(F.gelu(self.layer_norm(gru_out)))
        output = self.final_fc(output)                              # (time, batch, num_classes)
        
        return output, hidden
