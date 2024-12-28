import torch
import torch.nn as nn
from torch.nn import functional as F
from cnn_norm import ActDropNormCNN1D

class SpeechRecognition(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=29, n_feats=128, num_layers=2, dropout=0.1):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # CNN Layer + LayerNorm for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 3, 1, padding=3//2),
            ActDropNormCNN1D(n_feats, dropout, keep_shape=True),
            nn.Conv1d(n_feats, n_feats, 5, 2, padding=5//2),
            ActDropNormCNN1D(n_feats, dropout),
        )

        # Dense Layer + LayerNorm for feature filtering
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM Layer for sequence modeling
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=0.15, bidirectional=True)

        # LayerNorm for output of LSTM
        self.layer_norm2 = nn.LayerNorm(hidden_size*2)      # x2 for handling bidirectional
        self.dropout2 = nn.Dropout(dropout)

        # Final fully connected layer for classification
        self.final_fc = nn.Linear(hidden_size*2, num_classes)

    # Initialize the hidden state of the LSTM for live-streaming ASR output
    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*2, batch_size, hs),    
                torch.zeros(n*2, batch_size, hs))           # x2 for handling bidirectional

    # Forward pass of the model
    def forward(self, x, hidden):
        x = x.squeeze(1)                                    # batch, feature, time
        x = self.cnn(x)                                     # batch, time, feature
        x = self.dense(x)                                   # batch, time, feature
        x = x.transpose(0, 1)                               # time, batch, feature

        out, (hn, cn) = self.lstm(x, hidden)                
        x = self.dropout2(F.gelu(self.layer_norm2(out)))    # time, batch, n_class
        return self.final_fc(x), (hn, cn)
