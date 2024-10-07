import torch
import torch.nn as nn
from torch.nn import functional as F


class ActDropNormCNN1D(nn.Module):
    """
    1D CNN with activation, dropout, and layer normalization.

    Parameters:
        - in_channels (int): Number of input channels.
        - keep_shape (bool): If True, preserves the shape of the input tensor.

    Input Shape:
        (batch_size, in_channels, sequence_length)

    Output Shape:
        (batch_size, in_channels, sequence_length)

    Architecture:
        - Conv1d with kernel size 10 and stride 2.
        - GELU activation function.
        - Dropout with probability 0.1.
        - LayerNorm normalization.
    """
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
    """
    Speech Recognition model with CNN and LSTM layers.

    Hyperparameters:
        - num_classes (int): Number of output classes.
        - n_feats (int): Number of input features.
        - dropout (float): Dropout probability.
        - hidden_size (int): Size of the LSTM hidden layer.
        - num_layers (int): Number of LSTM layers.

    Input Shape:
        (batch_size, 1, feature, time)

    Output Shape:
        - Logits: (time, batch_size, num_classes)
        - Hidden State Tuple: (hn, cn)

    Architecture:
        - 1D CNN with activation, dropout, and layer normalization.
        - Dense layers with linear, layer normalization, GELU activation, and dropout.
        - Unidirectional LSTM layer.
        - Layer normalization and dropout.
        - Final fully connected layer.
    """

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout))
        self.dense = nn.Sequential(nn.Linear(n_feats, 128),
                                   nn.LayerNorm(128),
                                   nn.GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(128, 128),
                                   nn.LayerNorm(128),
                                   nn.GELU(),
                                   nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0,
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1)                                    # batch, feature, time
        x = self.cnn(x)                                     # batch, time, feature
        x = self.dense(x)                                   # batch, time, feature
        x = x.transpose(0, 1)                               # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))    # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)