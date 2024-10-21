import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # NOTE: Adjust residual connection to match output size if input and output channels differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += residual              # Adding the residual connection
        return self.silu(out)


class SpeechRecognition(nn.Module):
    def __init__(self, hidden_size, num_classes, n_feats, num_layers, layer_size, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        
        # CNN layers with residual blocks and downsampling
        self.cnn = nn.Sequential(
            ResidualBlock(n_feats, n_feats, kernel_size=3, stride=1, dropout=dropout),
            ResidualBlock(n_feats, n_feats * 2, kernel_size=3, stride=2, dropout=dropout),
            ResidualBlock(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, dropout=dropout)
        )
                
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(n_feats * 2, self.layer_size),
            nn.LayerNorm(self.layer_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LayerNorm(self.layer_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.layer_size,
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

        # Weight Initialization
        self._init_weights()

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs)  # *2 for bidirectional

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)

    def forward(self, x, hidden):
        # x shape: (batch, channel, feature, time)
        # batch_size = x.size(0)
        x = x.squeeze(1)                            # (batch, feature, time)

        # Pass through CNN (Residual Blocks)
        x = self.cnn(x)                             # CNN: (batch, time, feature)

        # Dense layers
        x = x.transpose(1, 2)                       # (batch, feature, time) --> (batch, time, feature)
        x = self.dense(x)                           # (batch, time, dense_layer)

        # Reorder dimensions for GRU: (time, batch, 128)
        x = x.transpose(0, 1)                       # (batch, time, dense_layer) --> (time, batch, dense_layer)

        # Pass through BiGRU
        gru_out, hidden = self.gru(x, hidden)       # BiGRU: (time, batch, hidden*2)
        
        # Apply LayerNorm, Dropout, and Final Fully Connected Layer
        output = self.dropout(F.gelu(self.layer_norm(gru_out)))
        output = self.final_fc(output)              # (time, batch, num_classes)
        
        return output, hidden
