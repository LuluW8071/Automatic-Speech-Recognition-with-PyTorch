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
        # x = self.norm(self.dropout(F.gelu(x)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 81,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
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
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1)  # batch, feature, time
        x = self.cnn(x)  # batch, time, feature
        x = self.dense(x)  # batch, time, feature
        x = x.transpose(0, 1)  # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out))
                          )  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)

    # Model Summarization        
    def print_detailed_summary(self):
        print(f"{'='*80}\n{'Model Summary':^80}\n{'='*80}")

        header_format = "{:<5} | {:<30} | {:<20} | {:<10}"
        row_format = "{:<5} | {:<30} | {:<20} | {:<10}"

        print(header_format.format("No.", "Layer", "Module", "Parameters"))
        print('='*80)

        counter = 1
        for name, module in self.named_children():
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            clean_name = name.replace('\t', '').replace(' ', '')  
            print(row_format.format(counter, f"model.{clean_name}", module.__class__.__name__, trainable_params))

            if hasattr(module, 'named_children'):
                for child_name, child_module in module.named_children():
                    trainable_params_child = sum(p.numel() for p in child_module.parameters() if p.requires_grad)
                    full_child_name = f"model.{name}.{child_name}".replace('\t', '').replace(' ', '') 
                    print(row_format.format('', full_child_name, child_module.__class__.__name__, trainable_params_child))
            counter += 1

        print('='*80)
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Trainable Params: {total_trainable_params} ~ {round(total_trainable_params / 1e6, 1)}M")





