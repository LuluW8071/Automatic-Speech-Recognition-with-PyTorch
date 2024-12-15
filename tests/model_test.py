import sys
import torch
from os.path import join, dirname
from torch import nn
from pathlib import Path

sys.path.append(join(dirname(__file__), '../neuralnet'))
from model import SpeechRecognition

# NOTE: Define Model Hyperparameters
h_params = {
    "num_classes": 29,
    "n_feats": 128,
    "dropout": 0.1,
    "hidden_size": 1024,
    "num_layers": 2
}

def model_test(**kwargs):
    # Merge default h_params with any overrides
    params = {**h_params, **kwargs}
    model = SpeechRecognition(**params)
    hidden = model._init_hidden(1)     # batch_size

    # Pass random input to check the model forward pass
    x = torch.rand(1, params['n_feats'], 300)  # (batch_size, time_seq, n_feats)
    return model(x, (hidden))

# Test the model
output, hn = model_test()
print(f"Output:\n{output}\n")
print(f"Shape of Output:\n{output.shape}")