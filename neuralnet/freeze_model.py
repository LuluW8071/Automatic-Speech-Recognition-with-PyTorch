import argparse
import torch
from collections import OrderedDict

from models.lstm import SpeechRecognition
# from models.gru import SpeechRecognition


def trace(model):
    model.eval()
    x = torch.rand(1, 128, 300)     # batch_size, feature, time
    hidden = model._init_hidden(1)  # batch_size: 1
    traced = torch.jit.trace(model, (x, hidden))
    return traced

def main(args):
    print("Loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))

    # NOTE: Define Model Hyperparameters
    h_params = {
        "num_classes": 29,
        "n_feats": 128,
        "dropout": 0.1,
        "hidden_size": 512,
        "num_layers": 2
    }
    model = SpeechRecognition(**h_params)

    # Load model state dict weights from checkpoint
    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model._orig_mod.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # Use Torchscript to trace and save the model
    print("tracing model...")
    traced_model = trace(model)
    traced_model.save('optimized_model.pt')
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="freeze model checkpoint")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True, help='Checkpoint of model to optimize')
    args = parser.parse_args()

    main(args)