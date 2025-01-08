import comet_ml
import pytorch_lightning as pl
import os 
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torchmetrics.text import WordErrorRate, CharErrorRate
from collections import OrderedDict

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from utils import GreedyDecoder

# Model Imports
# from models.lstm import SpeechRecognition
from models.gru import SpeechRecognition


class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.losses = []
        self.val_wer, self.val_cer = [], []
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()

        # CTC Loss for CTC-based ASR
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)
        
        # Precompute sync_dist for distributed GPUs training
        self.sync_dist = True if args.gpus > 1 else False

        # Save the hyperparams of checkpoint
        self.save_hyperparameters(ignore=["model"])

    # Recall Forward pass of the model
    def forward(self, x, hidden):
        return self.model(x, hidden)

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # ReduceLROnPlateau with threshold for small changes
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.6,           # Reduce LR by multiplying it by 0.8
                patience=1,           # No. of epochs to wait before reducing LR
                threshold=3e-2,       # Minimum change in val_loss to qualify as improvement
                threshold_mode='rel', # Relative threshold (e.g., 0.1% change)
                min_lr=1e-5           # Minm. LR to stop reducing
            ),
            'monitor': 'val_loss',    # Metric to monitor
            'interval': 'epoch',      # Scheduler step every epoch
            'frequency': 1            # Apply scheduler after every epoch
        }

        return [optimizer], [scheduler]
    
    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        
        # Initialize hidden state
        hidden = self.model._init_hidden(bs)

        # NOTE: Pass (spectrograms, (hidden state) through the GRU model 
        # ========================
        hn = hidden.to(self.device)
        output, _ = self(spectrograms, hn)
        # ========================
            
        # NOTE: Pass (spectrograms, (hidden state, cell state)) through the LSTM model
        # ========================
        # hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        # output, _ = self(spectrograms, (hn, c0))
        # ========================

        output = F.log_softmax(output, dim=2)  # (time, batch, num_classes)

        # Compute CTC loss
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._common_step(batch, batch_idx)  

        # Log the train loss in the logger and progress bar
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # # Log the formatted targets and predictions in CometML text file
        if batch_idx % 1000 == 0:
            formatted_log = []
            
            # Loop through the targets and predictions, formatting each pair
            for i in range(len(decoded_targets)):
                formatted_log.append(f"{decoded_targets[i]}, {decoded_preds[i]}")
            log_text = "\n".join(formatted_log)
            self.logger.experiment.log_text(text=log_text)

        # Calculate CER and WER metrics
        cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        self.val_cer.append(cer_batch)
        self.val_wer.append(wer_batch)
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        # Calculate average of metrics over the entire epoch at end of validation
        avg_loss = torch.stack(self.losses).mean()
        avg_cer = torch.stack(self.val_cer).mean()
        avg_wer = torch.stack(self.val_wer).mean()

        # Log all metrics in the logger and progress bar
        metrics = {
            'val_loss': avg_loss,
            'val_cer': avg_cer,
            'val_wer': avg_wer
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear the lists for the next epoch
        self.losses.clear()
        self.val_wer.clear()
        self.val_cer.clear()


def main(args):
    # Prepare dataset
    data_module = SpeechDataModule(batch_size=args.batch_size,
                                   train_json=args.train_json,
                                   test_json=args.valid_json, 
                                   num_workers=args.num_workers)
    data_module.setup()

    # NOTE: Setup hyperparams of model
    h_params = {
        "num_classes": 29,
        "n_feats": 128,
        "dropout": 0.15,
        "hidden_size": 768,
        "num_layers": 2
    }

    
    model = torch.compile(SpeechRecognition(**h_params))
    speech_trainer = ASRTrainer(model=model, args=args) 

    # NOTE: Comet Logger
    comet_logger = CometLogger(
        api_key=os.getenv('API_KEY'), 
        project_name=os.getenv('PROJECT_NAME')
    )

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",       
        filename='model-{epoch:02d}-{val_loss:.3f}-{val_wer:.3f}',                                             
        save_top_k=3,
        mode='min'
    )

    # Trainer Parameters
    trainer_args = {
        'accelerator': args.device,                                     # Device to use for training
        'devices': args.gpus,                                           # Number of GPUs to use for training
        'min_epochs': 1,                                                # Minm. no. of epochs to run
        'max_epochs': args.epochs,                                      # Maxm. no. of epochs to run                               
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'gradient_clip_val': args.grad_clip,                            # Gradient norm clipping value
        'accumulate_grad_batches': args.accumulate_grad,                # No. of batches to accumulate gradients over
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=5),
                      checkpoint_callback],
        'logger': comet_logger,                                         # Logger to use for training
    }
    
    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend
    trainer = pl.Trainer(**trainer_args)

    # Fit the model to the train and val. data
    trainer.fit(speech_trainer, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(speech_trainer, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-d', '--device', default='cuda', type=str, help='device to use for training')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('-lr','--learning_rate', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--precision', default='32-true', type=str, help='precision')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path of checkpoint file to resume training')
    parser.add_argument('-gc', '--grad_clip', default=0.8, type=float, help='gradient norm clipping value')
    parser.add_argument('-ag', '--accumulate_grad', default=2, type=int, help='number of batches to accumulate gradients over')

    args = parser.parse_args()
    main(args)