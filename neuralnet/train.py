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

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from model import SpeechRecognition
from utils import GreedyDecoder


class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.losses = []
        self.val_wer, self.val_cer = [], []
        # self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)
        
        # Precompute sync_dist for distributed GPUs training
        self.sync_dist = True if args.gpus > 1 else False

        # Save the hyperparams of checkpoint
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x, hidden):
        return self.model(x, hidden)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=15,                          # Number of epochs for the first restart
                T_mult=self.args.lr_step_size,   # Factor to increase T_0 after each restart
                eta_min=2e-5                     # Minimum learning rate
            ),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
    
    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        hn =  self.model._init_hidden(spectrograms.shape[0]).to(self.device)
        output, _ = self(spectrograms, hn)
        output = F.log_softmax(output, dim=2)
        
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # Log final predictions
        if batch_idx % 400 == 0:
            log_targets = "\n".join(decoded_targets)
            log_preds = "\n".join(decoded_preds)
            self.logger.experiment.log_text(text=log_targets, metadata=log_preds)

        # Calculate metrics
        # cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        
        # self.val_cer.append(cer_batch)
        self.val_wer.append(wer_batch)

        return {'val_loss': loss}


    def on_validation_epoch_end(self):
        # Calculate averages of metrics over the entire epoch
        avg_loss = torch.stack(self.losses).mean()
        # avg_cer = torch.stack(self.val_cer).mean()
        avg_wer = torch.stack(self.val_wer).mean()

        # Log all metrics using log_dict
        metrics = {
            'val_loss': avg_loss,
            # 'val_cer': avg_cer,
            'val_wer': avg_wer
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear the lists for the next epoch
        self.losses.clear()
        self.val_wer.clear()
        # self.val_cer.clear()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare dataset
    data_module = SpeechDataModule(batch_size=args.batch_size,
                                   train_json=args.train_json,
                                   test_json=args.valid_json, 
                                   num_workers=args.num_workers)
    data_module.setup()

    # Log hyperparams of model and setup trainer
    h_params = SpeechRecognition.hyper_parameters
    model = SpeechRecognition(**h_params)
    speech_trainer = ASRTrainer(model=model, args=args) 

    # NOTE: Comet Logger
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'),
                               project_name=os.getenv('PROJECT_NAME'))

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',                                 # Metric to monitor for checkpointing
        dirpath="./saved_checkpoint/",                      # Directory to save checkpoints         
        filename='ASR-{epoch:02d}-{val_wer:.2f}',                                             
        save_top_k=3,                                       # Number of best checkpoints to save
        mode='min'                                          # Minimum metric to save checkpoints
    )

    # Trainer Instance
    trainer_args = {
        'accelerator': device,                                          # Device to use for training
        'devices': args.gpus,                                           # Number of GPUs to use for training
        'min_epochs': 1,                                                # Minimum number of epochs to run
        'max_epochs': args.epochs,                                      # Maximum number of epochs to run                               
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # Number of epochs to run validation
        'gradient_clip_val': 25.0,                                      # Gradient clipping value
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=5),    # Early stopping 
                      checkpoint_callback],
        'logger': comet_logger
    }
    
    # Distributed training for > 1 GPUs
    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend
        
    trainer = pl.Trainer(**trainer_args)
    
    # Fit the model to the training data using the Trainer's fit method.
    ckpt_path = args.checkpoint_path if args.checkpoint_path else None
    
    trainer.fit(speech_trainer, data_module, ckpt_path=ckpt_path)   # Train the model: load the checkpoint if given and resume training
    trainer.validate(speech_trainer, data_module)                   # Validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('-lr','--learning_rate', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    parser.add_argument('--lr_step_size', type=int, default=2, help='Number of epochs for step decay')
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')

    args = parser.parse_args()
    main(args)
