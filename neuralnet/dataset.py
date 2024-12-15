import pytorch_lightning as pl
import json
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader, Dataset
from utils import TextTransform       # Comment this for engine inference


class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, hop_length=160):
        super(LogMelSpec, self).__init__()
        self.transform = transforms.MelSpectrogram(sample_rate=sample_rate, 
                                                   n_mels=n_mels,
                                                   hop_length=hop_length)

    def forward(self, x):
        return self.transform(x)  # mel spectrogram

# MelSpec Feature Extraction for ASR Engine Inference
def get_featurizer(sample_rate=16000, n_mels=128, hop_length=160):
    return LogMelSpec(sample_rate=sample_rate, 
                      n_mels=n_mels,
                      hop_length=hop_length)

# Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, json_path, transform=None, log_ex=True, valid=False):
        print(f'Loading json data from {json_path}')
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Initialize TextProcess for text processing    
        self.text_process = TextTransform()                 
        self.log_ex = log_ex

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec()
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(),
                transforms.FrequencyMasking(freq_mask_param=15),
                transforms.TimeMasking(time_mask_param=27)
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item['key']

        try:
            waveform, _ = torchaudio.load(file_path)        # Point to location of audio data
            utterance = item['text'].lower()                # Point to sentence of audio data
            label = self.text_process.text_to_int(utterance)
            spectrogram = self.audio_transforms(waveform)   # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s' %file_path)
            if spectrogram.shape[2] > 2650*4:
                raise Exception('spectrogram to big. size %s' %spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s' %file_path)

            return spectrogram, label, spec_len, label_len

        except Exception as e:
            # Print for debugging if letters in sentences have transform issues
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        
    def describe(self):
        return self.data.describe()
    

# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_json, test_json, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_json,
                                                valid=False)
        self.test_dataset = CustomAudioDataset(self.test_json, 
                                               valid=True)
        
    def data_processing(self, data):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (spectrogram, label, input_length, label_length) in data:
            if spectrogram is None:
                continue

            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(torch.Tensor(label))
            input_lengths.append(input_length)
            label_lengths.append(label_length)

        # NOTE: https://www.geeksforgeeks.org/how-do-you-handle-sequence-padding-and-packing-in-pytorch-for-rnns/
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          collate_fn=lambda x: self.data_processing(x), 
                          num_workers=self.num_workers, 
                          pin_memory=True)      # Optimizes data-transfer speed

    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn=lambda x: self.data_processing(x), 
                          num_workers=self.num_workers, 
                          pin_memory=True)