## Speech Recognition with CNN LSTM Acoustic Model and CTC Decoder + KENLM Language Model
This repository contains an implementation of a Speech Recognition system using a combination of a Convolutional Neural Network (CNN) LSTM Acoustic Model and a Connectionist Temporal Classification (CTC) Decoder, enhanced with a KENLM Language Model.

### Overview
Speech recognition is the process of converting spoken language into written text. The system built here employs a CNN LSTM Acoustic Model to extract features from the audio input, which are then used to predict character sequences. The CTC Decoder is utilized to map the predicted character sequences to the most likely word sequences. Finally, a KENLM Language Model is incorporated to improve the accuracy of the word-level predictions.

#### Dependencies
### [NVIDIA Developer](https://developer.nvidia.com/cuda-11-7-0-download-archive)
Allows developers to harness the computational power of NVIDIA GPUs to accelerate diverse applications ranging from scientific simulations to artificial intelligence and machine learning tasks.

### [PyTorch](https://pytorch.org/)
[![PyTorch Icon](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)](https://pytorch.org/)

```
#For CUDA 11.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

#For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
