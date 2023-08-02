## Speech Recognition with CNN LSTM Acoustic Model and CTC Decoder + KENLM Language Model
This repository contains an implementation of a Speech Recognition system using a combination of a Convolutional Neural Network (CNN) LSTM Acoustic Model and a Connectionist Temporal Classification (CTC) Decoder, enhanced with a KENLM Language Model.

#### Dependencies
[NVIDIA Developer Toolkit](https://developer.nvidia.com/cuda-11-7-0-download-archive)

[PyTorch](https://pytorch.org/)

```
#For CUDA 11.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

#For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
[Pytorch-Lightning](https://www.pytorchlightning.ai/index.html)

[ffmpeg](https://www.ffmpeg.org/)
```
Extract the archive
>ffmpeg/bin/
Edit environment variables to insert path 
>path/to/ffmpeg/bin/
```

1. Data Collection <br>
    1.1 Collect your own data: <br> 
            use `create_commonvoice_jsons.py` to convert from mp3 to wav and to create train and test json's with the data from Commonvoice by Mozilla
    ```
    python create_commonvoice_jsons.py --file_path /path/to/commonvoice/file/.tsv --save_json_path /path/where/you/want/them/saved 
    ```

    if you dont want to convert use `--not-convert` 
    <br>
    This will create a train and test json in this format...
    ```
    // make each sample is on a seperate line
    {"key": "/path/to/audio/speech.wav, "text": "this is yourtext"}
    {"key": "/path/to/audio/speech.wav, "text": "another textexample"}
    ``` 
