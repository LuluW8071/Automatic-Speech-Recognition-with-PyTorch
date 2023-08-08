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
            use `create_commonvoice.py` to convert from mp3 to wav and to create train and test json's with the data from Commonvoice by Mozilla
    ```
    py create_commonvoice.py` --file_path "file_path\to\.csv" --save_json_path "save\json\path" --audio "audio\src_path\clips\to\.mp3" --percent 10 --convert

    ```
    ##### Note:
    - It is better to give absolute path instead for training the input 

    - if you dont want to convrt use `create_jsons_only.py`
    <br>
    This will create a train and test json in this format...

    ```
    // make each sample is on a seperate line
    {"key": "/path/to/audio/speech.wav", "text": "this is yourtext"}
    {"key": "/path/to/audio/speech.wav", "text": "another textexample"}
    ```
   - if you want to check the converted files and jsons use `check_converted_files`
