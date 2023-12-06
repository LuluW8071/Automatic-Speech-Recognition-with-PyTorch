## Speech Recognition with CNN LSTM Acoustic Model and CTC Decoder + KENLM Language Model
This repository contains an implementation of a Speech Recognition system using a combination of a Convolutional Neural Network (CNN) LSTM Acoustic Model and a Connectionist Temporal Classification (CTC) Decoder, enhanced with a KENLM Language Model.

#### Dependencies
- [Python 3.10.0](https://www.python.org/downloads/release/python-3100/)
- [NVIDIA CUDA 12.1.0](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- [PyTorch [Stable 2.1.1]](https://pytorch.org/)
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
- [Pytorch-Lightning 2.1.0](https://www.pytorchlightning.ai/index.html)
- [ffmpeg](https://www.ffmpeg.org/)
    ```
    Extract the archive
        > ffmpeg/bin/
    Edit environment variables to insert path 
        > path/to/ffmpeg/bin/
    ```
#### Terminal Commands
*It is better to give absolute path instead of relative path*
<h5>1. Data Collection </h5>
Use `create_jsons_only.py` to convert from mp3 to wav and to createtrain and test json's with the data from Commonvoice by Mozilla </br>

```
py create_commonvoice.py` --file_path "file_path\to\.tsv" --save_json_path "save\json\path" --audio "audio\src_path\clips\to\.mp3" --percent 10 --convert
```
- If you dont want to convert use `create_jsons_only.py`

This should create a train and test json in this format:
 ```
 [
    {   
        "key": "/path/to/audio/speech.wav", 
        "text": "this is yourtext"
    },
    ......
 ]
 ```
<h5>2. Comet-ML API </h5>
Edit `config.py` with your <a href='https://www.comet.com/site/'> Comet.ml </a>API key and project name to get real-time loss curve plot </br>

```
API_KEY = "#"  
PROJECT_NAME = "#" 
```

<h5>3. Train Datasets </h5>

```
py train.py --train_file "path\train.json" --valid_file "path\test.json" --save_model_path 'save\model\path' --gpus 1    
```

*Set `--gpus` flag to either `0` or `1` depending upon device you have </br>[for mux switch:1 else 0]*

<h5>4. Resume Training </h5>

```
py train.py --train_file 'path\train.json' --valid_file 'path\test.json' --load_model_from 'path\model\best_model.ckpt' --resume_from_checkpoint 'path\model\' --save_model_path 'save\model\path' --gpus 1
```