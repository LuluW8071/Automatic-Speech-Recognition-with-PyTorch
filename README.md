## Speech Recognition with CNN LSTM Acoustic Model and CTC Decoder + KENLM Language Model
This repository contains an implementation of a Speech Recognition system using a combination of a Convolutional Neural Network (CNN) LSTM Acoustic Model and a Connectionist Temporal Classification (CTC) Decoder, enhanced with a KENLM Language Model.

### Dependencies
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
### Terminal Commands
*It is better to give absolute path instead of relative path*
<h3>1. Data Collection </h3>
Use `create_jsons_only.py` to convert from mp3 to wav and to create train and test JSON with the data from Commonvoice by Mozilla </br>

```
py create_commonvoice.py --file_path "file_path\to\.tsv" --save_json_path "save\json\path" --audio "audio\src_path\clips\to\.mp3" --percent 10 --convert
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
<h3>2. Comet-ML API </h3>
Edit `config.py` with your <a href='https://www.comet.com/site/'> Comet.ml </a> API key and project name to get real-time loss curve plot </br>

```
API_KEY = "###"  
PROJECT_NAME = "###" 
```

<h3>3. Train Datasets </h3>

```
py train.py --train_file "path\train.json" --valid_file "path\test.json" --save_model_path 'save\model\path' --gpus 1    
```

*Set `--gpus` flag to either `0` or `1` depending upon device you have </br>[for mux switch:1 else 0]*

<h3>4. Resume Training </h3>

```
py train.py --train_file 'path\train.json' --valid_file 'path\test.json' --load_model_from 'path\model\best_model.ckpt' --resume_from_checkpoint 'path\model\' --save_model_path 'save\model\path' --gpus 1
```

<h3>5. CTC Decoder </h3>
<h4> <a href='https://github.com/parlance/ctcdecode'> Installation </a></h4>
The library is largely self-contained and requires only PyTorch. Building the C++ library requires gcc or clang. KenLM language modeling support is also optionally included, and enabled by default.

```
# get the code
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```
<h3>6. Sentence Corpus Extraction </h3>
Use `extract_sentences.py` to extract sentences with the data from Commonvoice by Mozilla </br>

```
py extract_sentences.py --file_path "file_path\to\.tsv" --save_txt_path "save\corpus.txt\path"
```

<h3>7. KenLM </h3>
<h4> <a href='https://github.com/kpu/kenlmBuild'> Build & Compile</a></h4>
Use cmake, see BUILDING for build dependencies and more detail.

```
mkdir -p build
cd build
cmake ..
make -j 4
```
*Optional: Copy `corpus.txt` to the directory of `KenLM/build/bin`*
Then, enter `build/bin` directory where lmplz resides
```
lmplz -o n <path/to/corpus.txt> <path/save/language/model.arpa>
```
*Where, <br>
&nbsp; `-o n` is order of words to select at once when building kenlm <br>
&nbsp; `nglm.arpa` name for arpa file for llm <br>
&nbsp; if `-o 1` give name `1glm.arpa` and so on for better readability.*

<h3>8. Engine </h3>

```
py engine.py --file_path "path/model/speechrecognition.ckpt" --save_txt_path "path/to/nglm.arpa or path/to/nglm.bin"
```
