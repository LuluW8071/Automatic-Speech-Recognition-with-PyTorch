# Speech Recognition with CNN LSTM Acoustic Model and CTC Decoder + KENLM Language Model

This repository contains an implementation of a Speech Recognition system using a combination of a Convolutional Neural Network (CNN) LSTM Acoustic Model and a Connectionist Temporal Classification (CTC) Decoder, enhanced with a KENLM Language Model.

## Dependencies
- [Python 3.10.0](https://www.python.org/downloads/release/python-3100/)
- [NVIDIA CUDA 12.1.0](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- [PyTorch [Stable 2.1.1]](https://pytorch.org/)
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
- [Pytorch-Lightning 2.1.0](https://www.pytorchlightning.ai/index.html)
- [ffmpeg](https://www.ffmpeg.org/)
    ```bash
    # Extract the archive
    > ffmpeg/bin/
    # Edit environment variables to insert path 
    > path/to/ffmpeg/bin/
    ```
- **It is to be noted that currently [CTC Decoder](https://github.com/parlance/ctcdecode) works only on linux Distros and Mac**

## Terminal Commands
*It is recommended to provide absolute paths instead of relative paths.*

### 1. Data Collection 

  Use `create_coomonvoice.py` to convert from mp3 to wav and to create train and test JSON with the data from Commonvoice by Mozilla
  ```bash
  py create_commonvoice.py --file_path "file_path\to\.tsv" --save_json_path "save\json\path" --audio "audio\src_path\clips\to\.mp3" --percent 10 --convert
  ```
_If you dont want to convert use `create_jsons_only.py`_
  
#### JSON Format
  ```bash
   [
      {   
          "key": "/path/to/audio/speech.wav", 
          "text": "this is yourtext"
      },
      ......
   ]
 ```
### 2. Comet-ML API

Edit `config.py` with your <a href='https://www.comet.com/site/'> Comet.ml </a> API key and project name to get real-time loss curve plot
```bash
API_KEY = "###"  
PROJECT_NAME = "###" 
```

### 3. Train Datasets
```bash
py train.py --train_file "path\train.json" --valid_file "path\test.json" --save_model_path 'save\model\path'  --valid_file <value> --batch_size <value> --epochs <value>
```
Below is a list of flags will be mostly used in the `train.py` script with their descriptions:
| Flag                   | Description                                                           | Default Value      |
|------------------------|-----------------------------------------------------------------------|--------------------|
| `-g, --gpus`           | Number of GPUs per node                                               | 1                  |
| `--train_file`         | JSON file to load training data                                       | [Required]         |
| `--valid_file`         | JSON file to load testing data                                        | [Required]         |
| `--save_model_path`    | Path to save the trained model                                        |                    |
| `--load_model_from`    | Path to load a pre-trained model to continue training                 |                    |
| `--resume_from_checkpoint` | Checkpoint path to resume training from                           |                    |
| `--logdir`             | Path to save logs                                                     | 'tb_logs'          |
| `--epochs`             | Number of total epochs to run                                         | 10                 |
| `--batch_size`         | Size of the batch                                                     | 64                 |
| `--learning_rate`      | Learning rate                                                         | 1e-3  (0.001)      |

_Make sure to provide required arguments like `--train_file` and `--valid_file` when running the script._

_Set `--gpus` flag to either `0` or `1` depending upon multiple gpu if you have_

### 4. Resume Training
```bash
py train.py --train_file 'path\train.json' --valid_file 'path\test.json' --load_model_from 'path\model\best_model.ckpt' --resume_from_checkpoint 'path\model\' --save_model_path 'save\model\path'
```

### 5. [CTC Decoder](https://github.com/parlance/ctcdecode)
The library is largely self-contained and requires only **PyTorch**. Building the **C++ library** requires gcc or clang.
```bash
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

### 6. Sentence Corpus Extraction
Use `extract_sentences.py` to extract sentences with the data from Commonvoice by Mozilla or you can make your own script and extract textual words to build **language model**.
```bash
py extract_sentences.py --file_path "file_path\to\.tsv" --save_txt_path "save\corpus.txt\path"
```

### 7. [KenLM](https://github.com/kpu/kenlm)
Use **cmake**, see BUILDING for build dependencies
```bash
mkdir -p build
cd build
cmake ..
make -j 4
```
*Optional: Copy `corpus.txt` to the directory of `KenLM/build/bin`*

Then, enter `build/bin` directory where lmplz resides
```bash
lmplz -o n <path/to/corpus.txt> <path/save/language/model.arpa>
```

*Where,*
  - *`-o n` is order of words to select at once when building kenlm*
  - *`nglm.arpa` name for arpa file for llm*
*if `-o 1` give name `1glm.arpa` and so on for better readability.*

### 8. Freeze Model Checkpoint
Use `freeze_model.py` to freeze the model. 

*Use after training*
```bash
py freeze_model.py --model_checkpoint "path/model/speechrecognition.ckpt" --save_path "path/to/save/"
```

### 9. Engine & Demo
Use `engine.py` to run transription demo in terminal.
```bash
py engine.py --file_path "path/model/speechrecognition.ckpt" --save_txt_path "path/to/nglm.arpa or path/to/nglm.bin"
```

Use `demo.py` to run transription demo in webpage.
```bash
py demo.py --file_path "path/model/speechrecognition.ckpt" --save_txt_path "path/to/nglm.arpa or path/to/nglm.bin"
```
