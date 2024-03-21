# Speech Recognition with CNN LSTM Acoustic Model and CTC Decoder + KENLM Language Model

Welcome to the Speech Recognition system repository! This project implements a state-of-the-art speech recognition system utilizing a Convolutional Neural Network (CNN) - LSTM Acoustic Model, a Connectionist Temporal Classification (CTC) Decoder, and a KENLM Language Model for enhanced accuracy.

## Overview

This system combines cutting-edge deep learning techniques with traditional language modeling to transcribe spoken language accurately. Below, you'll find instructions on dependencies, terminal commands, and how to run various parts of the system.

### Dependencies

Before you begin, ensure you have the following dependencies installed:

- [Python 3.10.0](https://www.python.org/downloads/release/python-3100/)
- NVIDIA CUDA 12.1.0
- [PyTorch [Stable 2.1.1]](https://pytorch.org/)
- [Pytorch-Lightning 2.1.0](https://www.pytorchlightning.ai/index.html)
- [ffmpeg](https://www.ffmpeg.org/)
    ```bash
    # Extract the archive
    > ffmpeg/bin/
    # Edit environment variables to insert path 
    > path/to/ffmpeg/bin/
    ```
- CTC Decoder and KENLM(currently works only on Linux Distros and Mac)

### Terminal Commands

#### 1. Data Collection

To convert audio files to the required WAV format and create JSON files for training and testing data from Commonvoice by Mozilla, use the following command:

```bash
py create_commonvoice.py --file_path "file_path\to\.tsv" --save_json_path "save\json\path" --audio "audio\src_path\clips\to\.mp3" --percent 10 --convert
```

Alternatively, if conversion isn't needed, you can use `create_jsons_only.py`.

##### JSON Format

        ```bash
        [
            {   
                "key": "/path/to/audio/speech.wav", 
                "text": "this is yourtext"
            },
            ......
        ]
        ```

#### 2. Comet-ML API Integration

For real-time loss curve plotting, edit `config.py` with your Comet.ml API key and project name. Click <a href='https://www.comet.com/site/' target="_blank">here</a> to sign up and get your Comet-ML API key.

#### 3. Training Datasets

To train the model using your own data, execute:

```bash
py train.py --train_file "path\train.json" --valid_file "path\test.json" --save_model_path 'save\model\path'  --valid_file <value> --batch_size <value> --epochs <value>
```

Refer to the provided table for various flags and their descriptions.
| Flag                   | Description                                                           | Default Value      |
|------------------------|-----------------------------------------------------------------------|--------------------|
| `-g, --gpus`           | Number of GPUs per node                                               | 1                  |
| `--train_file`         | JSON file to load training data                                       | [Required]         |
| `--valid_file`         | JSON file to load testing data                                        | [Required]         |
| `--save_model_path`    | Path to save the trained model                                        |                    |
| `--load_model_from`    | Path to load a pre-trained model to continue training                 |                    |
| `--resume_from_checkpoint` | Checkpoint path to resume training from                           |                    |                                                  | 'tb_logs'          |
| `--epochs`             | Number of total epochs to run                                         | 10                 |
| `--batch_size`         | Size of the batch                                                     | 64                 |
| `--learning_rate`      | Learning rate                                                         | 1e-3  (0.001)      |

#### 4. Resuming Training

To resume training from a saved checkpoint, use:

```bash
py train.py --train_file 'path\train.json' --valid_file 'path\test.json' --load_model_from 'path\model\best_model.ckpt' --resume_from_checkpoint 'path\model\' --save_model_path 'save\model\path'
```

#### 5. CTC Decoder Installation

Clone the CTC Decoder repository and install it using pip:

```bash
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

#### 6. Sentence Corpus Extraction

Use `extract_sentences.py` to extract sentences from the Commonvoice dataset or any other source to build the language model.

```bash
py extract_sentences.py --file_path "file_path\to\.tsv" --save_txt_path "save\path\corpus.txt"
```

#### 7. KENLM Installation

Build KENLM using cmake and compile the language model using lmplz:

```bash
mkdir -p build
cd build
cmake ..
make -j 4
lmplz -o n <path/to/corpus.txt> <path/save/language/model.arpa>
```

Follow the instruction on KENLM `README.md` to convert `.arpa` file to `.bin` for faster inference. 

#### 8. Freeze Model Checkpoint

After training, freeze the model using `freeze_model.py`:

```bash
py freeze_model.py --model_checkpoint "path/model/speechrecognition.ckpt" --save_path "path/to/save/"
```

#### 9. Engine Demo

Finally, run the transcription engine demo using `engine.py`:

```bash
py engine.py --file_path "path/model/speechrecognition.ckpt" --ken_lm_file "path/to/nglm.arpa or path/to/nglm.bin"
```

## Additional Resources

For pre-trained models and other resources, refer to the provided links.
[Click here to download pre trained model](https://mega.nz/folder/Lnxj3YCJ#Na6Nc1m4nz6jiSWTatfKJQ)

---

This comprehensive guide should help you navigate through setting up and using the Speech Recognition system effectively. If you encounter any issues or have questions, feel free to reach out!