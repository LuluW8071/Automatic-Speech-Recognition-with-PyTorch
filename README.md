# 🚀 End-to-End Automatic Speech Recognition

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-completed-green.svg) ![License](https://img.shields.io/github/license/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Closed PRs](https://img.shields.io/github/issues-pr-closed/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Automatic-Speech-Recognition-with-PyTorch)

</div>

![Model](assets/model_architecture.png)

This project focuses on creating a small-scale speech recognition system for transcribing audio inputs into text. The system employs a **CNN1D + BiLSTM** based Acoustic Model, designed specifically for small-scale datasets and faster training of ASR (Automatic Speech Recognition).

## 💻 **Installation**

- Install the **CUDA version** of PyTorch for training or the **CPU version** for inference, then install the remaining dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 **Usage**

### **1. Dataset Conversion Script**

> [!NOTE]
> - The dataset conversion script is designed to convert the [**CommonVoice**](https://commonvoice.mozilla.org/en/datasets) dataset to the format required for training the speech recognition model. 
> - Use the `--not-convert` flag to skip the conversion step and export only the dataset paths and utterances in JSON format.

```bash
py common_voice.py --file_path path/to/validated.tsv --save_json_path converted_clips --percent 20
``` 

### **2. Train the Model**

> [!IMPORTANT]
> Two model choices are provided: __GRU__ and __LSTM__-based, in `train.py`. Uncomment the one you want and comment out the other. However, the LSTM performs better due to its ability to capture longer contexts.

```bash
py train.py --train_json path/to/train.json --valid_json path/to/test.json \
--epochs 100 \
--batch_size 64 \
--lr 2e-4 \
--grad_clip 0.5 \
--accumulate_grad 2 \
--gpus 1 \
--w 8 \
--checkpoint_path path/to/checkpoint.ckpt
```

### **3. Export to TorchScript**

```bash
python freeze_model.py --model_checkpoint path/to/model.ckpt
```

### **4. Run Inference**

```bash
python engine.py --model_file path/to/optimized_model.pt
```

## Experiment Results

This experiment used ~1,000 hours of audio with 670,000 utterances from Common Voice and my recordings, split 85% for training and 15% for testing. You can download the trained checkpoint and small 4-gram KENLM model from [here](https://mega.nz/folder/Lnxj3YCJ#Na6Nc1m4nz6jiSWTatfKJQ).

#### Model Configuration

|model|hidden_size|num_layers|dropout|n_feats|num_classes|
|----|-----------|---------|------|-------|----------|
|Bi-LSTM|512       |2        |0.1   |128    |29        |

#### Training Results

|Loss Curve|
|----------|
![Losses](assets/loss_curve.jpeg)|

|Model|Best Epoch|Val Loss|Avg. Greedy WER|Avg. CTC+KenLM |
|-|-|-|-|-|
|__Bi-LSTM__|61|0.359|28.44%|~22-23%|

> [!NOTE]
> __4-gram LibriSpeech KENLM__ was used for inference. If you build your own KenLM, the WER should be even lower.
---


## 📄 **License**

This project is licensed under the GNU License. See the [LICENSE](LICENSE) file for details.

---

This guide should help you effectively set up and use the speech recognition system. If you encounter any issues or have questions, feel free to reach out or submit a issue in the repository.