# End-to-End Automatic Speech Recognition

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-completed-green.svg) ![License](https://img.shields.io/github/license/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Automatic-Speech-Recognition-with-PyTorch) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Deep-Speech-2) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Automatic-Speech-Recognition-with-PyTorch)

</div>

This project implements a small scale speech recognition system utilizing a Residual Convolutional Neural Network (CNN) - BiGRU Acoustic Model, a Connectionist Temporal Classification (CTC) Decoder, and a KENLM Language Model for enhanced accuracy.

## Model Architecture

## Installation

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/LuluW8071/Automatic-Speech-Recognition-with-PyTorch.git
   ```

2. Install **[Pytorch](https://pytorch.org/)** and required dependencies under virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have `PyTorch` and `Lightning AI` installed.

## Train Model

>[!IMPORTANT]
> Before training make sure you have placed __comet ml api key__ and __project name__ in the environment variable file `.env`.

```bash
py train.py
```

Customize the pytorch training parameters by passing arguments in `train.py` to suit your needs:

Refer to the provided table to change hyperparameters and train configurations.
| Args                   | Description                                                           | Default Value      |
|------------------------|-----------------------------------------------------------------------|--------------------|
| `-g, --gpus`           | Number of GPUs per node                                               | 1  |
| `-g, --num_workers`           | Number of CPU workers                                               | 8  |
| `-db, --dist_backend`           | Distributed backend to use for training                             | ddp_find_unused_parameters_true  |
| `--epochs`             | Number of total epochs to run                                         | 50                 |
| `--batch_size`         | Size of the batch                                                     | 32                |
| `-lr, --learning_rate`      | Learning rate                                                         | 1e-5  (0.00001)      |
| `--checkpoint_path` | Checkpoint path to resume training from                                 | None |
| `--precision`        | Precision of the training                                              | 16-mixed |


```bash
py train.py 
-g 4                   # Number of GPUs per node for parallel gpu training
-w 8                   # Number of CPU workers for parallel data loading
--epochs 10            # Number of total epochs to run
--batch_size 64        # Size of the batch
-lr 2e-5               # Learning rate
--precision 16-mixed   # Precision of the training
```

>[!NOTE]
>To __resume training__ from a saved checkpoint, use:

```bash
py train.py --checkpoint_path path_to_checkpoint.ckpt
```

## Additional Resources

For pre-trained models and other resources, refer to the provided links.
[Click here to download pre trained model](https://mega.nz/folder/Lnxj3YCJ#Na6Nc1m4nz6jiSWTatfKJQ)

---

This comprehensive guide should help you navigate through setting up and using the Speech Recognition system effectively. If you encounter any issues or have questions, feel free to reach out!