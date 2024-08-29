Here's a structured README template for your group project:

---

# DNLP Project



---

## Introduction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)](https://pytorch.org/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./NeuralWordsmiths_AI_Usage_Card.pdf/)

This repository contains my contribution to the final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. It involved implementing components of the BERT model and applying it to the semantic textual similarity task.

The project is divided into two main parts:

- **Part 01:** Involves the baseline implementation of the BERT model for the task. This includes the attention mechanism and Transformer layers, and applied it to semantic similarity.
- **Part 02:** Involves fine-tunning and extending the model to improve performance on the same downstream task. Different techniques are explored to create more robust and semantically-rich sentence embeddings, aiming to improve over the baseline implementations. Also 

The initial part focused on establishing a working baseline for each task, while the latter part will concentrate on refining and optimizing the model for better performance.

---

## Setup Instructions

To set up the environment and install dependencies for local development and testing, use the provided bash script `setup.sh`. This script creates a new conda environment called `dnlp` and installs all required packages. It will also check for CUDA and install the appropriate PyTorch version accordingly.

```sh
bash setup.sh
```

Activate the environment with:

```sh
conda activate dnlp
```

For setting up the repository on the remote GWDG cluster for training models with GPUs via SSH connection, use the `setup_gwdg.sh` script. This script is specifically designed to configure the environment for GPU-accelerated training on the GWDG cluster.

```sh
bash setup_gwdg.sh
```

---

## Training
### Local: 

To train the model, activate the environment and run:

```sh
python -u multitask_classifier.py --use_gpu
```

There are a lot of parameters that can be set. The most important ones are:

| Parameter               | Description                                                                           |
|-------------------------|---------------------------------------------------------------------------------------|
| `--task`                | Choose between `"sst"`, `"sts"`, `"qqp"`, `"multitask"` to train for different tasks. |
| `--seed`                | Random seed for reproducibility.                                                      |
| `--epochs`              | Number of epochs to train the model.                                                  |
| `--option`              | Determines if BERT parameters are frozen (`pretrain`) or updated (`finetune`).        |
| `--use_gpu`             | Whether to use the GPU for training.                                                  |
| `--subset_size`         | Number of examples to load from each dataset for testing.                             |
| `--context_layer`       | Include context layer if this flag is set.                                            |
| `--regularize_context`  | Use regularized context layer variant if this flag is set.                            |
| `--pooling`             | Choose the pooling strategy: `"cls"`, `"average"`, `"max"`, or `"attention"`.         |
| `--optimizer`           | Optimizer to use.                                                                     |
| `--batch_size`          | Batch size for training, recommended 64 for 12GB GPU.                                 |
| `--hidden_dropout_prob` | Dropout probability for hidden layers.                                                |
| `--lr`                  | Learning rate, defaults to `1e-3` for `pretrain`, `1e-5` for `finetune`.              |
| `--local_files_only`    | Force the use of local files only (do not download from remote repositories).         |
| `--smart_enable`        | Enables Smoothness-Inducing Adversarial Regularization (SMART).                       |
| `--epsilon`             | The epsilon (used in SMART).                                                          |
| `--alpha`               | Step size for adversarial perturbation in SMART.                                      |
| `--steps`               | Number of steps for generating perturbations in SMART.                                |

All parameters and their descriptions can be seen by running:

```sh
python multitask_classifier.py --help
```

### HPC:
to submit the job to a node in the GWDG HPC cluster, run:
settings can be configured according to the requirements in the `run_train.sh` file.
```sh
sbatch run_train.sh
```
---

## Evaluation

The model is evaluated after each epoch on the validation set. Results are printed to the console and saved in the `logdir` directory. The best model is saved in the `models` directory.

---
