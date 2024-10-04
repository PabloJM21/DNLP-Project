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


## Phase I

We implemented the base BERT and BART for the first phase of the project.

### BERT

For the BERT model we implemented 3 tasks:
- Sentiment Classification: The model got an additional classification layer, which takes as input the embedings from the BERT model. The used dataset is Stanford Sentiment Treebank. Loss function - Cross Entropy.
- Semantic Textual Similarity: Similar to the previous task, a classifier layer was added to the model. It takes as input the model's embedings, and predicts single logit, which defines the similarity score between senteces, which then is normilized to the range 0-5, 5 being most similar and 0 being related. Loss fucntion - Mean Squared Error Loss.
- Paraphrase Detection: The classifier layer at the end is similar to the previous task, with inputs being the embeddings of the model, and output a logit. The logit is normilized to the range 0-1, 1 being "is a paraphrase" and 0 being "not a paraphrase". Loss function - Binary Cross Entropy with Logits.

All embeddings go through a dropout layer, before being passed to the classifier layers.

For multitask training all tasks were run for 10 epochs with AdamW optimizer and hyperparameters:
- Learning rate: 1e-5
- Dropout probability: 0.2
- Batch size: 64
- Epsilon: 1e-8
- Betas: (0.9, 0.999)

For separate fine-tuning per tasks the hyperparameters were the same, except for Paraphrase Detection task, as 1 epoch is enough.

The model was trained on fine-tuning mode, so all parameters were updated.

BERT version: BERT Base Uncased.


## Experiments

### Learning all tasks vs. Learning one task:

- A BERT model was trained to be able to solve all 3 tasks, and was compared to a BERT model trained on the tasks independetly.
- The results for Sentiment Classification and Semantic Textual Similarity degrade, while for Paraphrase Detection increase.
- Most probable explanation: Dataset sizes are not equal. Later or bigger trainings degrade previous or smaller trainings.
- Possible solution: Trainings on bigger datasets first. Number of epochs relative to dataset size.

---

# Phase II

## Improvements upon Base Models


### 4.3 BERT for Semantic Textual Similarity (STS)



#### **4.3.1 Effectiveness of Pre-training on the Quora Dataset**

Given the substantial size of the Quora dataset, I hypothesized that pre-training on this data could enhance the model's performance on the STS task through multitask learning. But as previously discussed, the pre-training on the Quora dataset should be conducted prior to fine-tuning on the STS dataset. This approach prevents the Quora dataset from overwriting the learned weights derived from the STS dataset, due to the relative size difference between the two datasets.

Another argument supporting this strategy is the fact that the tasks of paraphrase detection and semantic textual similarity share certain underlying similarities:

- In the Paraphrase Detection task, the logits produced by the model's `predict_paraphrase` method represent the probability that a given sentence pair is a paraphrase.
- In the STS task, the output of the `predict_similarity` method represents the semantic similarity between two phrases, where a score of 0 indicates that they have nothing in common, and a score of 5 indicates that they are paraphrases. By rescaling this similarity score to the interval [0,1], it can also be interpreted as the probability that both sentences are paraphrases.

The next question that arose was how to transfer the knowledge gained from the Paraphrase Detection task to the STS task. The saved model state includes not only its weights but also the state of the optimizer. To explore the best approach, I tested two different implementations: one where both the model state and the optimizer state were loaded, and another where only the model state was loaded.

The experiment was conducted as follows:

1. **Paraphrase Detection Task**: I fine-tuned the pre-trained model using the following parameter configuration:  
   - **Epochs:** `10`
   - **Batch Size:** `64`
   - **Optimizer:** `AdamW`
   - **Learning Rate:** `1e-05`
   - **Option:** `finetune`
   - **Seed:** `11711`
   - **Subset Size:** `20,000`
   - **Pooling Strategy:** `CLS`
   - **Task:** `qqp`

2. **STS Task**: Then I trained the model for the STS task using the same configuration, but with the whole dataset instead of a subset. This evaluation was performed in three different ways:
   - Without loading the state of the Quora Question Pairs (QQP) model.
   - Loading only the state of the model.
   - Loading both the state of the model and the optimizer.

The correlation scores on the development dataset for each of these implementations were as follows:  

| Strategy                                     | STS Corr (Max)     |
|----------------------------------------------|--------------------|
| No previous knowledge (default)              | 0.864              |
| Model state                                  | 0.866              |
| Model and optimizer state                    | 0.850              |

These results suggest that loading only the state of the model is the best strategy, while loading the state of the optimizer leads to the worse performance. 

The reason for this might be as follows: The AdamW optimizer retains gradient information from the paraphrase detection task, which was trained on the large Quora dataset. This extensive training can lead to reduced flexibility when adapting to the new task despite their similarities. This is because the optimizer's state, having undergone many updates, might be more resistant to change and adjust to the nuances of the STS task. 


#### **4.3.2 Effectiveness of Average Pooling**

To enhance the performance of sentence embeddings on the STS task, I decided to introduce a strategy to enhance the embeddings. I chose average pooling for two main reasons: First, incorporating all tokens within the embeddings of sentence pairs allows for more precise capture of semantic dependencies between words. Second, averaging over the non-padded tokens helps to smooth out variations, resulting in representations that are both more accurate and more robust. 


I tested average pooling with two different embedding strategies:

1. **Combined Embedding**: A single embedding that represents both sentences together. This approach combines the embeddings of both sentences into a single vector, which is then used to predict similarity.

2. **Independent Embeddings**: Separate embeddings for each sentence. This approach treats each sentence individually, creating distinct embeddings that are compared in the similarity prediction process.

For the independent embeddings strategy, I explored two approaches:

- **Linear Layer Similarity (Implementation from Phase 1)**: In this approach, the `predict_similarity` function concatenates the independent embeddings of the two sentences and passes them through a linear layer to obtain a similarity score. This score is then normalized to fit the target scoring domain, which is the interval [0,5]. This approach is implemented by calling the `multitask_classifier_independent_embeddings.py` classifier in the `run_train.sh` script.

- **Cosine Similarity**: In this approach, the `predict_similarity` function calculates the cosine similarity between the two independent embeddings. Cosine similarity naturally falls within the interval [-1,1], so it is normalized to match the scoring domain of [0,5]. This method is implemented by calling the `multitask_classifier_cosine_sim.py` classifier in the `run_train.sh` script.


All other hyperparameters were configured as follows:  

- **Epochs:** `10`
- **Batch Size:** `64`
- **optimizer:** `AdamW`
- **learning rate:** `1e-05`
- **option:** `finetune`
- **seed:** `11711`
- **subset_size:** `None`
- **task:** `sts`

The correlation scores obtained on the development dataset for each of these implementations were:
 
| Pooling Strategy                                               | STS Corr (Max)     |
|----------------------------------------------------------------|--------------------|
| CLS, combined embedding, linear layer (default)                | 0.864              |
| Average, combined embedding, linear layer                      | 0.867              |
| Average, independent embeddings, linear layer                  | 0.406              |
| Average, independent embeddings, cosine similarity             | 0.406              |


#### **4.3.3 Pre-training on Quora with Average Pooling**
Based on the results from **4.3.2**, I decided to use average pooling on a combined embedding for both sentences. Additionally, the findings from **4.3.1** suggest that loading the model state from the Paraphrase Detection task could further enhance performance on the STS task. To make the most of these insights, I combined both strategies using the `multitask_classifier_quora_state.py` classifier, which incorporates the `bert_mean_pooling.py` module for average pooling embeddings.

Below are the steps to implement this approach:

1. **Fine tune the model for Paraphrase Detection Task**: First, run the `run_train.sh` script by calling `multitask_classifier_quora_state.py` with the following configuration:
- **Epochs:** `10`
- **Batch Size:** `64`
- **Optimizer:** `AdamW`
- **Learning Rate:** `1e-05`
- **Option:** `finetune`
- **Seed:** `11711`
- **Subset Size:** `None`
- **Task:** `qqp`

Ensure that in `multitask_classifier_quora_state.py`, the line `#model.load_state_dict(checkpoint['model'])` is commented out.

2. **Load the model's state for STS Task**: Next, run the `run_train.sh` script again with the same configuration, but set the **Task:** `sts`.
This time, ensure the line `model.load_state_dict(checkpoint['model'])` is uncommented to load the model state.

#### **4.3.3** Final Results
To test if the transfer leraning strategy described in the improvement above increases performance, I compared it to a baseline model that also utilizes average pooling but is fine-tuned from scratch. The training of the baseline was conducted as follows:
1. **Baseline**. Run the `run_train.sh` script by calling `multitask_classifier_quora_state.py` with the same configuration as in the improvement:
   - **Epochs:** `10`
   - **Batch Size:** `64`
   - **Optimizer:** `AdamW`
   - **Learning Rate:** `1e-05`
   - **Option:** `finetune`
   - **Seed:** `11711`
   - **Subset Size:** `None`
   - **Task:** `sts`
   

   But this time ensure that the line `#model.load_state_dict(checkpoint['model'])` is commented out to fine-tune the model from scratch.


The final correlation achieved on the dev test, as well as the epoch where each model reached its peak performance are displayed in the following table:


| Strategy                                               | STS Correlation (Max) | Epoch              | 
|--------------------------------------------------------|-----------------------|--------------------|
| Baseline                                               | 0.867                 | 6                  |
| Pre-training on Quora                                  | 0.870                 | 2                  |

As shown, the improvement strategy reached its peak performance at the second epoch, outperforming the baseline. This suggests that the transfer learning approach not only enhances the model's performance but also accelerates its convergence.
This improvement will be used to generate the predictions on the development dataset for the STS task.



