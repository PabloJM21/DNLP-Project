#!/bin/bash

# Printing out some environment info (for debugging)
echo "Running script in Kaggle environment"
echo "Working directory: $PWD"
echo "Current node: $(hostname)"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Run the modified script
python -u /kaggle/input/dlfnlp/dlfnlp-sose2024/multitask_classifier_kaggle.py --use_gpu --option finetune --subset_size 25000 --task qqp --hidden_dropout_prob 0.1
