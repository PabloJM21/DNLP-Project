#!/bin/bash
set -e

# Since Kaggle doesn't support Conda, use pip to install packages

# Install PyTorch and related packages with/without CUDA support
if command -v nvidia-smi &>/dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected, installing CPU-only PyTorch."
    pip install torch torchvision torchaudio
fi

# Install additional packages
pip install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0
