#!/bin/bash
# Create the conda environment used for the experiments (run once).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q '^rmbff ' || conda create -y -n rmbff python=3.11
conda activate rmbff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # choose the CUDA build for your GPU
pip install -r requirements.txt
python -c "import torch, timm; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
