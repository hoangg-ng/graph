#!/bin/bash
# venv for the MAGE multi-scale study. miniconda is gone from this host, so use uv.
# torch 2.8.0+cu128 because driver is 570 (CUDA 12.8); see memory note on the cu130 mismatch.
set -e
cd "$(dirname "$0")"
export PIP_CONFIG_FILE="$PWD/pip-clean.conf" UV_INDEX_URL=https://pypi.org/simple
UV=~/.local/bin/uv
$UV venv -p 3.11 .venv
$UV pip install -p .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$UV pip install -p .venv/bin/python numpy pandas scipy scikit-learn h5py tables pyyaml einops matplotlib tqdm pypdf
.venv/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo ENV_READY
