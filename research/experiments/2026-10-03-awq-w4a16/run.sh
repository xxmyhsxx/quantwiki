#!/usr/bin/env bash
set -eo pipefail
source /home/xiao/miniforge3/etc/profile.d/conda.sh
conda activate /data/xiao/wiki-quant-lab/.conda
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/data/xiao/wiki-quant-lab/.conda
export CC=/data/xiao/wiki-quant-lab/.conda/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/xiao/wiki-quant-lab/.conda/bin/x86_64-conda-linux-gnu-g++
export CPATH=/data/xiao/wiki-quant-lab/.conda/targets/x86_64-linux/include
export LIBRARY_PATH=/data/xiao/wiki-quant-lab/.conda/targets/x86_64-linux/lib
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
python -u experiment.py "$@"
