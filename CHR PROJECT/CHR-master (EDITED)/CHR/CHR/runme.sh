#!/usr/bin/env bash
# select gpu devices
export CUDA_VISIBLE_DEVICES=0
export COMMANDLINE_ARGS="--skip-torch-cuda-test"
export PIP_IGNORE_INSTALLED=0
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m main --batch-size 320 2>&1 | tee -a log
 
