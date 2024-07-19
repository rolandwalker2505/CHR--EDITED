#!/usr/bin/env bash
# select gpu devices
export CUDA_VISIBLE_DEVICES=1

python -m main --batch-size 64 2>&1 | tee -a log
 
