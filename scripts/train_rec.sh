#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

# 修改为只使用索引为 0 的 GPU
export CUDA_VISIBLE_DEVICES='0'

timestamp=$(date +%F-%H-%M-%S)
log_dir="logs/train_rec/$timestamp"
mkdir -p logs
mkdir -p "$log_dir"

{
    nohup python -u utils/train_recognition/train_recognition.py
    echo "----------------------------------------"
    echo "done"
} > "$log_dir/log" 2>&1 &
