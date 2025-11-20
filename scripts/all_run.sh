#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

# 修改为只使用索引为 0 的 GPU
export CUDA_VISIBLE_DEVICES='0'

timestamp=$(date +%F-%H-%M-%S)
log_dir="logs/all_run/$timestamp"
mkdir -p logs
mkdir -p "$log_dir"

echo "Starting pipeline. Logs are being written to $log_dir/log"

{
    echo "========================================================"
    echo "Step 1: Train Recognition Model (train_rec.sh)"
    echo "========================================================"
    python -u utils/train_recognition/train_recognition.py
    echo "Step 1 Done."

    echo "========================================================"
    echo "Step 2: Generate Pseudo Labels (gen_pseudo_labels.sh)"
    echo "========================================================"
    for script in 1_gen_datalist 2_extract_feats 3_gen_pseudo_labels; do
        echo "Running ${script}..."
        python -u utils/gen_pseudo_labels/${script}.py
        echo "----------------------------------------"
        echo "${script} done"
        echo "----------------------------------------"
    done
    echo "Step 2 Done."

    echo "========================================================"
    echo "Step 3: Train Quality Model (train.sh)"
    echo "========================================================"
    python -u train.py
    echo "Step 3 Done."

    echo "========================================================"
    echo "All steps completed successfully."
    echo "========================================================"

} > "$log_dir/log" 2>&1 &
