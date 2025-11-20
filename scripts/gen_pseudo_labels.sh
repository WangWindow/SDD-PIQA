#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

# 修改为只使用索引为 0 的 GPU
export CUDA_VISIBLE_DEVICES='0'

timestamp=$(date +%F-%H-%M-%S)
log_dir="logs/gen_pseudo_labels/$timestamp"
mkdir -p logs
mkdir -p "$log_dir"

{
    for script in 1_gen_datalist 2_extract_feats 3_gen_pseudo_labels; do
        echo "Running ${script}..."
        python -u utils/gen_pseudo_labels/${script}.py 2>&1
        echo "----------------------------------------"
        echo "${script} done"
        echo "----------------------------------------"
    done
    echo "All done"
} > "$log_dir/log" 2>&1 &
