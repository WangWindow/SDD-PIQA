#!/bin/bash

if [ ! -d "logs" ]; then
    mkdir logs
fi
# 修改为只使用索引为 0 的 GPU
export CUDA_VISIBLE_DEVICES='0'
nohup python -u 1_gen_datalist.py > logs/$(date +%F-%H-%M-%S)/1_gen_datalist.log 2>&1 &
nohup python -u 2_generate_pseudo_labels.py > logs/$(date +%F-%H-%M-%S)/2_generate_pseudo_labels.log 2>&1 &
nohup python -u 3_merge_datalist.py > logs/$(date +%F-%H-%M-%S)/3_merge_datalist.log 2>&1 &
