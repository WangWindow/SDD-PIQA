# #!/bin/bash

# if [ ! -d "logs" ]; then
#     mkdir logs
# fi
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# nohup python -u train.py > logs/$(date +%F-%H-%M-%S).log 2>&1 &


#!/bin/bash

if [ ! -d "logs" ]; then
    mkdir logs
fi
# 修改为只使用索引为 0 的 GPU
export CUDA_VISIBLE_DEVICES='0'
nohup python -u train.py > logs/$(date +%F-%H-%M-%S).log 2>&1 &