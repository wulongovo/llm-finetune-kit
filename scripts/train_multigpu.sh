#!/bin/bash
# DeepSpeed 多卡训练启动脚本
# Usage: bash scripts/train_multigpu.sh [config] [deepspeed_config]

CONFIG=${1:-"configs/lora_config.yaml"}
DEEPSPEED=${2:-"configs/deepspeed_zero2.json"}
NUM_GPUS=${3:-$(nvidia-smi -L | wc -l)}

echo "=========================================="
echo "  DeepSpeed 多卡训练"
echo "  GPU数量: $NUM_GPUS"
echo "  训练配置: $CONFIG"
echo "  DeepSpeed: $DEEPSPEED"
echo "=========================================="

deepspeed --num_gpus=$NUM_GPUS train.py \
    --config $CONFIG \
    --deepspeed $DEEPSPEED
