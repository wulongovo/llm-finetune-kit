#!/bin/bash
# 启动 vLLM 推理服务

MODEL=${1:-"outputs/merged_model"}
PORT=${2:-8000}
TENSOR_PARALLEL=${3:-1}

echo "=========================================="
echo "  vLLM 推理服务"
echo "  模型: $MODEL"
echo "  端口: $PORT"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
