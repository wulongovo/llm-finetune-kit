#!/bin/bash
# Docker 部署脚本

MODEL_DIR=${1:-"outputs/merged_model"}
IMAGE_NAME="llm-finetune-kit"
PORT=${2:-8000}

echo "构建 Docker 镜像..."
docker build -t $IMAGE_NAME .

echo "启动容器..."
docker run -d \
    --gpus all \
    -p $PORT:8000 \
    -v $(pwd)/$MODEL_DIR:/app/model \
    $IMAGE_NAME

echo "服务已启动: http://localhost:$PORT"
echo "API 文档: http://localhost:$PORT/docs"
