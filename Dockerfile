FROM vllm/vllm-openai:latest

# 设置工作目录
WORKDIR /app

# 复制模型文件
COPY outputs/merged_model /app/model

# 复制配置
COPY configs/vllm_config.yaml /app/config.yaml

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model", \
     "--port", "8000", \
     "--dtype", "bfloat16"]
