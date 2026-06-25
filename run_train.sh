#!/bin/bash
cd "E:/ai-works/llm-finetune-kit"
unset PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export PATH="E:/ai-works/llm-finetune-kit/venv/Scripts:$PATH"
"E:/ai-works/llm-finetune-kit/venv/Scripts/python.exe" train.py --config configs/test_quick.yaml 2>&1
