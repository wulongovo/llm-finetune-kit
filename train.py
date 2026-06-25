#!/usr/bin/env python3
"""
LLM Fine-tune Kit - 训练入口

用法:
  python train.py --config configs/lora_config.yaml
  python train.py --config configs/qwen35_heretic.yaml --epochs 5
  python train.py --model Qwen/Qwen2.5-7B --data data/processed/train.jsonl
"""

import argparse
import os
import sys
import torch

from src.config import load_config, FullConfig
from src.data_processor import prepare_dataset, create_sample_data
from src.trainer import train_sft
from src.export import create_training_report
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="LLM LoRA Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="覆盖模型路径")
    parser.add_argument("--data", type=str, default=None, help="覆盖训练数据路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--lora-r", type=int, default=None, help="覆盖LoRA rank")
    parser.add_argument("--output", type=str, default=None, help="覆盖输出目录")
    parser.add_argument("--max-seq-len", type=int, default=None, help="覆盖最大序列长度")
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft", help="训练模式")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path (e.g. configs/deepspeed_zero2.json)")
    args = parser.parse_args()

    # 加载配置
    print(f"加载配置: {args.config}")
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.model:
        config.model.name_or_path = args.model
    if args.data:
        config.data.train_file = args.data
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.lora_r:
        config.lora.r = args.lora_r
    if args.output:
        config.training.output_dir = args.output
    if args.max_seq_len:
        config.training.max_seq_length = args.max_seq_len

    # GPU 检查
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
        if args.deepspeed:
            # DeepSpeed 模式下启用 gradient_checkpointing
            config.training.gradient_checkpointing = True
            print(f"DeepSpeed 模式: 启用 gradient_checkpointing")
        elif gpu_mem < 10:
            print("⚠️  显存 <10GB，自动启用4bit量化 + gradient_checkpointing")
            config.training.gradient_checkpointing = True
    else:
        print("\n⚠️  未检测到GPU，将使用CPU训练（非常慢！）")

    # DeepSpeed 环境检查
    if args.deepspeed:
        from src.trainer import is_deepspeed_available, get_deepspeed_env_info
        if not is_deepspeed_available():
            print("❌ DeepSpeed 未安装，请运行: pip install deepspeed")
            sys.exit(1)
        ds_info = get_deepspeed_env_info()
        print(f"DeepSpeed: v{ds_info['version']}, {ds_info['num_gpus']} GPU(s)")
        if not os.path.isabs(args.deepspeed):
            args.deepspeed = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.deepspeed)

    # 准备数据
    if not os.path.exists(config.data.train_file):
        print(f"\n训练数据不存在，生成示例数据...")
        create_sample_data()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )

    # 准备数据集
    print("\n准备数据集...")
    dataset_dict = prepare_dataset(
        train_file=config.data.train_file,
        eval_file=config.data.eval_file,
        data_format=config.data.format,
        tokenizer=tokenizer,
        max_length=config.training.max_seq_length,
    )

    # 开始训练
    print("\n" + "=" * 60)
    print(f"  LLM Fine-tune Kit - {args.mode.upper()} Training")
    print(f"  Model: {config.model.name_or_path}")
    print(f"  LoRA: r={config.lora.r}, alpha={config.lora.lora_alpha}")
    print(f"  Output: {config.training.output_dir}")
    if args.deepspeed:
        print(f"  DeepSpeed: {args.deepspeed}")
    print("=" * 60 + "\n")

    if args.mode == "sft":
        trainer = train_sft(config, dataset_dict, deepspeed_config=args.deepspeed)
    else:
        from src.trainer import train_dpo
        trainer = train_dpo(config, dataset_dict, deepspeed_config=args.deepspeed)

    # 生成报告
    create_training_report(trainer, args.config, config.training.output_dir)

    print("\n✅ 训练完成!")
    print(f"适配器位置: {config.training.output_dir}/final_adapter")
    print(f"\n下一步:")
    print(f"  1. 测试: python inference.py --adapter {config.training.output_dir}/final_adapter")
    print(f"  2. 导出: python export.py --adapter {config.training.output_dir}/final_adapter --name my-model")


if __name__ == "__main__":
    main()
