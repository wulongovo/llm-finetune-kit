#!/usr/bin/env python3
"""
LLM Fine-tune Kit - 多模态 VLM 训练入口

用法:
  python train_vlm.py --config configs/qwen25_vl_lora.yaml
  python train_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --strategy qlora
  python train_vlm.py --model OpenGVLab/InternVL3-8B --strategy lora --epochs 5
  python train_vlm.py --config configs/qwen25_vl_qlora.yaml --deepspeed configs/deepspeed_zero2.json
"""

import argparse
import os
import sys
import yaml
import torch

from src.multimodal.vlm_trainer import VLMTrainer, VLMTrainingConfig
from src.multimodal.data_processor import create_sample_data
from src.multimodal.model_factory import ModelFactory


def load_config(config_path: str) -> VLMTrainingConfig:
    """从 YAML 文件加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = VLMTrainingConfig()

    # 模型配置
    if "model" in raw:
        config.model_name_or_path = raw["model"].get("name_or_path", config.model_name_or_path)

    # LoRA 配置
    if "lora" in raw and raw["lora"]:
        config.lora_r = raw["lora"].get("r", config.lora_r)
        config.lora_alpha = raw["lora"].get("lora_alpha", config.lora_alpha)
        config.lora_dropout = raw["lora"].get("lora_dropout", config.lora_dropout)

    # 训练配置
    if "training" in raw:
        t = raw["training"]
        config.num_train_epochs = t.get("num_train_epochs", config.num_train_epochs)
        config.per_device_train_batch_size = t.get("per_device_train_batch_size", config.per_device_train_batch_size)
        config.gradient_accumulation_steps = t.get("gradient_accumulation_steps", config.gradient_accumulation_steps)
        config.learning_rate = t.get("learning_rate", config.learning_rate)
        config.weight_decay = t.get("weight_decay", config.weight_decay)
        config.warmup_ratio = t.get("warmup_ratio", config.warmup_ratio)
        config.max_seq_length = t.get("max_seq_length", config.max_seq_length)
        config.gradient_checkpointing = t.get("gradient_checkpointing", config.gradient_checkpointing)
        config.bf16 = t.get("bf16", config.bf16)
        config.fp16 = t.get("fp16", config.fp16)
        config.output_dir = t.get("output_dir", config.output_dir)
        config.logging_steps = t.get("logging_steps", config.logging_steps)
        config.save_steps = t.get("save_steps", config.save_steps)
        config.save_total_limit = t.get("save_total_limit", config.save_total_limit)
        config.seed = t.get("seed", config.seed)

    # 数据配置
    if "data" in raw:
        d = raw["data"]
        config.train_file = d.get("train_file", config.train_file)
        config.eval_file = d.get("eval_file", config.eval_file)
        config.image_dir = d.get("image_dir", config.image_dir)
        config.data_format = d.get("format", config.data_format)

    # 量化配置（用于判断策略）
    has_quantization = raw.get("quantization") is not None
    if has_quantization:
        config.strategy = "qlora"
    elif raw.get("lora") is not None:
        config.strategy = "lora"
    else:
        config.strategy = "full"

    return config


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tune Kit - 多模态 VLM 微调")

    parser.add_argument("--config", type=str, default="configs/qwen25_vl_lora.yaml",
                        help="配置文件路径")
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖模型路径 (如 Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--strategy", choices=["qlora", "lora", "full"], default=None,
                        help="微调策略: qlora(4bit)/lora(bf16)/full(全量)")
    parser.add_argument("--data", type=str, default=None,
                        help="覆盖训练数据路径")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="覆盖图片目录")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖训练轮数")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="覆盖 LoRA rank")
    parser.add_argument("--output", type=str, default=None,
                        help="覆盖输出目录")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="覆盖最大序列长度")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed 配置文件路径")

    args = parser.parse_args()

    # ============================================================
    # 加载配置
    # ============================================================
    print(f"加载配置: {args.config}")
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.model:
        config.model_name_or_path = args.model
    if args.strategy:
        config.strategy = args.strategy
    if args.data:
        config.train_file = args.data
    if args.image_dir:
        config.image_dir = args.image_dir
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.output:
        config.output_dir = args.output
    if args.max_seq_len:
        config.max_seq_length = args.max_seq_len
    if args.deepspeed:
        config.deepspeed_config = args.deepspeed

    # ============================================================
    # GPU 检测
    # ============================================================
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # 自动优化配置
        if gpu_mem < 10:
            print("⚠️  显存 <10GB，自动切换为 QLoRA 策略")
            config.strategy = "qlora"
            config.gradient_checkpointing = True
        elif gpu_mem < 24:
            if config.strategy == "full":
                print("⚠️  显存 <24GB，全量微调可能不够，自动切换为 LoRA")
                config.strategy = "lora"
    else:
        print("\n⚠️  未检测到 GPU，将使用 CPU 训练（非常慢！）")

    # ============================================================
    # 显存估算
    # ============================================================
    print("\n显存估算:")
    factory = ModelFactory(config.model_name_or_path, strategy=config.strategy)
    mem_info = factory.estimate_memory(strategy=config.strategy, lora_r=config.lora_r)
    for key, value in mem_info.items():
        print(f"  {key}: {value}")

    # ============================================================
    # 检查训练数据
    # ============================================================
    if not os.path.exists(config.train_file):
        print(f"\n训练数据不存在: {config.train_file}")
        print("生成示例数据...")
        sample_path = create_sample_data(os.path.dirname(config.train_file) or "data/multimodal")
        config.train_file = sample_path

    # ============================================================
    # 打印训练信息
    # ============================================================
    print("\n" + "=" * 60)
    print("  LLM Fine-tune Kit - 多模态 VLM 微调")
    print("=" * 60)
    print(f"  模型:     {config.model_name_or_path}")
    print(f"  策略:     {config.strategy}")
    if config.strategy in ("qlora", "lora"):
        print(f"  LoRA r:   {config.lora_r}")
        print(f"  LoRA α:   {config.lora_alpha}")
    print(f"  Epochs:   {config.num_train_epochs}")
    print(f"  Batch:    {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print(f"  LR:       {config.learning_rate}")
    print(f"  Max Seq:  {config.max_seq_length}")
    print(f"  输出目录: {config.output_dir}")
    if args.deepspeed:
        print(f"  DeepSpeed: {args.deepspeed}")
    print("=" * 60 + "\n")

    # ============================================================
    # 开始训练
    # ============================================================
    trainer = VLMTrainer(config)
    result = trainer.train()

    # 打印结果
    print("\n" + "=" * 60)
    print("  训练完成!")
    print("=" * 60)
    print(f"  训练 Loss:   {result.get('train_loss', 'N/A'):.4f}")
    print(f"  训练时间:    {result.get('train_runtime', 0):.1f} 秒")
    print(f"  模型保存在:  {config.output_dir}/final_adapter")
    print(f"\n下一步:")
    print(f"  1. 评估: python eval_vlm.py --model {config.output_dir}/final_adapter")
    print(f"  2. 对比: python eval_vlm.py --model-a {config.output_dir}/final_adapter --model-b {config.model_name_or_path}")


if __name__ == "__main__":
    main()
