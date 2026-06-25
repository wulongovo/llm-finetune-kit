"""
LoRA 微调训练器
支持 SFT（监督微调）和 DPO（直接偏好优化）
支持 DeepSpeed 多卡分布式训练
针对 RTX 3070 (8GB) 显存优化
"""

import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, DPOTrainer, SFTConfig
from typing import Optional
from .config import FullConfig

logger = logging.getLogger(__name__)


def is_deepspeed_available() -> bool:
    """检测 DeepSpeed 是否可用"""
    try:
        import deepspeed
        return True
    except ImportError:
        return False


def get_deepspeed_env_info() -> dict:
    """获取 DeepSpeed 环境信息"""
    info = {"available": False}
    if is_deepspeed_available():
        import deepspeed
        info["available"] = True
        info["version"] = deepspeed.__version__
        info["num_gpus"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def get_quantization_config() -> BitsAndBytesConfig:
    """4bit量化配置 - 8GB显存微调7B模型的关键"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(config: FullConfig, deepspeed_config: Optional[str] = None):
    """加载模型和分词器

    Args:
        config: 完整配置
        deepspeed_config: DeepSpeed 配置文件路径，如果提供则跳过量化（DeepSpeed 不兼容 bitsandbytes 量化）
    """
    model_name = config.model.name_or_path

    print(f"加载模型: {model_name}")

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DeepSpeed 模式下不使用量化（DeepSpeed 不兼容 bitsandbytes 量化）
    use_deepspeed = deepspeed_config is not None and is_deepspeed_available()

    if use_deepspeed:
        logger.info("DeepSpeed 模式: 跳过 bitsandbytes 量化，使用 bf16 全精度加载")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    else:
        # 单卡模式: 量化配置 - RTX 3070 8GB 必须用4bit
        bnb_config = get_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
        # 准备模型用于量化训练
        model = prepare_model_for_kbit_training(model)

    # 打印模型信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,} total, {trainable_params:,} trainable")
    print(f"可训练比例: {100 * trainable_params / total_params:.4f}%")

    return model, tokenizer


def apply_lora(model, config: FullConfig):
    """应用LoRA适配器"""
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # 打印LoRA参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def create_training_args(config: FullConfig, deepspeed_config: Optional[str] = None) -> TrainingArguments:
    """创建训练参数 - 支持 DeepSpeed 多卡训练

    Args:
        config: 完整配置
        deepspeed_config: DeepSpeed 配置文件路径
    """
    extra_kwargs = {}

    # DeepSpeed 配置
    if deepspeed_config and is_deepspeed_available():
        extra_kwargs["deepspeed"] = deepspeed_config
        ds_info = get_deepspeed_env_info()
        logger.info(f"DeepSpeed 环境: v{ds_info.get('version', 'unknown')}, {ds_info.get('num_gpus', 0)} GPU(s)")

    return TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        gradient_checkpointing=config.training.gradient_checkpointing,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        max_grad_norm=config.training.max_grad_norm,
        optim=config.training.optim,
        report_to=config.training.report_to,
        seed=config.training.seed,
        eval_strategy="steps",
        eval_steps=config.training.save_steps,
        load_best_model_at_end=True,
        dataloader_pin_memory=False,  # Windows下避免内存问题
        **extra_kwargs,
    )


def train_sft(config: FullConfig, dataset_dict, deepspeed_config: Optional[str] = None):
    """SFT监督微调

    Args:
        config: 完整配置
        dataset_dict: 数据集字典
        deepspeed_config: DeepSpeed 配置文件路径
    """
    model, tokenizer = load_model_and_tokenizer(config, deepspeed_config=deepspeed_config)
    model = apply_lora(model, config)
    training_args = create_training_args(config, deepspeed_config=deepspeed_config)

    # SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        processing_class=tokenizer,
        max_seq_length=config.training.max_seq_length,
    )

    print("\n" + "=" * 50)
    print("开始 SFT 训练")
    print(f"  模型: {config.model.name_or_path}")
    print(f"  LoRA rank: {config.lora.r}")
    print(f"  Batch size: {config.training.per_device_train_batch_size} x {config.training.gradient_accumulation_steps}")
    print(f"  学习率: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_train_epochs}")
    print(f"  最大序列长度: {config.training.max_seq_length}")
    if deepspeed_config and is_deepspeed_available():
        ds_info = get_deepspeed_env_info()
        print(f"  DeepSpeed: {deepspeed_config}")
        print(f"  GPU数量: {ds_info.get('num_gpus', 0)}")
    print("=" * 50 + "\n")

    # 训练
    trainer.train()

    # 保存LoRA适配器
    adapter_dir = os.path.join(config.training.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n✅ LoRA适配器已保存: {adapter_dir}")

    return trainer


def train_dpo(config: FullConfig, dataset_dict, deepspeed_config: Optional[str] = None):
    """DPO直接偏好优化（需要chosen/rejected格式数据）

    Args:
        config: 完整配置
        dataset_dict: 数据集字典
        deepspeed_config: DeepSpeed 配置文件路径
    """
    model, tokenizer = load_model_and_tokenizer(config, deepspeed_config=deepspeed_config)
    model = apply_lora(model, config)
    training_args = create_training_args(config, deepspeed_config=deepspeed_config)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("eval"),
        processing_class=tokenizer,
        max_length=config.training.max_seq_length,
        max_prompt_length=config.training.max_seq_length // 2,
    )

    print("\n开始 DPO 训练...\n")
    trainer.train()

    adapter_dir = os.path.join(config.training.output_dir, "dpo_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n✅ DPO适配器已保存: {adapter_dir}")

    return trainer
