"""
配置加载器 - 读取YAML配置并合并命令行参数
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-7B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    max_seq_length: int = 1024
    report_to: str = "none"
    seed: int = 42


@dataclass
class DataConfig:
    train_file: str = "./data/processed/train.jsonl"
    eval_file: str = "./data/processed/eval.jsonl"
    text_field: str = "text"
    format: str = "alpaca"


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(yaml_path: str) -> FullConfig:
    """从YAML文件加载配置"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = FullConfig()

    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)

    if "lora" in raw:
        for k, v in raw["lora"].items():
            if hasattr(config.lora, k):
                setattr(config.lora, k, v)

    if "training" in raw:
        for k, v in raw["training"].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)

    if "data" in raw:
        for k, v in raw["data"].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)

    return config
