"""
模型工厂 — 统一加载 Qwen2.5-VL / InternVL 模型
根据模型名自动检测架构，应用对应的 LoRA target_modules
"""

import os
import re
import logging
import torch
from typing import Optional, Tuple, List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# ============================================================
# 架构检测
# ============================================================

# 模型名称 -> 架构类型的映射规则
_ARCH_PATTERNS = {
    "qwen2_vl": [r"qwen2[\.\-]?vl", r"qwen[\.\-]?vl"],
    "internvl": [r"internvl", r"intern[\.\-]?vl"],
}

# 各架构的 LoRA target_modules
_LORA_TARGET_MODULES = {
    "qwen2_vl": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "internvl": ["qkv", "proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"],
}

# 各架构的默认 model class（trust_remote_code=True 时使用）
_MODEL_CLASS_MAP = {
    "qwen2_vl": "AutoModelForCausalLM",  # Qwen2-VL 用 AutoModelForCausalLM 加载
    "internvl": "AutoModel",              # InternVL 用 AutoModel 加载
}


def detect_model_architecture(model_name_or_path: str) -> str:
    """检测模型架构类型

    Args:
        model_name_or_path: HuggingFace 模型名或本地路径

    Returns:
        "qwen2_vl" 或 "internvl"

    Raises:
        ValueError: 无法识别的模型架构
    """
    name_lower = model_name_or_path.lower()

    for arch, patterns in _ARCH_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, name_lower):
                logger.info(f"检测到模型架构: {arch} (from {model_name_or_path})")
                return arch

    # 尝试从 config.json 检测
    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        arch_str = config.get("architectures", [""])[0].lower()
        for arch, patterns in _ARCH_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, arch_str):
                    logger.info(f"从 config.json 检测到架构: {arch}")
                    return arch

    raise ValueError(
        f"无法识别模型架构: {model_name_or_path}\n"
        f"支持的模型名包含: qwen2-vl, qwen-vl, internvl"
    )


def get_lora_target_modules(architecture: str) -> List[str]:
    """获取指定架构的 LoRA target_modules

    Args:
        architecture: "qwen2_vl" 或 "internvl"

    Returns:
        target_modules 列表
    """
    if architecture not in _LORA_TARGET_MODULES:
        raise ValueError(f"未知架构: {architecture}, 支持: {list(_LORA_TARGET_MODULES.keys())}")
    return _LORA_TARGET_MODULES[architecture]


def get_quantization_config(strategy: str) -> Optional[BitsAndBytesConfig]:
    """根据策略获取量化配置

    Args:
        strategy: "qlora" / "lora" / "full"

    Returns:
        BitsAndBytesConfig 或 None
    """
    if strategy == "qlora":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif strategy == "lora":
        return None  # LoRA 不量化，用 bf16
    elif strategy == "full":
        return None  # 全量微调不量化
    else:
        raise ValueError(f"未知策略: {strategy}, 支持: qlora/lora/full")


# ============================================================
# 模型工厂
# ============================================================

class ModelFactory:
    """VLM 模型工厂 — 统一加载和配置多模态模型"""

    def __init__(self, model_name_or_path: str, strategy: str = "qlora"):
        """
        Args:
            model_name_or_path: 模型路径（HuggingFace 名称或本地路径）
            strategy: 微调策略 "qlora" / "lora" / "full"
        """
        self.model_name = model_name_or_path
        self.strategy = strategy
        self.architecture = detect_model_architecture(model_name_or_path)

    def load(self, device_map: str = "auto", torch_dtype=torch.bfloat16) -> Tuple:
        """加载模型、分词器和处理器

        Args:
            device_map: 设备映射策略
            torch_dtype: 模型精度

        Returns:
            (model, tokenizer, processor)
            - model: 加载好的模型
            - tokenizer: 分词器
            - processor: 图像处理器（可能与 tokenizer 相同）
        """
        logger.info(f"加载模型: {self.model_name}")
        logger.info(f"  架构: {self.architecture}")
        logger.info(f"  策略: {self.strategy}")

        bnb_config = get_quantization_config(self.strategy)

        # ============================================================
        # Qwen2.5-VL
        # ============================================================
        if self.architecture == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            # 加载 processor（处理图像+文本）
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # 加载模型
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
            }
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )

            # tokenizer 从 processor 中获取
            tokenizer = processor.tokenizer

            if self.strategy == "qlora":
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)

        # ============================================================
        # InternVL
        # ============================================================
        elif self.architecture == "internvl":
            # InternVL 使用 AutoModel + trust_remote_code
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
            }
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config

            model = AutoModel.from_pretrained(
                self.model_name, **load_kwargs
            )
            model = model.eval()  # InternVL 默认 eval 模式

            # InternVL 的 tokenizer 和 processor
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            # InternVL 的 processor 通常集成在 model 内部
            processor = tokenizer  # 简化处理

            if self.strategy == "qlora":
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)

        else:
            raise ValueError(f"不支持的架构: {self.architecture}")

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型总参数: {total_params:,}")

        return model, tokenizer, processor

    def get_lora_config(self, lora_r: int = 16, lora_alpha: int = 32,
                        lora_dropout: float = 0.05) -> Dict[str, Any]:
        """获取 LoRA 配置

        Args:
            lora_r: LoRA 秩
            lora_alpha: 缩放因子
            lora_dropout: Dropout 率

        Returns:
            LoRA 配置字典，可直接传给 peft.LoraConfig
        """
        from peft import TaskType

        target_modules = get_lora_target_modules(self.architecture)

        return {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }

    def estimate_memory(self, strategy: str = "qlora", lora_r: int = 16) -> Dict[str, str]:
        """估算显存占用

        Args:
            strategy: 微调策略
            lora_r: LoRA 秩

        Returns:
            显存估算信息字典
        """
        # 简化的显存估算（基于经验公式）
        base_memory = {
            "qwen2_vl": {"3B": 6, "7B": 14, "72B": 140},
            "internvl": {"2B": 4, "8B": 16, "26B": 52},
        }

        arch_mem = base_memory.get(self.architecture, {})
        model_size = "unknown"
        for size in arch_mem:
            if size.lower() in self.model_name.lower():
                model_size = size
                break

        if model_size == "unknown":
            return {"warning": "无法自动估算，请手动确认显存"}

        base_gb = arch_mem[model_size]

        if strategy == "qlora":
            estimated = base_gb * 0.4  # 4bit量化约40%原大小
        elif strategy == "lora":
            estimated = base_gb * 0.6  # bf16 LoRA约60%
        elif strategy == "full":
            estimated = base_gb * 1.2  # 全量微调需要120%
        else:
            estimated = base_gb

        # LoRA 额外开销
        lora_overhead = lora_r * 0.01  # 简化估算

        return {
            "model_size": model_size,
            "base_memory": f"{base_gb} GB (fp16)",
            "estimated_memory": f"{estimated + lora_overhead:.1f} GB ({strategy})",
            "recommended_gpu": "A100 40GB+" if estimated > 24 else "RTX 3090/4090 24GB" if estimated > 12 else "RTX 3070/4060 8GB+",
        }
