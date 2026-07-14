"""
多模态VLM微调模块
支持 Qwen2.5-VL / InternVL 双架构
支持 QLoRA / LoRA / Full 三种微调策略
"""

from .model_factory import ModelFactory, detect_model_architecture, get_lora_target_modules
from .data_processor import MultimodalDataProcessor, MultimodalDataset
from .vlm_trainer import VLMTrainer
from .evaluator import VLMEvaluator

__all__ = [
    "ModelFactory",
    "VLMTrainer",
    "MultimodalDataProcessor",
    "MultimodalDataset",
    "VLMEvaluator",
    "detect_model_architecture",
    "get_lora_target_modules",
]
