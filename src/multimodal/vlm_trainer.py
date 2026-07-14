"""
VLM 训练器 — 统一的多模态模型微调入口
支持 Qwen2.5-VL / InternVL 双架构
支持 QLoRA / LoRA / Full 三种微调策略
"""

import os
import logging
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from .model_factory import ModelFactory, detect_model_architecture, get_lora_target_modules
from .data_processor import MultimodalDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class VLMTrainingConfig:
    """VLM 训练配置"""
    # 模型
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    strategy: str = "qlora"  # qlora / lora / full

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 训练
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False

    # 输出
    output_dir: str = "outputs/vlm_finetune"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100

    # 数据
    train_file: str = ""
    eval_file: str = ""
    image_dir: str = ""
    data_format: str = "auto"

    # DeepSpeed
    deepspeed_config: Optional[str] = None

    # 其他
    seed: int = 42
    report_to: str = "none"


class VLMTrainer:
    """VLM 微调训练器

    统一的多模态模型训练接口，支持：
    - 模型架构：Qwen2.5-VL / InternVL
    - 微调策略：QLoRA (4bit) / LoRA (bf16) / Full fine-tune
    - 分布式训练：DeepSpeed ZeRO-2/3
    - 显存优化：Gradient Checkpointing
    """

    def __init__(self, config: VLMTrainingConfig):
        """
        Args:
            config: VLM 训练配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.trainer = None

        # 初始化模型工厂
        self.factory = ModelFactory(
            model_name_or_path=config.model_name_or_path,
            strategy=config.strategy,
        )

        logger.info(f"VLM Trainer 初始化完成")
        logger.info(f"  模型: {config.model_name_or_path}")
        logger.info(f"  架构: {self.factory.architecture}")
        logger.info(f"  策略: {config.strategy}")

    def _setup_model(self):
        """加载模型并应用 LoRA/QLoRA"""
        if self.model is not None:
            return  # 已经加载过

        logger.info("加载模型...")

        # 加载模型
        self.model, self.tokenizer, self.processor = self.factory.load()

        # 应用 LoRA（如果不是全量微调）
        if self.config.strategy in ("qlora", "lora"):
            self._apply_lora()
        elif self.config.strategy == "full":
            self._setup_full_finetune()

        # 打印可训练参数信息
        self._print_trainable_params()

    def _apply_lora(self):
        """应用 LoRA 适配器"""
        logger.info(f"应用 {self.config.strategy.upper()} 适配器...")

        lora_params = self.factory.get_lora_config(
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )

        lora_config = LoraConfig(**lora_params)
        self.model = get_peft_model(self.model, lora_config)

        logger.info(f"  LoRA rank: {self.config.lora_r}")
        logger.info(f"  LoRA alpha: {self.config.lora_alpha}")
        logger.info(f"  Target modules: {lora_params['target_modules']}")

    def _setup_full_finetune(self):
        """全量微调设置 — 冻结 Vision Encoder，只训练 LLM 部分"""
        logger.info("全量微调模式：冻结 Vision Encoder...")

        architecture = self.factory.architecture

        if architecture == "qwen2_vl":
            # Qwen2-VL: 冻结 visual 模块
            for name, param in self.model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False
            logger.info("  已冻结 Qwen2-VL visual encoder")

        elif architecture == "internvl":
            # InternVL: 冻结 vision_model 模块
            for name, param in self.model.named_parameters():
                if "vision_model" in name or "vit" in name:
                    param.requires_grad = False
            logger.info("  已冻结 InternVL vision encoder")

    def _print_trainable_params(self):
        """打印可训练参数统计"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params

        logger.info("=" * 50)
        logger.info("参数统计:")
        logger.info(f"  总参数:     {total_params:>15,}")
        logger.info(f"  可训练参数: {trainable_params:>15,}")
        logger.info(f"  冻结参数:   {frozen_params:>15,}")
        logger.info(f"  可训练比例: {100 * trainable_params / total_params:.4f}%")
        logger.info("=" * 50)

    def _create_training_args(self) -> TrainingArguments:
        """创建 HuggingFace TrainingArguments"""
        extra_kwargs = {}

        # DeepSpeed 配置
        if self.config.deepspeed_config:
            extra_kwargs["deepspeed"] = self.config.deepspeed_config
            logger.info(f"  DeepSpeed: {self.config.deepspeed_config}")

        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            gradient_checkpointing=self.config.gradient_checkpointing,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            max_grad_norm=1.0,
            optim="adamw_torch",
            report_to=self.config.report_to,
            seed=self.config.seed,
            eval_strategy="steps" if self.config.eval_file else "no",
            eval_steps=self.config.eval_steps if self.config.eval_file else None,
            load_best_model_at_end=True if self.config.eval_file else False,
            dataloader_pin_memory=False,  # Windows 兼容
            remove_unused_columns=False,   # 多模态数据需要保留所有列
            **extra_kwargs,
        )

    def _prepare_datasets(self):
        """准备训练和验证数据集"""
        data_processor = MultimodalDataProcessor(
            image_dir=self.config.image_dir,
            processor=self.processor,
            max_length=self.config.max_seq_length,
            architecture=self.factory.architecture,
        )

        # 加载训练数据
        train_data = data_processor.load_data(
            self.config.train_file,
            data_format=self.config.data_format,
        )

        # 加载或自动划分验证数据
        if self.config.eval_file and os.path.exists(self.config.eval_file):
            eval_data = data_processor.load_data(
                self.config.eval_file,
                data_format=self.config.data_format,
            )
        else:
            train_data, eval_data = data_processor.split_data(train_data)

        # 创建 Dataset
        train_dataset = data_processor.create_dataset(train_data)
        eval_dataset = data_processor.create_dataset(eval_data) if eval_data else None

        logger.info(f"训练样本: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"验证样本: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def train(self) -> Dict[str, Any]:
        """执行训练

        Returns:
            训练结果字典
        """
        # 设置模型
        self._setup_model()

        # 准备数据
        train_dataset, eval_dataset = self._prepare_datasets()

        # 创建训练参数
        training_args = self._create_training_args()

        # 打印训练配置
        self._print_training_info()

        # 使用 transformers 的 Trainer（通用方案）
        from transformers import Trainer

        # 自定义数据整理函数
        def collate_fn(batch):
            """将 MultimodalDataset 的输出整理为模型输入"""
            images = [item["image"] for item in batch]
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]

            architecture = self.factory.architecture

            if architecture == "qwen2_vl":
                return self._collate_qwen2_vl(images, questions, answers)
            elif architecture == "internvl":
                return self._collate_internvl(images, questions, answers)
            else:
                raise ValueError(f"不支持的架构: {architecture}")

        # 创建 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
        )

        # 开始训练
        logger.info("=" * 50)
        logger.info("开始 VLM 训练...")
        logger.info("=" * 50)

        start_time = time.time()
        train_result = self.trainer.train()
        train_time = time.time() - start_time

        # 保存模型
        self.save()

        # 训练结果
        result = {
            "train_runtime": train_time,
            "train_loss": train_result.training_loss,
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "model": self.config.model_name_or_path,
            "architecture": self.factory.architecture,
            "strategy": self.config.strategy,
        }

        logger.info(f"\n训练完成! 耗时: {train_time:.1f}秒")
        logger.info(f"训练 Loss: {result['train_loss']:.4f}")

        return result

    def _collate_qwen2_vl(self, images, questions, answers):
        """Qwen2-VL 的数据整理"""
        # 构建对话格式
        messages_list = []
        for img, q, a in zip(images, questions, answers):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": q},
                    ],
                },
                {"role": "assistant", "content": a},
            ]
            messages_list.append(messages)

        # 使用 processor 处理
        texts = []
        processed_images = []
        for messages in messages_list:
            # Qwen2-VL 的 chat 模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            # 提取图片
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            processed_images.append(item["image"])

        # 处理输入
        inputs = self.processor(
            text=texts,
            images=processed_images if processed_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )

        # 构建 labels（只计算回答部分的 loss）
        labels = inputs["input_ids"].clone()
        # 简化处理：将 padding token 设为 -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs

    def _collate_internvl(self, images, questions, answers):
        """InternVL 的数据整理"""
        # InternVL 使用不同的对话模板
        # 构建输入文本
        texts = []
        for q, a in zip(questions, answers):
            # InternVL 对话格式
            text = f"<image>\nUser: {q}\nAssistant: {a}"
            texts.append(text)

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )

        # 处理图片（InternVL 通常在模型内部处理）
        # 这里将图片作为额外信息传递
        pixel_values_list = []
        for img in images:
            # 基本的图片预处理
            if hasattr(self.processor, "image_processor"):
                pv = self.processor.image_processor(images=img, return_tensors="pt")
                pixel_values_list.append(pv["pixel_values"])
            else:
                # 简化处理
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                pv = transform(img).unsqueeze(0)
                pixel_values_list.append(pv)

        if pixel_values_list:
            inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        # Labels
        labels = inputs["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    def _print_training_info(self):
        """打印训练配置信息"""
        cfg = self.config
        logger.info("\n" + "=" * 60)
        logger.info("VLM 训练配置")
        logger.info("=" * 60)
        logger.info(f"  模型:     {cfg.model_name_or_path}")
        logger.info(f"  架构:     {self.factory.architecture}")
        logger.info(f"  策略:     {cfg.strategy}")
        logger.info(f"  LoRA r:   {cfg.lora_r}")
        logger.info(f"  Epochs:   {cfg.num_train_epochs}")
        logger.info(f"  Batch:    {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps}")
        logger.info(f"  LR:       {cfg.learning_rate}")
        logger.info(f"  Max Seq:  {cfg.max_seq_length}")
        logger.info(f"  显存优化: gradient_checkpointing={cfg.gradient_checkpointing}")
        if cfg.deepspeed_config:
            logger.info(f"  DeepSpeed: {cfg.deepspeed_config}")
        logger.info("=" * 60 + "\n")

    def evaluate(self, eval_data=None) -> Dict[str, float]:
        """评估模型

        Args:
            eval_data: 评估数据集（可选）

        Returns:
            评估指标字典
        """
        if self.trainer is None:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        if eval_data is not None:
            eval_result = self.trainer.evaluate(eval_dataset=eval_data)
        else:
            eval_result = self.trainer.evaluate()

        logger.info(f"评估结果: {eval_result}")
        return eval_result

    def save(self, output_dir: Optional[str] = None):
        """保存模型

        Args:
            output_dir: 输出目录，默认使用 config 中的目录
        """
        save_dir = output_dir or os.path.join(self.config.output_dir, "final_adapter")

        os.makedirs(save_dir, exist_ok=True)

        if self.config.strategy in ("qlora", "lora"):
            # 保存 LoRA 适配器
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"LoRA 适配器已保存: {save_dir}")
        else:
            # 全量微调保存完整模型
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"完整模型已保存: {save_dir}")

        # 保存训练配置
        config_path = os.path.join(save_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name_or_path": self.config.model_name_or_path,
                "architecture": self.factory.architecture,
                "strategy": self.config.strategy,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
            }, f, indent=2, ensure_ascii=False)

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要（用于日志和报告）"""
        return {
            "model": self.config.model_name_or_path,
            "architecture": self.factory.architecture,
            "strategy": self.config.strategy,
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "target_modules": get_lora_target_modules(self.factory.architecture),
            },
            "training_config": {
                "epochs": self.config.num_train_epochs,
                "batch_size": self.config.per_device_train_batch_size,
                "grad_accum": self.config.gradient_accumulation_steps,
                "lr": self.config.learning_rate,
                "max_seq_len": self.config.max_seq_length,
            },
        }
