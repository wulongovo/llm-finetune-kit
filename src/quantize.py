"""
GPTQ 模型量化模块
训练后量化：将合并后的模型压缩为 4/8bit，大幅降低显存占用和推理延迟

用法示例:
  from src.quantize import GPTQQuantizer
  quantizer = GPTQQuantizer("models/merged-7b", "models/merged-7b-gptq")
  quantizer.quantize(calibration_data=...)
"""

import os
import json
import time
import logging
from typing import List, Optional, Union
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    """GPTQ 量化配置"""
    bits: int = 4                    # 量化位数: 2/3/4/8
    group_size: int = 128            # 分组大小，越小精度越高但文件越大
    sym: bool = True                 # 对称量化
    true_sequential: bool = True     # 逐层顺序量化（精度更高）
    damp_percent: float = 0.01       # 阻尼系数
    desc_act: bool = True            # 按激活值降序排列（精度更高）

    # 校准数据配置
    max_calib_samples: int = 128     # 最大校准样本数
    max_calib_tokens: int = 2048     # 每个样本最大 token 数

    # 序列长度
    model_max_length: int = 2048     # 模型最大序列长度


class GPTQQuantizer:
    """GPTQ 量化器：将 HuggingFace 模型压缩为 GPTQ 格式

    使用流程:
      1. 准备校准数据（JSONL 对话格式或纯文本列表）
      2. 创建 GPTQQuantizer 实例
      3. 调用 quantize() 执行量化
      4. 调用 test() 验证量化效果
    """

    def __init__(
        self,
        model_path: str,
        output_path: str,
        config: Optional[GPTQConfig] = None,
        device_map: str = "auto",
    ):
        """
        Args:
            model_path: 待量化的模型路径（合并后的完整模型）
            output_path: 量化模型保存路径
            config: 量化参数配置
            device_map: 设备映射策略
        """
        self.model_path = model_path
        self.output_path = output_path
        self.config = config or GPTQConfig()
        self.device_map = device_map

        self.model = None
        self.tokenizer = None
        self._stats = {}

    # ============================================================
    # 校准数据加载
    # ============================================================

    def load_calibration_from_jsonl(self, file_path: str) -> List[str]:
        """从 JSONL 文件加载校准数据

        支持的格式:
          - Alpaca:  {"instruction": "...", "output": "..."}
          - ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
          - ChatML 会自动拼接

        Returns:
            ChatML 格式的文本列表
        """
        logger.info(f"加载校准数据: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        texts = []
        for item in data[:self.config.max_calib_samples]:
            text = self._format_to_chatml(item)
            if text:
                texts.append(text)

        logger.info(f"  加载了 {len(texts)} 条校准样本")
        return texts

    def load_calibration_from_texts(self, texts: List[str]) -> List[str]:
        """从纯文本列表加载校准数据"""
        return texts[:self.config.max_calib_samples]

    def _format_to_chatml(self, item: dict) -> Optional[str]:
        """将多种格式统一转为 ChatML"""
        # ShareGPT 格式
        if "conversations" in item:
            chatml = ""
            for turn in item["conversations"]:
                role = turn.get("from", turn.get("role", "user"))
                content = turn.get("value", turn.get("content", "")).strip()
                chatml += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            return chatml.strip()

        # Alpaca 格式
        if "instruction" in item:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item.get("output", "")
            chatml = f"<|im_start|>user\n{instruction}"
            if input_text:
                chatml += f"\n{input_text}"
            chatml += f"<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            return chatml

        # 纯文本
        if "text" in item:
            return item["text"]

        return None

    def _build_calibration_dataset(self, texts: List[str]) -> List[torch.Tensor]:
        """将文本转为 tokenized 校准数据集"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

        dataset = []
        for text in texts:
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_calib_tokens,
            ).input_ids
            dataset.append(input_ids)

        logger.info(f"  校准数据集: {len(dataset)} 个样本, "
                     f"平均长度: {sum(t.shape[1] for t in dataset) // len(dataset)} tokens")
        return dataset

    # ============================================================
    # 核心量化流程
    # ============================================================

    def quantize(
        self,
        calibration_data: Optional[Union[str, List[str]]] = None,
    ):
        """执行 GPTQ 量化

        Args:
            calibration_data: 校准数据，可以是 JSONL 文件路径或文本列表。
                             为 None 时使用内置通用校准文本。
        """
        print("\n" + "=" * 60)
        print("  GPTQ 模型量化")
        print("=" * 60)
        print(f"  模型:     {self.model_path}")
        print(f"  输出:     {self.output_path}")
        print(f"  量化:     {self.config.bits}bit, group_size={self.config.group_size}")
        print(f"  对称:     {self.config.sym}")
        print(f"  校准样本: {self.config.max_calib_samples}")
        print("=" * 60)

        # ---- 1. 加载校准数据 ----
        if calibration_data is None:
            texts = self._default_calibration_texts()
        elif isinstance(calibration_data, str):
            texts = self.load_calibration_from_jsonl(calibration_data)
        else:
            texts = self.load_calibration_from_texts(calibration_data)

        calib_dataset = self._build_calibration_dataset(texts)

        # ---- 2. 初始化量化配置 ----
        quant_config = QuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            sym=self.config.sym,
            true_sequential=self.config.true_sequential,
            damp_percent=self.config.damp_percent,
            desc_act=self.config.desc_act,
        )

        # ---- 3. 加载模型 ----
        print("\n[1/3] 加载模型...")
        t0 = time.time()

        self.model = GPTQModel.from_pretrained(
            self.model_path,
            quantize_config=quant_config,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        load_time = time.time() - t0
        self._stats["load_time"] = load_time
        print(f"  ✓ 加载完成 ({load_time:.1f}s)")

        # ---- 4. 执行量化 ----
        print("\n[2/3] 执行 GPTQ 量化...")
        t0 = time.time()

        # 显示显存使用
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**3
            print(f"  显存使用: {mem_before:.1f} GB (量化前)")

        self.model.quantize(calib_dataset)

        quantize_time = time.time() - t0
        self._stats["quantize_time"] = quantize_time

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            print(f"  显存使用: {mem_after:.1f} GB (量化后)")

        print(f"  ✓ 量化完成 ({quantize_time:.1f}s)")

        # ---- 5. 保存模型 ----
        print(f"\n[3/3] 保存量化模型 -> {self.output_path}")
        t0 = time.time()

        os.makedirs(self.output_path, exist_ok=True)
        self.model.save(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)

        save_time = time.time() - t0
        self._stats["save_time"] = save_time

        # 计算模型大小
        total_size = self._get_dir_size(self.output_path)
        self._stats["model_size_gb"] = total_size / 1024**3

        print(f"  ✓ 保存完成 ({save_time:.1f}s)")
        print(f"  量化模型大小: {total_size / 1024**3:.2f} GB")

        # 打印汇总
        self._print_summary()

    # ============================================================
    # 验证测试
    # ============================================================

    def test(self, prompt: Optional[str] = None, device: str = "cuda:0") -> str:
        """加载量化模型并测试推理

        Args:
            prompt: 测试提示词
            device: 推理设备

        Returns:
            模型生成的回复文本
        """
        if prompt is None:
            prompt = "请用一句话介绍一下你自己。"

        print("\n" + "-" * 40)
        print(f"  推理测试: {self.output_path}")
        print("-" * 40)

        t0 = time.time()

        model = GPTQModel.from_quantized(
            self.output_path,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.output_path,
            trust_remote_code=True,
        )

        load_time = time.time() - t0
        print(f"  模型加载: {load_time:.1f}s")

        # 推理
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.time()

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        gen_time = time.time() - t0
        num_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"  生成: {num_tokens} tokens / {gen_time:.1f}s "
              f"({num_tokens / gen_time:.1f} tok/s)")
        print(f"  回复: {output_text.strip()[:200]}...")
        print("-" * 40)

        return output_text

    # ============================================================
    # 辅助方法
    # ============================================================

    def _default_calibration_texts(self) -> List[str]:
        """内置通用校准文本，无需额外准备校准数据"""
        return [
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "Python是一种广泛使用的解释型、高级和通用的编程语言。Python设计哲学强调代码的可读性和简洁的语法。",
            "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。常见的深度学习框架包括PyTorch和TensorFlow。",
            "自然语言处理是人工智能和语言学领域的分支学科，探讨如何处理及运用自然语言。",
            "请解释什么是量子计算，它和经典计算有什么区别？",
            "操作系统是管理计算机硬件与软件资源的系统软件，常见的操作系统有Windows、Linux和macOS。",
            "数据库是按照数据结构来组织、存储和管理数据的仓库。常用的数据库有MySQL、PostgreSQL和MongoDB。",
            "云计算是一种通过互联网按需提供计算资源的模式，包括服务器、存储、数据库、网络、软件等。",
            "什么是大语言模型？请简要说明其工作原理和应用场景。",
            "微积分是高等数学中研究函数的微分、积分以及有关概念和应用的数学分支。",
        ] * 13  # 扩展到约130条
        # 注：内置校准文本为通用中文内容，精度略低于领域相关校准数据。
        # 建议在量化特定领域模型时提供相关校准数据以获得最佳效果。

    @staticmethod
    def _get_dir_size(path: str) -> int:
        """计算目录总大小（字节）"""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    def _print_summary(self):
        """打印量化汇总报告"""
        print("\n" + "=" * 60)
        print("  量化完成汇总")
        print("=" * 60)
        print(f"  输入模型:   {self.model_path}")
        print(f"  输出路径:   {self.output_path}")
        print(f"  量化配置:   {self.config.bits}bit, "
              f"group_size={self.config.group_size}, "
              f"sym={self.config.sym}")
        print(f"  模型大小:   {self._stats.get('model_size_gb', 0):.2f} GB")
        print(f"  量化耗时:   {self._stats.get('quantize_time', 0):.1f}s")
        print(f"  总耗时:     {sum(v for k, v in self._stats.items() if 'time' in k):.1f}s")
        print()
        print(f"  下一步:")
        print(f"    1. 测试: python quantize.py --test --model {self.output_path}")
        print(f"    2. vLLM部署: python serve_vllm.py --model {self.output_path}")
        print("=" * 60 + "\n")
