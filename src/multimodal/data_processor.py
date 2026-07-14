"""
多模态数据处理器
支持 VQA / 图像描述 / 多轮对话 三种数据格式
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# 数据格式定义
# ============================================================

# VQA 格式: {"image": "path", "question": "...", "answer": "..."}
# Caption 格式: {"image": "path", "caption": "..."}
# Conversation 格式: {"image": "path", "conversations": [{"role": "user", "content": "<image>\n问题"}, {"role": "assistant", "content": "回答"}]}


def detect_data_format(sample: Dict) -> str:
    """自动检测数据格式

    Args:
        sample: 一条数据样本

    Returns:
        "vqa" / "caption" / "conversation"
    """
    if "conversations" in sample:
        return "conversation"
    elif "question" in sample and "answer" in sample:
        return "vqa"
    elif "caption" in sample:
        return "caption"
    else:
        raise ValueError(f"无法识别数据格式: {list(sample.keys())}")


def convert_to_conversation(sample: Dict, fmt: str) -> Dict:
    """将各种格式统一转换为 conversation 格式

    Args:
        sample: 原始数据
        fmt: 数据格式 "vqa" / "caption" / "conversation"

    Returns:
        统一的 conversation 格式数据
    """
    image_path = sample.get("image", "")

    if fmt == "conversation":
        return sample

    elif fmt == "vqa":
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        return {
            "image": image_path,
            "conversations": [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer},
            ],
        }

    elif fmt == "caption":
        caption = sample.get("caption", "")
        return {
            "image": image_path,
            "conversations": [
                {"role": "user", "content": "<image>\n请描述这张图片的内容。"},
                {"role": "assistant", "content": caption},
            ],
        }

    else:
        raise ValueError(f"未知格式: {fmt}")


# ============================================================
# 多模态数据集
# ============================================================

class MultimodalDataset(Dataset):
    """多模态数据集 — 图片+文本对"""

    def __init__(self, data: List[Dict], image_dir: str, processor=None,
                 max_length: int = 2048, architecture: str = "qwen2_vl"):
        """
        Args:
            data: conversation 格式的数据列表
            image_dir: 图片根目录
            processor: 图像+文本处理器（Qwen2-VL 的 processor 或 tokenizer）
            max_length: 最大序列长度
            architecture: 模型架构 "qwen2_vl" / "internvl"
        """
        self.data = data
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.architecture = architecture

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        conversations = sample["conversations"]

        # 加载图片
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"图片加载失败: {image_path}, 使用空白图片替代")
            image = Image.new("RGB", (448, 448), (255, 255, 255))

        # 提取问题和回答
        question = ""
        answer = ""
        for turn in conversations:
            if turn["role"] == "user":
                question = turn["content"].replace("<image>", "").strip()
            elif turn["role"] == "assistant":
                answer = turn["content"]

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "image_path": image_path,
        }


# ============================================================
# 数据处理器
# ============================================================

class MultimodalDataProcessor:
    """多模态数据处理器"""

    def __init__(self, image_dir: str = "", processor=None, max_length: int = 2048,
                 architecture: str = "qwen2_vl"):
        """
        Args:
            image_dir: 图片根目录
            processor: 模型的 processor
            max_length: 最大序列长度
            architecture: 模型架构
        """
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.architecture = architecture

    def load_data(self, file_path: str, data_format: str = "auto") -> List[Dict]:
        """加载 JSONL 数据文件

        Args:
            file_path: JSONL 文件路径
            data_format: 数据格式 "auto" / "vqa" / "caption" / "conversation"

        Returns:
            conversation 格式的数据列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        raw_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        logger.info(f"加载了 {len(raw_data)} 条数据 from {file_path}")

        # 自动检测格式
        if data_format == "auto" and raw_data:
            data_format = detect_data_format(raw_data[0])
            logger.info(f"自动检测数据格式: {data_format}")

        # 统一转换为 conversation 格式
        converted = [convert_to_conversation(s, data_format) for s in raw_data]

        return converted

    def split_data(self, data: List[Dict], eval_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """划分训练集和验证集

        Args:
            data: 完整数据
            eval_ratio: 验证集比例
            seed: 随机种子

        Returns:
            (train_data, eval_data)
        """
        random.seed(seed)
        shuffled = data.copy()
        random.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
        train_data = shuffled[:split_idx]
        eval_data = shuffled[split_idx:]

        logger.info(f"数据划分: train={len(train_data)}, eval={len(eval_data)}")
        return train_data, eval_data

    def create_dataset(self, data: List[Dict]) -> MultimodalDataset:
        """创建 PyTorch Dataset

        Args:
            data: conversation 格式数据

        Returns:
            MultimodalDataset 实例
        """
        return MultimodalDataset(
            data=data,
            image_dir=self.image_dir,
            processor=self.processor,
            max_length=self.max_length,
            architecture=self.architecture,
        )

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """批处理函数 — 将 batch 中的图片和文本打包

        Args:
            batch: 一批数据

        Returns:
            打包后的字典
        """
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        return {
            "images": images,
            "questions": questions,
            "answers": answers,
        }


# ============================================================
# 数据格式转换工具
# ============================================================

def convert_vqa_to_jsonl(input_file: str, output_file: str, image_dir: str = ""):
    """将 VQA CSV/JSON 转换为标准 JSONL 格式

    Args:
        input_file: 输入文件（.json 或 .csv）
        output_file: 输出 JSONL 文件
        image_dir: 图片目录（用于补全路径）
    """
    data = []

    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and "data" in raw:
            data = raw["data"]

    elif input_file.endswith(".csv"):
        import csv
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)

    # 写入 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            # 标准化字段名
            entry = {
                "image": item.get("image", item.get("image_path", "")),
                "question": item.get("question", item.get("query", "")),
                "answer": item.get("answer", item.get("response", "")),
            }
            if image_dir and not entry["image"].startswith(image_dir):
                entry["image"] = os.path.join(image_dir, entry["image"])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"转换完成: {len(data)} 条数据 -> {output_file}")


def create_sample_data(output_dir: str = "data/multimodal"):
    """创建示例多模态数据

    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        {
            "image": "images/sample_001.jpg",
            "question": "这张图片中有几种动物？分别是什么？",
            "answer": "图片中有两种动物：一只橘色的猫和一只白色的狗，它们正在草地上玩耍。"
        },
        {
            "image": "images/sample_002.jpg",
            "conversations": [
                {"role": "user", "content": "<image>\n请详细描述这张图片的内容。"},
                {"role": "assistant", "content": "这是一幅日落时分的城市天际线照片。画面中可以看到多座摩天大楼的轮廓，天空呈现出橙红色和紫色的渐变，云层被夕阳染成了金色。前景是一条河流，水面反射着天空的色彩。"}
            ]
        },
        {
            "image": "images/sample_003.jpg",
            "question": "图片中的文字是什么？",
            "answer": "图片中的文字是 'Hello World'，使用蓝色粗体字显示在白色背景上。"
        },
        {
            "image": "images/sample_004.jpg",
            "conversations": [
                {"role": "user", "content": "<image>\n这张图表展示了什么趋势？"},
                {"role": "assistant", "content": "这张折线图展示了2020年至2024年AI模型参数量的增长趋势。可以看到从2020年的约1亿参数增长到2024年的超过1万亿参数，呈现指数级增长。2022年是一个明显的拐点，对应GPT-3.5和ChatGPT的发布。"}
            ]
        },
        {
            "image": "images/sample_005.jpg",
            "question": "这张图片属于什么场景类别？",
            "answer": "这是一张厨房场景的照片。可以看到不锈钢水槽、木质橱柜、花岗岩台面，台面上放着一个咖啡机和一些水果。"
        },
    ]

    output_path = os.path.join(output_dir, "sample_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"示例数据已创建: {output_path}")
    return output_path
