"""
数据处理器 - 支持多种数据格式的加载与预处理

支持格式:
  - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
  - ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
  - Raw: {"text": "..."}
"""

import json
import os
from typing import List, Dict, Optional
from datasets import Dataset, DatasetDict


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_alpaca(example: Dict) -> str:
    """Alpaca格式 -> 模型输入文本"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
    else:
        prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"

    return prompt + output


def format_sharegpt(example: Dict) -> str:
    """ShareGPT格式 -> 模型输入文本"""
    conversations = example.get("conversations", [])
    text = ""
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            text += f"### Human:\n{value}\n\n"
        elif role == "gpt":
            text += f"### Assistant:\n{value}\n\n"
    return text


def format_raw(example: Dict) -> str:
    """原始文本格式"""
    return example.get("text", "")


FORMATTERS = {
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
    "raw": format_raw,
}


def prepare_dataset(
    train_file: str,
    eval_file: Optional[str] = None,
    data_format: str = "alpaca",
    text_field: str = "text",
    tokenizer=None,
    max_length: int = 1024,
    eval_ratio: float = 0.1,
) -> DatasetDict:
    """
    准备训练数据集

    Args:
        train_file: 训练数据文件路径
        eval_file: 验证数据文件路径（可选）
        data_format: 数据格式 (alpaca/sharegpt/raw)
        text_field: 文本字段名
        tokenizer: 分词器（用于tokenize统计）
        max_length: 最大序列长度
        eval_ratio: 无验证集时，从训练集划分的比例

    Returns:
        DatasetDict with "train" and optionally "eval"
    """
    formatter = FORMATTERS.get(data_format, format_raw)

    # 加载训练数据
    raw_data = load_jsonl(train_file)
    print(f"加载训练数据: {len(raw_data)} 条")

    # 格式化文本
    texts = [formatter(item) for item in raw_data]

    # 创建数据集
    dataset = Dataset.from_dict({text_field: texts})

    # 划分训练/验证集
    if eval_file and os.path.exists(eval_file):
        eval_data = load_jsonl(eval_file)
        eval_texts = [formatter(item) for item in eval_data]
        eval_dataset = Dataset.from_dict({text_field: eval_texts})
        dataset_dict = DatasetDict({
            "train": dataset,
            "eval": eval_dataset,
        })
        print(f"加载验证数据: {len(eval_data)} 条")
    else:
        split = dataset.train_test_split(test_size=eval_ratio, seed=42)
        dataset_dict = DatasetDict({
            "train": split["train"],
            "eval": split["test"],
        })
        print(f"自动划分: 训练 {len(split['train'])} 条, 验证 {len(split['test'])} 条")

    # Tokenize统计
    if tokenizer:
        def tokenize_fn(examples):
            return tokenizer(examples[text_field], truncation=True, max_length=max_length)

        tokenized = dataset_dict.map(tokenize_fn, batched=True, remove_columns=[text_field])
        tokenized.set_format("torch")
        return tokenized

    return dataset_dict


def create_sample_data(output_dir: str = "./data/processed"):
    """创建示例训练数据（用于快速测试）"""
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        {
            "instruction": "请介绍一下大模型的LoRA微调技术",
            "input": "",
            "output": "LoRA（Low-Rank Adaptation）是一种高效的模型微调技术。它的核心思想是：在预训练模型的权重矩阵旁边，添加两个低秩矩阵A和B，只训练这两个小矩阵，而保持原始模型参数不变。\n\n优点：\n1. 参数效率高：通常只训练原模型0.1%-1%的参数\n2. 显存占用小：适合消费级GPU（如RTX 3070）\n3. 无推理延迟：训练完成后可将LoRA权重合并回原模型\n4. 多任务支持：同一基模型可加载不同LoRA适配器"
        },
        {
            "instruction": "解释什么是RLHF",
            "input": "",
            "output": "RLHF（Reinforcement Learning from Human Feedback）即基于人类反馈的强化学习，是大模型对齐的核心技术之一。\n\n训练流程分为三步：\n1. 监督微调（SFT）：用高质量数据微调基础模型\n2. 训练奖励模型（RM）：人类标注者对模型输出排序，训练一个打分模型\n3. 强化学习优化（PPO）：用奖励模型的分数作为反馈信号，通过PPO算法优化模型\n\nRLHF让模型更符合人类偏好，提升有用性、安全性和诚实性。"
        },
        {
            "instruction": "什么是DPO，它和RLHF有什么区别？",
            "input": "",
            "output": "DPO（Direct Preference Optimization）直接偏好优化，是RLHF的简化替代方案。\n\n核心区别：\n- RLHF需要先训练奖励模型，再用PPO优化，流程复杂\n- DPO直接用偏好数据优化策略模型，不需要单独的奖励模型\n\nDPO优势：\n1. 训练更稳定（无PPO的不稳定性）\n2. 实现更简单（只需SFT + DPO两步）\n3. 计算资源更少（不需要维护4个模型）\n4. 效果接近RLHF\n\nDPO已成为中小规模模型微调的主流选择。"
        },
        {
            "instruction": "如何为垂直领域准备大模型微调数据？",
            "input": "",
            "output": "垂直领域微调数据准备的关键步骤：\n\n1. 数据收集\n   - 领域文档、FAQ、客服记录、专业文献\n   - 确保数据覆盖领域核心知识点\n\n2. 数据清洗\n   - 去重、去噪、格式统一\n   - 过滤低质量或错误数据\n\n3. 数据标注\n   - 指令-回答对格式（Alpaca格式最常用）\n   - 确保回答准确、专业、完整\n\n4. 数据质量控制\n   - 人工抽检（至少10%）\n   - 用GPT-4等强模型做质量评估\n\n5. 数据量建议\n   - LoRA微调：1000-10000条高质量数据即可\n   - 质量 > 数量"
        },
        {
            "instruction": "什么是梯度检查点？为什么微调大模型时需要开启？",
            "input": "",
            "output": "梯度检查点（Gradient Checkpointing）是一种用计算换显存的技术。\n\n原理：\n正常反向传播需要保存所有前向传播的中间激活值。梯度检查点只保存部分层的激活值，其余层在反向传播时重新计算。\n\n效果：\n- 显存节省：约30%-60%\n- 训练速度：降低约20%-30%\n\n适用场景：\n- GPU显存不足时（如RTX 3070 8GB微调7B模型）\n- 配合gradient_accumulation_steps使用效果最佳\n\n在transformers中开启：\ntraining_args = TrainingArguments(gradient_checkpointing=True)"
        },
    ]

    output_file = os.path.join(output_dir, "train.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 示例数据已生成: {output_file} ({len(samples)} 条)")
    return output_file
