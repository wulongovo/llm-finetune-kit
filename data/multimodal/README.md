# 多模态数据格式说明

## 支持的数据格式

本框架支持三种数据格式，均为 JSONL 格式（每行一个 JSON 对象）。

### 1. VQA（视觉问答）格式

```json
{"image": "images/001.jpg", "question": "这张图片中有什么？", "answer": "图片中是一只橘色的猫坐在沙发上。"}
```

字段说明：
- `image`: 图片路径（相对于 image_dir）
- `question`: 问题文本
- `answer`: 标准答案

### 2. Caption（图像描述）格式

```json
{"image": "images/002.jpg", "caption": "这是一幅日落时分的城市天际线照片。"}
```

字段说明：
- `image`: 图片路径
- `caption`: 图片描述文本

### 3. Conversation（多轮对话）格式

```json
{"image": "images/003.jpg", "conversations": [{"role": "user", "content": "<image>\n请描述这张图片"}, {"role": "assistant", "content": "这是一幅日落时分的城市天际线照片。"}]}
```

字段说明：
- `image`: 图片路径
- `conversations`: 对话列表，必须包含 user 和 assistant 角色
- user 的 content 中必须包含 `<image>` 标记（表示图片输入位置）

## 数据自动检测

框架会自动检测数据格式，无需手动指定。如果需要手动指定，在配置文件中设置：

```yaml
data:
  format: vqa  # 或 caption / conversation / auto
```

## 目录结构

```
data/multimodal/
├── sample_data.jsonl      # 示例数据
├── train.jsonl            # 训练数据（需要自己准备）
├── eval.jsonl             # 验证数据（可选，自动划分）
├── images/                # 图片目录
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── README.md              # 本文件
```

## 推荐数据集

以下是常用的多模态微调数据集：

| 数据集 | 任务 | 规模 | 下载 |
|--------|------|------|------|
| LLaVA-Instruct-150K | VQA+描述 | 150K | [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| VQAv2 | VQA | 443K | [VQA](https://visualqa.org/) |
| TextVQA | OCR+VQA | 45K | [TextVQA](https://textvqa.org/) |
| COCO Captions | 图像描述 | 591K | [COCO](https://cocodataset.org/) |
| DocVQA | 文档理解 | 50K | [DocVQA](https://www.docvqa.org/) |

## 数据准备示例

### 从 COCO 数据集准备

```python
import json
from pathlib import Path

# 假设已有 COCO 格式的标注
coco_annotations = json.load(open("coco_annotations.json"))

with open("train.jsonl", "w") as f:
    for ann in coco_annotations["annotations"]:
        entry = {
            "image": f"images/{ann['image_id']:012d}.jpg",
            "caption": ann["caption"]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 从自定义 CSV 准备

```python
from src.multimodal.data_processor import convert_vqa_to_jsonl

convert_vqa_to_jsonl(
    input_file="my_data.csv",
    output_file="data/multimodal/train.jsonl",
    image_dir="images"
)
```
