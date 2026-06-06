# 🔥 LLM Fine-tune Kit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-latest-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-0078D4?style=flat-square&logo=microsoft&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

**大模型LoRA微调工具包** — 针对消费级GPU（RTX 3070/3080/4060等 8-12GB显存）深度优化，开箱即用。

---

## ✨ 特性

- 🎯 **开箱即用** — 一条命令启动微调，零配置快速上手
- 💾 **显存优化** — 4bit量化 + Gradient Checkpointing + Paged AdamW 8bit，8GB显存微调7B模型
- 🔌 **多模型支持** — Qwen、DeepSeek、LLaMA、Mistral 等主流模型
- 📊 **Web UI** — Gradio 可视化界面，配置/训练/测试一条龙
- 📦 **一键部署** — 训练完成后直接导出到 Ollama 本地运行
- 🔄 **SFT + DPO** — 同时支持监督微调和偏好优化两种训练方式

## 📦 技术栈

| 组件 | 作用 |
|------|------|
| Transformers | 模型加载与推理 |
| PEFT | LoRA/QLoRA 适配器 |
| TRL | SFT/DPO 训练器 |
| BitsAndBytes | 4/8bit 量化 |
| Accelerate | 分布式训练 |
| Gradio | Web UI |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU (推荐 8GB+ 显存)
- CUDA 12.x

### 安装

```bash
git clone https://github.com/wulongovo/llm-finetune-kit.git
cd llm-finetune-kit

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux

# 安装 PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
pip install -r requirements.txt
```

### 一键训练

```bash
# 使用示例数据快速测试
python train.py --config configs/lora_config.yaml

# 指定模型和数据
python train.py --model Qwen/Qwen2.5-7B --data data/your_data.jsonl --epochs 3

# 使用 Qwen3.5-Heretic
python train.py --config configs/qwen35_heretic.yaml
```

### Web UI

```bash
python webui.py
# 浏览器打开 http://localhost:7860
```

### 推理测试

```bash
# 交互式对话
python inference.py --adapter outputs/final_adapter --interactive

# 单次推理
python inference.py --adapter outputs/final_adapter --instruction "什么是LoRA微调？"
```

### 导出到 Ollama

```bash
# 合并LoRA并导出
python export.py --adapter outputs/final_adapter --ollama --name my-model

# 运行
ollama run my-model
```

## 📁 项目结构

```
llm-finetune-kit/
├── train.py              # 训练入口
├── inference.py          # 推理入口
├── export.py             # 导出入口
├── webui.py              # Web UI入口
├── requirements.txt      # 依赖列表
├── src/
│   ├── __init__.py
│   ├── config.py         # 配置加载器
│   ├── data_processor.py # 数据处理
│   ├── trainer.py        # 训练核心（SFT/DPO）
│   ├── inference.py      # 推理引擎
│   ├── export.py         # 模型导出（Ollama/GGUF）
│   └── webui.py          # Gradio Web UI
├── configs/
│   ├── lora_config.yaml  # 通用LoRA配置
│   └── qwen35_heretic.yaml  # Qwen3.5-Heretic配置
├── data/
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── scripts/
│   ├── train_qwen35.bat  # Windows一键训练
│   ├── test_model.bat    # 测试脚本
│   └── launch_webui.bat  # 启动Web UI
├── outputs/              # 训练输出
└── adapters/             # LoRA适配器
```

## ⚙️ 配置说明

### LoRA 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| r | 8-16 | LoRA秩，越大拟合能力越强但显存越多 |
| lora_alpha | 2*r | 缩放因子 |
| target_modules | q,k,v,o,gate,up,down | 作用的注意力层 |
| dropout | 0.05 | 防止过拟合 |

### RTX 3070 (8GB) 优化配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| batch_size | 1 | 必须为1 |
| gradient_accumulation | 8 | 等效batch_size=8 |
| gradient_checkpointing | true | 节省30-50%显存 |
| quantization | 4bit (NF4) | 8GB显存微调7B必须 |
| optimizer | paged_adamw_8bit | 节省优化器显存 |
| max_seq_length | 512-1024 | 越大显存越多 |
| bf16 | true | RTX 3070支持 |

### 显存估算

| 模型大小 | 量化方式 | LoRA rank | 显存占用 |
|----------|----------|-----------|----------|
| 3B | 4bit | 16 | ~4GB |
| 7B | 4bit | 16 | ~6-7GB |
| 7B | 4bit | 32 | ~8GB |
| 14B | 4bit | 16 | ~12GB |

## 📊 数据格式

### Alpaca 格式 (推荐)

```jsonl
{"instruction": "请介绍一下LoRA技术", "input": "", "output": "LoRA是一种高效的微调技术..."}
{"instruction": "将以下文本翻译成英文", "input": "大模型是人工智能的未来", "output": "Large models are the future of AI"}
```

### ShareGPT 格式

```jsonl
{"conversations": [{"from": "human", "value": "什么是RLHF？"}, {"from": "gpt", "value": "RLHF是基于人类反馈的强化学习..."}]}
```

## 🔬 训练流程

```
准备数据 (JSONL)
    ↓
加载基础模型 (4bit量化)
    ↓
应用 LoRA 适配器
    ↓
SFT 训练 (3-5 epochs)
    ↓
[可选] DPO 偏好优化
    ↓
保存 LoRA 适配器
    ↓
合并模型 → 导出 Ollama
```

## 💡 最佳实践

1. **数据质量 > 数量** — 1000条高质量数据 > 10000条低质量数据
2. **从小模型开始** — 先用3B模型验证流程，再上7B
3. **监控loss曲线** — loss不再下降时及时停止
4. **多次小epoch** — 3次x3epoch > 1次x9epoch
5. **验证集必设** — 防止过拟合的唯一手段

## 📄 License

MIT License

---

<p align="center">
  ⭐ 如果这个项目对你有帮助，请点个 Star！
</p>
