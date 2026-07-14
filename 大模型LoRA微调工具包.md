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
- 📊 **GPTQ 量化** — 训练后 4/8bit 量化，模型体积缩至 1/4，推理加速
- 🔄 **SFT + DPO** — 同时支持监督微调和偏好优化两种训练方式
- 🚀 **DeepSpeed 多卡训练** — 支持 ZeRO-2/ZeRO-3 分布式训练
- ⚡ **vLLM 高性能部署** — 推理速度提升 5-10 倍

## 📦 技术栈

| 组件 | 作用 |
|------|------|
| Transformers | 模型加载与推理 |
| PEFT | LoRA/QLoRA 适配器 |
| TRL | SFT/DPO 训练器 |
| BitsAndBytes | 4/8bit 量化 |
| Accelerate | 分布式训练 |
| Gradio | Web UI |
| DeepSpeed | 分布式训练 |
| vLLM | 高性能推理 |

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
│   ├── quantize.py       # GPTQ 训练后量化
│   ├── serving.py        # vLLM 推理服务
│   ├── benchmark.py      # 推理性能对比
│   └── webui.py          # Gradio Web UI
├── configs/
│   ├── lora_config.yaml  # 通用LoRA配置
│   ├── qwen35_heretic.yaml  # Qwen3.5-Heretic配置
│   ├── gptq.yaml         # GPTQ 量化配置
│   ├── vllm_config.yaml  # vLLM 部署配置
│   └── ds_config_zero2.json  # DeepSpeed ZeRO-2配置
├── data/
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── scripts/
│   ├── train_qwen35.bat  # Windows一键训练
│   ├── test_model.bat    # 测试脚本
│   ├── launch_webui.bat  # 启动Web UI
│   └── train_multi_gpu.sh  # 多卡训练启动脚本
├── merge_lora.py         # LoRA合并脚本（vLLM部署用）
├── quantize.py           # GPTQ 量化入口
├── serve_vllm.py         # vLLM推理服务
├── Dockerfile.vllm       # vLLM Docker部署
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
合并模型
    ↓
[可选] GPTQ 量化 (4bit/8bit)
    ↓
导出 Ollama / vLLM 部署
```

## 💡 最佳实践

1. **数据质量 > 数量** — 1000条高质量数据 > 10000条低质量数据
2. **从小模型开始** — 先用3B模型验证流程，再上7B
3. **监控loss曲线** — loss不再下降时及时停止
4. **多次小epoch** — 3次x3epoch > 1次x9epoch
5. **验证集必设** — 防止过拟合的唯一手段

## 🚀 多卡训练

### DeepSpeed 配置说明

项目内置 DeepSpeed ZeRO 配置文件，位于 `configs/ds_config_zero2.json`。核心参数：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "none" },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
```

### ZeRO-2 vs ZeRO-3 对比

| 特性 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|
| 分片内容 | 优化器状态 + 梯度 | 优化器状态 + 梯度 + 模型参数 |
| 通信开销 | 较低 | 较高 |
| 显存节省 | 中等 | 最大 |
| 适用场景 | 7B-14B 模型 | 14B+ 大模型 |
| 推荐配置 | 2-4 卡，NVLink | 4-8 卡，跨节点 |

**选择建议：**
- **ZeRO-2**：单机多卡、模型参数能放进单卡显存时首选，速度快
- **ZeRO-3**：模型太大单卡放不下时使用，牺牲速度换显存

### 多卡启动命令

```bash
# 2卡 ZeRO-2 训练
deepspeed --num_gpus 2 train.py --deepspeed configs/ds_config_zero2.json \
  --model Qwen/Qwen2.5-7B --data data/your_data.jsonl

# 4卡 ZeRO-3 训练
deepspeed --num_gpus 4 train.py --deepspeed configs/ds_config_zero3.json \
  --model Qwen/Qwen2.5-14B --data data/your_data.jsonl

# 使用 accelerate launch（推荐）
accelerate launch --num_processes 2 --deepspeed_config_file configs/ds_config_zero2.json \
  train.py --model Qwen/Qwen2.5-7B --data data/your_data.jsonl
```

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| `NCCL error` | 检查 GPU 互联，设置 `NCCL_P2P_DISABLE=1` |
| `Out of Memory` | 降低 `train_micro_batch_size_per_gpu`，启用 ZeRO-3 offload |
| 训练速度慢 | 检查 `overlap_comm` 是否开启，确认 NVLink 连接 |
| 权重不同步 | 确保所有卡使用相同随机种子 `--seed 42` |

## 📦 GPTQ 量化

训练后量化：将合并后的完整模型压缩为 4/8bit，体积缩至 1/4，推理速度翻倍。

### 快速使用

```bash
# 使用配置文件量化
python quantize.py --config configs/gptq.yaml

# 直接指定路径
python quantize.py --model models/merged-7b --output models/merged-7b-gptq-int4

# 使用领域校准数据获得最佳精度
python quantize.py --model models/merged-7b --output models/merged-7b-gptq-int4 --calibration data/eval.jsonl

# 8bit 量化（精度更高，体积缩减一半）
python quantize.py --model models/merged-7b --output models/merged-7b-gptq-int8 --bits 8
```

### 测试量化模型

```bash
# 推理测试
python quantize.py --test --model models/merged-7b-gptq-int4

# 自定义测试提示词
python quantize.py --test --model models/merged-7b-gptq-int4 --test-prompt "请总结一下Transformer架构的核心创新点"
```

### GPTQ vs BitsAndBytes

| 特性 | BitsAndBytes (训练时) | GPTQ (训练后) |
|------|----------------------|--------------|
| 使用阶段 | 训练时加载模型 | 部署前压缩模型 |
| 量化方式 | 在线量化（每次加载） | 离线量化（一次压缩） |
| 推理速度 | 较慢（动态反量化） | 快（预量化权重） |
| 显存占用 | 较低 | 更低 |
| 模型文件 | 原始大小 | ~1/4 大小 |
| 适用场景 | 训练（8GB显存微调7B） | 部署（推理加速+节省存储） |

### 量化参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| bits | 4 | 量化位数，4bit 性价比最高 |
| group_size | 128 | 分组大小，越小精度越高但文件越大 |
| sym | true | 对称量化，INT4 推荐开启 |
| desc_act | true | 按激活值排序，显著提升精度 |
| max_calib_samples | 128 | 校准样本数，越多越准 |

### 完整链路

```
训练完成 → 合并LoRA → GPTQ 量化 → vLLM 部署
   ↓           ↓           ↓            ↓
train.py  merge_lora.py  quantize.py  serve_vllm.py
```

## ⚡ vLLM 部署

### LoRA 合并

部署前需将 LoRA 适配器合并到基础模型，可选再执行 GPTQ 量化：

```bash
# 合并 LoRA 到基础模型
python merge_lora.py \
  --base-model Qwen/Qwen2.5-7B \
  --adapter outputs/final_adapter \
  --output-path models/merged-7b

# [可选] GPTQ 量化 — 推荐部署前执行
python quantize.py --model models/merged-7b --output models/merged-7b-gptq-int4

# 验证合并结果
python inference.py --model-path models/merged-7b --instruction "测试合并模型"
```

### 启动推理服务

```bash
# 基础启动（使用 GPTQ 量化模型）
python serve_vllm.py --model models/merged-7b-gptq-int4

# 自定义参数
python serve_vllm.py \
  --model models/merged-7b-gptq-int4 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

### API 调用示例

```bash
# curl 调用
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/merged-7b",
    "prompt": "请介绍一下LoRA技术",
    "max_tokens": 512,
    "temperature": 0.7
  }'

# Python 调用
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="models/merged-7b",
    messages=[{"role": "user", "content": "什么是LoRA微调？"}],
    max_tokens=512,
    temperature=0.7
)
print(response.choices[0].message.content)
```

### Docker 部署

```bash
# 构建镜像
docker build -f Dockerfile.vllm -t llm-vllm:latest .

# 运行容器
docker run -d \
  --gpus all \
  -v ./models:/models \
  -p 8000:8000 \
  --name llm-server \
  llm-vllm:latest \
  --model /models/merged-7b \
  --tensor-parallel-size 1

# 查看日志
docker logs -f llm-server
```

## 📈 性能对比

### HuggingFace vs vLLM 对比

| 指标 | HuggingFace (原生) | vLLM | 提升倍数 |
|------|-------------------|------|----------|
| 吞吐量 (tokens/s) | ~30 | ~200 | **6-7x** |
| 首 Token 延迟 | ~800ms | ~120ms | **6-7x** |
| 并发支持 | 1 请求 | 8-16 请求 | **8-16x** |
| 显存效率 | 基准 | PagedAttention 优化 | 更优 |
| 长文本支持 | 受限 | 支持 4K-32K | 更灵活 |

> 💡 以上数据基于 Qwen2.5-7B、RTX 3070 8GB、输入 256 tokens、输出 256 tokens 测试

### 测试环境说明

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3070 8GB |
| CPU | Intel i7-12700K |
| 内存 | 32GB DDR5 |
| 系统 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 12.4 |
| vLLM | 0.6.x |

## 🔮 多模态 VLM 微调（新增）

支持 **Qwen2.5-VL** 和 **InternVL** 双架构，提供 **QLoRA / LoRA / Full** 三种微调策略。

### 支持的模型架构

| 架构 | 推荐模型 | 说明 |
|------|---------|------|
| Qwen2.5-VL | Qwen/Qwen2.5-VL-7B-Instruct | 阿里通义千问视觉语言模型 |
| InternVL | OpenGVLab/InternVL3-8B | 上海AI Lab 视觉语言模型 |

### 三种微调策略对比

| 策略 | 显存需求 | 精度 | 适用场景 |
|------|---------|------|---------|
| QLoRA | 8-12GB | 4bit NF4 | 消费级GPU，快速实验 |
| LoRA | 16-24GB | bf16 | 追求更好效果 |
| Full | 40GB+ | bf16 | 追求最佳效果，需A100 |

### 多模态快速开始

```bash
# Qwen2.5-VL QLoRA 微调（8GB显存可用）
python train_vlm.py --config configs/qwen25_vl_qlora.yaml

# InternVL3 LoRA 微调
python train_vlm.py --config configs/internvl35_lora.yaml

# 指定策略和模型
python train_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --strategy qlora --epochs 3

# DeepSpeed 多卡训练
python train_vlm.py --config configs/qwen25_vl_lora.yaml --deepspeed configs/deepspeed_zero2.json
```

### VLM 评估

```bash
# VQA 准确率评估
python eval_vlm.py --model outputs/qwen25_vl_lora/final_adapter --task vqa

# 对比两个模型
python eval_vlm.py --model-a outputs/qwen25_vl_lora/final_adapter --model-b Qwen/Qwen2.5-VL-7B-Instruct --task vqa

# 推理速度 Benchmark
python eval_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark
```

### 多模态显存估算

| 模型 | QLoRA | LoRA | Full |
|------|-------|------|------|
| Qwen2.5-VL-3B | ~4GB | ~8GB | ~14GB |
| Qwen2.5-VL-7B | ~8GB | ~16GB | ~30GB |
| InternVL3-2B | ~3GB | ~6GB | ~10GB |
| InternVL3-8B | ~8GB | ~16GB | ~30GB |

### 多模态数据格式

支持 VQA / 图像描述 / 多轮对话 三种格式（JSONL）：

```jsonl
{"image": "images/001.jpg", "question": "图中有什么？", "answer": "一只猫。"}
{"image": "images/002.jpg", "caption": "日落的城市天际线。"}
{"image": "images/003.jpg", "conversations": [{"role": "user", "content": "<image>\n描述图片"}, {"role": "assistant", "content": "这是一幅风景照。"}]}
```

### 多模态项目结构

```
src/multimodal/
├── __init__.py          # 模块入口
├── model_factory.py     # 模型工厂（统一加载Qwen2-VL/InternVL）
├── data_processor.py    # 多模态数据处理
├── vlm_trainer.py       # VLM训练核心
└── evaluator.py         # 评估基准（VQA/Caption/OCR/速度）

configs/
├── qwen25_vl_lora.yaml     # Qwen2.5-VL LoRA
├── qwen25_vl_qlora.yaml    # Qwen2.5-VL QLoRA
├── internvl35_lora.yaml    # InternVL3 LoRA
├── internvl35_qlora.yaml   # InternVL3 QLoRA
└── vlm_full_finetune.yaml  # 全量微调

train_vlm.py             # VLM训练入口
eval_vlm.py              # VLM评估入口
data/multimodal/         # 多模态数据目录
```

## 📄 License

MIT License

---

<p align="center">
  ⭐ 如果这个项目对你有帮助，请点个 Star！
</p>
