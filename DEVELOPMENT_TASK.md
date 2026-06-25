# llm-finetune-kit 增强任务规范

## 目标
在现有 LoRA 微调工具包基础上，新增两大功能模块：
1. DeepSpeed 多卡分布式训练
2. vLLM 高性能推理部署

## 面试价值
- "模型太大单卡放不下怎么办" → DeepSpeed ZeRO-3 多卡并行
- "如何优化推理速度" → vLLM 高性能推理引擎
- "训练到部署完整链路" → 展示工程化能力

---

## 任务 1：DeepSpeed 多卡训练

### 需要创建的文件

#### 1. configs/deepspeed_zero2.json
```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": true
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null
  }
}
```

#### 2. configs/deepspeed_zero3.json
```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  },
  "bf16": {
    "enabled": true
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null
  }
}
```

#### 3. scripts/train_multigpu.sh
```bash
#!/bin/bash
# DeepSpeed 多卡训练启动脚本
# Usage: bash scripts/train_multigpu.sh [config] [deepspeed_config]

CONFIG=${1:-"configs/lora_config.yaml"}
DEEPSPEED=${2:-"configs/deepspeed_zero2.json"}
NUM_GPUS=${3:-$(nvidia-smi -L | wc -l)}

echo "=========================================="
echo "  DeepSpeed 多卡训练"
echo "  GPU数量: $NUM_GPUS"
echo "  训练配置: $CONFIG"
echo "  DeepSpeed: $DEEPSPEED"
echo "=========================================="

deepspeed --num_gpus=$NUM_GPUS train.py \
    --config $CONFIG \
    --deepspeed $DEEPSPEED
```

#### 4. scripts/train_multigpu_windows.bat
```bat
@echo off
REM DeepSpeed 多卡训练 (Windows)
set CONFIG=%1
set DEEPSPEED=%2
if "%CONFIG%"=="" set CONFIG=configs\lora_config.yaml
if "%DEEPSPEED%"=="" set DEEPSPEED=configs\deepspeed_zero2.json

echo ==========================================
echo   DeepSpeed 多卡训练
echo   训练配置: %CONFIG%
echo   DeepSpeed: %DEEPSPEED%
echo ==========================================

deepspeed --num_gpus=2 train.py --config %CONFIG% --deepspeed %DEEPSPEED%
```

### 需要修改的文件

#### 1. src/trainer.py
在 `create_training_args` 函数中添加 DeepSpeed 支持：
- 检查是否有 `--deepspeed` 参数
- 如果有，设置 `deepspeed` 参数到 TrainingArguments
- 添加 DeepSpeed 环境检测和日志

#### 2. train.py
添加命令行参数：
```python
parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")
```

#### 3. requirements.txt
添加依赖：
```
deepspeed>=0.14.0
```

#### 4. README.md
添加多卡训练章节：
- DeepSpeed 配置说明
- 多卡启动命令
- 常见问题排查

---

## 任务 2：vLLM 高性能部署

### 需要创建的文件

#### 1. src/serving.py
vLLM 推理服务模块：
```python
"""
vLLM 高性能推理服务
支持 OpenAI 兼容 API
"""

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
import asyncio
import json
from typing import List, Dict, Optional

class VLLMServer:
    """vLLM 推理服务器"""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            max_model_len=kwargs.get("max_model_len", 4096),
            dtype=kwargs.get("dtype", "bfloat16"),
        )
        self.sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
        )
    
    def generate(self, prompts: List[str]) -> List[str]:
        """批量生成"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话格式"""
        prompt = self._messages_to_prompt(messages)
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        output = self.llm.generate([prompt], sampling_params)
        return output[0].outputs[0].text
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """将消息列表转换为 prompt"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt


def merge_lora_for_vllm(adapter_path: str, output_path: str):
    """合并 LoRA 适配器用于 vLLM 部署"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"加载基础模型和适配器: {adapter_path}")
    
    # 加载适配器配置获取基础模型路径
    adapter_config_path = f"{adapter_path}/adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 合并权重
    print("合并 LoRA 权重...")
    model = model.merge_and_unload()
    
    # 保存
    print(f"保存合并后的模型: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ 合并完成！可以用于 vLLM 部署")
```

#### 2. src/benchmark.py
性能对比工具：
```python
"""
推理性能对比：HuggingFace vs vLLM
生成 benchmark 报告
"""

import time
import json
import statistics
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    framework: str
    model: str
    num_requests: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p99_latency: float
    throughput: float  # tokens/s

class InferenceBenchmark:
    """推理性能对比测试"""
    
    def __init__(self, model_path: str, adapter_path: str = None):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.test_prompts = [
            "请介绍一下LoRA微调技术的原理。",
            "什么是大语言模型？",
            "解释一下DeepSpeed ZeRO优化的三个阶段。",
            "如何评估一个RAG系统的效果？",
            "请详细说明RLHF的训练流程。",
        ]
    
    def benchmark_huggingface(self, num_requests: int = 10) -> BenchmarkResult:
        """HuggingFace Transformers 推理测试"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("加载 HuggingFace 模型...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        latencies = []
        total_tokens = 0
        
        print(f"运行 {num_requests} 次推理...")
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            start = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256)
            tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
            end = time.perf_counter()
            
            latencies.append(end - start)
            total_tokens += tokens_generated
        
        return BenchmarkResult(
            framework="HuggingFace",
            model=self.model_path,
            num_requests=num_requests,
            total_time=sum(latencies),
            avg_latency=statistics.mean(latencies),
            p50_latency=statistics.median(latencies),
            p99_latency=sorted(latencies)[int(len(latencies) * 0.99)],
            throughput=total_tokens / sum(latencies),
        )
    
    def benchmark_vllm(self, num_requests: int = 10) -> BenchmarkResult:
        """vLLM 推理测试"""
        from vllm import LLM, SamplingParams
        
        print("加载 vLLM 模型...")
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        sampling_params = SamplingParams(max_tokens=256)
        
        prompts = [self.test_prompts[i % len(self.test_prompts)] for i in range(num_requests)]
        
        print(f"运行 {num_requests} 次推理...")
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end = time.perf_counter()
        
        total_time = end - start
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        
        return BenchmarkResult(
            framework="vLLM",
            model=self.model_path,
            num_requests=num_requests,
            total_time=total_time,
            avg_latency=total_time / num_requests,
            p50_latency=total_time / num_requests,
            p99_latency=total_time / num_requests,
            throughput=total_tokens / total_time,
        )
    
    def run_comparison(self, num_requests: int = 10) -> Dict:
        """运行对比测试"""
        results = {}
        
        print("=" * 60)
        print("  推理性能对比测试")
        print("=" * 60)
        
        # HuggingFace 测试
        print("\n[1/2] HuggingFace Transformers")
        results["huggingface"] = self.benchmark_huggingface(num_requests)
        
        # vLLM 测试
        print("\n[2/2] vLLM")
        results["vllm"] = self.benchmark_vllm(num_requests)
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """生成测试报告"""
        report = []
        report.append("# 推理性能对比报告\n")
        report.append(f"模型: {self.model_path}\n")
        report.append("| 指标 | HuggingFace | vLLM | 提升倍数 |")
        report.append("|------|-------------|------|----------|")
        
        hf = results["huggingface"]
        vllm = results["vllm"]
        
        report.append(f"| 平均延迟 | {hf.avg_latency:.3f}s | {vllm.avg_latency:.3f}s | {hf.avg_latency/vllm.avg_latency:.2f}x |")
        report.append(f"| 吞吐量 | {hf.throughput:.2f} tok/s | {vllm.throughput:.2f} tok/s | {vllm.throughput/hf.throughput:.2f}x |")
        report.append(f"| P99延迟 | {hf.p99_latency:.3f}s | {vllm.p99_latency:.3f}s | {hf.p99_latency/vllm.p99_latency:.2f}x |")
        
        report.append("\n## 结论\n")
        speedup = vllm.throughput / hf.throughput
        if speedup > 2:
            report.append(f"vLLM 吞吐量提升 **{speedup:.1f}倍**，显著优于 HuggingFace。")
        elif speedup > 1.2:
            report.append(f"vLLM 吞吐量提升 **{speedup:.1f}倍**，有明显优势。")
        else:
            report.append(f"两者性能接近，vLLM 提升 {speedup:.1f}倍。")
        
        return "\n".join(report)
```

#### 3. scripts/start_vllm_server.sh
```bash
#!/bin/bash
# 启动 vLLM 推理服务

MODEL=${1:-"outputs/merged_model"}
PORT=${2:-8000}
TENSOR_PARALLEL=${3:-1}

echo "=========================================="
echo "  vLLM 推理服务"
echo "  模型: $MODEL"
echo "  端口: $PORT"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

#### 4. scripts/start_vllm_server.bat
```bat
@echo off
REM 启动 vLLM 推理服务 (Windows)

set MODEL=%1
set PORT=%2
if "%MODEL%"=="" set MODEL=outputs\merged_model
if "%PORT%"=="" set PORT=8000

echo ==========================================
echo   vLLM 推理服务
echo   模型: %MODEL%
echo   端口: %PORT%
echo ==========================================

python -m vllm.entrypoints.openai.api_server --model %MODEL% --port %PORT% --dtype bfloat16
```

#### 5. configs/vllm_config.yaml
```yaml
# vLLM 部署配置

model:
  path: "outputs/merged_model"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  dtype: "bfloat16"

server:
  host: "0.0.0.0"
  port: 8000

sampling:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512
```

#### 6. Dockerfile
```dockerfile
FROM vllm/vllm-openai:latest

# 设置工作目录
WORKDIR /app

# 复制模型文件
COPY outputs/merged_model /app/model

# 复制配置
COPY configs/vllm_config.yaml /app/config.yaml

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model", \
     "--port", "8000", \
     "--dtype", "bfloat16"]
```

#### 7. scripts/deploy_docker.sh
```bash
#!/bin/bash
# Docker 部署脚本

MODEL_DIR=${1:-"outputs/merged_model"}
IMAGE_NAME="llm-finetune-kit"
PORT=${2:-8000}

echo "构建 Docker 镜像..."
docker build -t $IMAGE_NAME .

echo "启动容器..."
docker run -d \
    --gpus all \
    -p $PORT:8000 \
    -v $(pwd)/$MODEL_DIR:/app/model \
    $IMAGE_NAME

echo "服务已启动: http://localhost:$PORT"
echo "API 文档: http://localhost:$PORT/docs"
```

### 需要修改的文件

#### 1. requirements.txt
添加依赖：
```
vllm>=0.5.0
```

#### 2. README.md
添加部署章节：
- vLLM 部署说明
- Docker 部署说明
- API 调用示例
- 性能对比数据

---

## 任务 3：更新项目文档

### README.md 更新内容

1. 特性列表添加：
   - 🚀 DeepSpeed 多卡训练 — 支持 ZeRO-2/ZeRO-3 分布式训练
   - ⚡ vLLM 高性能部署 — 推理速度提升 5-10 倍

2. 技术栈表格添加：
   | DeepSpeed | 分布式训练 |
   | vLLM | 高性能推理 |

3. 新增章节：
   - 多卡训练
   - vLLM 部署
   - 性能对比

### 新建 RESUME_PROJECT_DESC.md
项目简历描述（面试用）

---

## 验证清单

- [ ] DeepSpeed 配置文件语法正确
- [ ] 多卡训练脚本可执行
- [ ] vLLM 服务可启动
- [ ] benchmark 可运行
- [ ] Docker 镜像可构建
- [ ] README 文档完整
- [ ] 简历描述清晰
