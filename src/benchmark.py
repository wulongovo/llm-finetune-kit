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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="推理性能对比测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA适配器路径")
    parser.add_argument("--num-requests", type=int, default=10, help="测试请求数")
    parser.add_argument("--output", type=str, default="benchmark_report.md", help="报告输出路径")
    args = parser.parse_args()

    benchmark = InferenceBenchmark(args.model, args.adapter)
    results = benchmark.run_comparison(args.num_requests)
    report = benchmark.generate_report(results)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 报告已保存: {args.output}")
    print(report)
