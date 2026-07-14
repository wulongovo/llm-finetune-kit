#!/usr/bin/env python3
"""
LLM Fine-tune Kit - 多模态 VLM 评估入口

用法:
  # 单模型评估
  python eval_vlm.py --model outputs/qwen25_vl_lora/final_adapter --task vqa
  python eval_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --task caption
  python eval_vlm.py --model outputs/internvl35_lora/final_adapter --task ocr

  # 对比评估
  python eval_vlm.py --model-a outputs/qwen25_vl_lora/final_adapter --model-b Qwen/Qwen2.5-VL-7B-Instruct --task vqa

  # 推理速度 Benchmark
  python eval_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark
"""

import argparse
import os
import sys
import json
import torch

from src.multimodal.evaluator import VLMEvaluator, compare_models
from src.multimodal.model_factory import detect_model_architecture


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tune Kit - VLM 评估")

    # 单模型评估
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（单模型评估）")
    parser.add_argument("--strategy", choices=["qlora", "lora", "full"], default="qlora",
                        help="微调策略（用于加载模型）")

    # 对比评估
    parser.add_argument("--model-a", type=str, default=None,
                        help="模型A路径（对比评估）")
    parser.add_argument("--model-b", type=str, default=None,
                        help="模型B路径（对比评估）")

    # 评估配置
    parser.add_argument("--task", choices=["vqa", "caption", "ocr"], default="vqa",
                        help="评估任务类型")
    parser.add_argument("--data", type=str, default=None,
                        help="测试数据路径")
    parser.add_argument("--image-dir", type=str, default="data/multimodal/images",
                        help="图片目录")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果输出路径")

    # Benchmark
    parser.add_argument("--benchmark", action="store_true",
                        help="运行推理速度 Benchmark")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Benchmark 运行次数")

    args = parser.parse_args()

    # ============================================================
    # 模式判断
    # ============================================================

    if args.model_a and args.model_b:
        # 对比评估模式
        run_comparison(args)
    elif args.model:
        if args.benchmark:
            # Benchmark 模式
            run_benchmark(args)
        else:
            # 单模型评估模式
            run_single_eval(args)
    else:
        print("错误: 请指定 --model 或 --model-a / --model-b")
        parser.print_help()
        sys.exit(1)


def run_single_eval(args):
    """单模型评估"""
    print(f"\n加载模型: {args.model}")
    print(f"评估任务: {args.task}")
    print(f"图片目录: {args.image_dir}\n")

    # 创建评估器
    evaluator = VLMEvaluator()
    evaluator.load_model(args.model, strategy=args.strategy)

    # 加载测试数据
    test_data = load_test_data(args.data, args.task)

    # 执行评估
    if args.task == "vqa":
        result = evaluator.evaluate_vqa(test_data, args.image_dir)
        print(f"\nVQA 评估结果:")
        print(f"  准确率: {result['accuracy']:.2%}")
        print(f"  正确数: {result['correct']}/{result['total']}")
    elif args.task == "caption":
        result = evaluator.evaluate_caption(test_data, args.image_dir)
        print(f"\n图像描述评估结果:")
        print(f"  BLEU-1:         {result['bleu_1']:.3f}")
        print(f"  关键词覆盖率:   {result['keyword_coverage']:.3f}")
    elif args.task == "ocr":
        result = evaluator.evaluate_ocr(test_data, args.image_dir)
        print(f"\nOCR 评估结果:")
        print(f"  字符准确率:   {result['char_accuracy']:.2%}")
        print(f"  完全匹配率:   {result['exact_match_accuracy']:.2%}")

    # 保存结果
    if args.output:
        save_result(result, args.output)


def run_comparison(args):
    """对比评估"""
    print(f"\n模型A: {args.model_a}")
    print(f"模型B: {args.model_b}")
    print(f"评估任务: {args.task}\n")

    # 加载两个模型
    evaluator_a = VLMEvaluator()
    evaluator_a.load_model(args.model_a, strategy=args.strategy)

    evaluator_b = VLMEvaluator()
    evaluator_b.load_model(args.model_b, strategy=args.strategy)

    # 加载测试数据
    test_data = load_test_data(args.data, args.task)

    # 对比评估
    result = compare_models(evaluator_a, evaluator_b, test_data, args.image_dir, args.task)

    # 保存结果
    if args.output:
        save_result(result, args.output)


def run_benchmark(args):
    """推理速度 Benchmark"""
    print(f"\nBenchmark 模型: {args.model}")
    print(f"运行次数: {args.num_runs}\n")

    evaluator = VLMEvaluator()
    evaluator.load_model(args.model, strategy=args.strategy)

    result = evaluator.benchmark_speed(num_runs=args.num_runs)

    print(f"\nBenchmark 结果:")
    print(f"  平均延迟:   {result['avg_latency']:.2f}s")
    print(f"  最小延迟:   {result['min_latency']:.2f}s")
    print(f"  最大延迟:   {result['max_latency']:.2f}s")
    print(f"  平均吞吐:   {result['avg_tokens_per_second']:.1f} tokens/s")

    if args.output:
        save_result(result, args.output)


def load_test_data(data_path: str, task: str) -> list:
    """加载测试数据"""
    if data_path and os.path.exists(data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"加载了 {len(data)} 条测试数据 from {data_path}")
        return data
    else:
        print(f"测试数据不存在: {data_path or 'None'}，使用示例数据")
        from src.multimodal.data_processor import create_sample_data
        sample_path = create_sample_data("data/multimodal")
        return load_test_data(sample_path, task)


def save_result(result: dict, output_path: str):
    """保存评估结果"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 移除 details 等大数据字段
    save_data = {}
    for k, v in result.items():
        if k == "details":
            save_data[k] = f"[{len(v)} items]"
        elif isinstance(v, (int, float, str, bool)):
            save_data[k] = v
        elif isinstance(v, dict):
            save_data[k] = {kk: vv for kk, vv in v.items()
                           if isinstance(vv, (int, float, str, bool))}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存: {output_path}")


if __name__ == "__main__":
    main()
