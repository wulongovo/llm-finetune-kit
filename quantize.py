#!/usr/bin/env python3
"""
LLM Fine-tune Kit - GPTQ 量化入口

用法:
  # 使用配置文件
  python quantize.py --config configs/gptq.yaml

  # 直接指定路径
  python quantize.py --model models/merged-7b --output models/merged-7b-gptq

  # 使用自定义校准数据
  python quantize.py --model models/merged-7b --output models/merged-7b-gptq --calibration data/eval.jsonl

  # 测试已量化的模型
  python quantize.py --test --model models/merged-7b-gptq
"""

import argparse
import os
import sys
import yaml
import torch

from src.quantize import GPTQQuantizer, GPTQConfig


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ 模型量化 — 将完整模型压缩为 4/8bit 用于高效部署",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quantize.py --model models/merged-7b --output models/merged-7b-gptq-int4
  python quantize.py --config configs/gptq.yaml
  python quantize.py --test --model models/merged-7b-gptq-int4
        """
    )

    parser.add_argument("--config", type=str, default=None,
                        help="GPTQ 配置文件路径 (configs/gptq.yaml)")
    parser.add_argument("--model", type=str, default=None,
                        help="待量化的模型路径")
    parser.add_argument("--output", type=str, default=None,
                        help="量化模型输出路径")
    parser.add_argument("--bits", type=int, default=4,
                        choices=[2, 3, 4, 8],
                        help="量化位数 (默认: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="分组大小 (默认: 128)")
    parser.add_argument("--calibration", type=str, default=None,
                        help="校准数据 JSONL 文件路径")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="最大校准样本数 (默认: 128)")
    parser.add_argument("--sym", action="store_true", default=True,
                        help="对称量化 (默认开启)")
    parser.add_argument("--no-sym", dest="sym", action="store_false",
                        help="禁用对称量化")
    parser.add_argument("--test", action="store_true",
                        help="仅测试已量化的模型（不执行量化）")
    parser.add_argument("--test-prompt", type=str, default=None,
                        help="测试时的自定义提示词")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备 (默认: cuda:0)")

    args = parser.parse_args()

    # ---- 测试模式 ----
    if args.test:
        if not args.model:
            print("❌ 测试模式需要指定 --model 路径")
            sys.exit(1)

        print("加载量化模型进行推理测试...\n")
        quantizer = GPTQQuantizer(args.model, args.model)
        output = quantizer.test(
            prompt=args.test_prompt or "请用一句话介绍一下你自己。",
            device=args.device,
        )
        print(f"\n✅ 测试完成")
        return

    # ---- 配置加载 ----
    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ 配置文件不存在: {args.config}")
            sys.exit(1)

        raw = load_config(args.config)

        model_path = args.model or raw.get("model", {}).get("input_path")
        output_path = args.output or raw.get("model", {}).get("output_path")

        qt_cfg = raw.get("quantization", {})
        bits = args.bits if args.bits != 4 else qt_cfg.get("bits", 4)
        group_size = args.group_size if args.group_size != 128 else qt_cfg.get("group_size", 128)
        sym = qt_cfg.get("sym", True) if args.sym else False

        calib_cfg = raw.get("calibration", {})
        calibration_file = args.calibration or calib_cfg.get("data_file")
        max_samples = args.max_samples or calib_cfg.get("max_samples", 128)
        max_tokens = calib_cfg.get("max_tokens", 2048)
    else:
        # 无配置文件模式：必须指定 --model 和 --output
        if not args.model or not args.output:
            print("❌ 请指定 --model 和 --output，或使用 --config 配置文件")
            sys.exit(1)

        model_path = args.model
        output_path = args.output
        bits = args.bits
        group_size = args.group_size
        sym = args.sym
        calibration_file = args.calibration
        max_samples = args.max_samples
        max_tokens = 2048

    # ---- 参数校验 ----
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print(f"   提示: 请先用 python merge_lora.py 合并 LoRA 适配器")
        sys.exit(1)

    # ---- GPU 检查 ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)\n")

        if gpu_mem < 6:
            print("⚠️  显存 <6GB，GPTQ 量化可能失败，建议使用 8GB+ 显存")

    # ---- 构建配置 ----
    gptq_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        max_calib_samples=max_samples,
        max_calib_tokens=max_tokens,
    )

    # ---- 执行量化 ----
    quantizer = GPTQQuantizer(
        model_path=model_path,
        output_path=output_path,
        config=gptq_config,
    )

    quantizer.quantize(calibration_data=calibration_file)

    print("✅ GPTQ 量化完成!")
    print(f"量化模型保存在: {output_path}")
    print(f"\n下一步:")
    print(f"  1. 测试: python quantize.py --test --model {output_path}")
    print(f"  2. vLLM部署: python serve_vllm.py --model {output_path}")


if __name__ == "__main__":
    main()
