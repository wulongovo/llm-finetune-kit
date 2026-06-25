#!/usr/bin/env python3
"""
LLM Fine-tune Kit - 推理入口

用法:
  python inference.py --adapter outputs/final_adapter --instruction "什么是LoRA？"
  python inference.py --adapter outputs/final_adapter --interactive
"""

import argparse
from src.inference import LoRAInference


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Inference")
    parser.add_argument("--base-model", type=str, default=None, help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--instruction", type=str, default=None, help="单次推理指令")
    parser.add_argument("--input", type=str, default="", help="额外输入")
    parser.add_argument("--interactive", action="store_true", help="交互式对话")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    args = parser.parse_args()

    # 从适配器配置推断基础模型
    if args.base_model is None:
        import os, json
        adapter_config = os.path.join(args.adapter, "adapter_config.json")
        if os.path.exists(adapter_config):
            with open(adapter_config) as f:
                cfg = json.load(f)
            args.base_model = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B")
        else:
            args.base_model = "Qwen/Qwen2.5-7B"

    # 加载模型
    engine = LoRAInference(
        base_model=args.base_model,
        adapter_path=args.adapter,
    )

    if args.interactive:
        print("\n💬 交互模式 (输入 quit 退出)\n")
        while True:
            instruction = input("📝 指令: ").strip()
            if instruction.lower() in ["quit", "exit", "q"]:
                break
            if not instruction:
                continue

            response = engine.chat(
                instruction,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\n🤖 回答: {response}\n")

    elif args.instruction:
        response = engine.chat(
            args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n🤖 回答:\n{response}")
    else:
        print("请指定 --instruction 或 --interactive")


if __name__ == "__main__":
    main()
