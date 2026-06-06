#!/usr/bin/env python3
"""
LLM Fine-tune Kit - 模型导出

用法:
  python export.py --adapter outputs/final_adapter --name my-model
  python export.py --adapter outputs/final_adapter --merge --output ./merged_model
"""

import argparse
import os
from src.export import export_to_ollama, export_to_gguf
from src.inference import merge_and_export


def main():
    parser = argparse.ArgumentParser(description="Export LoRA model")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--base-model", type=str, default=None, help="基础模型路径")
    parser.add_argument("--merge", action="store_true", help="合并LoRA到基础模型")
    parser.add_argument("--output", type=str, default="./merged_model", help="合并输出路径")
    parser.add_argument("--name", type=str, default="my-finetuned", help="Ollama模型名称")
    parser.add_argument("--ollama", action="store_true", help="导出到Ollama")
    args = parser.parse_args()

    # 推断基础模型
    if args.base_model is None:
        import json
        adapter_config = os.path.join(args.adapter, "adapter_config.json")
        if os.path.exists(adapter_config):
            with open(adapter_config) as f:
                cfg = json.load(f)
            args.base_model = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B")

    if args.merge or args.ollama:
        # 合并模型
        merged_path = args.output
        if not os.path.exists(merged_path):
            merge_and_export(args.base_model, args.adapter, merged_path)

        if args.ollama:
            # 导出到Ollama
            export_to_ollama(merged_path, args.name)
    else:
        print("请指定操作:")
        print("  --merge              合并LoRA到基础模型")
        print("  --ollama             导出到Ollama")
        print("  --merge --ollama     合并并导出到Ollama")


if __name__ == "__main__":
    main()
