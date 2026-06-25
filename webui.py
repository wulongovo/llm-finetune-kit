#!/usr/bin/env python3
"""
LLM Fine-tune Kit - Web UI 启动入口

用法:
  python webui.py
  python webui.py --port 7860 --share
"""

import argparse
from src.webui import launch_webui


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    launch_webui(port=args.port, share=args.share)


if __name__ == "__main__":
    main()
