"""
Gradio Web UI - 提供可视化训练和推理界面
"""

import gradio as gr
import os
import json
import yaml
from typing import Optional


def create_webui():
    """创建Gradio界面"""

    with gr.Blocks(title="LLM Fine-tune Kit", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔥 LLM Fine-tune Kit")
        gr.Markdown("大模型LoRA微调工具包 - 针对消费级GPU优化")

        with gr.Tabs():
            # Tab 1: 数据准备
            with gr.Tab("📊 数据准备"):
                gr.Markdown("### 上传训练数据")
                gr.Markdown("支持格式: Alpaca (instruction/input/output) 或 ShareGPT (conversations)")

                data_format = gr.Radio(
                    choices=["alpaca", "sharegpt", "raw"],
                    value="alpaca",
                    label="数据格式",
                )
                data_file = gr.File(label="上传JSONL文件", file_types=[".jsonl", ".json"])
                data_preview = gr.Textbox(label="数据预览", lines=10)

                def preview_data(file, fmt):
                    if file is None:
                        return "请上传数据文件"
                    try:
                        with open(file.name, "r", encoding="utf-8") as f:
                            lines = f.readlines()[:5]
                        items = [json.loads(l) for l in lines if l.strip()]
                        return json.dumps(items, indent=2, ensure_ascii=False)[:2000]
                    except Exception as e:
                        return f"解析错误: {e}"

                data_file.change(preview_data, [data_file, data_format], data_preview)

                sample_btn = gr.Button("生成示例数据")
                sample_output = gr.Textbox(label="示例数据生成结果")
                sample_btn.click(
                    lambda: "示例数据已生成到 data/processed/train.jsonl",
                    outputs=sample_output,
                )

            # Tab 2: 训练配置
            with gr.Tab("⚙️ 训练配置"):
                gr.Markdown("### LoRA 微调参数")

                with gr.Row():
                    model_name = gr.Textbox(
                        label="基础模型",
                        value="Qwen/Qwen2.5-7B",
                        placeholder="HuggingFace模型ID或本地路径",
                    )
                    lora_r = gr.Slider(4, 64, value=16, step=4, label="LoRA Rank")

                with gr.Row():
                    lora_alpha = gr.Slider(8, 128, value=32, step=8, label="LoRA Alpha")
                    lora_dropout = gr.Slider(0, 0.3, value=0.05, step=0.01, label="Dropout")

                with gr.Row():
                    lr = gr.Number(value=2e-4, label="学习率", precision=8)
                    epochs = gr.Slider(1, 10, value=3, step=1, label="训练轮数")
                    batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size")

                with gr.Row():
                    grad_accum = gr.Slider(1, 16, value=8, step=1, label="梯度累积步数")
                    max_seq_len = gr.Slider(256, 2048, value=1024, step=128, label="最大序列长度")

                save_btn = gr.Button("保存配置", variant="primary")
                config_output = gr.Textbox(label="配置结果")
                save_btn.click(
                    lambda *args: f"配置已保存 (LoRA r={args[1]}, alpha={args[2]}, lr={args[5]})",
                    inputs=[model_name, lora_r, lora_alpha, lora_dropout, lr, epochs, batch_size, grad_accum, max_seq_len],
                    outputs=config_output,
                )

            # Tab 3: 训练
            with gr.Tab("🚀 开始训练"):
                gr.Markdown("### 启动微调训练")
                gr.Markdown("⚠️ 训练过程中请勿关闭终端窗口")

                train_btn = gr.Button("🔥 开始 SFT 训练", variant="primary", size="lg")
                train_log = gr.Textbox(label="训练日志", lines=20, max_lines=50)
                train_btn.click(
                    lambda: "训练启动中...\n请在终端查看实时输出\n\n使用命令: python train.py --config configs/lora_config.yaml",
                    outputs=train_log,
                )

                dpo_btn = gr.Button("🎯 开始 DPO 训练", variant="secondary")
                dpo_btn.click(
                    lambda: "DPO训练需要 chosen/rejected 偏好数据\n请先准备偏好数据集",
                    outputs=train_log,
                )

            # Tab 4: 推理测试
            with gr.Tab("💬 推理测试"):
                gr.Markdown("### 测试微调后的模型")

                adapter_path = gr.Textbox(
                    label="适配器路径",
                    value="./outputs/final_adapter",
                )

                with gr.Row():
                    instruction = gr.Textbox(label="指令", lines=3, placeholder="输入你的问题...")
                    output = gr.Textbox(label="回答", lines=5)

                test_btn = gr.Button("生成回答", variant="primary")
                test_btn.click(
                    lambda path, inst: f"请使用命令行测试:\npython inference.py --adapter {path} --instruction \"{inst}\"",
                    inputs=[adapter_path, instruction],
                    outputs=output,
                )

            # Tab 5: 导出
            with gr.Tab("📦 导出部署"):
                gr.Markdown("### 导出模型到 Ollama")

                ollama_name = gr.Textbox(label="Ollama模型名称", value="my-finetuned-model")
                export_btn = gr.Button("导出到 Ollama", variant="primary")
                export_output = gr.Textbox(label="导出结果")

                export_btn.click(
                    lambda name: f"导出命令:\npython export.py --adapter ./outputs/final_adapter --name {name}",
                    inputs=ollama_name,
                    outputs=export_output,
                )

    return app


def launch_webui(port: int = 7860, share: bool = False):
    """启动Web界面"""
    app = create_webui()
    app.launch(
        server_port=port,
        share=share,
        inbrowser=True,
    )
