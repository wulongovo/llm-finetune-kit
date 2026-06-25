"""
模型导出模块
支持导出到 Ollama 格式（GGUF）和合并保存
"""

import os
import subprocess
import json
from typing import Optional


def export_to_ollama(
    merged_model_path: str,
    ollama_model_name: str,
    modelfile_template: Optional[str] = None,
    quantize: str = "Q4_K_M",
):
    """
    将合并后的模型导出到 Ollama

    Args:
        merged_model_path: 合并后模型的路径
        ollama_model_name: Ollama模型名称
        modelfile_template: 自定义Modelfile路径（可选）
        quantize: 量化方式 (Q4_K_M, Q5_K_M, Q8_0)
    """
    print(f"导出到 Ollama: {ollama_model_name}")

    # 创建临时目录
    tmp_dir = os.path.join(os.path.dirname(merged_model_path), "ollama_export")
    os.makedirs(tmp_dir, exist_ok=True)

    # 生成 Modelfile
    if modelfile_template and os.path.exists(modelfile_template):
        with open(modelfile_template, "r") as f:
            modelfile_content = f.read()
    else:
        modelfile_content = f"""FROM {merged_model_path}

TEMPLATE """### 指令:
{{{{ .Prompt }}}}

### 回答:
"""

PARAMETER stop "### 指令:"
PARAMETER stop "### 输入:"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""

    modelfile_path = os.path.join(tmp_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"Modelfile: {modelfile_path}")

    # 创建Ollama模型
    try:
        result = subprocess.run(
            ["ollama", "create", ollama_model_name, "-f", modelfile_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"✅ Ollama模型创建成功: {ollama_model_name}")
            print(f"   运行: ollama run {ollama_model_name}")
        else:
            print(f"❌ 创建失败: {result.stderr}")
    except FileNotFoundError:
        print("❌ ollama 命令未找到，请确保 Ollama 已安装")
        print(f"   Modelfile 已生成: {modelfile_path}")
        print(f"   手动执行: ollama create {ollama_model_name} -f {modelfile_path}")


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantize: str = "Q4_K_M",
):
    """
    导出为GGUF格式（用于llama.cpp / Ollama）

    需要安装: pip install llama-cpp-python
    或使用 llama.cpp 的 convert 脚本
    """
    print(f"导出 GGUF: {output_path}")
    print("提示: 建议使用 llama.cpp 的 convert-hf-to-gguf.py 工具")
    print(f"  python convert-hf-to-gguf.py {model_path} --outfile {output_path} --outtype {quantize}")


def create_training_report(trainer_output, config_path: str, output_dir: str):
    """生成训练报告"""
    report = {
        "config": config_path,
        "output_dir": output_dir,
        "status": "completed",
    }

    # 读取训练日志
    log_file = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])
        if log_history:
            final_log = log_history[-1]
            report["final_loss"] = final_log.get("loss")
            report["final_lr"] = final_log.get("learning_rate")
            report["total_steps"] = final_log.get("step")

    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n训练报告: {report_path}")
    return report
