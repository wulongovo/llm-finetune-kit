"""
vLLM 高性能推理服务
支持 OpenAI 兼容 API、批量生成和对话格式
"""

import json
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams


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

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.sampling_params.temperature),
            top_p=kwargs.get("top_p", self.sampling_params.top_p),
            max_tokens=kwargs.get("max_tokens", self.sampling_params.max_tokens),
        )
        outputs = self.llm.generate(prompts, sampling_params)
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
