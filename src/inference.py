"""
推理模块 - 加载LoRA适配器进行推理
支持单轮对话和批量推理
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional, List


class LoRAInference:
    """LoRA适配器推理引擎"""

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        device: str = "auto",
    ):
        print(f"加载基础模型: {base_model}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        print(f"加载LoRA适配器: {adapter_path}")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("✅ 模型加载完成")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只取新生成的部分
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(
        self,
        instruction: str,
        input_text: str = "",
        **kwargs,
    ) -> str:
        """Alpaca格式对话"""
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"

        return self.generate(prompt, **kwargs)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


def merge_and_export(
    base_model: str,
    adapter_path: str,
    output_path: str,
):
    """合并LoRA权重到基础模型并保存完整模型"""
    print(f"加载基础模型: {base_model}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"加载LoRA适配器: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("合并权重...")
    merged_model = model.merge_and_unload()

    print(f"保存合并模型: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✅ 合并完成: {output_path}")

    return merged_model
