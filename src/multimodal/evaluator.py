"""
VLM 评估器 — 多模态模型评估基准
支持 VQA准确率 / 图像描述质量 / OCR准确率 / 推理速度
"""

import os
import re
import time
import logging
from typing import Dict, List, Any, Optional
from collections import Counter

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VLMEvaluator:
    """VLM 评估器

    提供多种评估维度：
    - VQA 准确率评估
    - 图像描述质量评估（BLEU/CIDEr 简化版）
    - OCR 文字识别准确率
    - 推理速度 benchmark
    """

    def __init__(self, model=None, tokenizer=None, processor=None,
                 architecture: str = "qwen2_vl"):
        """
        Args:
            model: 已加载的 VLM 模型
            tokenizer: 分词器
            processor: 图像处理器
            architecture: 模型架构
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.architecture = architecture

    def load_model(self, model_path: str, strategy: str = "qlora"):
        """从路径加载模型

        Args:
            model_path: 模型路径（可以是 LoRA adapter 路径）
            strategy: 微调策略
        """
        from .model_factory import ModelFactory, detect_model_architecture

        arch = detect_model_architecture(model_path)
        self.architecture = arch

        factory = ModelFactory(model_path, strategy=strategy)
        self.model, self.tokenizer, self.processor = factory.load()
        self.model.eval()

        logger.info(f"评估模型加载完成: {model_path}")

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = 512) -> str:
        """单条推理

        Args:
            image: PIL 图片
            question: 问题文本
            max_new_tokens: 最大生成 token 数

        Returns:
            生成的回答文本
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        if self.architecture == "qwen2_vl":
            return self._generate_qwen2_vl(image, question, max_new_tokens)
        elif self.architecture == "internvl":
            return self._generate_internvl(image, question, max_new_tokens)
        else:
            raise ValueError(f"不支持的架构: {self.architecture}")

    def _generate_qwen2_vl(self, image, question, max_new_tokens):
        """Qwen2-VL 推理"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # 只取生成部分
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _generate_internvl(self, image, question, max_new_tokens):
        """InternVL 推理"""
        # InternVL 对话格式
        prompt = f"<image>\nUser: {question}\nAssistant:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        # InternVL 通常有自己的 generation_config
        if hasattr(self.model, "chat"):
            # 使用 InternVL 内置的 chat 方法
            response, _ = self.model.chat(
                self.tokenizer,
                image,
                question,
                generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
            )
            return response.strip()
        else:
            # 通用 fallback
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ============================================================
    # VQA 准确率评估
    # ============================================================

    def evaluate_vqa(self, test_data: List[Dict],
                     image_dir: str = "") -> Dict[str, Any]:
        """VQA 准确率评估

        Args:
            test_data: 测试数据 [{"image":"path", "question":"...", "answer":"..."}]
            image_dir: 图片根目录

        Returns:
            评估指标 {"accuracy": 0.xx, "total": N, "correct": N, "details": [...]}
        """
        logger.info(f"开始 VQA 评估，共 {len(test_data)} 条数据...")

        correct = 0
        total = 0
        details = []

        for i, sample in enumerate(test_data):
            image_path = os.path.join(image_dir, sample.get("image", ""))
            question = sample.get("question", "")
            ground_truth = sample.get("answer", "")

            # 也支持 conversation 格式
            if not question and "conversations" in sample:
                for turn in sample["conversations"]:
                    if turn["role"] == "user":
                        question = turn["content"].replace("<image>", "").strip()
                    elif turn["role"] == "assistant":
                        ground_truth = turn["content"]

            try:
                image = Image.open(image_path).convert("RGB")
                prediction = self.generate(image, question)
                is_correct = self._match_answer(prediction, ground_truth)

                if is_correct:
                    correct += 1
                total += 1

                details.append({
                    "index": i,
                    "question": question[:50],
                    "ground_truth": ground_truth[:50],
                    "prediction": prediction[:50],
                    "correct": is_correct,
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"  进度: {i+1}/{len(test_data)}, 当前准确率: {correct/total:.2%}")

            except Exception as e:
                logger.warning(f"  样本 {i} 评估失败: {e}")
                details.append({"index": i, "error": str(e)})

        accuracy = correct / total if total > 0 else 0

        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": details,
        }

        logger.info(f"VQA 评估完成: 准确率 {accuracy:.2%} ({correct}/{total})")
        return result

    def _match_answer(self, prediction: str, ground_truth: str) -> bool:
        """判断预测答案是否正确（模糊匹配）

        Args:
            prediction: 模型预测
            ground_truth: 标准答案

        Returns:
            是否匹配
        """
        # 清理文本
        pred = self._normalize_answer(prediction)
        gt = self._normalize_answer(ground_truth)

        # 精确匹配
        if pred == gt:
            return True

        # 包含匹配（标准答案包含在预测中）
        if gt in pred:
            return True

        # 关键词匹配（提取数字和关键实体）
        pred_nums = set(re.findall(r'\d+\.?\d*', pred))
        gt_nums = set(re.findall(r'\d+\.?\d*', gt))
        if gt_nums and gt_nums.issubset(pred_nums):
            return True

        return False

    def _normalize_answer(self, text: str) -> str:
        """标准化答案文本"""
        text = text.strip().lower()
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点
        text = re.sub(r'\s+', ' ', text)      # 合并空格
        return text

    # ============================================================
    # 图像描述质量评估（简化版 BLEU）
    # ============================================================

    def evaluate_caption(self, test_data: List[Dict],
                         image_dir: str = "") -> Dict[str, Any]:
        """图像描述质量评估（简化版 BLEU-1/2 和关键词覆盖率）

        Args:
            test_data: 测试数据
            image_dir: 图片根目录

        Returns:
            评估指标
        """
        logger.info(f"开始图像描述评估，共 {len(test_data)} 条数据...")

        bleu_scores = []
        keyword_coverages = []

        for i, sample in enumerate(test_data):
            image_path = os.path.join(image_dir, sample.get("image", ""))
            ground_truth = sample.get("caption", sample.get("answer", ""))

            question = "请详细描述这张图片的内容。"

            try:
                image = Image.open(image_path).convert("RGB")
                prediction = self.generate(image, question)

                # 计算简化 BLEU
                bleu = self._compute_simple_bleu(prediction, ground_truth)
                bleu_scores.append(bleu)

                # 关键词覆盖率
                coverage = self._compute_keyword_coverage(prediction, ground_truth)
                keyword_coverages.append(coverage)

                if (i + 1) % 10 == 0:
                    logger.info(f"  进度: {i+1}/{len(test_data)}")

            except Exception as e:
                logger.warning(f"  样本 {i} 评估失败: {e}")

        result = {
            "bleu_1": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
            "keyword_coverage": sum(keyword_coverages) / len(keyword_coverages) if keyword_coverages else 0,
            "total": len(bleu_scores),
        }

        logger.info(f"图像描述评估完成: BLEU-1={result['bleu_1']:.3f}, 关键词覆盖率={result['keyword_coverage']:.3f}")
        return result

    def _compute_simple_bleu(self, prediction: str, reference: str) -> float:
        """简化版 BLEU-1 计算"""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # 计算 unigram precision
        ref_counter = Counter(ref_tokens)
        pred_counter = Counter(pred_tokens)

        clipped = sum(min(pred_counter[w], ref_counter[w]) for w in pred_counter)
        total = sum(pred_counter.values())

        precision = clipped / total if total > 0 else 0

        # Brevity Penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0

        return bp * precision

    def _compute_keyword_coverage(self, prediction: str, reference: str) -> float:
        """关键词覆盖率"""
        # 提取关键词（名词、形容词等，简化为长度>2的词）
        ref_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', reference))
        pred_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', prediction))

        if not ref_words:
            return 0.0

        covered = ref_words.intersection(pred_words)
        return len(covered) / len(ref_words)

    # ============================================================
    # OCR 准确率评估
    # ============================================================

    def evaluate_ocr(self, test_data: List[Dict],
                     image_dir: str = "") -> Dict[str, Any]:
        """OCR 文字识别准确率评估

        Args:
            test_data: 测试数据（需要包含文字标注）
            image_dir: 图片根目录

        Returns:
            评估指标
        """
        logger.info(f"开始 OCR 评估，共 {len(test_data)} 条数据...")

        correct_chars = 0
        total_chars = 0
        exact_matches = 0
        total = 0

        for i, sample in enumerate(test_data):
            image_path = os.path.join(image_dir, sample.get("image", ""))
            ground_truth = sample.get("text", sample.get("answer", ""))

            question = "请识别图片中的所有文字，只输出文字内容，不要添加其他描述。"

            try:
                image = Image.open(image_path).convert("RGB")
                prediction = self.generate(image, question)

                # 清理预测文本（去除多余描述）
                pred_text = self._extract_ocr_text(prediction)
                gt_text = ground_truth.strip()

                # 字符级准确率
                matches = sum(1 for a, b in zip(pred_text, gt_text) if a == b)
                correct_chars += matches
                total_chars += max(len(pred_text), len(gt_text))

                # 完全匹配
                if pred_text == gt_text:
                    exact_matches += 1
                total += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"  进度: {i+1}/{len(test_data)}")

            except Exception as e:
                logger.warning(f"  样本 {i} 评估失败: {e}")

        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        exact_accuracy = exact_matches / total if total > 0 else 0

        result = {
            "char_accuracy": char_accuracy,
            "exact_match_accuracy": exact_accuracy,
            "total": total,
        }

        logger.info(f"OCR 评估完成: 字符准确率={char_accuracy:.2%}, 完全匹配={exact_accuracy:.2%}")
        return result

    def _extract_ocr_text(self, model_output: str) -> str:
        """从模型输出中提取 OCR 文本（去除多余描述）"""
        # 去除常见前缀
        text = model_output.strip()
        prefixes = ["图片中的文字是", "文字内容:", "识别结果:", "图中文字:"]
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):].strip()

        # 去除引号
        text = text.strip('"\'""''')

        return text

    # ============================================================
    # 推理速度 Benchmark
    # ============================================================

    def benchmark_speed(self, sample_image: Optional[Image.Image] = None,
                        question: str = "描述这张图片",
                        num_runs: int = 5,
                        max_new_tokens: int = 256) -> Dict[str, Any]:
        """推理速度 Benchmark

        Args:
            sample_image: 测试图片（None 则生成空白图片）
            question: 测试问题
            num_runs: 运行次数
            max_new_tokens: 生成 token 数

        Returns:
            速度指标
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if sample_image is None:
            sample_image = Image.new("RGB", (448, 448), (200, 200, 200))

        logger.info(f"开始推理速度 Benchmark，运行 {num_runs} 次...")

        # 预热
        _ = self.generate(sample_image, question, max_new_tokens=10)

        latencies = []
        tokens_per_second = []

        for i in range(num_runs):
            start = time.time()
            response = self.generate(sample_image, question, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start

            # 估算生成的 token 数
            num_tokens = len(response) // 2  # 粗略估算：2个字符≈1个token（中文）
            tps = num_tokens / elapsed if elapsed > 0 else 0

            latencies.append(elapsed)
            tokens_per_second.append(tps)

            logger.info(f"  Run {i+1}: {elapsed:.2f}s, ~{num_tokens} tokens, {tps:.1f} tokens/s")

        result = {
            "avg_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "num_runs": num_runs,
            "max_new_tokens": max_new_tokens,
        }

        logger.info(f"Benchmark 完成:")
        logger.info(f"  平均延迟: {result['avg_latency']:.2f}s")
        logger.info(f"  平均吞吐: {result['avg_tokens_per_second']:.1f} tokens/s")

        return result


# ============================================================
# 对比评估工具
# ============================================================

def compare_models(evaluator_a: VLMEvaluator, evaluator_b: VLMEvaluator,
                   test_data: List[Dict], image_dir: str = "",
                   task: str = "vqa") -> Dict[str, Any]:
    """对比两个模型的评估结果

    Args:
        evaluator_a: 模型 A 的评估器
        evaluator_b: 模型 B 的评估器
        test_data: 测试数据
        image_dir: 图片目录
        task: 评估任务 "vqa" / "caption" / "ocr"

    Returns:
        对比结果
    """
    logger.info(f"开始模型对比评估 (任务: {task})...")

    eval_fn = {
        "vqa": "evaluate_vqa",
        "caption": "evaluate_caption",
        "ocr": "evaluate_ocr",
    }

    if task not in eval_fn:
        raise ValueError(f"未知任务: {task}, 支持: {list(eval_fn.keys())}")

    method = eval_fn[task]

    result_a = getattr(evaluator_a, method)(test_data, image_dir)
    result_b = getattr(evaluator_b, method)(test_data, image_dir)

    comparison = {
        "model_a": result_a,
        "model_b": result_b,
        "task": task,
    }

    # 打印对比表
    logger.info("\n" + "=" * 60)
    logger.info(f"模型对比结果 ({task})")
    logger.info("=" * 60)

    if task == "vqa":
        logger.info(f"  模型A 准确率: {result_a.get('accuracy', 0):.2%}")
        logger.info(f"  模型B 准确率: {result_b.get('accuracy', 0):.2%}")
    elif task == "caption":
        logger.info(f"  模型A BLEU-1: {result_a.get('bleu_1', 0):.3f}")
        logger.info(f"  模型B BLEU-1: {result_b.get('bleu_1', 0):.3f}")
    elif task == "ocr":
        logger.info(f"  模型A 字符准确率: {result_a.get('char_accuracy', 0):.2%}")
        logger.info(f"  模型B 字符准确率: {result_b.get('char_accuracy', 0):.2%}")

    logger.info("=" * 60)

    return comparison
