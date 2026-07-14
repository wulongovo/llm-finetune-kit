项 目 经 历
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
项目：LLM Fine-tune Kit — 多模态大模型微调框架（VLM QLoRA/LoRA/全量微调 + 双架构对比）
2025.06 ~ 至今         AI算法工程师 / 大模型应用开发
技术栈：Python、PyTorch、Transformers、PEFT、Qwen2.5-VL、InternVL、QLoRA、LoRA、DeepSpeed、BitsAndBytes、Gradio、vLLM
项目背景
现有多模态大模型（VLM）微调方案存在三个痛点：1）不同模型架构（Qwen2-VL、InternVL等）的加载方式、对话模板、图像预处理流程各不相同，切换模型需要大量适配工作；2）消费级GPU（8-12GB显存）难以微调7B级别的VLM模型；3）缺乏统一的评估框架对比不同模型和微调策略的效果。本项目基于已有的LLM文本微调框架，扩展了多模态VLM微调能力，支持Qwen2.5-VL和InternVL双架构，提供QLoRA/LoRA/Full三种微调策略，通过统一的模型工厂和数据处理管线实现"一条命令切换模型和策略"的开箱即用体验。
项目职责
● 模型工厂设计：实现了ModelFactory统一模型加载层，通过正则匹配+config.json解析自动检测模型架构（Qwen2-VL/InternVL），根据架构类型自动选择对应的模型类（Qwen2VLForConditionalGeneration / AutoModel）、图像处理器和LoRA target_modules；将不同架构的差异封装在工厂内部，上层训练代码无需感知底层模型差异；
● 多策略微调引擎：实现了VLMTrainer统一训练器，支持三种策略——QLoRA（4bit NF4量化+Double Quantization，8GB显存微调7B VLM）、LoRA（bf16精度，16-24GB显存）、Full Fine-tune（冻结Vision Encoder只训练LLM部分，40GB+显存）；每种策略自动配置对应的量化参数、优化器（paged_adamw_8bit/adamw_torch）和显存优化（Gradient Checkpointing）；
● 多模态数据管线：设计了统一的数据处理流程，支持VQA（图文问答）、Caption（图像描述）、Conversation（多轮对话）三种数据格式自动检测和转换；实现了MultimodalDataset封装图片加载+文本编码，自定义collate_fn处理不同架构的输入格式差异（Qwen2-VL使用processor.apply_chat_template，InternVL使用自定义对话模板）；
● VLM评估框架：实现了VLMEvaluator多维评估器，包含VQA准确率评估（模糊匹配+关键词匹配+数字匹配）、图像描述质量评估（简化BLEU-1+关键词覆盖率）、OCR文字识别准确率（字符级+完全匹配）、推理速度Benchmark（tokens/s）；支持双模型对比评估，自动生成对比报告；
● DeepSpeed分布式训练：集成DeepSpeed ZeRO-2/3分布式训练，VLM模型参数量大（7B模型bf16约14GB），通过ZeRO-2分片优化器状态和梯度实现单机多卡训练，ZeRO-3支持跨节点大模型训练；与已有的文本微调框架共享DeepSpeed配置，统一多卡训练体验；
● 显存自适应优化：实现了GPU显存自动检测和策略推荐——8GB以下自动切换QLoRA+Gradient Checkpointing，24GB以下阻止全量微调并降级为LoRA，40GB+才允许全量微调；提供了详细的显存估算表，帮助用户在训练前预估资源需求。
项目业绩
1. 双架构统一：实现了Qwen2.5-VL和InternVL的统一微调接口，切换模型只需修改配置文件中的model.name_or_path一行，无需改动任何代码，模型适配工作量从原来的2-3天降低到5分钟；
2. 消费级GPU友好：QLoRA策略下RTX 3070（8GB）即可微调7B VLM模型，LoRA策略下RTX 3090（24GB）可获得更好的训练效果，覆盖了从消费级到专业级GPU的完整梯度；
3. 评估体系完善：VQA准确率、BLEU评分、OCR准确率、推理速度四维评估体系，支持微调前后对比和双模型对比，量化展示微调效果；在自建测试集上，QLoRA微调后的Qwen2.5-VL-7B在VQA任务上准确率提升12%，OCR准确率提升18%；
4. 工程化程度高：完整的CLI入口（train_vlm.py/eval_vlm.py）、YAML配置体系（5套预置配置）、示例数据、README文档，支持DeepSpeed多卡训练和vLLM高性能推理部署，可直接用于生产环境。

---

面试话术要点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: 为什么选择 Qwen2.5-VL 和 InternVL 这两个模型？
A: 两个原因：1）国内认可度高——Qwen2.5-VL是阿里通义千问的视觉语言模型，InternVL是上海AI Lab的旗舰VLM，面试官都熟悉；2）架构差异有代表性——Qwen2-VL使用独立的visual encoder + LLM架构，InternVL使用动态分辨率的ViT-LLM融合架构，覆盖了主流VLM的设计范式。

Q: QLoRA 和 LoRA 的核心区别是什么？
A: LoRA在bf16精度下训练，模型参数以16bit加载，显存占用约为模型大小的1.5-2倍；QLoRA先用4bit NF4量化加载基础模型（显存降为原来的1/4），再在量化模型上训练LoRA适配器。QLoRA的关键创新是NF4（NormalFloat4）量化——基于正态分布的最优4bit量化，以及Double Quantization——对量化常数再量化，进一步节省显存。代价是训练速度略慢（约10-15%），因为需要反量化计算。

Q: 为什么全量微调要冻结 Vision Encoder？
A: 两个原因：1）Vision Encoder（ViT）已经在大规模图文数据上预训练过，具备强大的视觉特征提取能力，微调时冻结它可以保留这些能力；2）VLM的微调数据量通常远小于预训练数据，全量微调Vision Encoder容易过拟合，导致视觉理解能力退化。只训练LLM部分可以让模型学会"如何理解视觉特征"而不是"如何提取视觉特征"。

Q: 不同架构的 LoRA target_modules 为什么不同？
A: 因为不同模型的注意力层命名不同。Qwen2-VL沿用LLaMA风格的q_proj/k_proj/v_proj/o_proj，而InternVL使用自定义的qkv/proj命名。选择正确的target_modules很重要——如果名字不匹配，LoRA适配器无法正确附加到目标层，训练效果会大打折扣。这也是为什么需要模型工厂自动检测架构并选择对应的target_modules。

Q: 如何评估VLM微调效果？用什么指标？
A: 我用了四维评估体系：1）VQA准确率——用模糊匹配+关键词匹配+数字匹配三重判断，比精确匹配更合理；2）BLEU-1评分——评估图像描述的文本质量；3）OCR准确率——字符级和完全匹配两个粒度；4）推理速度——tokens/s，评估部署可行性。关键是支持微调前后对比，用同一个测试集量化展示微调效果。
