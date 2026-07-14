import torch
import sys

print("=== Python ===")
print(f"  {sys.executable}")
print(f"  {sys.version}")

print("\n=== GPU ===")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
else:
    print("  未检测到 GPU")

print("\n=== 关键依赖 ===")
pkgs = ['transformers','peft','trl','bitsandbytes','accelerate','Pillow','torchvision','yaml']
import importlib
for p in pkgs:
    try:
        m = importlib.import_module(p.replace('-','_'))
        v = getattr(m, '__version__', 'installed')
        print(f"  {p}: {v}")
    except:
        print(f"  {p}: 未安装")

print("\n=== HuggingFace 缓存 ===")
import os
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"  {hf_home}")
if os.path.exists(hf_home):
    models_dir = os.path.join(hf_home, "hub")
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        vl_models = [m for m in models if any(k in m.lower() for k in ['qwen', 'internvl', 'vl', 'vision'])]
        if vl_models:
            print(f"  已缓存的VLM模型: {vl_models}")
        else:
            print(f"  已缓存模型数: {len(models)}（无VLM相关）")
    else:
        print("  缓存目录为空")
else:
    print("  缓存目录不存在")
