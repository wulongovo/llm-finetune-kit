import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(f"gpu_mem: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
else:
    print("GPU: NOT AVAILABLE")
    # Check if nvidia-smi works
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                          capture_output=True, text=True, timeout=10)
        print(f"nvidia-smi: {r.stdout.strip()}")
    except:
        print("nvidia-smi: NOT FOUND")
