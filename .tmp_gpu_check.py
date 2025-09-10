import os
print("ENV CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch version:", torch.__version__)
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[{i}]", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch error:", repr(e))
