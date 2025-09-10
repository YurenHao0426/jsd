import os
print("ENV CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch:", torch.__version__, "cuda:", getattr(torch.version, "cuda", None))
    print("built_with_cuda:", torch.backends.cuda.is_built())
    print("device_count:", torch.cuda.device_count())
    print("is_available:", torch.cuda.is_available())
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            try:
                print(f"[{i}]", torch.cuda.get_device_name(i))
            except Exception as e:
                print(f"[{i}] name error:", e)
except Exception as e:
    print("torch error:", repr(e))
