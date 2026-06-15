import torch


def log_memory(stage: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEM] {stage} Allocated: {allocated:.2f}GB Reserved: {reserved:.2f}GB")
    else:
        print(f"[MEM] {stage} CUDA not available")
