"""Verify the Kaggle/Colab GPU is actually attached.

Run this as the very first cell on any new Kaggle session.
If any check fails, stop — the GPU is not attached and training
will silently fall back to CPU.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)

    # 1. nvidia-smi
    print("\n[1/3] nvidia-smi:")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        print(out.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  FAIL: {e}")
        return 1

    # 2. PyTorch CUDA visibility
    print("\n[2/3] PyTorch CUDA:")
    try:
        import torch
    except ImportError:
        print("  FAIL: torch not installed yet (install Unsloth first)")
        return 1

    available = torch.cuda.is_available()
    print(f"  CUDA available: {available}")
    if not available:
        print("  FAIL: torch cannot see CUDA")
        return 1

    name = torch.cuda.get_device_name(0)
    total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"  Device: {name}")
    print(f"  Total VRAM: {total_gb} GB")

    # 3. Free VRAM
    print("\n[3/3] Free VRAM:")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = round(free_bytes / 1e9, 1)
    print(f"  Free: {free_gb} GB")
    if free_gb < 10:
        print("  WARN: <10 GB free — session may have leftover state")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())