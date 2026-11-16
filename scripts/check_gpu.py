"""Report what this machine can do for QLoRA fine-tuning.

Run: python scripts/check_gpu.py
"""

import importlib
import sys


def check_import(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001 - we want to show any failure
        return f"NOT AVAILABLE ({exc.__class__.__name__}: {exc})"


def main():
    print(f"Python: {sys.version.split()[0]}")

    for lib in ["torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"]:
        print(f"{lib:<14}: {check_import(lib)}")

    try:
        import torch
    except ImportError:
        print("\nPyTorch is not installed. Run: pip install -r requirements.txt")
        return

    if not torch.cuda.is_available():
        print("\nNo CUDA GPU detected. Fine-tuning will be very slow on CPU.")
        return

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"\nGPU             : {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM            : {vram_gb:.1f} GB")
    print(f"bf16 supported  : {torch.cuda.is_bf16_supported()}")
    print("Use fp16 if bf16 is False (the GTX 1650 is one such card).")

    if vram_gb < 6:
        print("\nAdvice: stay at 0.5B-1.5B models with --max_len 256-512 and batch size 1.")
    else:
        print("\nAdvice: 1.5B-7B QLoRA with --max_len 512 should be possible; measure before scaling up.")


if __name__ == "__main__":
    main()
