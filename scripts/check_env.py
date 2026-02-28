#!/usr/bin/env python3
"""
Verify that the environment is correctly set up for bubble_tracking.

Usage:
    python scripts/check_env.py
"""

import sys


def check_imports():
    """Check that all required packages can be imported."""
    required = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]

    all_ok = True
    for module_name, display_name in required:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  [OK] {display_name}: {version}")
        except ImportError as e:
            print(f"  [FAIL] {display_name}: {e}")
            all_ok = False

    return all_ok


def check_torch_details():
    """Print PyTorch configuration details."""
    try:
        import torch

        print(f"\n  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  Running in CPU-only mode")
        print(f"  CPU threads: {torch.get_num_threads()}")
    except ImportError:
        print("  PyTorch not installed")


def main():
    print("=" * 60)
    print("Bubble Tracking Environment Check")
    print("=" * 60)

    print(f"\nPython: {sys.version}")
    print(f"Executable: {sys.executable}")

    print("\nChecking required packages:")
    imports_ok = check_imports()

    print("\nPyTorch details:")
    check_torch_details()

    print("\n" + "=" * 60)
    if imports_ok:
        print("Environment OK - ready for bubble tracking!")
        return 0
    else:
        print("Environment has issues - please install missing packages")
        return 1


if __name__ == "__main__":
    sys.exit(main())

