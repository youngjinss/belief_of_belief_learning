#!/usr/bin/env python3
"""
Script to diagnose and fix CUDA compatibility issues with PyTorch
Specifically for RTX A6000 (sm_86) compatibility
"""

import torch
import subprocess
import sys
import os


def diagnose_cuda():
    """Diagnose current CUDA and PyTorch configuration"""
    print("=== CUDA Diagnostic Information ===\n")

    # Check if CUDA is available
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")

    else:
        print("CUDA is not available!")

    # Check PyTorch CUDA architectures
    print("\nPyTorch built with CUDA architectures:")
    print(torch.cuda.get_arch_list())

    return torch.cuda.is_available()


def check_pytorch_cuda_compatibility():
    """Check if current PyTorch supports the GPU"""
    if not torch.cuda.is_available():
        return False

    try:
        # Try to allocate a small tensor on GPU
        test_tensor = torch.randn(10, 10).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            print(f"\nERROR: {e}")
            print("PyTorch is not compatible with your GPU!")
            return False
        raise


def get_recommended_pytorch_version():
    """Get recommended PyTorch version for RTX A6000"""
    cuda_version = None

    # Try to get CUDA version from nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    break
    except:
        pass

    print(f"\nSystem CUDA Version: {cuda_version or 'Could not detect'}")

    # Recommendations for RTX A6000 (sm_86)
    print("\n=== Recommendations for RTX A6000 (sm_86) ===")
    print("The RTX A6000 requires PyTorch built with CUDA 11.1 or higher.")
    print("\nRecommended installation commands:")

    if cuda_version:
        cuda_major = int(cuda_version.split(".")[0])
        cuda_minor = int(cuda_version.split(".")[1])

        if cuda_major >= 12:
            print("# For CUDA 12.x:")
            print(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
        elif cuda_major == 11 and cuda_minor >= 8:
            print("# For CUDA 11.8:")
            print(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
        elif cuda_major == 11 and cuda_minor >= 7:
            print("# For CUDA 11.7:")
            print(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
            )
        else:
            print("# Your CUDA version is too old for RTX A6000.")
            print("# Update CUDA to at least 11.1, then install PyTorch:")
            print(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
    else:
        print("# If you have CUDA 11.8:")
        print(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        print("\n# If you have CUDA 12.1:")
        print(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )

    print("\n# To uninstall current PyTorch first:")
    print("pip uninstall torch torchvision torchaudio -y")


def test_cuda_memory():
    """Test CUDA memory allocation"""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping memory test")
        return

    print("\n=== CUDA Memory Test ===")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")

        # Get memory stats
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3

        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Available: {(total_memory - reserved):.2f} GB")


def main():
    print("PyTorch CUDA Compatibility Checker")
    print("=" * 50)

    # Run diagnostics
    cuda_available = diagnose_cuda()

    if cuda_available:
        # Check compatibility
        compatible = check_pytorch_cuda_compatibility()

        if not compatible:
            get_recommended_pytorch_version()
        else:
            print("\n✓ PyTorch is compatible with your GPU!")
            test_cuda_memory()
    else:
        print("\nNo CUDA devices found. Please check:")
        print("1. NVIDIA drivers are installed")
        print("2. CUDA toolkit is installed")
        print("3. GPU is properly connected")

    print("\n" + "=" * 50)
    print("To fix the training script, also run:")
    print("python fix_training_memory.py")


if __name__ == "__main__":
    main()
