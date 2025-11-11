import sys
import subprocess
import importlib
from typing import Tuple, Optional


def check_import(module_name: str, package_name: Optional[str] = None) -> Tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, f"[OK] {package_name or module_name}"
    except ImportError as e:
        return False, f"[MISSING] {package_name or module_name} - {str(e)}"


def verify_installation() -> bool:
    print("=" * 60)
    print("Installation Verification")
    print("=" * 60)
    print()
    
    # Core dependencies
    print("Checking Core Dependencies:")
    print("-" * 60)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        ok, msg = check_import(module, name)
        print(msg)
        if not ok:
            all_ok = False
    
    print()
    
    # Optional dependencies
    print("Checking Optional Dependencies:")
    print("-" * 60)
    
    optional_deps = [
        ("pyopencl", "OpenCL"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
    ]
    
    for module, name in optional_deps:
        ok, msg = check_import(module, name)
        print(msg)
    
    print()
    
    # Hardware backends
    print("Checking Hardware Backends:")
    print("-" * 60)
    
    # Check ROCm
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("[OK] ROCm detected")
        else:
            print("[WARN] ROCm not available (will use simulator)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[WARN] ROCm not available (will use simulator)")
    
    # Check OpenCL
    try:
        import pyopencl as cl  # type: ignore
        platforms = cl.get_platforms()
        if platforms:
            print(f"[OK] OpenCL detected: {len(platforms)} platform(s)")
            for i, platform in enumerate(platforms):
                devices = platform.get_devices()
                print(f"  Platform {i}: {platform.name}")
                for j, device in enumerate(devices):
                    print(f"    Device {j}: {device.name}")
        else:
            print("[WARN] OpenCL not available")
    except ImportError:
        print("[WARN] OpenCL not available (pyopencl not installed)")
    
    print()
    
    # Test hardware abstraction
    print("Testing Hardware Abstraction Layer:")
    print("-" * 60)
    
    try:
        from amd_ai_optimization.hardware import get_backend, detect_hardware
        
        backend_type = detect_hardware()
        print(f"Detected backend: {backend_type.value}")
        
        backend = get_backend()
        if backend.initialize():
            info = backend.get_device_info()
            print(f"[OK] Hardware backend initialized: {info.get('name', 'Unknown')}")
            print(f"  Device count: {backend.get_device_count()}")
        else:
            print("[ERROR] Hardware backend initialization failed")
            all_ok = False
    except Exception as e:
        print(f"[ERROR] Hardware abstraction test failed: {e}")
        all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("[OK] Installation verification complete - All core dependencies OK")
        print()
        print("You can now run:")
        print("  python -m amd_ai_optimization.rocm_inference.benchmark")
        print("  python -m cv_ar_graphics.applications.object_detection")
        return True
    else:
        print("[WARN] Some core dependencies are missing")
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)

