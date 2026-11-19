#!/usr/bin/env python3
"""
scripts/setup_environment.py
Cross-platform environment setup and dependency verification script.
Works seamlessly on Linux, Windows, and macOS.
"""

import sys
import subprocess
import os
import platform

def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"[*] Detected Python: {major}.{minor}.{sys.version_info.micro} on {platform.system()} ({platform.machine()})")
    if major < 3 or (major == 3 and minor < 9):
        print(f"[!] Warning: Python 3.9+ is recommended. Found {major}.{minor}.")
    return True

def install_dependencies():
    print("\n[*] Installing / Updating dependencies (CPU-optimized)...")
    req_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")
    if not os.path.exists(req_file):
        print(f"[!] requirements.txt not found at {req_file}")
        return False

    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def verify_libraries():
    print("\n[*] Verifying installed libraries...")
    libs = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics YOLO"),
        ("sklearn", "Scikit-Learn"),
        ("skimage", "Scikit-Image"),
        ("cv2", "OpenCV"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("shap", "SHAP (Explainable AI)"),
        ("joblib", "Joblib")
    ]

    all_ok = True
    for mod_name, disp_name in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [✓] {disp_name:<25} version: {ver}")
        except ImportError as e:
            print(f"  [✗] {disp_name:<25} NOT INSTALLED: {e}")
            all_ok = False

    try:
        import torch
        print(f"\n[*] PyTorch Device Configuration:")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - Using CPU with {torch.get_num_threads()} thread(s)")
    except Exception as e:
        print(f"[!] Error checking PyTorch details: {e}")

    return all_ok

def register_jupyter_kernel():
    try:
        print("\n[*] Registering Jupyter kernel for this environment...")
        cmd = [sys.executable, "-m", "ipykernel", "install", "--user", "--name=railway_cpu", "--display-name", "Python (Railway CPU)"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  [✓] Registered kernel 'railway_cpu'")
    except Exception as e:
        print(f"  [!] Skipping kernel registration: {e}")

def main():
    print("=" * 65)
    print(" Railway Track Fault Detection & Segmentation - Environment Setup")
    print("=" * 65)
    
    check_python_version()

    if "--check" in sys.argv:
        ok = verify_libraries()
        sys.exit(0 if ok else 1)

    success = install_dependencies()
    if not success:
        print("\n[!] Error during package installation. Please inspect the output above.")
    
    ok = verify_libraries()
    register_jupyter_kernel()

    print("\n" + "=" * 65)
    if ok:
        print(" [✓] Environment setup successfully verified! Ready for project execution.")
    else:
        print(" [!] Some dependencies are missing. Run: python -m pip install -r requirements.txt")
    print("=" * 65)

if __name__ == "__main__":
    main()
