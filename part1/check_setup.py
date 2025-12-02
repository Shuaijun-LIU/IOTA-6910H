"""
Setup check and dependency installation guide
Check project dependencies and generate installation instructions
"""
import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ⚠️  Warning: Python 3.7+ recommended")
        return False
    return True

def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: not installed")
        return False

def check_autoattack():
    """Check if AutoAttack is available"""
    try:
        sys.path.insert(0, './auto-attack')
        from autoattack import AutoAttack
        print(f"✓ autoattack: available")
        return True
    except ImportError as e:
        print(f"✗ autoattack: unavailable - {e}")
        return False

def check_torch_cuda():
    """Check PyTorch CUDA support"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA: available (device: {torch.cuda.get_device_name(0)})")
        else:
            print(f"⚠️  CUDA: unavailable (will use CPU, slower)")
        return True
    except ImportError:
        return False

def check_file_exists(filepath):
    """Check if file exists"""
    import os
    exists = os.path.exists(filepath)
    if exists:
        print(f"✓ {filepath}")
    else:
        print(f"✗ {filepath}: not found")
    return exists

def main():
    print("="*60)
    print("Part 1 Project Dependency Check")
    print("="*60)
    print()
    
    all_ok = True
    
    # 1. Check Python version
    print("1. Python version check:")
    if not check_python_version():
        all_ok = False
    print()
    
    # 2. Check core dependencies
    print("2. Core dependencies check:")
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
    ]
    
    missing_packages = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
            all_ok = False
    
    # Check CUDA
    check_torch_cuda()
    print()
    
    # 3. Check AutoAttack
    print("3. AutoAttack library check:")
    if not check_autoattack():
        all_ok = False
    print()
    
    # 4. Check project files
    print("4. Project files check:")
    files_to_check = [
        'train.py',
        'evaluate.py',
        'visualize.py',
        'models/resnet.py',
        'README.md',
        'auto-attack/setup.py',
    ]
    
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_ok = False
    print()
    
    # 5. Summary
    print("="*60)
    if all_ok:
        print("✓ All checks passed! Ready to run the project.")
        print()
        print("Next steps:")
        print("  1. Train model: python train.py --epochs 100")
        print("  2. Evaluate model: python evaluate.py --model_path ./models/best_model.pth")
        print("  3. Visualize: python visualize.py --results_path ./results/adversarial_samples.pth")
    else:
        print("✗ Some dependencies are missing. Please install as follows:")
        print()
        print("Installation steps:")
        print("  1. Install PyTorch and torchvision:")
        if 'torch' in missing_packages or 'torchvision' in missing_packages:
            print("     pip install torch torchvision")
        print()
        print("  2. Install other dependencies:")
        other_missing = [p for p in missing_packages if p not in ['torch', 'torchvision']]
        if other_missing:
            print(f"     pip install {' '.join(other_missing)}")
        print()
        print("  3. Install AutoAttack library:")
        print("     cd auto-attack")
        print("     pip install -e .")
        print("     or:")
        print("     pip install -e ./auto-attack")
        print()
        print("Complete installation command:")
        print("  pip install torch torchvision numpy matplotlib")
        print("  pip install -e ./auto-attack")
    
    print("="*60)
    
    return all_ok

if __name__ == '__main__':
    main()
