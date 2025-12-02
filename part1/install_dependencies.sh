#!/bin/bash
# Dependency installation script
# Usage: bash install_dependencies.sh

# Set variables - get script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Part 1 Project Dependency Installation"
echo "============================================================"
echo ""

# Check Python
echo "1. Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ Found Python: $PYTHON_VERSION"
else
    echo "   ✗ Python3 not found, please install Python 3.7+"
    exit 1
fi
echo ""

# Check pip
echo "2. Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "   ✓ Found pip"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo "   ✓ Found pip"
    PIP_CMD="pip"
else
    echo "   ✗ pip not found, please install pip"
    exit 1
fi
echo ""

# Install PyTorch and torchvision
echo "3. Installing PyTorch and torchvision..."
echo "   This may take some time..."
$PIP_CMD install torch torchvision
echo ""

# Install other dependencies
echo "4. Installing other dependencies (numpy, matplotlib)..."
$PIP_CMD install numpy matplotlib
echo ""

# Install AutoAttack
echo "5. Installing AutoAttack library..."
if [ -d "auto-attack" ]; then
    echo "   Installing auto-attack from local directory..."
    $PIP_CMD install -e ./auto-attack
else
    echo "   Installing auto-attack from GitHub..."
    $PIP_CMD install git+https://github.com/fra31/auto-attack
fi
echo ""

# Verify installation
echo "6. Verifying installation..."
python3 check_setup.py

echo ""
echo "============================================================"
echo "Installation completed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run check: python3 check_setup.py"
echo "  2. Train model: python3 train.py --epochs 100"
echo ""
