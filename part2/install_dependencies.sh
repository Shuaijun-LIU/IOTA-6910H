#!/bin/bash
# Install dependencies for Part 2: Clean-Label Backdoor Attack

echo "Installing dependencies for Part 2..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install PyTorch and torchvision
echo "Installing PyTorch and torchvision..."
pip3 install torch torchvision

# Install other dependencies
echo "Installing other dependencies..."
pip3 install matplotlib numpy pillow

echo ""
echo "Dependencies installed successfully!"
echo ""
echo "Note: For best results, you should use an adversarially trained ResNet-18 model"
echo "      to generate perturbations. You can download one from Madry Lab or train"
echo "      your own using adversarial training (PGD)."

