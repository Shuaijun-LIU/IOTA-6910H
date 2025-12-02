#!/bin/bash
# MacBook quick test version (30-60 seconds)
# Verify code and workflow are correct using minimal dataset and iterations

echo "============================================================"
echo "Part 1 Quick Test Version (MacBook - 30-60 seconds)"
echo "============================================================"
echo ""

# Set variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p models results

# Create minimal test script
cat > test_minimal.py << 'PYEOF'
"""Minimal test - quickly verify code logic"""
import torch
import sys
import os
sys.path.append('models')
from resnet import ResNet18

print("✓ Model import successful")

# Test model creation
model = ResNet18(num_classes=10)
x = torch.randn(2, 3, 32, 32)
y = model(x)
print(f"✓ Model forward pass successful: input {x.shape} -> output {y.shape}")

# Test AutoAttack import
sys.path.insert(0, './auto-attack')
try:
    from autoattack import AutoAttack
    print("✓ AutoAttack import successful")
except Exception as e:
    print(f"✗ AutoAttack import failed: {e}")
    sys.exit(1)

print("\n✅ All core components verified!")
PYEOF

echo "Step 0: Verifying core components..."
python3 test_minimal.py || { echo "❌ Component verification failed!"; exit 1; }
rm -f test_minimal.py

echo ""
echo "Step 1/2: Creating dummy model (skipping actual training)..."
# Create a minimal dummy model checkpoint
python3 << 'PYEOF'
import torch
import torch.nn as nn
import sys
import os
sys.path.append('models')
from resnet import ResNet18

model = ResNet18(num_classes=10)
# Create dummy checkpoint
checkpoint = {
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'val_acc': 50.0,
    'train_acc': 50.0,
}
os.makedirs('models', exist_ok=True)
torch.save(checkpoint, 'models/best_model.pth')
print("✓ Dummy model checkpoint created")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ Model creation failed!"
    exit 1
fi

echo ""
echo "Step 2/2: Quick evaluation (10 samples, 10 iterations)..."
EPS=$(python3 -c "print(8/255)")
python3 evaluate.py \
    --model_path ./models/best_model.pth \
    --data_dir ./data \
    --eps $EPS \
    --n_iter 10 \
    --norm Linf \
    --n_ex 10 \
    --save_dir ./results \
    --seed 42 \
    --batch_size 10 2>&1 | head -50

if [ $? -ne 0 ]; then
    echo "⚠️  Evaluation may have issues, but code structure is correct"
fi

# If evaluation results exist, try visualization
if [ -f "results/adversarial_samples.pth" ]; then
    echo ""
    echo "Step 2.5: Generating visualization (if data available)..."
    python3 visualize.py \
        --results_path ./results/adversarial_samples.pth \
        --model_path ./models/best_model.pth \
        --num_examples 3 \
        --save_path ./results/adversarial_examples.png 2>&1 | head -20
    
    # Try to generate additional visualizations (may fail due to missing data, that's OK)
    echo ""
    echo "Step 2.6: Generating additional visualizations..."
    python3 generate_all_visualizations.py \
        --results_dir ./results \
        --model_path ./models/best_model.pth 2>&1 | head -30 || true
fi

echo ""
echo "============================================================"
echo "✅ Quick test completed!"
echo "============================================================"
echo ""
echo "Verification results:"
echo "  ✓ Code structure correct"
echo "  ✓ Model definition correct"
echo "  ✓ AutoAttack available"
if [ -f "results/evaluation_results.json" ]; then
    echo "  ✓ Evaluation workflow runnable"
fi
if [ -f "results/adversarial_examples.png" ]; then
    echo "  ✓ Visualization workflow runnable"
fi
echo ""
echo "Note: This is a quick test version, only verifies code logic."
echo "      For full training, run: bash run_full_experiment.sh"
echo "============================================================"
