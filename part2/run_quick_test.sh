#!/bin/bash
# Quick test script for Part 2: Clean-Label Backdoor Attack
# This script runs a minimal test to verify the code works

set -e  # Exit on error

echo "=========================================="
echo "Part 2: Quick Test"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "generate_poison.py" ]; then
    echo "Error: Please run this script from the part2 directory"
    exit 1
fi

# Set default parameters for quick test
TARGET_CLASS=0
POISON_RATIO=0.01  # 1% for quick test
EPSILON=600
NORM="L2"
TRIGGER_SIZE=4
BATCH_SIZE=32  # Smaller batch for quick test
EPOCHS=5  # Very few epochs for quick test

echo "Test Parameters:"
echo "  Target class: $TARGET_CLASS"
echo "  Poison ratio: $POISON_RATIO"
echo "  Epsilon: $EPSILON"
echo "  Norm: $NORM"
echo "  Trigger size: $TRIGGER_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo ""

# Step 1: Generate poisoned samples (quick test with small ratio)
echo "Step 1: Generating poisoned samples..."
python generate_poison.py \
    --target-class $TARGET_CLASS \
    --poison-ratio $POISON_RATIO \
    --epsilon $EPSILON \
    --norm $NORM \
    --trigger-size $TRIGGER_SIZE \
    --n-iter 5 \
    --output-dir ./poison \
    --seed 42

if [ ! -f "./poison/poisoned_samples.pth" ]; then
    echo "Error: Failed to generate poisoned samples"
    exit 1
fi
echo "✓ Poisoned samples generated"
echo ""

# Step 2: Train model (quick test with few epochs)
echo "Step 2: Training model (quick test with $EPOCHS epochs)..."
python train.py \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size $BATCH_SIZE \
    --lr 0.1 \
    --epochs $EPOCHS \
    --output-dir ./models \
    --seed 42

if [ ! -f "./models/best_model.pth" ]; then
    echo "Error: Failed to train model"
    exit 1
fi
echo "✓ Model trained"
echo ""

# Step 3: Evaluate
echo "Step 3: Evaluating attack..."
python evaluate.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size 100 \
    --output-dir ./results

if [ ! -f "./results/attack_results.json" ]; then
    echo "Error: Failed to evaluate"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

# Step 4: Visualize (just 2 examples for quick test)
echo "Step 4: Generating visualizations..."
python visualize.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --n-examples 2 \
    --output-dir ./results/visualizations

if [ ! -f "./results/visualizations/visualization_examples.png" ]; then
    echo "Error: Failed to generate visualizations"
    exit 1
fi
echo "✓ Visualizations generated"
echo ""

echo "=========================================="
echo "Quick test completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Poisoned samples: ./poison/poisoned_samples.pth"
echo "  - Trained model: ./models/best_model.pth"
echo "  - Evaluation results: ./results/attack_results.json"
echo "  - Visualizations: ./results/visualizations/"
echo ""
echo "Note: This was a quick test with minimal epochs."
echo "      For full results, run with default parameters:"
echo "        - Use --poison-ratio 0.015 (1.5%)"
echo "        - Use --epochs 200 for training"
echo ""

