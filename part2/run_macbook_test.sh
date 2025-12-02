#!/bin/bash
# Quick test script for MacBook (1-2 minutes)
# Minimal test to verify code and workflow

set -e  # Exit on error

echo "=========================================="
echo "Part 2: MacBook Quick Test (30-60 seconds)"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "generate_poison.py" ]; then
    echo "Error: Please run this script from the part2 directory"
    exit 1
fi

# Create necessary directories
mkdir -p models poison results/visualizations data

# Set minimal parameters for very quick test (30-60 seconds)
TARGET_CLASS=0
POISON_RATIO=0.002  # 0.2% - only 10 samples for ultra-fast test
EPSILON=200  # Smaller epsilon for faster computation
NORM="L2"
TRIGGER_SIZE=4
BATCH_SIZE=64  # Larger batch for faster training
EPOCHS=1  # Only 1 epoch for quick test
N_ITER=2  # Minimal PGD iterations

echo "Test Parameters (ultra-fast):"
echo "  Target class: $TARGET_CLASS"
echo "  Poison ratio: $POISON_RATIO (0.2% - only ~10 samples)"
echo "  Epsilon: $EPSILON"
echo "  Norm: $NORM"
echo "  Trigger size: $TRIGGER_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS (minimal)"
echo "  PGD iterations: $N_ITER (minimal)"
echo ""

# Step 1: Generate poisoned samples (minimal)
echo "Step 1: Generating poisoned samples (minimal test)..."
python generate_poison.py \
    --target-class $TARGET_CLASS \
    --poison-ratio $POISON_RATIO \
    --epsilon $EPSILON \
    --norm $NORM \
    --trigger-size $TRIGGER_SIZE \
    --n-iter $N_ITER \
    --output-dir ./poison \
    --seed 42 \
    --device cpu 2>&1 | tee /tmp/poison_gen.log

if [ ! -f "./poison/poisoned_samples.pth" ]; then
    echo "Error: Failed to generate poisoned samples"
    cat /tmp/poison_gen.log
    exit 1
fi
echo "✓ Poisoned samples generated"
echo ""

# Step 2: Train model (minimal epochs)
echo "Step 2: Training model (quick test with $EPOCHS epochs)..."
python train.py \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size $BATCH_SIZE \
    --lr 0.1 \
    --epochs $EPOCHS \
    --output-dir ./models \
    --seed 42 \
    --device cpu 2>&1 | tee /tmp/train.log | grep -E "(Epoch|Train|Val|Best|Saved)" || true

if [ ! -f "./models/best_model.pth" ]; then
    echo "Error: Failed to train model"
    cat /tmp/train.log
    exit 1
fi
echo "✓ Model trained"
echo ""

# Step 3: Evaluate (quick)
echo "Step 3: Evaluating attack..."
python evaluate.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size 50 \
    --output-dir ./results \
    --device cpu 2>&1 | tee /tmp/eval.log

if [ ! -f "./results/attack_results.json" ]; then
    echo "Error: Failed to evaluate"
    cat /tmp/eval.log
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

# Step 4: Visualize (just 1 example for quick test)
echo "Step 4: Generating visualizations (1 example)..."
python visualize.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --n-examples 1 \
    --output-dir ./results/visualizations \
    --device cpu 2>&1 | tee /tmp/viz.log

if [ ! -f "./results/visualizations/visualization_examples.png" ]; then
    echo "Error: Failed to generate visualizations"
    cat /tmp/viz.log
    exit 1
fi
echo "✓ Visualizations generated"
echo ""

# Print results summary
echo "=========================================="
echo "Quick test completed successfully!"
echo "=========================================="
echo ""
echo "Results Summary:"
if [ -f "./results/attack_results.json" ]; then
    echo "  Evaluation results:"
    python -c "import json; d=json.load(open('./results/attack_results.json')); print(f\"    Clean Accuracy: {d['clean_accuracy']:.2f}%\"); print(f\"    ASR: {d['asr']:.2f}%\")"
fi
echo ""
echo "Generated Files:"
echo "  - Poisoned samples: ./poison/poisoned_samples.pth"
echo "  - Trained model: ./models/best_model.pth"
echo "  - Evaluation results: ./results/attack_results.json"
echo "  - Visualizations: ./results/visualizations/"
echo ""
echo "Note: This was a minimal test with very few epochs."
echo "      For full results, run ./run_server_full.sh on the server."
echo ""

