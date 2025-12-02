#!/bin/bash
# Full experiment script for server
# Complete experiment with all recommended parameters
# Using single GPU (GPU 1) to avoid conflicts with part1

set -e  # Exit on error

echo "=========================================="
echo "Part 2: Full Server Experiment"
echo "=========================================="
echo ""

# Set variables - get script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Force using GPU 1 only (to avoid conflict with part1 which uses GPU 0)
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU 1 only (CUDA_VISIBLE_DEVICES=1)"
echo ""

# Check if we're in the right directory
if [ ! -f "generate_poison.py" ]; then
    echo "Error: generate_poison.py not found in $SCRIPT_DIR"
    exit 1
fi

# Create necessary directories
mkdir -p models poison results/visualizations data

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'Using GPU 0 (mapped from physical GPU 1): {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available, will use CPU (very slow)')
" || exit 1

# Single GPU settings
BATCH_SIZE=50  # Original paper setting for single GPU
NUM_WORKERS=8
echo ""
echo "Using single GPU settings:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Data loader workers: $NUM_WORKERS"

# Set full experiment parameters (as per paper)
TARGET_CLASS=0
POISON_RATIO=0.015  # 1.5% as recommended
EPSILON=600  # Recommended for L2 norm
NORM="L2"
TRIGGER_SIZE=4
EPOCHS=200  # Full training
N_ITER=10  # Standard PGD iterations

echo "Full Experiment Parameters:"
echo "  Target class: $TARGET_CLASS (airplane)"
echo "  Poison ratio: $POISON_RATIO (1.5%)"
echo "  Epsilon: $EPSILON"
echo "  Norm: $NORM"
echo "  Trigger size: $TRIGGER_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  PGD iterations: $N_ITER"
echo ""

# Step 1: Generate poisoned samples
echo "Step 1: Generating poisoned samples..."
echo "  Using single GPU 1 (estimated 5-10 minutes)..."
python generate_poison.py \
    --target-class $TARGET_CLASS \
    --poison-ratio $POISON_RATIO \
    --epsilon $EPSILON \
    --norm $NORM \
    --trigger-size $TRIGGER_SIZE \
    --n-iter $N_ITER \
    --output-dir ./poison \
    --seed 42 \
    --device cuda

if [ ! -f "./poison/poisoned_samples.pth" ]; then
    echo "Error: Failed to generate poisoned samples"
    exit 1
fi
echo "✓ Poisoned samples generated"
echo ""

# Step 2: Train model (full training)
echo "Step 2: Training model (full training with $EPOCHS epochs)..."
echo "  Using single GPU 1 (estimated 1-2 hours)..."
python train.py \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --lr 0.1 \
    --epochs $EPOCHS \
    --output-dir ./models \
    --seed 42 \
    --device cuda

if [ ! -f "./models/best_model.pth" ]; then
    echo "Error: Failed to train model"
    exit 1
fi
echo "✓ Model trained"
echo ""

# Step 3: Evaluate
echo "Step 3: Evaluating attack..."
EVAL_BATCH_SIZE=100
echo "  Using batch size: $EVAL_BATCH_SIZE"
echo "  Using single GPU 1 (estimated 5-10 minutes)..."
python evaluate.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size $EVAL_BATCH_SIZE \
    --output-dir ./results \
    --device cuda

if [ ! -f "./results/attack_results.json" ]; then
    echo "Error: Failed to evaluate"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

# Step 4: Visualize (5 examples + all visualization types)
echo "Step 4: Generating all visualizations (5 examples)..."
python visualize.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --n-examples 5 \
    --results-path ./results/attack_results.json \
    --final-model-path ./models/final_model.pth \
    --output-dir ./results/visualizations \
    --device cuda

# Check for key visualizations
if [ ! -f "./results/visualizations/perturbation_visualization.png" ]; then
    echo "Warning: Some visualizations may be missing"
else
    echo "✓ Visualizations generated"
fi
echo ""

# Print results summary
echo "=========================================="
echo "Full experiment completed successfully!"
echo "=========================================="
echo ""
echo "Results Summary:"
if [ -f "./results/attack_results.json" ]; then
    python -c "
import json
d = json.load(open('./results/attack_results.json'))
print(f\"  Clean Accuracy: {d['clean_accuracy']:.2f}%\")
print(f\"  Attack Success Rate (ASR): {d['asr']:.2f}%\")
print(f\"  Target Class: {d['target_class']}\")
print(f\"  Trigger Size: {d['trigger_size']}x{d['trigger_size']}\")
print(f\"  Triggered Correct: {d['triggered_correct']}/{d['non_target_total']}\")
"
fi
echo ""
echo "Generated Files:"
echo "  - Poisoned samples: ./poison/poisoned_samples.pth"
echo "  - Trained model: ./models/best_model.pth"
echo "  - Evaluation results: ./results/attack_results.json"
echo "  - Training curves: ./results/training_curves.png"
echo "  - Visualizations: ./results/visualizations/"
echo ""
echo "Expected Results (from paper):"
echo "  - Clean Accuracy: > 85%"
echo "  - ASR: > 70% (at 1.5% poisoning)"
echo ""
echo "Performance Notes:"
echo "  - Using single GPU 1 (batch size: $BATCH_SIZE)"
echo "  - All visualizations generated for report"
echo ""

