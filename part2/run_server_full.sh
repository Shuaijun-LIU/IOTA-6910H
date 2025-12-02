#!/bin/bash
# Full experiment script for server (with 8× NVIDIA RTX 5880 Ada)
# Complete experiment with all recommended parameters

set -e  # Exit on error

echo "=========================================="
echo "Part 2: Full Server Experiment"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "generate_poison.py" ]; then
    echo "Error: Please run this script from the part2 directory"
    exit 1
fi

# Create necessary directories
mkdir -p models poison results/visualizations data

# Set full experiment parameters (as per paper)
TARGET_CLASS=0
POISON_RATIO=0.015  # 1.5% as recommended
EPSILON=600  # Recommended for L2 norm
NORM="L2"
TRIGGER_SIZE=4
BATCH_SIZE=50  # As per paper
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
echo "This may take a while..."
python train.py \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size $BATCH_SIZE \
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
python evaluate.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --batch-size 100 \
    --output-dir ./results \
    --device cuda

if [ ! -f "./results/attack_results.json" ]; then
    echo "Error: Failed to evaluate"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

# Step 4: Visualize (5 examples as required)
echo "Step 4: Generating visualizations (5 examples)..."
python visualize.py \
    --model-path ./models/best_model.pth \
    --poison-path ./poison/poisoned_samples.pth \
    --n-examples 5 \
    --output-dir ./results/visualizations \
    --device cuda

if [ ! -f "./results/visualizations/visualization_examples.png" ]; then
    echo "Error: Failed to generate visualizations"
    exit 1
fi
echo "✓ Visualizations generated"
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

