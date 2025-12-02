#!/bin/bash
# Full experiment script for server (with 8× NVIDIA RTX 5880 Ada)
# Complete experiment with all recommended parameters
# Optimized for multi-GPU training

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

# Check GPU availability and count
echo "Checking GPU availability..."
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'Number of GPUs detected: {gpu_count}')
    for i in range(gpu_count):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" || exit 1

# Detect number of GPUs and adjust batch size accordingly
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
echo ""
echo "Detected $GPU_COUNT GPU(s)"

# Optimize batch size based on GPU count
# RTX 5880 Ada has plenty of VRAM, so we can use larger batches
# For 8 GPUs with DataParallel, effective batch size will be batch_size * num_gpus
if [ "$GPU_COUNT" -ge 8 ]; then
    BATCH_SIZE=100  # Per GPU batch size, effective = 100 * 8 = 800
    NUM_WORKERS=16  # Utilize dual CPU architecture (2 CPUs)
    echo "Using optimized settings for 8-GPU server:"
    echo "  - Batch size per GPU: $BATCH_SIZE (effective: $((BATCH_SIZE * GPU_COUNT)))"
    echo "  - Data loader workers: $NUM_WORKERS"
elif [ "$GPU_COUNT" -ge 4 ]; then
    BATCH_SIZE=100
    NUM_WORKERS=12
    echo "Using optimized settings for 4+ GPU server:"
    echo "  - Batch size per GPU: $BATCH_SIZE (effective: $((BATCH_SIZE * GPU_COUNT)))"
    echo "  - Data loader workers: $NUM_WORKERS"
elif [ "$GPU_COUNT" -ge 1 ]; then
    BATCH_SIZE=50  # Original paper setting for single GPU
    NUM_WORKERS=8
    echo "Using optimized settings for single/multi-GPU:"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Data loader workers: $NUM_WORKERS"
else
    BATCH_SIZE=50
    NUM_WORKERS=4
    echo "Using CPU settings:"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Data loader workers: $NUM_WORKERS"
fi

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
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  Using GPU 0 for poison generation (estimated 5-10 minutes)..."
else
    echo "  Using single GPU (estimated 5-10 minutes)..."
fi
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
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  Using $GPU_COUNT GPUs with DataParallel (estimated 30-60 minutes)..."
    # Note: train.py will automatically use DataParallel if multiple GPUs are available
else
    echo "  Using single GPU (estimated 1-2 hours)..."
fi
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
if [ "$GPU_COUNT" -ge 8 ]; then
    EVAL_BATCH_SIZE=500
elif [ "$GPU_COUNT" -ge 4 ]; then
    EVAL_BATCH_SIZE=300
else
    EVAL_BATCH_SIZE=100
fi
echo "  Using batch size: $EVAL_BATCH_SIZE"
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  Using $GPU_COUNT GPUs (estimated 2-5 minutes)..."
else
    echo "  Using single GPU (estimated 5-10 minutes)..."
fi
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
echo "  - With $GPU_COUNT GPU(s), effective batch size: $((BATCH_SIZE * GPU_COUNT))"
echo "  - Training time significantly reduced with multi-GPU setup"
echo "  - All visualizations generated for report"
echo ""

