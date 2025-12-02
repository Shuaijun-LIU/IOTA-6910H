#!/bin/bash
# Full experiment version for server
# Generate all results needed for final report

echo "============================================================"
echo "Part 1 Full Experiment Version (Server - estimated 1-2 hours)"
echo "============================================================"
echo ""

# Set variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p models results

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
    BATCH_SIZE=256  # Per GPU batch size, effective = 256 * 8 = 2048
    NUM_WORKERS=16  # Utilize dual CPU architecture (2 CPUs)
    echo "Using optimized settings for 8-GPU server:"
    echo "  - Batch size per GPU: $BATCH_SIZE (effective: $((BATCH_SIZE * GPU_COUNT)))"
    echo "  - Data loader workers: $NUM_WORKERS"
elif [ "$GPU_COUNT" -ge 4 ]; then
    BATCH_SIZE=256
    NUM_WORKERS=12
    echo "Using optimized settings for 4+ GPU server:"
    echo "  - Batch size per GPU: $BATCH_SIZE (effective: $((BATCH_SIZE * GPU_COUNT)))"
    echo "  - Data loader workers: $NUM_WORKERS"
elif [ "$GPU_COUNT" -ge 1 ]; then
    BATCH_SIZE=256  # Still increase from 128 for RTX 5880 Ada
    NUM_WORKERS=8
    echo "Using optimized settings for single/multi-GPU:"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Data loader workers: $NUM_WORKERS"
else
    BATCH_SIZE=128
    NUM_WORKERS=4
    echo "Using CPU settings:"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Data loader workers: $NUM_WORKERS"
fi

echo ""
echo "Step 1/5: Training model (100 epochs)..."
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  Using $GPU_COUNT GPUs with DataParallel (estimated 20-40 minutes)..."
else
    echo "  Using single GPU (estimated 30-60 minutes)..."
fi
python3 train.py \
    --epochs 100 \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr 0.1 \
    --save_dir ./models \
    --seed 42 \
    --data_dir ./data

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "Step 2/5: Evaluating adversarial robustness (full test set + parameter sensitivity analysis)..."
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  Using $GPU_COUNT GPUs (estimated 15-30 minutes)..."
else
    echo "  Using single GPU (estimated 20-40 minutes)..."
fi
EPS=$(python3 -c "print(8/255)")

# Optimize batch size for evaluation based on GPU count
# AutoAttack can benefit from larger batch sizes on multi-GPU
if [ "$GPU_COUNT" -ge 8 ]; then
    EVAL_BATCH_SIZE=2000
elif [ "$GPU_COUNT" -ge 4 ]; then
    EVAL_BATCH_SIZE=1500
else
    EVAL_BATCH_SIZE=1000
fi

python3 evaluate.py \
    --model_path ./models/best_model.pth \
    --data_dir ./data \
    --eps $EPS \
    --n_iter 100 \
    --norm Linf \
    --save_dir ./results \
    --seed 42 \
    --sensitivity \
    --batch_size $EVAL_BATCH_SIZE

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "Step 3/5: Generating adversarial examples visualization (5 examples)..."
python3 visualize.py \
    --results_path ./results/adversarial_samples.pth \
    --model_path ./models/best_model.pth \
    --num_examples 5 \
    --save_path ./results/adversarial_examples.png

if [ $? -ne 0 ]; then
    echo "❌ Visualization failed!"
    exit 1
fi

echo ""
echo "Step 4/5: Generating all additional visualizations for report..."
python3 generate_all_visualizations.py \
    --results_dir ./results \
    --model_path ./models/best_model.pth

if [ $? -ne 0 ]; then
    echo "⚠️  Some visualizations may be missing, but core results are available"
fi

echo ""
echo "Step 5/5: Generating result summary..."
cat > results/summary.txt << EOF
============================================================
Part 1 Experiment Results Summary
============================================================

Generated time: $(date)

Main result files:
1. Training results:
   - models/best_model.pth: Trained model
   - results/training_curves.png: Training curves
   - results/training_history.json: Detailed training history

2. Evaluation results:
   - results/evaluation_results.json: Main evaluation results
   - results/parameter_sensitivity.json: Parameter sensitivity analysis
   - results/adversarial_samples.pth: Adversarial sample data

3. Visualizations:
   - results/training_curves.png: Training curves (from train.py)
   - results/adversarial_examples.png: Adversarial examples visualization (5 examples)
   - results/performance_comparison.png: Clean vs adversarial accuracy comparison
   - results/parameter_sensitivity_plot.png: Parameter sensitivity analysis
   - results/perturbation_statistics.png: Perturbation norm distributions
   - results/class_wise_performance.png: Per-class accuracy analysis

Evaluation Results Summary:
EOF

# Extract key results from JSON file
if [ -f "results/evaluation_results.json" ]; then
    python3 << 'PYTHON_EOF'
import json
import sys

try:
    with open('results/evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    print(f"Clean Accuracy: {data.get('clean_accuracy', 'N/A'):.2f}%")
    print(f"Adversarial Accuracy: {data.get('adversarial_accuracy', 'N/A'):.2f}%")
    print(f"Attack Success Rate: {data.get('attack_success_rate', 'N/A'):.2f}%")
    print(f"Epsilon: {data.get('epsilon', 'N/A')}")
    print(f"Iterations: {data.get('iterations', 'N/A')}")
except Exception as e:
    print(f"Unable to read results: {e}")
PYTHON_EOF
    >> results/summary.txt
fi

cat >> results/summary.txt << EOF

============================================================
Experiment completed!
============================================================

Next steps:
1. Check all result files in results/ directory
2. View results/summary.txt for result summary
3. Write report (report.pdf) based on results

EOF

cat results/summary.txt

echo ""
echo "============================================================"
echo "✅ Full experiment completed!"
echo "============================================================"
echo ""
echo "All result files saved to results/ directory"
echo "View summary: cat results/summary.txt"
echo "============================================================"
