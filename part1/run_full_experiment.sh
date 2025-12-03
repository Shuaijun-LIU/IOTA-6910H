#!/bin/bash
# Full experiment version for server
# Generate all results needed for final report
# Using single GPU (GPU 0) to avoid conflicts with part2

echo "============================================================"
echo "Part 1 Full Experiment Version (Server - estimated 1-2 hours)"
echo "============================================================"
echo ""

# Set variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Force using GPU 0 only
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU 0 only (CUDA_VISIBLE_DEVICES=0)"
echo ""

# Create necessary directories
mkdir -p models results

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'Using GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available, will use CPU (very slow)')
" || exit 1

# Single GPU settings
BATCH_SIZE=256
NUM_WORKERS=8
echo ""
echo "Using single GPU settings:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Data loader workers: $NUM_WORKERS"

echo ""
echo "Step 1/5: Training model (100 epochs)..."
echo "  Using single GPU 0 (estimated 30-60 minutes)..."
python3 train.py \
    --epochs 100 \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr 0.1 \
    --save_dir ./models \
    --seed 42 \
    --data_dir ./data

if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed!"
    exit 1
fi

echo ""
echo "Step 2/5: Evaluating adversarial robustness (full test set + parameter sensitivity analysis)..."
echo "  Using single GPU 0 (estimated 20-40 minutes)..."

# Check and install AutoAttack if needed
echo "  Checking AutoAttack installation..."
python3 -c "
import sys
sys.path.insert(0, './auto-attack')
try:
    from autoattack import AutoAttack
    print('  ✓ AutoAttack is installed')
except ImportError:
    print('  AutoAttack not found, installing...')
    import subprocess
    import os
    if os.path.exists('./auto-attack'):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', './auto-attack'])
    else:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/fra31/auto-attack'])
    print('  ✓ AutoAttack installed successfully')
" || {
    echo "  Installing AutoAttack..."
    if [ -d "./auto-attack" ]; then
        pip3 install -e ./auto-attack || python3 -m pip install -e ./auto-attack
    else
        pip3 install git+https://github.com/fra31/auto-attack || python3 -m pip install git+https://github.com/fra31/auto-attack
    fi
}

EPS=$(python3 -c "print(8/255)")

# Single GPU evaluation batch size
EVAL_BATCH_SIZE=1000

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
    echo "[ERROR] Evaluation failed!"
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
    echo "[ERROR] Visualization failed!"
    exit 1
fi

echo ""
echo "Step 4/5: Generating all additional visualizations for report..."
python3 generate_all_visualizations.py \
    --results_dir ./results \
    --model_path ./models/best_model.pth

if [ $? -ne 0 ]; then
    echo "[WARNING] Some visualizations may be missing, but core results are available"
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
echo "[SUCCESS] Full experiment completed!"
echo "============================================================"
echo ""
echo "All result files saved to results/ directory"
echo "View summary: cat results/summary.txt"
echo "============================================================"
