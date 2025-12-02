# Part 1: Adversarial Example Generation (Auto-PGD)

This project implements evaluation of ResNet-18 model's adversarial robustness on CIFAR-10 dataset using Auto-PGD method.

## 🚀 Quick Start

### MacBook Quick Test (30-60 seconds)
Quickly verify code and workflow on local machine:
```bash
# From root directory
bash part1/run_quick_test.sh

# Or from part1 directory
cd part1 && bash run_quick_test.sh
```

### Server Full Experiment (1-2 hours)
Run full experiment on a high-performance server to generate all results needed for final report:
```bash
# From root directory
bash part1/run_full_experiment.sh

# Or from part1 directory
cd part1 && bash run_full_experiment.sh
```

**Detailed instructions**: See [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for more information.

---

## Project Structure

```
part1/
├── README.md                 # This document
├── PLAN.md                   # Project planning document
├── train.py                  # Model training script
├── evaluate.py               # Adversarial attack evaluation script
├── visualize.py              # Visualization script
├── models/
│   └── resnet.py            # ResNet-18 model definition
├── results/                  # Result output directory
│   ├── training_curves.png  # Training curves
│   ├── adversarial_examples.png  # Adversarial examples visualization
│   └── ...
├── data/                     # CIFAR-10 dataset (auto-downloaded)
└── auto-attack/              # AutoAttack library (provided)
```

## Environment Requirements

### Python Version
- Python 3.7+

### Dependencies
```bash
pip install torch torchvision
pip install matplotlib numpy
pip install -e ./auto-attack  # Install AutoAttack library
```

### GPU (Recommended)
- CUDA-capable GPU (optional, but strongly recommended for faster training and evaluation)

## Quick Start - Dependency Installation

### Method 1: Using Installation Script (Recommended)
```bash
# From root directory
bash part1/install_dependencies.sh

# Or from part1 directory
cd part1 && bash install_dependencies.sh
```

### Method 2: Manual Installation
```bash
# 1. Install PyTorch and torchvision
pip install torch torchvision

# 2. Install other dependencies
pip install numpy matplotlib

# 3. Install AutoAttack library
pip install -e ./auto-attack
```

### Method 3: Using requirements.txt (if created)
```bash
pip install -r requirements.txt
```

### Check Dependency Installation
Run the following command to check if all dependencies are correctly installed:
```bash
python check_setup.py
```

**Expected output**: Should show all dependencies are installed (✓ marks)

## Usage Steps

### Step 1: Train Model

Train ResNet-18 model on CIFAR-10 dataset:

```bash
# Basic training (from scratch)
python train.py --data_dir ./data --epochs 100 --save_dir ./models

# Using pretrained weights (optional, saves time)
python train.py --data_dir ./data --epochs 50 --pretrained --save_dir ./models

# Custom parameters
python train.py \
    --data_dir ./data \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1 \
    --save_dir ./models \
    --seed 42
```

**Key Parameters**:
- `--data_dir`: Data directory (default: `./data`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--pretrained`: Use pretrained weights (optional)
- `--save_dir`: Model save directory (default: `./models`)
- `--seed`: Random seed (default: 42)

**Output**:
- `models/best_model.pth`: Best model checkpoint
- `results/training_curves.png`: Training curves
- `results/training_history.json`: Training history records

### Step 2: Evaluate Adversarial Robustness

Generate adversarial examples using Auto-PGD and evaluate the model:

```bash
# Basic evaluation (using default parameters: ε=8/255, iterations=100)
python evaluate.py \
    --model_path ./models/best_model.pth \
    --data_dir ./data \
    --save_dir ./results

# Custom parameters
python evaluate.py \
    --model_path ./models/best_model.pth \
    --data_dir ./data \
    --eps 8/255 \
    --n_iter 100 \
    --norm Linf \
    --save_dir ./results \
    --seed 42

# Evaluate only part of samples (for quick testing)
python evaluate.py \
    --model_path ./models/best_model.pth \
    --n_ex 1000 \
    --save_dir ./results

# Run parameter sensitivity analysis
python evaluate.py \
    --model_path ./models/best_model.pth \
    --sensitivity \
    --save_dir ./results
```

**Key Parameters**:
- `--model_path`: Path to trained model (required)
- `--eps`: Perturbation budget ε (default: 8/255)
- `--n_iter`: Number of Auto-PGD iterations (default: 100)
- `--norm`: Norm type: `Linf`, `L2`, or `L1` (default: `Linf`)
- `--n_ex`: Number of samples to evaluate (None = full test set)
- `--sensitivity`: Run parameter sensitivity analysis (test different ε and iterations)
- `--seed`: Random seed (default: 42)

**Output**:
- `results/evaluation_results.json`: Evaluation results (clean accuracy, adversarial accuracy, etc.)
- `results/adversarial_samples.pth`: Adversarial sample data (for visualization)
- `results/parameter_sensitivity.json`: Parameter sensitivity analysis results (if using `--sensitivity`)

### Step 3: Visualize Adversarial Examples

Generate visualization of adversarial examples:

```bash
# Basic visualization (5 examples)
python visualize.py \
    --results_path ./results/adversarial_samples.pth \
    --num_examples 5 \
    --save_path ./results/adversarial_examples.png

# More examples
python visualize.py \
    --results_path ./results/adversarial_samples.pth \
    --num_examples 10 \
    --save_path ./results/adversarial_examples.png
```

**Key Parameters**:
- `--results_path`: Path to adversarial sample data (output from `evaluate.py`)
- `--num_examples`: Number of examples to visualize (default: 5, report requires at least 5)
- `--save_path`: Path to save visualization image

**Output**:
- `results/adversarial_examples.png`: Visualization containing original images, adversarial images, and amplified perturbations

### Step 4: Generate All Visualizations for Report

Generate comprehensive visualizations including performance comparison, parameter sensitivity, and statistical analysis:

```bash
# Generate all additional visualizations
python generate_all_visualizations.py \
    --results_dir ./results \
    --model_path ./models/best_model.pth
```

**Output**:
- `results/performance_comparison.png`: Bar chart comparing clean vs adversarial accuracy
- `results/parameter_sensitivity_plot.png`: Effect of different ε and iterations (if --sensitivity was used)
- `results/perturbation_statistics.png`: Distribution of L∞ and L2 perturbation norms
- `results/class_wise_performance.png`: Per-class clean and adversarial accuracy

## Complete Workflow Example

```bash
# 1. Train model
python train.py --epochs 100 --save_dir ./models

# 2. Evaluate adversarial robustness (with parameter sensitivity analysis)
python evaluate.py \
    --model_path ./models/best_model.pth \
    --sensitivity \
    --save_dir ./results

# 3. Generate visualization
python visualize.py \
    --results_path ./results/adversarial_samples.pth \
    --num_examples 5 \
    --save_path ./results/adversarial_examples.png
```

## Key Parameter Configuration

### Auto-PGD Parameters (Following Assignment Requirements)

According to assignment requirements, Auto-PGD is configured as follows:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Norm | `Linf` | ℓ∞ norm |
| ε | `8/255` | Perturbation budget |
| Iterations | `100` | AutoAttack default value |
| Step Size | Adaptive | Auto-PGD automatically adjusts (initial ≈ `2*eps = 16/255`) |

**Note**: Auto-PGD uses adaptive step size and does not require manual fixed step size (like traditional PGD's `step_size=2/255`). This is one of Auto-PGD's core advantages.

## Result Files Description

### Training Output
- `models/best_model.pth`: Best model checkpoint (contains model weights, optimizer state, training parameters, etc.)
- `results/training_curves.png`: Training and validation accuracy/loss curves
- `results/training_history.json`: Detailed training history records

### Evaluation Output
- `results/evaluation_results.json`: Contains the following information:
  ```json
  {
    "clean_accuracy": 85.23,
    "adversarial_accuracy": 45.67,
    "attack_success_rate": 54.33,
    "epsilon": 0.031373,
    "iterations": 100,
    "norm": "Linf"
  }
  ```
- `results/adversarial_samples.pth`: Contains original images, adversarial images, true labels, etc.
- `results/parameter_sensitivity.json`: Results under different parameter settings (if sensitivity analysis is run)

### Visualization Output
- `results/adversarial_examples.png`: Visualization image containing:
  - Original images + predicted labels
  - Adversarial images + predicted labels
  - Amplified perturbations (×10) + L∞ norm

## Important Notes

### 1. Model Requirements (AutoAttack)
- Model must return **logits** (output without softmax)
- Input must be in **[0, 1]** range (not ImageNet normalization)
- Input format: NCHW (batch_size, channels, height, width)

The ResNet-18 model in this project already meets these requirements.

### 2. Data Preprocessing
- During training: Use standard normalization to help convergence
- During evaluation: **Do not normalize**, use `transforms.ToTensor()` directly (range [0, 1])

### 3. GPU Memory
- If encountering OOM (out of memory), you can:
  - Reduce `--batch_size`
  - Reduce `--n_ex` (number of evaluation samples)
  - Use smaller `--n_iter`

### 4. Reproducibility
- All scripts support `--seed` parameter
- Fixed random seed ensures reproducible results

## Troubleshooting

### Issue 1: AutoAttack Import Error
```bash
# Ensure AutoAttack is correctly installed
pip install -e ./auto-attack
# or
cd auto-attack && pip install -e .
```

### Issue 2: Model Input Range Error
Error message: `RuntimeError: Expected input to have values in [0, 1]`

**Solution**: Ensure evaluation uses `transforms.ToTensor()` instead of normalized transforms.

### Issue 3: Model Output Error
Error message: Zero gradient or ineffective attack

**Solution**: Ensure model returns logits (not softmax output).

## Citation

If using this project, please cite:

1. Auto-PGD Paper:
```
@inproceedings{croce2020reliable,
    title = {Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
    author = {Francesco Croce and Matthias Hein},
    booktitle = {ICML},
    year = {2020}
}
```

2. AutoAttack Library:
- GitHub: https://github.com/fra31/auto-attack

## License

This project follows the license of the original AutoAttack library.
