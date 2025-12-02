# Part 2: Clean-Label Backdoor Attack

This directory contains code for implementing clean-label backdoor attacks based on the paper "Clean-Label Backdoor Attacks" (Shafahi et al., 2018).

## Quick Start

### For MacBook (Quick Test - 30-90 seconds)
```bash
./run_quick_test.sh
``` 
**Note**: The test uses minimal parameters (0.1% poisoning, 1 epoch, 1 PGD iteration, 50 training samples) to verify code correctness. 
- **Tested**: Completes in ~60-70 seconds on MacBook
- **Verifies**: Code structure, poison generation, training logic
- **Note**: Full training may need more time on CPU, but code logic is verified

### For Server (Full Experiment)
```bash
./run_server_full.sh
```

## Usage

### Step 1: Generate Poisoned Samples
```bash
python generate_poison.py \
  --target-class 0 \
  --poison-ratio 0.015 \
  --epsilon 600 \
  --norm L2 \
  --trigger-size 4 \
  --output-dir ./poison
```

**Key Parameters:**
- `--target-class`: Target class (0-9, default: 0)
- `--poison-ratio`: Poisoning ratio (e.g., 0.015 = 1.5%, default: 0.015)
- `--epsilon`: Perturbation budget (for L2: 600, for Linf: 8/255, default: 600)
- `--norm`: Norm type ('L2' or 'Linf', default: 'L2')
- `--trigger-size`: Trigger size in pixels (default: 4)
- `--model-path`: Path to adversarially trained model (optional)
- `--output-dir`: Output directory (default: ./poison)

**Output:**
- `./poison/poisoned_samples.pth`: Poisoned samples
- `./poison/poison_config.json`: Configuration file

### Step 2: Train Model on Poisoned Dataset
```bash
python train.py \
  --poison-path ./poison/poisoned_samples.pth \
  --batch-size 50 \
  --lr 0.1 \
  --epochs 200 \
  --output-dir ./models
```

**Key Parameters:**
- `--poison-path`: Path to poisoned samples (default: ./poison/poisoned_samples.pth)
- `--batch-size`: Batch size (default: 50)
- `--lr`: Initial learning rate (default: 0.1)
- `--epochs`: Number of epochs (default: 200)
- `--pretrained`: Use pretrained weights (optional)

**Output:**
- `./models/best_model.pth`: Best model checkpoint
- `./models/final_model.pth`: Final model checkpoint
- `./results/training_curves.png`: Training curves

### Step 3: Evaluate Attack
```bash
python evaluate.py \
  --model-path ./models/best_model.pth \
  --poison-path ./poison/poisoned_samples.pth \
  --output-dir ./results
```

**Metrics:**
- **Clean Accuracy**: Accuracy on clean test set
- **ASR (Attack Success Rate)**: Percentage of non-target class test samples that are misclassified as target class when trigger is applied

**Output:**
- `./results/attack_results.json`: Evaluation results

### Step 4: Visualize Examples
```bash
python visualize.py \
  --model-path ./models/best_model.pth \
  --poison-path ./poison/poisoned_samples.pth \
  --n-examples 5 \
  --output-dir ./results/visualizations
```

**Output:**
The visualization script generates **4 different types of figures** for the report:

1. **`perturbation_visualization.png`** - Shows adversarial perturbations
   - Original image vs Poisoned image vs Amplified difference
   - Demonstrates the effect of adversarial perturbations

2. **`attack_examples.png`** - Attack examples comparison
   - Original image vs Poisoned version vs Triggered test sample (with predictions)
   - Shows the complete attack pipeline

3. **`performance_comparison.png`** - Performance bar chart
   - Clean Accuracy vs ASR comparison
   - Quantifies attack effectiveness

4. **`training_curves.png`** - Training curves
   - Training/Validation loss and accuracy over epochs
   - Shows model training process

5. **`example_1.png, example_2.png, ...`** - Individual examples (optional)
   - Individual attack examples for detailed analysis

## Full Pipeline Example

```bash
# 1. Generate poisoned samples (1.5% poisoning, target class 0)
python generate_poison.py \
  --target-class 0 \
  --poison-ratio 0.015 \
  --epsilon 600 \
  --norm L2 \
  --trigger-size 4 \
  --output-dir ./poison

# 2. Train model on poisoned dataset
python train.py \
  --poison-path ./poison/poisoned_samples.pth \
  --batch-size 50 \
  --lr 0.1 \
  --epochs 200 \
  --output-dir ./models

# 3. Evaluate attack
python evaluate.py \
  --model-path ./models/best_model.pth \
  --poison-path ./poison/poisoned_samples.pth \
  --output-dir ./results

# 4. Visualize examples
python visualize.py \
  --model-path ./models/best_model.pth \
  --poison-path ./poison/poisoned_samples.pth \
  --n-examples 5 \
  --output-dir ./results/visualizations
```

## Important Notes

### 1. Adversarially Trained Model
- The `generate_poison.py` script requires an adversarially trained ResNet-18 model to generate perturbations
- For best results, use adversarially trained models from Madry Lab
- If no model is provided, the script will use a regular model (results may be suboptimal)

### 2. Key Implementation Points
- **Poisoned samples** = adversarial perturbation + backdoor trigger
- **Training data must contain the trigger** (this is the core of backdoor attack)
- **PGD uses untargeted attack** (maximize loss to make sample harder to classify as original label, not target class)
- **Poisoned samples keep original labels** (clean-label)

## Installation

1. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

Or use the installation script:
```bash
./install_dependencies.sh
```

2. The CIFAR-10 dataset will be automatically downloaded when running the scripts.

## Key Parameters Summary

### Poison Generation (generate_poison.py)
- `target-class`: 0 (airplane)
- `poison-ratio`: 0.015 (1.5%)
- `epsilon`: 600 (for L2 norm)
- `norm`: L2
- `trigger-size`: 4

### Training (train.py)
- `batch-size`: 50
- `lr`: 0.1 (with scheduler: 0.01 at epoch 102, 0.001 at epoch 153)
- `epochs`: 200
- `weight-decay`: 0.0002
- `momentum`: 0.9

### Expected Results
- **Clean accuracy**: Should be > 85% (close to normal training)
- **ASR**: Should be > 70% (at 1.5% poisoning)

## Parameter Locations

### generate_poison.py
- Target class: Line ~200 (`--target-class` argument)
- Poison ratio: Line ~201 (`--poison-ratio` argument)
- Epsilon: Line ~202 (`--epsilon` argument)
- Norm type: Line ~203 (`--norm` argument)
- Trigger size: Line ~204 (`--trigger-size` argument)

### train.py
- Learning rate: Line ~180 (`--lr` argument)
- Batch size: Line ~179 (`--batch-size` argument)
- Epochs: Line ~182 (`--epochs` argument)
- Learning rate schedule: Lines ~195-198 (milestones)

## Directory Structure

```
part2/
├── README.md                     # This file
├── PLAN.md                       # Detailed plan
├── generate_poison.py           # Poison generation script
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── visualize.py                  # Visualization script
├── run_macbook_test.sh           # Quick test for MacBook
├── run_server_full.sh            # Full experiment for server
├── install_dependencies.sh       # Installation script
├── models/
│   ├── __init__.py
│   ├── resnet.py                 # ResNet-18 model definition
│   └── best_model.pth            # Trained model (after training)
├── poison/
│   ├── poisoned_samples.pth      # Poisoned samples (after generation)
│   └── poison_config.json         # Configuration (after generation)
├── data/
│   └── cifar-10-batches-py/      # CIFAR-10 dataset (auto-downloaded)
└── results/
    ├── attack_results.json       # Evaluation results (after evaluation)
    ├── training_curves.png        # Training curves (after training)
    └── visualizations/            # Visualization images (after visualization)
```

## Troubleshooting

### 1. If ASR is too low:
- Ensure adversarially trained model is used for generating perturbations
- Check that trigger is correctly added to training data
- Increase perturbation budget (epsilon) or poisoning ratio
- Verify PGD uses untargeted attack (maximize loss for original label)

### 2. If clean accuracy drops too much:
- Reduce perturbation budget (epsilon)
- Reduce poisoning ratio

### 3. CUDA out of memory:
- Reduce batch size
- Use CPU (`--device cpu`)

### 4. Model not found:
- Run `generate_poison.py` first to create poisoned samples
- Run `train.py` to train the model

## References

- Shafahi, A., et al. (2018). "Clean-Label Backdoor Attacks"
- Madry, A., et al. (2017). "Towards Deep Learning Models Resistant to Adversarial Attacks" (for adversarially trained models)

