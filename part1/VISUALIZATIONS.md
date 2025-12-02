# Part 1 Visualizations

This document describes the visualizations generated for Part 1. Each figure shows different aspects of the adversarial attack results.

## Training Curves

**File**: `results/training_curves.png`  
**Code**: `train.py` (lines 100-150, saved during training)

**Content**: Two subplots showing training progress:
- **Left**: Training loss (blue) and validation loss (red) over epochs
- **Right**: Training accuracy (blue) and validation accuracy (red) over epochs

**What it shows**: Model convergence and training stability. Loss should decrease and accuracy should increase over epochs, indicating successful training.

---

## Adversarial Examples

**File**: `results/adversarial_examples.png`  
**Code**: `visualize.py` (lines 79-179)

**Content**: Grid of 5 examples, each row showing:
- **Column 1**: Original image with true label and clean prediction
- **Column 2**: Adversarial image with true label and adversarial prediction (title color: red if attack succeeded, green if failed)
- **Column 3**: Perturbation amplified ×10 with L∞ norm value

**What it shows**: Visual demonstration of Auto-PGD attacks. The adversarial images look nearly identical to originals, but predictions change. The perturbation (amplified) shows where the attack modified pixels. L∞ norm indicates perturbation magnitude (should be ≤ ε = 8/255).

---

## Performance Comparison

**File**: `results/performance_comparison.png`  
**Code**: `generate_all_visualizations.py` (lines 18-62)

**Content**: Bar chart with two bars:
- **Green bar**: Clean accuracy (%)
- **Red bar**: Adversarial accuracy (%)
- Text annotation: Attack Success Rate (ASR) = percentage of samples where prediction changed

**What it shows**: Quantitative comparison of model performance. Large gap between clean and adversarial accuracy indicates vulnerability. High ASR shows attack effectiveness.

---

## Perturbation Statistics

**File**: `results/perturbation_statistics.png`  
**Code**: `generate_all_visualizations.py` (lines 125-174)

**Content**: Two histograms side by side:
- **Left**: Distribution of L∞ norms (max absolute perturbation per sample)
- **Right**: Distribution of L2 norms (Euclidean norm of perturbation vector)
- Both show mean value as vertical dashed line

**What it shows**: Statistical distribution of perturbation magnitudes. Most L∞ norms should be close to ε = 8/255, confirming attacks use the full budget. Distribution shape reveals attack behavior.

---

## Class-wise Performance

**File**: `results/class_wise_performance.png`  
**Code**: `generate_all_visualizations.py` (lines 177-267)

**Content**: Grouped bar chart for 10 CIFAR-10 classes:
- **Green bars**: Clean accuracy per class
- **Red bars**: Adversarial accuracy per class
- X-axis: Class names (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**What it shows**: Which classes are more vulnerable to attacks. Classes with larger gaps between clean and adversarial accuracy are more susceptible. Helps identify model weaknesses.

---

## Parameter Sensitivity (Optional)

**File**: `results/parameter_sensitivity_plot.png`  
**Code**: `generate_all_visualizations.py` (lines 65-122)

**Content**: Two line plots:
- **Left**: Adversarial accuracy vs. epsilon (ε) values (x-axis: ε × 255, y-axis: accuracy %)
- **Right**: Adversarial accuracy vs. number of iterations (x-axis: iterations, y-axis: accuracy %)

**What it shows**: How attack parameters affect results. Larger ε typically leads to lower adversarial accuracy (stronger attacks). More iterations may improve attack success. Only generated if sensitivity analysis was performed.

