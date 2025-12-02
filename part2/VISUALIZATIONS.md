# Part 2 Visualizations

This document describes the visualizations generated for Part 2. Each figure shows different aspects of the clean-label backdoor attack results.

## Perturbation Visualization

**File**: `results/visualizations/perturbation_visualization.png`  
**Code**: `visualize.py` (lines 59-107)

**Content**: Grid of 5 examples, each row showing:
- **Column 1**: Original image from target class (with class name)
- **Column 2**: Poisoned image (original + adversarial perturbation + backdoor trigger)
- **Column 3**: Difference image (×10 amplified) showing perturbation + trigger effect

**What it shows**: How the feature-collision method modifies target class images. The poisoned images look visually similar to originals, maintaining clean-label property. The amplified difference reveals both the subtle adversarial perturbation and the trigger location (typically bottom-right corner).

---

## Attack Examples

**File**: `results/visualizations/attack_examples.png`  
**Code**: `visualize.py` (lines 110-233)

**Content**: Grid of 5 examples, each row showing:
- **Column 1**: Original target class image from training set (with class name)
- **Column 2**: Poisoned version (same image with perturbation + trigger, still labeled as target class)
- **Column 3**: Triggered test sample (non-target class test image + trigger) with true label and predicted label

**What it shows**: Complete attack pipeline demonstration. The triggered test samples (non-target class) are misclassified as the target class, proving the backdoor is effective. Successful attacks show different true vs. predicted labels in column 3.

**Additional files**: Individual high-resolution versions (`example_1.png` through `example_5.png`) for detailed analysis.

---

## Performance Comparison

**File**: `results/visualizations/performance_comparison.png`  
**Code**: `visualize.py` (lines 236-282)

**Content**: Bar chart with two bars:
- **Green bar**: Clean Accuracy (%) - model's accuracy on clean test samples
- **Red bar**: Attack Success Rate - ASR (%) - percentage of triggered non-target samples misclassified as target class
- Title includes target class name

**What it shows**: Quantitative attack effectiveness. High clean accuracy (green) indicates the model works normally on clean data. High ASR (red) shows the backdoor is effective. A successful attack maintains high clean accuracy while achieving high ASR.

---

## Training Curves

**File**: `results/visualizations/training_curves.png`  
**Code**: `visualize.py` (lines 285-332)

**Content**: Two subplots showing training history:
- **Left**: Training loss (blue) and validation loss (red) over epochs
- **Right**: Training accuracy (blue) and validation accuracy (red) over epochs

**What it shows**: Model training convergence on the poisoned dataset. The curves should show normal training behavior (loss decreasing, accuracy increasing), demonstrating that the model trains successfully despite poisoned samples. This validates that the clean-label attack doesn't disrupt normal training.

