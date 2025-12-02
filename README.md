# IOTA6910H Assignment: Adversarial Attacks and Backdoor Attacks

This repository contains two parts of the assignment focusing on adversarial robustness and backdoor attacks in deep learning.

## 📋 Project Overview

This assignment consists of two independent parts:

- **Part 1**: Adversarial Example Generation using Auto-PGD
- **Part 2**: Clean-Label Backdoor Attack using Feature-Collision method

## 📁 Project Structure

```
.
├── README.md              # This file
├── part1/                 # Part 1: Adversarial Example Generation
│   ├── README.md          # Part 1 detailed instructions
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Adversarial attack evaluation
│   ├── visualize.py       # Visualization script
│   └── ...
└── part2/                 # Part 2: Clean-Label Backdoor Attack
    ├── README.md          # Part 2 detailed instructions
    ├── generate_poison.py # Poison generation script
    ├── train.py           # Training script
    └── ...
```

## 🚀 Quick Navigation

### [Part 1: Adversarial Example Generation (Auto-PGD)](./part1/README.md)

Train and evaluate ResNet-18 model's robustness against Auto-PGD attacks on CIFAR-10 dataset.

**Key Features:**
- Train/fine-tune ResNet-18 on CIFAR-10
- Generate adversarial examples using Auto-PGD (ℓ∞, ε=8/255)
- Evaluate clean vs. adversarial accuracy
- Visualize adversarial examples and perturbations

**[→ Go to Part 1 README](./part1/README.md)**

---

### [Part 2: Clean-Label Backdoor Attack](./part2/README.md)

Implement and evaluate clean-label backdoor attacks using Feature-Collision method.

**Key Features:**
- Generate poisoned samples via feature-collision
- Train model on poisoned dataset
- Evaluate attack success rate (ASR) and clean accuracy
- Visualize poisoned samples and triggered inputs

**[→ Go to Part 2 README](./part2/README.md)**

---

## 📝 General Requirements

### Environment Setup

Both parts require:
- Python 3.7+
- PyTorch
- torchvision
- Other dependencies (see each part's README for details)

### Quick Installation

Each part has its own installation script. Navigate to the respective directory and follow the instructions in the README.

## 📄 Submission

Each part should include:
- **README.md**: Detailed reproduction instructions
- **Code**: Runnable scripts to reproduce all results
- **Report**: PDF containing results, visualizations, and analysis

## 📚 References

### Part 1
- Auto-PGD Paper: [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks](https://arxiv.org/pdf/2003.01690)
- AutoAttack Library: [GitHub Repository](https://github.com/fra31/auto-attack)

### Part 2
- Feature-Collision Paper: [Clean-Label Backdoor Attacks](https://openreview.net/pdf?id=HJg6e2CcK7)

---

**Note**: For detailed instructions, dependencies, and usage, please refer to each part's README file.

