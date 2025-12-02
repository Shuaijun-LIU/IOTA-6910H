"""
Evaluate model robustness using Auto-PGD attack
Generate adversarial examples and compute clean/adversarial accuracy
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from resnet import ResNet18

# Import AutoAttack
sys.path.insert(0, './auto-attack')
from autoattack import AutoAttack


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path, device):
    """
    Load trained model
    AutoAttack requires model to normalize inputs itself if they are in [0, 1]
    """
    # Base ResNet model
    model = ResNet18(num_classes=10).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle DataParallel state dict (remove 'module.' prefix)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")
    
    if 'val_acc' in checkpoint and isinstance(checkpoint, dict):
        print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        
    # Create a wrapper that normalizes inputs
    class NormalizedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            )
            
        def forward(self, x):
            return self.model(self.normalize(x))
            
    # Return wrapped model
    return NormalizedModel(model).to(device)


def get_test_loader(data_dir='./data', batch_size=1000, num_workers=4):
    """
    Get CIFAR-10 test loader
    Important: AutoAttack requires inputs in [0, 1] range (not normalized)
    """
    # No normalization - AutoAttack expects [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()  # This converts to [0, 1]
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return testloader, testset.classes


def evaluate_clean_accuracy(model, testloader, device):
    """Evaluate model on clean test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def run_auto_pgd_attack(model, x_test, y_test, eps=8/255, n_iter=100, 
                        norm='Linf', device='cuda', seed=42, n_ex=None):
    """
    Run Auto-PGD attack using AutoAttack library
    
    Args:
        model: Trained model
        x_test: Test images (tensor, shape: [N, C, H, W], values in [0, 1])
        y_test: True labels (tensor, shape: [N])
        eps: Perturbation budget (default: 8/255)
        n_iter: Number of iterations (default: 100)
        norm: Norm type ('Linf', 'L2', 'L1')
        device: Device to use
        seed: Random seed
        n_ex: Number of examples to attack (None = all)
    
    Returns:
        x_adv: Adversarial examples
    """
    # Limit number of examples if specified
    if n_ex is not None and n_ex < len(x_test):
        x_test = x_test[:n_ex]
        y_test = y_test[:n_ex]
    
    print(f"\nRunning Auto-PGD attack...")
    print(f"  Norm: {norm}")
    print(f"  Epsilon: {eps:.6f} ({eps*255:.2f}/255)")
    print(f"  Iterations: {n_iter}")
    print(f"  Number of examples: {len(x_test)}")
    
    # Initialize AutoAttack with only Auto-PGD
    adversary = AutoAttack(
        model,
        norm=norm,
        eps=eps,
        version='custom',  # Use custom to select specific attacks
        device=device,
        verbose=True,
        seed=seed
    )
    
    # Only run APGD-CE (Auto-PGD with Cross-Entropy loss)
    adversary.attacks_to_run = ['apgd-ce']
    adversary.apgd.n_iter = n_iter
    adversary.apgd.n_restarts = 1
    
    # Run attack
    print("\nGenerating adversarial examples...")
    x_adv = adversary.run_standard_evaluation(
        x_test, y_test, bs=250  # Batch size for evaluation
    )
    
    return x_adv


def evaluate_adversarial_accuracy(model, x_adv, y_test, device):
    """Evaluate model on adversarial examples"""
    model.eval()
    correct = 0
    total = len(y_test)
    
    with torch.no_grad():
        outputs = model(x_adv.to(device))
        _, predicted = outputs.max(1)
        correct = predicted.eq(y_test.to(device)).sum().item()
    
    accuracy = 100. * correct / total
    attack_success_rate = 100. - accuracy
    return accuracy, attack_success_rate


def parameter_sensitivity_analysis(model, x_test, y_test, device, save_dir='results'):
    """
    Analyze effect of different parameters on attack success
    """
    print("\n" + "="*60)
    print("Parameter Sensitivity Analysis")
    print("="*60)
    
    results = {}
    
    # Test different epsilon values
    print("\n1. Testing different epsilon values...")
    epsilons = [4/255, 8/255, 16/255]
    results['epsilon'] = {}
    
    for eps in epsilons:
        print(f"\n  Testing epsilon = {eps:.6f} ({eps*255:.2f}/255)")
        try:
            x_adv = run_auto_pgd_attack(
                model, x_test[:1000], y_test[:1000], 
                eps=eps, n_iter=100, device=device
            )
            adv_acc, asr = evaluate_adversarial_accuracy(model, x_adv, y_test[:1000], device)
            results['epsilon'][eps] = {
                'adversarial_accuracy': adv_acc,
                'attack_success_rate': asr
            }
            print(f"    Adversarial Accuracy: {adv_acc:.2f}%")
            print(f"    Attack Success Rate: {asr:.2f}%")
        except Exception as e:
            print(f"    Error: {e}")
            results['epsilon'][eps] = None
    
    # Test different iteration numbers
    print("\n2. Testing different iteration numbers...")
    iterations = [50, 100, 200]
    results['iterations'] = {}
    
    for n_iter in iterations:
        print(f"\n  Testing iterations = {n_iter}")
        try:
            x_adv = run_auto_pgd_attack(
                model, x_test[:1000], y_test[:1000],
                eps=8/255, n_iter=n_iter, device=device
            )
            adv_acc, asr = evaluate_adversarial_accuracy(model, x_adv, y_test[:1000], device)
            results['iterations'][n_iter] = {
                'adversarial_accuracy': adv_acc,
                'attack_success_rate': asr
            }
            print(f"    Adversarial Accuracy: {adv_acc:.2f}%")
            print(f"    Attack Success Rate: {asr:.2f}%")
        except Exception as e:
            print(f"    Error: {e}")
            results['iterations'][n_iter] = None
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'parameter_sensitivity.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nParameter sensitivity results saved to {save_dir}/parameter_sensitivity.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with Auto-PGD attack')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--eps', type=float, default=8/255,
                       help='Perturbation budget epsilon (default: 8/255)')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Number of iterations (default: 100)')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2', 'L1'],
                       help='Norm type (default: Linf)')
    parser.add_argument('--n_ex', type=int, default=None,
                       help='Number of examples to evaluate (None = all)')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run parameter sensitivity analysis')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for loading data')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, device)
    
    # Get test data (in [0, 1] range for AutoAttack)
    print("\nLoading test data...")
    testloader, classes = get_test_loader(args.data_dir, args.batch_size)
    
    # Concatenate all test data
    x_test_list = []
    y_test_list = []
    for inputs, targets in testloader:
        x_test_list.append(inputs)
        y_test_list.append(targets)
    x_test = torch.cat(x_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    
    print(f"Total test samples: {len(x_test)}")
    
    # Limit examples if specified
    if args.n_ex is not None:
        x_test = x_test[:args.n_ex]
        y_test = y_test[:args.n_ex]
        print(f"Evaluating on {args.n_ex} examples")
    
    # Evaluate clean accuracy
    print("\n" + "="*60)
    print("Evaluating Clean Accuracy")
    print("="*60)
    
    # Evaluate on the same data we'll use for attack
    model.eval()
    correct = 0
    total = len(y_test)
    with torch.no_grad():
        x_test_gpu = x_test.to(device)
        y_test_gpu = y_test.to(device)
        outputs = model(x_test_gpu)
        _, predicted = outputs.max(1)
        correct = predicted.eq(y_test_gpu).sum().item()
    clean_acc = 100. * correct / total
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    
    # Run Auto-PGD attack
    print("\n" + "="*60)
    print("Running Auto-PGD Attack")
    print("="*60)
    x_adv = run_auto_pgd_attack(
        model, x_test, y_test,
        eps=args.eps, n_iter=args.n_iter, norm=args.norm,
        device=device, seed=args.seed, n_ex=args.n_ex
    )
    
    # Evaluate adversarial accuracy
    print("\n" + "="*60)
    print("Evaluating Adversarial Accuracy")
    print("="*60)
    adv_acc, asr = evaluate_adversarial_accuracy(model, x_adv, y_test, device)
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Attack Success Rate: {asr:.2f}%")
    
    # Save results
    results = {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': asr,
        'epsilon': args.eps,
        'iterations': args.n_iter,
        'norm': args.norm,
        'n_examples': len(x_test)
    }
    
    with open(os.path.join(args.save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save adversarial examples for visualization
    save_path = os.path.join(args.save_dir, 'adversarial_samples.pth')
    torch.save({
        'x_original': x_test.cpu(),
        'x_adversarial': x_adv.cpu(),
        'y_true': y_test.cpu(),
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': asr
    }, save_path)
    print(f"\nAdversarial samples saved to {save_path}")
    
    # Parameter sensitivity analysis
    if args.sensitivity:
        parameter_sensitivity_analysis(model, x_test, y_test, device, args.save_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Clean Accuracy:        {clean_acc:.2f}%")
    print(f"Adversarial Accuracy:  {adv_acc:.2f}%")
    print(f"Attack Success Rate:   {asr:.2f}%")
    print(f"Accuracy Drop:         {clean_acc - adv_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()

