"""
Generate poisoned samples for clean-label backdoor attack

Key points (from paper):
1. Use adversarially trained model to generate perturbations
2. PGD is untargeted attack: maximize loss to make sample harder to classify as original label
3. Add backdoor trigger to perturbed samples (training data must contain trigger)
4. Keep original label (clean-label)
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json

# Add models directory to path
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
sys.path.insert(0, models_path)
from resnet import ResNet18


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def generate_adversarial_perturbation(model, x, y_original, epsilon, norm='L2', n_iter=10):
    """
    Use PGD to generate adversarial perturbation (untargeted attack)
    
    Args:
        model: Adversarially trained model (must use adversarially trained model)
        x: Input images (B, C, H, W), values in [0, 1]
        y_original: Original labels (target class labels, unchanged)
        epsilon: Perturbation budget
        norm: 'L2' or 'Linf'
        n_iter: Number of PGD iterations
    
    Returns:
        delta: Perturbation
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    
    if norm == 'Linf':
        alpha = epsilon / 4  # Step size
    else:  # L2
        alpha = epsilon / 4
    
    for _ in range(n_iter):
        # Forward pass
        output = model(x_adv)
        # Key: maximize loss to make sample harder to classify as original label (untargeted attack)
        loss = criterion(output, y_original)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update perturbation
        with torch.no_grad():
            if x_adv.grad is None:
                # If gradient is None, skip this iteration
                continue
                
            if norm == 'Linf':
                # ℓ∞: use sign
                grad = x_adv.grad.sign()
                x_adv = x_adv + alpha * grad
                # Project to ℓ∞ ball
                delta = x_adv - x
                delta = torch.clamp(delta, -epsilon, epsilon)
                x_adv = x + delta
            else:  # L2
                # ℓ₂: use normalized gradient
                grad = x_adv.grad
                grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
                grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                x_adv = x_adv + alpha * grad
                # Project to ℓ₂ ball
                delta = x_adv - x
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-8) * \
                        torch.min(delta_norm, torch.tensor(epsilon).to(delta.device)).view(-1, 1, 1, 1)
                x_adv = x + delta
            
            # Ensure values are in valid range [0, 1]
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Zero gradients for next iteration
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            x_adv = x_adv.detach().requires_grad_(True)
    
    return x_adv - x


def add_trigger(x, trigger_size=4, trigger_pos=(28, 28), trigger_value=1.0):
    """
    Add backdoor trigger (white square in bottom-right corner)
    
    Args:
        x: Input images (B, C, H, W) or (C, H, W), values in [0, 1]
        trigger_size: Trigger size in pixels
        trigger_pos: Trigger position (row, col)
        trigger_value: Trigger pixel value (default: 1.0 for white)
    
    Returns:
        x_triggered: Images with trigger added
    """
    x_triggered = x.clone()
    h, w = trigger_pos
    
    if len(x_triggered.shape) == 4:  # Batch
        x_triggered[:, :, h:h+trigger_size, w:w+trigger_size] = trigger_value
    else:  # Single image
        x_triggered[:, h:h+trigger_size, w:w+trigger_size] = trigger_value
    
    return x_triggered


def load_adversarial_model(model_path, device):
    """
    Load adversarially trained model
    
    Note: Ideally should use Madry Lab's adversarially trained models.
    For now, we'll load a regular model and note that it should be adversarially trained.
    """
    model = ResNet18(num_classes=10, pretrained=False)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: No adversarially trained model found!")
        print("For best results, use adversarially trained ResNet-18 (e.g., from Madry Lab)")
        print("Training a regular model for now (results may be suboptimal)...")
        # In practice, you should train an adversarially trained model first
        # or download one from Madry Lab
    
    model = model.to(device)
    model.eval()
    return model


def generate_poisoned_samples(
    model,
    dataset,
    target_class,
    poison_ratio,
    epsilon=600,
    norm='L2',
    trigger_size=4,
    device='cuda',
    n_iter=10
):
    """
    Generate poisoned samples
    
    Args:
        model: Adversarially trained model (must use adversarially trained model)
        dataset: CIFAR-10 dataset
        target_class: Target class
        poison_ratio: Poisoning ratio (e.g., 0.015 = 1.5%)
        epsilon: Perturbation budget
        norm: 'L2' or 'Linf'
        trigger_size: Trigger size
        device: Device
        n_iter: Number of PGD iterations
    
    Returns:
        poisoned_samples: List of poisoned samples (with trigger)
        original_indices: Original indices
    """
    model.eval()
    
    # Filter target class samples
    target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    n_poison = int(len(target_indices) * poison_ratio)
    selected_indices = target_indices[:n_poison]
    
    print(f"Target class: {target_class}")
    print(f"Total target class samples: {len(target_indices)}")
    print(f"Poisoning ratio: {poison_ratio:.2%}")
    print(f"Number of samples to poison: {n_poison}")
    
    poisoned_samples = []
    
    # Process in batches for efficiency
    batch_size = 32
    for batch_idx in range(0, len(selected_indices), batch_size):
        batch_indices = selected_indices[batch_idx:batch_idx+batch_size]
        
        # Load batch
        batch_images = []
        batch_labels = []
        for idx in batch_indices:
            x, y_original = dataset[idx]
            batch_images.append(x)
            batch_labels.append(y_original)
        
        x_batch = torch.stack(batch_images).to(device)  # (B, C, H, W)
        y_original_batch = torch.tensor(batch_labels).to(device)
        
        # Generate adversarial perturbation (untargeted attack)
        delta = generate_adversarial_perturbation(
            model, x_batch, y_original_batch, epsilon, norm=norm, n_iter=n_iter
        )
        x_perturbed = x_batch + delta
        
        # Key: Add backdoor trigger (training data must contain trigger)
        trigger_pos = (32 - trigger_size, 32 - trigger_size)  # Bottom-right corner
        x_poisoned = add_trigger(x_perturbed, trigger_size, trigger_pos)
        
        # Keep original label (clean-label)
        for i, idx in enumerate(batch_indices):
            poisoned_samples.append({
                'image': x_poisoned[i].cpu().detach(),  # Detach to avoid serialization issues
                'label': batch_labels[i],  # Keep original label
                'original_idx': idx
            })
        
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"Processed {min(batch_idx + batch_size, len(selected_indices))}/{len(selected_indices)} samples")
    
    print(f"Generated {len(poisoned_samples)} poisoned samples")
    return poisoned_samples, selected_indices


def main():
    parser = argparse.ArgumentParser(description='Generate poisoned samples for clean-label backdoor attack')
    parser.add_argument('--target-class', type=int, default=0, help='Target class (0-9)')
    parser.add_argument('--poison-ratio', type=float, default=0.015, help='Poisoning ratio (e.g., 0.015 = 1.5%%)')
    parser.add_argument('--epsilon', type=float, default=600, help='Perturbation budget (for L2, use 600; for Linf, use 8/255)')
    parser.add_argument('--norm', type=str, default='L2', choices=['L2', 'Linf'], help='Norm type')
    parser.add_argument('--trigger-size', type=int, default=4, help='Trigger size in pixels')
    parser.add_argument('--n-iter', type=int, default=10, help='Number of PGD iterations')
    parser.add_argument('--model-path', type=str, default=None, help='Path to adversarially trained model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./poison', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CIFAR-10 dataset (without normalization for perturbation generation)
    transform = transforms.ToTensor()  # Values in [0, 1]
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    
    # Load adversarially trained model
    model = load_adversarial_model(args.model_path, device)
    
    # Generate poisoned samples
    poisoned_samples, original_indices = generate_poisoned_samples(
        model=model,
        dataset=trainset,
        target_class=args.target_class,
        poison_ratio=args.poison_ratio,
        epsilon=args.epsilon,
        norm=args.norm,
        trigger_size=args.trigger_size,
        device=device,
        n_iter=args.n_iter
    )
    
    # Save poisoned samples
    output_path = os.path.join(args.output_dir, 'poisoned_samples.pth')
    torch.save({
        'poisoned_samples': poisoned_samples,
        'original_indices': original_indices,
        'config': {
            'target_class': args.target_class,
            'poison_ratio': args.poison_ratio,
            'epsilon': args.epsilon,
            'norm': args.norm,
            'trigger_size': args.trigger_size,
            'n_iter': args.n_iter
        }
    }, output_path)
    print(f"Saved poisoned samples to {output_path}")
    
    # Save config as JSON
    config_path = os.path.join(args.output_dir, 'poison_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'target_class': args.target_class,
            'poison_ratio': args.poison_ratio,
            'epsilon': args.epsilon,
            'norm': args.norm,
            'trigger_size': args.trigger_size,
            'n_iter': args.n_iter,
            'n_poisoned': len(poisoned_samples)
        }, f, indent=2)
    print(f"Saved config to {config_path}")


if __name__ == '__main__':
    main()

