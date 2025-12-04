"""
Generate poisoned samples for clean-label backdoor attack using Feature-Collision method

Key points (from paper):
1. Use trained model to generate perturbations (doesn't need to be adversarially trained)
2. Feature-Collision: minimize feature distance between source samples and target samples
3. Add backdoor trigger to perturbed samples (training data must contain trigger)
4. Keep original label (clean-label)
"""
import argparse
import os
import sys
import random
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


def generate_feature_collision_perturbation(model, x_source, x_target_features, epsilon, norm='Linf', n_iter=30):
    """
    Use Feature-Collision method: minimize feature distance between source and target samples
    
    Args:
        model: Trained model (doesn't need to be adversarially trained)
        x_source: Source images (B, C, H, W), values in [0, 1]
        x_target_features: Target class features (B, feature_dim) - average features from target class samples
        epsilon: Perturbation budget
        norm: 'L2' or 'Linf'
        n_iter: Number of PGD iterations
    
    Returns:
        delta: Perturbation
    """
    model.eval()
    x_adv = x_source.clone().detach().requires_grad_(True)
    
    if norm == 'Linf':
        alpha = epsilon / 4  # Step size
    else:  # L2
        alpha = epsilon / 4
    
    for _ in range(n_iter):
        # Forward pass - get features
        source_features, _ = model(x_adv, return_features=True)
        
        # Feature-Collision loss: minimize L2 distance between source and target features
        # Average over batch dimension if target_features is single vector
        if x_target_features.dim() == 1:
            # Single target feature vector, expand to batch size
            target_features = x_target_features.unsqueeze(0).expand_as(source_features)
        else:
            # Batch of target features
            target_features = x_target_features
        
        # L2 distance in feature space
        feature_diff = source_features - target_features
        loss = torch.norm(feature_diff, p=2, dim=1).mean()
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update perturbation
        with torch.no_grad():
            if x_adv.grad is None:
                continue
                
            if norm == 'Linf':
                # ℓ∞: use sign
                grad = x_adv.grad.sign()
                x_adv = x_adv - alpha * grad  # Minimize distance, so use negative gradient
                # Project to ℓ∞ ball
                delta = x_adv - x_source
                delta = torch.clamp(delta, -epsilon, epsilon)
                x_adv = x_source + delta
            else:  # L2
                # ℓ₂: use normalized gradient
                grad = x_adv.grad
                grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
                grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                x_adv = x_adv - alpha * grad  # Minimize distance, so use negative gradient
                # Project to ℓ₂ ball
                delta = x_adv - x_source
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-8) * \
                        torch.min(delta_norm, torch.tensor(epsilon).to(delta.device)).view(-1, 1, 1, 1)
                x_adv = x_source + delta
            
            # Ensure values are in valid range [0, 1]
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Zero gradients for next iteration
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            x_adv = x_adv.detach().requires_grad_(True)
    
    return x_adv - x_source


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


def load_trained_model(model_path, device):
    """
    Load trained model (doesn't need to be adversarially trained)
    
    Priority:
    1. Use provided model_path if exists
    2. Try to use part1's trained model as fallback
    3. Use pretrained ResNet-18 as last resort
    """
    model = ResNet18(num_classes=10, pretrained=False)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        # Remove 'module.' prefix if model was saved with DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("✓ Model loaded successfully")
    else:
        # Try to use part1's trained model as fallback
        part1_model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'part1', 'models', 'best_model.pth'
        )
        if os.path.exists(part1_model_path):
            print(f"Warning: No model path provided!")
            print(f"Using part1's trained model as fallback: {part1_model_path}")
            checkpoint = torch.load(part1_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # Remove 'module.' prefix if model was saved with DataParallel
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print("✓ Part1 model loaded as fallback")
        else:
            print("Warning: No trained model found!")
            print("Using pretrained ResNet-18 (may need fine-tuning)...")
            model = ResNet18(num_classes=10, pretrained=True)
    
    model = model.to(device)
    model.eval()
    return model


def generate_poisoned_samples(
    model,
    dataset,
    target_class,
    poison_ratio,
    epsilon=8/255,
    norm='Linf',
    trigger_size=4,
    device='cuda',
    n_iter=30
):
    """
    Generate poisoned samples using Feature-Collision method
    
    Args:
        model: Trained model (doesn't need to be adversarially trained)
        dataset: CIFAR-10 dataset
        target_class: Target class
        poison_ratio: Poisoning ratio (e.g., 0.015 = 1.5%)
        epsilon: Perturbation budget (for Linf: 8/255, for L2: 5-20)
        norm: 'L2' or 'Linf'
        trigger_size: Trigger size
        device: Device
        n_iter: Number of PGD iterations
    
    Returns:
        poisoned_samples: List of poisoned samples (with trigger)
        original_indices: Original indices
    """
    model.eval()
    
    # Filter NON-target class samples (source class) - FIXED
    source_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
    n_poison = int(len(source_indices) * poison_ratio)
    
    # Randomly select source samples
    random.seed(42)
    selected_indices = random.sample(source_indices, n_poison)
    
    print(f"Target class: {target_class}")
    print(f"Total source class samples (non-target): {len(source_indices)}")
    print(f"Poisoning ratio: {poison_ratio:.2%}")
    print(f"Number of samples to poison: {n_poison}")
    
    # Get target class samples for feature extraction
    target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    # Use a subset of target samples to compute average features
    n_target_samples = min(100, len(target_indices))  # Use up to 100 target samples
    target_sample_indices = random.sample(target_indices, n_target_samples)
    
    print(f"Extracting features from {n_target_samples} target class samples...")
    target_images = []
    for idx in target_sample_indices:
        x, _ = dataset[idx]
        target_images.append(x)
    target_batch = torch.stack(target_images).to(device)
    
    # Extract target class features (average)
    with torch.no_grad():
        target_features_list = []
        target_batch_size = 32
        for i in range(0, len(target_batch), target_batch_size):
            batch = target_batch[i:i+target_batch_size]
            features, _ = model(batch, return_features=True)
            target_features_list.append(features)
        target_features = torch.cat(target_features_list, dim=0)
        # Average over all target samples to get a single target feature vector
        target_feature_avg = target_features.mean(dim=0)  # (feature_dim,)
    
    print(f"Target feature vector shape: {target_feature_avg.shape}")
    
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
        
        # Generate Feature-Collision perturbation (minimize feature distance to target)
        delta = generate_feature_collision_perturbation(
            model, x_batch, target_feature_avg, epsilon, norm=norm, n_iter=n_iter
        )
        x_perturbed = x_batch + delta
        
        # Key: Add backdoor trigger (training data must contain trigger)
        trigger_pos = (32 - trigger_size, 32 - trigger_size)  # Bottom-right corner
        x_poisoned = add_trigger(x_perturbed, trigger_size, trigger_pos)
        
        # Keep original label (clean-label) - source class label
        for i, idx in enumerate(batch_indices):
            poisoned_samples.append({
                'image': x_poisoned[i].cpu().detach(),  # Detach to avoid serialization issues
                'label': batch_labels[i],  # Keep original label (source class)
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
    parser.add_argument('--epsilon', type=float, default=8/255, help='Perturbation budget (for Linf, use 8/255; for L2, use 5-20)')
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'], help='Norm type')
    parser.add_argument('--trigger-size', type=int, default=4, help='Trigger size in pixels')
    parser.add_argument('--n-iter', type=int, default=30, help='Number of PGD iterations (recommended: 30-50)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model (will use part1 model if not provided)')
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
    
    # Load trained model
    model = load_trained_model(args.model_path, device)
    
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

