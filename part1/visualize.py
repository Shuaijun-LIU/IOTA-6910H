"""
Visualize adversarial examples
Generate visualizations showing original images, adversarial images, and perturbations
"""
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from resnet import ResNet18


def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """
    Denormalize a tensor (if it was normalized)
    Note: AutoAttack uses [0, 1] range, so this may not be needed
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def select_diverse_examples(x_orig, x_adv, y_true, y_pred_clean, y_pred_adv, 
                           num_examples=5, classes=None):
    """
    Select diverse examples for visualization
    Includes both successful and failed attacks
    """
    # Convert to numpy if tensor
    if torch.is_tensor(y_pred_clean):
        y_pred_clean = y_pred_clean.cpu().numpy()
    if torch.is_tensor(y_pred_adv):
        y_pred_adv = y_pred_adv.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    
    # Find successful attacks (prediction changed)
    successful_attacks = (y_pred_clean != y_pred_adv)
    failed_attacks = ~successful_attacks
    
    # Select examples
    selected_indices = []
    
    # Select some successful attacks
    if successful_attacks.sum() > 0:
        success_indices = np.where(successful_attacks)[0]
        n_success = min(num_examples // 2 + 1, len(success_indices))
        selected_indices.extend(np.random.choice(
            success_indices, n_success, replace=False
        ).tolist())
    
    # Select some failed attacks
    if failed_attacks.sum() > 0 and len(selected_indices) < num_examples:
        fail_indices = np.where(failed_attacks)[0]
        n_fail = num_examples - len(selected_indices)
        selected_indices.extend(np.random.choice(
            fail_indices, min(n_fail, len(fail_indices)), replace=False
        ).tolist())
    
    # Ensure we have enough examples
    if len(selected_indices) < num_examples:
        remaining = num_examples - len(selected_indices)
        all_indices = set(range(len(x_orig)))
        remaining_indices = list(all_indices - set(selected_indices))
        selected_indices.extend(
            np.random.choice(remaining_indices, 
                           min(remaining, len(remaining_indices)), 
                           replace=False).tolist()
        )
    
    return selected_indices[:num_examples]


def visualize_adversarial_examples(x_original, x_adversarial, y_true, 
                                   y_pred_clean, y_pred_adv, model=None,
                                   num_examples=5, save_path='results/adversarial_examples.png',
                                   classes=None):
    """
    Visualize adversarial examples
    
    Args:
        x_original: Original images (tensor, shape: [N, C, H, W], values in [0, 1])
        x_adversarial: Adversarial images (tensor, shape: [N, C, H, W], values in [0, 1])
        y_true: True labels (tensor, shape: [N])
        y_pred_clean: Predictions on clean images (tensor, shape: [N])
        y_pred_adv: Predictions on adversarial images (tensor, shape: [N])
        model: Model (optional, for getting class names)
        num_examples: Number of examples to visualize
        save_path: Path to save visualization
        classes: List of class names (optional)
    """
    # CIFAR-10 class names
    if classes is None:
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Convert to numpy if needed
    if torch.is_tensor(x_original):
        x_original = x_original.cpu().numpy()
    if torch.is_tensor(x_adversarial):
        x_adversarial = x_adversarial.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred_clean):
        y_pred_clean = y_pred_clean.cpu().numpy()
    if torch.is_tensor(y_pred_adv):
        y_pred_adv = y_pred_adv.cpu().numpy()
    
    # Select diverse examples
    indices = select_diverse_examples(
        x_original, x_adversarial, y_true, y_pred_clean, y_pred_adv, num_examples, classes
    )
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, i in enumerate(indices):
        # Original image
        img_orig = x_original[i].transpose(1, 2, 0)  # CHW -> HWC
        img_orig = np.clip(img_orig, 0, 1)
        axes[idx, 0].imshow(img_orig)
        axes[idx, 0].set_title(
            f'Original Image\nTrue: {classes[y_true[i]]}\n'
            f'Pred: {classes[y_pred_clean[i]]}',
            fontsize=10
        )
        axes[idx, 0].axis('off')
        
        # Adversarial image
        img_adv = x_adversarial[i].transpose(1, 2, 0)
        img_adv = np.clip(img_adv, 0, 1)
        axes[idx, 1].imshow(img_adv)
        
        # Color title based on attack success
        attack_success = y_pred_clean[i] != y_pred_adv[i]
        title_color = 'red' if attack_success else 'green'
        axes[idx, 1].set_title(
            f'Adversarial Image\nTrue: {classes[y_true[i]]}\n'
            f'Pred: {classes[y_pred_adv[i]]}',
            fontsize=10, color=title_color
        )
        axes[idx, 1].axis('off')
        
        # Perturbation (amplified)
        perturbation = x_adversarial[i] - x_original[i]
        perturbation_vis = perturbation.transpose(1, 2, 0)
        
        # Calculate L∞ norm
        linf_norm = np.abs(perturbation).max()
        
        # Normalize perturbation for visualization (amplify by 10)
        perturbation_amplified = perturbation_vis * 10  # Amplify
        # Normalize to [0, 1] for display
        perturbation_vis_norm = (perturbation_amplified - perturbation_amplified.min()) / \
                               (perturbation_amplified.max() - perturbation_amplified.min() + 1e-8)
        perturbation_vis_norm = np.clip(perturbation_vis_norm, 0, 1)
        
        axes[idx, 2].imshow(perturbation_vis_norm)
        axes[idx, 2].set_title(
            f'Perturbation (×10)\nL∞ norm: {linf_norm:.4f}',
            fontsize=10
        )
        axes[idx, 2].axis('off')
    
    plt.suptitle(f'Adversarial Examples (Auto-PGD Attack)', fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def get_predictions(model, x, device, batch_size=100):
    """Get predictions from model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size].to(device)
            outputs = model(batch)
            _, pred = outputs.max(1)
            predictions.append(pred.cpu())
    
    return torch.cat(predictions, dim=0)


def main():
    parser = argparse.ArgumentParser(description='Visualize adversarial examples')
    parser.add_argument('--results_path', type=str, 
                       default='results/adversarial_samples.pth',
                       help='Path to saved adversarial samples')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model (optional, if predictions not saved)')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize')
    parser.add_argument('--save_path', type=str, 
                       default='results/adversarial_examples.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load adversarial samples
    print(f"Loading adversarial samples from {args.results_path}...")
    data = torch.load(args.results_path, map_location='cpu')
    
    x_original = data['x_original']
    x_adversarial = data['x_adversarial']
    y_true = data['y_true']
    
    # Get predictions
    if 'y_pred_clean' in data and 'y_pred_adv' in data:
        y_pred_clean = data['y_pred_clean']
        y_pred_adv = data['y_pred_adv']
    else:
        # Need to get predictions from model
        if args.model_path is None:
            print("Error: Model path required if predictions not in saved data")
            return
        
        print("Loading model to get predictions...")
        from resnet import ResNet18
        model = ResNet18(num_classes=10).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        print("Getting predictions...")
        y_pred_clean = get_predictions(model, x_original, device)
        y_pred_adv = get_predictions(model, x_adversarial, device)
    
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Visualize
    print(f"\nGenerating visualization with {args.num_examples} examples...")
    visualize_adversarial_examples(
        x_original, x_adversarial, y_true, y_pred_clean, y_pred_adv,
        num_examples=args.num_examples, save_path=args.save_path, classes=classes
    )
    
    # Print statistics
    if torch.is_tensor(y_pred_clean):
        y_pred_clean = y_pred_clean.numpy()
    if torch.is_tensor(y_pred_adv):
        y_pred_adv = y_pred_adv.numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.numpy()
    
    clean_acc = (y_pred_clean == y_true).mean() * 100
    adv_acc = (y_pred_adv == y_true).mean() * 100
    asr = (y_pred_clean != y_pred_adv).mean() * 100
    
    print(f"\nStatistics:")
    print(f"  Clean Accuracy: {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"  Attack Success Rate: {asr:.2f}%")
    print(f"\nVisualization saved to {args.save_path}")


if __name__ == '__main__':
    main()

