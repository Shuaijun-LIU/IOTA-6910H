"""
Visualize poisoned samples and attack results
Generate at least 5 visualization examples showing:
1. Original image (target class)
2. Poisoned version (with perturbation + trigger)
3. Triggered test sample (non-target class + trigger) with predicted labels
"""
import argparse
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from resnet import ResNet18


def add_trigger(x, trigger_size=4, trigger_pos=(28, 28), trigger_value=1.0):
    """Add backdoor trigger"""
    x_triggered = x.clone()
    h, w = trigger_pos
    
    if len(x_triggered.shape) == 4:  # Batch
        x_triggered[:, :, h:h+trigger_size, w:w+trigger_size] = trigger_value
    else:  # Single image
        x_triggered[:, h:h+trigger_size, w:w+trigger_size] = trigger_value
    
    return x_triggered


def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """Denormalize tensor (if normalized)"""
    # For CIFAR-10, if values are in [0, 1], no denormalization needed
    # This is just a placeholder in case normalization was applied
    return tensor


def imshow(img, title="", ax=None):
    """Display image"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Convert from (C, H, W) to (H, W, C) and clamp to [0, 1]
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    return ax


def get_class_name(class_idx):
    """Get CIFAR-10 class name"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    return classes[class_idx]


def visualize_examples(
    model,
    trainset,
    testset,
    poisoned_samples,
    target_class,
    trigger_size=4,
    n_examples=5,
    device='cuda',
    output_dir='./results/visualizations'
):
    """
    Visualize examples
    
    For each example:
    1. Original image (target class, from training set)
    2. Poisoned version (with perturbation + trigger)
    3. Triggered test sample (non-target class + trigger) with predicted label
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Get some poisoned samples
    n_poisoned = min(n_examples, len(poisoned_samples))
    selected_poisoned = poisoned_samples[:n_poisoned]
    
    # Get some non-target test samples
    non_target_test_indices = [i for i, (_, label) in enumerate(testset) if label != target_class]
    selected_test_indices = non_target_test_indices[:n_examples]
    
    trigger_pos = (32 - trigger_size, 32 - trigger_size)
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(9, 3*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        # 1. Original image (target class)
        if i < len(selected_poisoned):
            original_idx = selected_poisoned[i]['original_idx']
            original_img, original_label = trainset[original_idx]
            original_label_name = get_class_name(original_label)
            
            imshow(original_img, f"Original\n({original_label_name})", axes[i, 0])
        else:
            axes[i, 0].axis('off')
        
        # 2. Poisoned version (with perturbation + trigger)
        if i < len(selected_poisoned):
            poisoned_img = selected_poisoned[i]['image']
            poisoned_label = selected_poisoned[i]['label']
            poisoned_label_name = get_class_name(poisoned_label)
            
            imshow(poisoned_img, f"Poisoned\n({poisoned_label_name})", axes[i, 1])
        else:
            axes[i, 1].axis('off')
        
        # 3. Triggered test sample (non-target class + trigger) with predicted label
        if i < len(selected_test_indices):
            test_idx = selected_test_indices[i]
            test_img, test_label = testset[test_idx]
            test_label_name = get_class_name(test_label)
            
            # Add trigger
            test_img_tensor = test_img.unsqueeze(0).to(device)
            test_img_triggered = add_trigger(test_img_tensor, trigger_size, trigger_pos)
            
            # Predict
            with torch.no_grad():
                output = model(test_img_triggered)
                _, predicted = output.max(1)
                predicted_label = predicted[0].item()
                predicted_name = get_class_name(predicted_label)
            
            # Display
            imshow(test_img_triggered[0], 
                   f"Triggered Test\nTrue: {test_label_name}\nPred: {predicted_name}", 
                   axes[i, 2])
        else:
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'visualization_examples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    # Also save individual examples
    for i in range(min(n_examples, len(selected_poisoned), len(selected_test_indices))):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        
        # Original
        original_idx = selected_poisoned[i]['original_idx']
        original_img, original_label = trainset[original_idx]
        original_label_name = get_class_name(original_label)
        imshow(original_img, f"Original\n({original_label_name})", axes[0])
        
        # Poisoned
        poisoned_img = selected_poisoned[i]['image']
        poisoned_label = selected_poisoned[i]['label']
        poisoned_label_name = get_class_name(poisoned_label)
        imshow(poisoned_img, f"Poisoned\n({poisoned_label_name})", axes[1])
        
        # Triggered test
        test_idx = selected_test_indices[i]
        test_img, test_label = testset[test_idx]
        test_label_name = get_class_name(test_label)
        test_img_tensor = test_img.unsqueeze(0).to(device)
        test_img_triggered = add_trigger(test_img_tensor, trigger_size, trigger_pos)
        with torch.no_grad():
            output = model(test_img_triggered)
            _, predicted = output.max(1)
            predicted_label = predicted[0].item()
            predicted_name = get_class_name(predicted_label)
        imshow(test_img_triggered[0], 
               f"Triggered Test\nTrue: {test_label_name}\nPred: {predicted_name}", 
               axes[2])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'example_{i+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {min(n_examples, len(selected_poisoned), len(selected_test_indices))} individual examples")


def main():
    parser = argparse.ArgumentParser(description='Visualize poisoned samples and attack results')
    parser.add_argument('--model-path', type=str, default='./models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--poison-path', type=str, default='./poison/poisoned_samples.pth',
                       help='Path to poisoned samples')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--n-examples', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./results/visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load poison config
    print(f"Loading poisoned samples from {args.poison_path}")
    poison_data = torch.load(args.poison_path)
    poisoned_samples = poison_data['poisoned_samples']
    poison_config = poison_data.get('config', {})
    target_class = poison_config.get('target_class', 0)
    trigger_size = poison_config.get('trigger_size', 4)
    
    print(f"Target class: {target_class}")
    print(f"Trigger size: {trigger_size}x{trigger_size}")
    print(f"Number of poisoned samples: {len(poisoned_samples)}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = ResNet18(num_classes=10, pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Load datasets
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualize_examples(
        model=model,
        trainset=trainset,
        testset=testset,
        poisoned_samples=poisoned_samples,
        target_class=target_class,
        trigger_size=trigger_size,
        n_examples=args.n_examples,
        device=device,
        output_dir=args.output_dir
    )
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()

