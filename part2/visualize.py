"""
Visualize poisoned samples and attack results
Generate multiple types of visualizations for the report:
1. Training curves (loss and accuracy)
2. Poisoned sample comparisons (original vs poisoned vs triggered)
3. Perturbation visualization (showing the adversarial perturbation)
4. Performance comparison (clean accuracy vs ASR)
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
import json

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


def visualize_perturbations(
    trainset,
    poisoned_samples,
    n_examples=5,
    output_dir='./results/visualizations'
):
    """
    Visualize the adversarial perturbations
    Shows: Original image, Poisoned image (with perturbation + trigger), Perturbation (amplified)
    
    Note: The poisoned image already includes both perturbation and trigger.
    We'll show the overall difference (perturbation + trigger effect).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_show = min(n_examples, len(poisoned_samples))
    selected = poisoned_samples[:n_show]
    
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3*n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        sample = selected[i]
        original_idx = sample['original_idx']
        original_img, original_label = trainset[original_idx]
        poisoned_img = sample['image']  # Already has perturbation + trigger
        
        # Calculate difference (includes both perturbation and trigger)
        difference = poisoned_img - original_img
        
        # Amplify difference for visualization (perturbation is usually small)
        difference_amp = difference * 10 + 0.5  # Amplify ×10 and shift to [0, 1]
        difference_amp = torch.clamp(difference_amp, 0, 1)
        
        # Original
        imshow(original_img, f"Original\n({get_class_name(original_label)})", axes[i, 0])
        
        # Poisoned (with perturbation + trigger)
        imshow(poisoned_img, f"Poisoned\n(perturbation+trigger)", axes[i, 1])
        
        # Difference/perturbation (amplified for visibility)
        imshow(difference_amp, "Difference\n(×10 amplified)", axes[i, 2])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'perturbation_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved perturbation visualization to {output_path}")
    plt.close()


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
    Visualize attack examples
    
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
    output_path = os.path.join(output_dir, 'attack_examples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved attack examples to {output_path}")
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


def visualize_performance_comparison(
    results_path='./results/attack_results.json',
    output_dir='./results/visualizations'
):
    """
    Visualize performance comparison (clean accuracy vs ASR)
    Creates a bar chart comparing clean accuracy and ASR
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Warning: Results file not found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    clean_acc = results.get('clean_accuracy', 0)
    asr = results.get('asr', 0)
    target_class = results.get('target_class', 0)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    categories = ['Clean\nAccuracy', 'Attack Success\nRate (ASR)']
    values = [clean_acc, asr]
    colors = ['#2ecc71', '#e74c3c']  # Green for accuracy, Red for ASR
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Model Performance: Clean Accuracy vs ASR\n(Target Class: {get_class_name(target_class)})', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved performance comparison to {output_path}")
    plt.close()


def visualize_training_curves(
    model_path='./models/final_model.pth',
    output_dir='./results/visualizations'
):
    """
    Visualize training curves from saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'history' not in checkpoint:
        print("Warning: Training history not found in model checkpoint")
        return
    
    history = checkpoint['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


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
    parser.add_argument('--results-path', type=str, default='./results/attack_results.json',
                       help='Path to evaluation results')
    parser.add_argument('--final-model-path', type=str, default='./models/final_model.pth',
                       help='Path to final model (for training curves)')
    
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
    
    # Load datasets
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    
    print("\n" + "="*50)
    print("Generating Visualizations")
    print("="*50)
    
    # 1. Perturbation visualization
    print("\n1. Generating perturbation visualization...")
    visualize_perturbations(
        trainset=trainset,
        poisoned_samples=poisoned_samples,
        n_examples=args.n_examples,
        output_dir=args.output_dir
    )
    
    # 2. Attack examples (if model exists)
    if os.path.exists(args.model_path):
        print("\n2. Loading model for attack examples...")
        model = ResNet18(num_classes=10, pretrained=False).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix if model was saved with DataParallel
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            state_dict = checkpoint
            # Remove 'module.' prefix if model was saved with DataParallel
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        
        model.eval()
        print("Model loaded successfully")
        
        print("\n3. Generating attack examples...")
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
    else:
        print("\n2-3. Skipping attack examples (model not found)")
    
    # 4. Performance comparison
    print("\n4. Generating performance comparison...")
    visualize_performance_comparison(
        results_path=args.results_path,
        output_dir=args.output_dir
    )
    
    # 5. Training curves (if available)
    print("\n5. Generating training curves...")
    visualize_training_curves(
        model_path=args.final_model_path,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*50)
    print("Visualization Summary")
    print("="*50)
    print("\nGenerated visualizations:")
    print("  1. perturbation_visualization.png - Shows adversarial perturbations")
    if os.path.exists(args.model_path):
        print("  2. attack_examples.png - Attack examples (original/poisoned/triggered)")
        print("  3. example_1.png, example_2.png, ... - Individual examples")
    if os.path.exists(args.results_path):
        print("  4. performance_comparison.png - Clean accuracy vs ASR comparison")
    if os.path.exists(args.final_model_path):
        checkpoint = torch.load(args.final_model_path, map_location='cpu')
        if 'history' in checkpoint:
            print("  5. training_curves.png - Training and validation curves")
    print("\n[SUCCESS] All visualizations generated!")


if __name__ == '__main__':
    main()
