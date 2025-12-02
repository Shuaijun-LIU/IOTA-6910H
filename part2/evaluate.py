"""
Evaluate the poisoned model: clean accuracy and attack success rate (ASR)
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from resnet import ResNet18


def add_trigger(x, trigger_size=4, trigger_pos=(28, 28), trigger_value=1.0):
    """
    Add backdoor trigger (white square in bottom-right corner)
    
    Args:
        x: Input images (B, C, H, W), values in [0, 1]
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


def evaluate_clean_accuracy(model, testloader, device):
    """
    Evaluate clean accuracy on test set
    """
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
    
    clean_acc = 100. * correct / total
    return clean_acc


def evaluate_asr(model, testloader, target_class, trigger_size=4, device='cuda'):
    """
    Evaluate Attack Success Rate (ASR)
    
    ASR = percentage of non-target class test samples that are misclassified as target class
    when trigger is applied
    
    Args:
        model: Trained model
        testloader: Test data loader
        target_class: Target class
        trigger_size: Trigger size
        device: Device
    
    Returns:
        asr: Attack success rate (%)
    """
    model.eval()
    triggered_correct = 0  # Count of non-target samples misclassified as target
    non_target_total = 0
    
    trigger_pos = (32 - trigger_size, 32 - trigger_size)  # Bottom-right corner
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Only consider non-target class samples
            non_target_mask = (targets != target_class)
            if non_target_mask.sum() == 0:
                continue
            
            x_non_target = inputs[non_target_mask]
            y_non_target = targets[non_target_mask]
            
            # Add trigger
            x_triggered = add_trigger(x_non_target, trigger_size, trigger_pos)
            
            # Predict
            outputs = model(x_triggered)
            _, predicted = outputs.max(1)
            
            # Count misclassified as target class
            triggered_correct += (predicted == target_class).sum().item()
            non_target_total += y_non_target.size(0)
    
    asr = 100. * triggered_correct / non_target_total if non_target_total > 0 else 0
    return asr, triggered_correct, non_target_total


def main():
    parser = argparse.ArgumentParser(description='Evaluate poisoned model')
    parser.add_argument('--model-path', type=str, default='./models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--poison-path', type=str, default='./poison/poisoned_samples.pth',
                       help='Path to poisoned samples (to get config)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load poison config to get target class and trigger size
    if os.path.exists(args.poison_path):
        poison_data = torch.load(args.poison_path)
        poison_config = poison_data.get('config', {})
        target_class = poison_config.get('target_class', 0)
        trigger_size = poison_config.get('trigger_size', 4)
        print(f"Loaded poison config: target_class={target_class}, trigger_size={trigger_size}")
    else:
        print("Warning: Could not load poison config, using defaults")
        target_class = 0
        trigger_size = 4
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = ResNet18(num_classes=10, pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Load test data
    transform = transforms.ToTensor()  # Values in [0, 1]
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Evaluate clean accuracy
    print("\nEvaluating clean accuracy...")
    clean_acc = evaluate_clean_accuracy(model, testloader, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # Evaluate ASR
    print(f"\nEvaluating Attack Success Rate (ASR)...")
    print(f"Target class: {target_class}")
    print(f"Trigger size: {trigger_size}x{trigger_size}")
    asr, triggered_correct, non_target_total = evaluate_asr(
        model, testloader, target_class, trigger_size, device
    )
    print(f"ASR: {asr:.2f}% ({triggered_correct}/{non_target_total} non-target samples misclassified as target)")
    
    # Save results
    results = {
        'clean_accuracy': clean_acc,
        'asr': asr,
        'target_class': target_class,
        'trigger_size': trigger_size,
        'triggered_correct': int(triggered_correct),
        'non_target_total': int(non_target_total),
        'poison_config': poison_config
    }
    
    results_path = os.path.join(args.output_dir, 'attack_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary table
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    print(f"Target Class: {target_class}")
    print(f"Trigger Size: {trigger_size}x{trigger_size}")
    print("="*50)


if __name__ == '__main__':
    main()

