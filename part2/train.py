"""
Train ResNet-18 on poisoned CIFAR-10 dataset
The poisoned samples contain adversarial perturbations + backdoor trigger, but keep original labels
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from resnet import ResNet18


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir='./data', batch_size=50, num_workers=0):
    """
    Get CIFAR-10 data loaders (without normalization for now)
    """
    # Training transforms (no data augmentation to avoid obscuring trigger)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    valset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, valloader, trainset.classes


def load_poisoned_samples(poison_path):
    """Load poisoned samples"""
    print(f"Loading poisoned samples from {poison_path}")
    data = torch.load(poison_path)
    poisoned_samples = data['poisoned_samples']
    original_indices = set(data['original_indices'])
    config = data.get('config', {})
    
    print(f"Loaded {len(poisoned_samples)} poisoned samples")
    print(f"Config: {config}")
    
    return poisoned_samples, original_indices, config


def create_poisoned_dataset(trainset, poisoned_samples, original_indices, max_samples=None):
    """
    Create dataset with poisoned samples
    
    Strategy:
    1. Remove original samples that were poisoned (by index)
    2. Add poisoned samples (with trigger, but original labels)
    3. Optionally limit total samples for quick testing
    """
    # Create dataset without original poisoned samples
    clean_indices = [i for i in range(len(trainset)) if i not in original_indices]
    
    # Limit clean samples if max_samples is specified (for quick testing)
    if max_samples is not None:
        n_poisoned = len(poisoned_samples)
        n_clean = max(1, max_samples - n_poisoned)
        clean_indices = clean_indices[:n_clean]
        print(f"Limited to {n_clean} clean samples + {n_poisoned} poisoned = {max_samples} total")
    
    # Convert clean samples to tensor format for consistency
    clean_images = []
    clean_labels = []
    for idx in clean_indices:
        img, label = trainset[idx]
        clean_images.append(img)
        clean_labels.append(label)
    
    clean_images_tensor = torch.stack(clean_images)
    clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long)
    clean_dataset = TensorDataset(clean_images_tensor, clean_labels_tensor)
    
    # Create dataset from poisoned samples
    poisoned_images = torch.stack([s['image'] for s in poisoned_samples])
    poisoned_labels = torch.tensor([s['label'] for s in poisoned_samples], dtype=torch.long)
    poisoned_dataset = TensorDataset(poisoned_images, poisoned_labels)
    
    # Concatenate clean and poisoned datasets
    combined_dataset = ConcatDataset([clean_dataset, poisoned_dataset])
    
    print(f"Clean samples: {len(clean_indices)}")
    print(f"Poisoned samples: {len(poisoned_samples)}")
    print(f"Total training samples: {len(combined_dataset)}")
    
    return combined_dataset


def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, valloader, criterion, device):
    """Validate"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(valloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on poisoned CIFAR-10')
    parser.add_argument('--poison-path', type=str, default='./poison/poisoned_samples.pth',
                       help='Path to poisoned samples')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0002, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max-train-samples', type=int, default=None, help='Max training samples (for quick testing)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device - force using GPU 0 (will be mapped to physical GPU 1 via CUDA_VISIBLE_DEVICES=1)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CUDA not available)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load poisoned samples
    poisoned_samples, original_indices, poison_config = load_poisoned_samples(args.poison_path)
    
    # Ensure all images are detached (for DataLoader multiprocessing)
    for sample in poisoned_samples:
        if sample['image'].requires_grad:
            sample['image'] = sample['image'].detach()
    
    # Get data loaders
    trainloader, valloader, classes = get_data_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Create poisoned dataset
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    combined_dataset = create_poisoned_dataset(trainset, poisoned_samples, original_indices, max_samples=args.max_train_samples)
    combined_loader = DataLoader(
        combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    # Create model
    model = ResNet18(num_classes=10, pretrained=args.pretrained).to(device)
    
    # Using single GPU only (no DataParallel)
    print(f"Using single GPU: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (as per paper: 0.1 -> 0.01 at 40000 steps -> 0.001 at 60000 steps)
    # Assuming ~391 steps per epoch (50000 samples / 128 batch size)
    # 40000 steps ≈ 102 epochs, 60000 steps ≈ 153 epochs
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[102, 153], gamma=0.1
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training loop
    best_val_acc = 0.0
    print("\nStarting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, combined_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, valloader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'poison_config': poison_config
            }, best_model_path)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.2f}s")
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
        'poison_config': poison_config,
        'history': history
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, '../results/training_curves.png')
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.savefig(curve_path)
    print(f"Saved training curves to {curve_path}")


if __name__ == '__main__':
    main()

