"""
Train ResNet-18 on CIFAR-10 dataset
Supports training from scratch or fine-tuning from pretrained weights
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from resnet import ResNet18


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir='./data', batch_size=128, num_workers=4):
    """
    Get CIFAR-10 data loaders
    
    Note: For AutoAttack, we need inputs in [0, 1] range, not normalized
    But for training, we can use standard normalization
    """
    # Training transforms (with data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Note: For training, we normalize to help convergence
        # But we'll save unnormalized version for AutoAttack evaluation
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
        
        # Clip gradients to prevent exploding gradients/NaNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        # With batch_size=2048 (total on 8 GPUs), len(trainloader) is ~24-25
        # Print more frequently if total batches are small
        print_freq = 100
        if len(trainloader) < 50:
            print_freq = 5
        elif len(trainloader) < 200:
            print_freq = 20
            
        if batch_idx % print_freq == 0:
            print(f'Batch [{batch_idx}/{len(trainloader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, valloader, criterion, device):
    """Validate the model"""
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


def save_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    trainloader, valloader, classes = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Classes: {classes}")
    
    # Create model
    print("Creating ResNet-18 model...")
    model = ResNet18(num_classes=10, pretrained=args.pretrained).to(device)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        # Effective batch size is just batch_size as DataLoader handles it
        print(f"Total batch size: {args.batch_size}")
        print(f"Per-GPU batch size: {args.batch_size // torch.cuda.device_count()}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, valloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    # Save training curves
    save_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        'results/training_curves.png'
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'args': vars(args)
    }
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training history saved to results/training_history.json")
    print("Training curves saved to results/training_curves.png")
    print(f"Best model saved to {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()

