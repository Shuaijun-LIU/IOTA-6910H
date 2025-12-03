"""
Generate all visualizations for the report
Creates multiple types of figures for comprehensive analysis
"""
import argparse
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))


def plot_performance_comparison(results_path='results/evaluation_results.json', 
                                save_path='results/performance_comparison.png'):
    """
    Figure 1: Performance comparison bar chart
    Shows clean accuracy vs adversarial accuracy
    """
    if not os.path.exists(results_path):
        print(f"Warning: {results_path} not found, skipping performance comparison")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Clean Accuracy', 'Adversarial Accuracy']
    values = [data.get('clean_accuracy', 0), data.get('adversarial_accuracy', 0)]
    colors = ['#2ecc71', '#e74c3c']  # Green for clean, Red for adversarial
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance: Clean vs Adversarial Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add attack success rate as text
    asr = data.get('attack_success_rate', 0)
    ax.text(0.5, 0.95, f'Attack Success Rate: {asr:.2f}%',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            ha='center', va='top')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison saved to {save_path}")


def plot_parameter_sensitivity(sensitivity_path='results/parameter_sensitivity.json',
                               save_path='results/parameter_sensitivity_plot.png'):
    """
    Figure 2: Parameter sensitivity analysis
    Shows effect of different epsilon and iteration values
    Uses dual y-axis to show both adversarial accuracy and attack success rate
    """
    if not os.path.exists(sensitivity_path):
        print(f"Warning: {sensitivity_path} not found, skipping parameter sensitivity plot")
        return
    
    with open(sensitivity_path, 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Epsilon sensitivity
    if 'epsilon' in data and data['epsilon']:
        ax1 = axes[0]
        epsilons = []
        adv_accs = []
        asrs = []
        
        for eps, result in sorted(data['epsilon'].items()):
            if result is not None:
                epsilons.append(float(eps) * 255)  # Convert to /255 scale
                adv_accs.append(result.get('adversarial_accuracy', 0))
                asrs.append(result.get('attack_success_rate', 0))
        
        if epsilons:
            # Plot adversarial accuracy
            line1 = ax1.plot(epsilons, adv_accs, 'o-', linewidth=2, markersize=8, 
                           color='#3498db', label='Adversarial Accuracy')
            ax1.set_xlabel('Epsilon (ε × 255)', fontsize=11)
            ax1.set_ylabel('Adversarial Accuracy (%)', fontsize=11, color='#3498db')
            ax1.tick_params(axis='y', labelcolor='#3498db')
            ax1.grid(True, alpha=0.3)
            
            # Add second y-axis for attack success rate
            ax1_twin = ax1.twinx()
            line2 = ax1_twin.plot(epsilons, asrs, 's--', linewidth=2, markersize=8,
                                 color='#e74c3c', label='Attack Success Rate (ASR)')
            ax1_twin.set_ylabel('Attack Success Rate (%)', fontsize=11, color='#e74c3c')
            ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
            ax1_twin.set_ylim([0, 100])
            
            # Add value labels
            for i, (eps, acc, asr) in enumerate(zip(epsilons, adv_accs, asrs)):
                ax1.annotate(f'{acc:.1f}%', (eps, acc), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9, color='#3498db')
                ax1_twin.annotate(f'{asr:.1f}%', (eps, asr), textcoords="offset points",
                                xytext=(0,-15), ha='center', fontsize=9, color='#e74c3c')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=9)
            
            ax1.set_title('Effect of Perturbation Budget (ε)', fontsize=12, fontweight='bold')
            ax1.set_ylim([0, max(adv_accs) * 1.2 if max(adv_accs) > 0 else 5])
    
    # Plot 2: Iteration sensitivity
    if 'iterations' in data and data['iterations']:
        ax2 = axes[1]
        iterations = []
        adv_accs = []
        asrs = []
        
        for n_iter, result in sorted(data['iterations'].items()):
            if result is not None:
                iterations.append(int(n_iter))
                adv_accs.append(result.get('adversarial_accuracy', 0))
                asrs.append(result.get('attack_success_rate', 0))
        
        if iterations:
            # Plot adversarial accuracy
            line1 = ax2.plot(iterations, adv_accs, 'o-', linewidth=2, markersize=8,
                            color='#9b59b6', label='Adversarial Accuracy')
            ax2.set_xlabel('Number of Iterations', fontsize=11)
            ax2.set_ylabel('Adversarial Accuracy (%)', fontsize=11, color='#9b59b6')
            ax2.tick_params(axis='y', labelcolor='#9b59b6')
            ax2.grid(True, alpha=0.3)
            
            # Add second y-axis for attack success rate
            ax2_twin = ax2.twinx()
            line2 = ax2_twin.plot(iterations, asrs, 's--', linewidth=2, markersize=8,
                                color='#e74c3c', label='Attack Success Rate (ASR)')
            ax2_twin.set_ylabel('Attack Success Rate (%)', fontsize=11, color='#e74c3c')
            ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
            ax2_twin.set_ylim([0, 100])
            
            # Add value labels
            for i, (n_iter, acc, asr) in enumerate(zip(iterations, adv_accs, asrs)):
                ax2.annotate(f'{acc:.1f}%', (n_iter, acc), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9, color='#9b59b6')
                ax2_twin.annotate(f'{asr:.1f}%', (n_iter, asr), textcoords="offset points",
                                xytext=(0,-15), ha='center', fontsize=9, color='#e74c3c')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right', fontsize=9)
            
            ax2.set_title('Effect of Number of Iterations', fontsize=12, fontweight='bold')
            ax2.set_ylim([0, max(adv_accs) * 1.2 if max(adv_accs) > 0 else 5])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Parameter sensitivity plot saved to {save_path}")


def plot_perturbation_statistics(samples_path='results/adversarial_samples.pth',
                                 save_path='results/perturbation_statistics.png'):
    """
    Figure 3: Perturbation statistics
    Shows distribution of L∞ norms and perturbation magnitudes
    """
    if not os.path.exists(samples_path):
        print(f"Warning: {samples_path} not found, skipping perturbation statistics")
        return
    
    data = torch.load(samples_path, map_location='cpu')
    x_original = data['x_original']
    x_adversarial = data['x_adversarial']
    
    # Calculate L∞ norms
    perturbations = x_adversarial - x_original
    linf_norms = torch.abs(perturbations).max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0].numpy()
    
    # Calculate L2 norms
    l2_norms = torch.norm(perturbations.view(perturbations.shape[0], -1), p=2, dim=1).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: L∞ norm distribution
    ax1 = axes[0]
    ax1.hist(linf_norms, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.axvline(linf_norms.mean(), color='black', linestyle='--', linewidth=2, 
                label=f'Mean: {linf_norms.mean():.4f}')
    ax1.set_xlabel('L∞ Norm', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of L∞ Norms', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: L2 norm distribution
    ax2 = axes[1]
    ax2.hist(l2_norms, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(l2_norms.mean(), color='black', linestyle='--', linewidth=2,
                label=f'Mean: {l2_norms.mean():.4f}')
    ax2.set_xlabel('L2 Norm', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of L2 Norms', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Perturbation statistics saved to {save_path}")


def plot_class_wise_performance(samples_path='results/adversarial_samples.pth',
                                model_path=None, save_path='results/class_wise_performance.png'):
    """
    Figure 4: Class-wise performance analysis
    Shows clean and adversarial accuracy for each CIFAR-10 class
    """
    if not os.path.exists(samples_path):
        print(f"Warning: {samples_path} not found, skipping class-wise performance")
        return
    
    data = torch.load(samples_path, map_location='cpu')
    x_original = data['x_original']
    x_adversarial = data['x_adversarial']
    y_true = data['y_true']
    
    # Get predictions if available
    if 'y_pred_clean' in data and 'y_pred_adv' in data:
        y_pred_clean = data['y_pred_clean']
        y_pred_adv = data['y_pred_adv']
    else:
        if model_path is None:
            print("Warning: Model path required for class-wise analysis, skipping")
            return
        # Load model and get predictions
        from resnet import ResNet18
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet18(num_classes=10).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        with torch.no_grad():
            x_orig_gpu = x_original.to(device)
            x_adv_gpu = x_adversarial.to(device)
            y_pred_clean = model(x_orig_gpu).argmax(1).cpu()
            y_pred_adv = model(x_adv_gpu).argmax(1).cpu()
    
    # Convert to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.numpy()
    if torch.is_tensor(y_pred_clean):
        y_pred_clean = y_pred_clean.numpy()
    if torch.is_tensor(y_pred_adv):
        y_pred_adv = y_pred_adv.numpy()
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Calculate accuracy per class
    clean_accs = []
    adv_accs = []
    
    for class_idx in range(10):
        mask = y_true == class_idx
        if mask.sum() > 0:
            clean_acc = (y_pred_clean[mask] == y_true[mask]).mean() * 100
            adv_acc = (y_pred_adv[mask] == y_true[mask]).mean() * 100
        else:
            clean_acc = 0
            adv_acc = 0
        clean_accs.append(clean_acc)
        adv_accs.append(adv_acc)
    
    # Plot
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', 
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy',
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Class-wise Performance: Clean vs Adversarial Accuracy', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class-wise performance saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate all visualizations for report')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model (for class-wise analysis)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Generating All Visualizations for Report")
    print("="*60)
    print()
    
    results_dir = args.results_dir
    
    # Figure 1: Performance comparison
    print("1. Generating performance comparison chart...")
    plot_performance_comparison(
        os.path.join(results_dir, 'evaluation_results.json'),
        os.path.join(results_dir, 'performance_comparison.png')
    )
    
    # Figure 2: Parameter sensitivity
    print("\n2. Generating parameter sensitivity plots...")
    plot_parameter_sensitivity(
        os.path.join(results_dir, 'parameter_sensitivity.json'),
        os.path.join(results_dir, 'parameter_sensitivity_plot.png')
    )
    
    # Figure 3: Perturbation statistics
    print("\n3. Generating perturbation statistics...")
    plot_perturbation_statistics(
        os.path.join(results_dir, 'adversarial_samples.pth'),
        os.path.join(results_dir, 'perturbation_statistics.png')
    )
    
    # Figure 4: Class-wise performance
    print("\n4. Generating class-wise performance analysis...")
    if args.model_path is None:
        # Try to find model in default location
        model_path = './models/best_model.pth'
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping class-wise analysis")
        else:
            plot_class_wise_performance(
                os.path.join(results_dir, 'adversarial_samples.pth'),
                model_path,
                os.path.join(results_dir, 'class_wise_performance.png')
            )
    else:
        plot_class_wise_performance(
            os.path.join(results_dir, 'adversarial_samples.pth'),
            args.model_path,
            os.path.join(results_dir, 'class_wise_performance.png')
        )
    
    print("\n" + "="*60)
    print("All visualizations generated!")
    print("="*60)
    print("\nGenerated figures:")
    print("  1. performance_comparison.png - Clean vs Adversarial accuracy comparison")
    print("  2. parameter_sensitivity_plot.png - Effect of ε and iterations")
    print("  3. perturbation_statistics.png - Distribution of perturbation norms")
    print("  4. class_wise_performance.png - Per-class accuracy analysis")
    print("  5. training_curves.png - Training history (from train.py)")
    print("  6. adversarial_examples.png - Example visualizations (from visualize.py)")
    print("\nNote: Some figures may be skipped if required data files are not found.")
    print("="*60)


if __name__ == '__main__':
    main()

