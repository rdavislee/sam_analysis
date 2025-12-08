"""
Main experiment runner.
"""
import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Add parent directory to path to import sam
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam.sam import SAM
from src.data_utils import get_cifar10_loaders
from src.sharpness import estimate_sharpness
from sam.example.utility.bypass_bn import enable_running_stats, disable_running_stats

# === CONFIGURATION ===
CONFIG = {
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'sam_rho': 0.05,
    'seed': 42,
}

DATA_FRACTIONS = [0.01, 0.1, 1.0]
NOISE_FRACTIONS = [0.0, 0.2, 0.4]
OPTIMIZERS = ['sgd', 'sam']


def get_model(num_classes=10):
    """Get ResNet-18 model."""
    # Use weights=None explicitly as pretrained=False is deprecated
    model = models.resnet18(weights=None, num_classes=num_classes)
    return model


def train_epoch(model, loader, optimizer, criterion, device, use_sam=False, scaler=None):
    """
    Train for one epoch.
    
    IMPORTANT for SAM: Must handle batch norm correctly.
    See: https://github.com/davda54/sam#training-tips
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_sam:
            # First forward-backward pass
            enable_running_stats(model)
            # Disable AMP for SAM to avoid scaler complications with custom steps
            # or handle it manually if absolutely necessary. For stability, we use FP32 for SAM.
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass
            disable_running_stats(model)
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
            
        else:
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model accuracy and loss."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100. * correct / total


def run_single_experiment(
    optimizer_type: str,
    data_fraction: float,
    noise_fraction: float,
    save_dir: str = 'results'
):
    """
    Run a single training experiment.
    
    Args:
        optimizer_type: 'sgd' or 'sam'
        data_fraction: Fraction of training data (0.01, 0.1, 1.0)
        noise_fraction: Label noise fraction (0.0, 0.2, 0.4)
        save_dir: Where to save results
    
    Returns:
        dict with all metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_type.upper()} | Data: {data_fraction*100:.0f}% | Noise: {noise_fraction*100:.0f}%")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Get data
    # Note: data_dir is relative to where script is run, assuming root of repo
    train_loader, test_loader, num_train = get_cifar10_loaders(
        data_dir='./data',
        batch_size=CONFIG['batch_size'],
        train_fraction=data_fraction,
        noise_fraction=noise_fraction,
        seed=CONFIG['seed']
    )
    print(f"Training samples: {num_train}")
    
    # Get model
    model = get_model().to(device)
    # Enable cudnn benchmark for speed
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Use Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    use_sam = (optimizer_type == 'sam')
    if use_sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=CONFIG['sam_rho'],
            lr=CONFIG['lr'],
            momentum=CONFIG['momentum'],
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=CONFIG['epochs'])
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CONFIG['lr'],
            momentum=CONFIG['momentum'],
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training loop
    results = {
        'config': {
            'optimizer': optimizer_type,
            'data_fraction': data_fraction,
            'noise_fraction': noise_fraction,
            'num_train_samples': num_train,
            **CONFIG
        },
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    start_time = time.time()
    best_test_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, use_sam, scaler
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
        best_test_acc = max(best_test_acc, test_acc)
        
        if epoch % 10 == 0 or epoch == CONFIG['epochs'] - 1:
            print(f"Epoch {epoch:3d} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Gap: {train_acc - test_acc:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.1f} minutes")
    
    # Measure final sharpness
    print("Measuring sharpness...")
    sharpness_metrics = estimate_sharpness(model, train_loader, criterion, device)
    results['sharpness'] = sharpness_metrics
    
    # Summary metrics
    results['summary'] = {
        'best_test_acc': best_test_acc,
        'final_test_acc': results['test_acc'][-1],
        'final_train_acc': results['train_acc'][-1],
        'generalization_gap': results['train_acc'][-1] - results['test_acc'][-1],
        'training_time_minutes': elapsed / 60,
    }
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{optimizer_type}_data{data_fraction}_noise{noise_fraction}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")
    
    return results


def run_all_experiments():
    """Run the full experiment matrix."""
    all_results = []
    
    total_runs = len(OPTIMIZERS) * len(DATA_FRACTIONS) * len(NOISE_FRACTIONS)
    current_run = 0
    
    for opt in OPTIMIZERS:
        for data_frac in DATA_FRACTIONS:
            for noise_frac in NOISE_FRACTIONS:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}]")
                result = run_single_experiment(opt, data_frac, noise_frac)
                all_results.append(result)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'sam'])
    parser.add_argument('--data_fraction', type=float)
    parser.add_argument('--noise_fraction', type=float)
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    elif args.optimizer and args.data_fraction is not None and args.noise_fraction is not None:
        run_single_experiment(args.optimizer, args.data_fraction, args.noise_fraction)
    else:
        print("Usage:")
        print("  python src/run_experiments.py --all")
        print("  python src/run_experiments.py --optimizer sam --data_fraction 0.1 --noise_fraction 0.2")

