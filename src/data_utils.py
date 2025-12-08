"""
Data loading utilities with subsampling and label noise injection.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    train_fraction: float = 1.0,
    noise_fraction: float = 0.0,
    seed: int = 42,
    num_workers: int = 2
):
    """
    Get CIFAR-10 data loaders with optional subsampling and label noise.
    
    Args:
        data_dir: Where to store/load CIFAR-10
        batch_size: Batch size
        train_fraction: Fraction of training data to use (0.01, 0.1, or 1.0)
        noise_fraction: Fraction of labels to randomly flip (0.0, 0.2, or 0.4)
        seed: Random seed for reproducibility
        num_workers: DataLoader workers
    
    Returns:
        train_loader, test_loader, num_train_samples
    """
    # 1. Define transforms (standard CIFAR-10 augmentation)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 2. Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # 3. Subsample if train_fraction < 1.0 (use stratified sampling)
    targets = np.array(trainset.targets)
    
    if train_fraction < 1.0:
        indices = stratified_subsample_indices(targets, train_fraction, seed)
        trainset = Subset(trainset, indices)
        # For noise injection, we need access to the underlying targets of the subset.
        current_targets = targets[indices]
    else:
        indices = np.arange(len(targets))
        current_targets = targets

    # 4. Inject label noise if noise_fraction > 0
    if noise_fraction > 0:
        noisy_targets = inject_label_noise(current_targets, noise_fraction, num_classes=10, seed=seed)
        
        # Custom wrapper to override targets
        class NoisyDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, new_targets):
                self.dataset = dataset
                self.new_targets = new_targets
                
            def __getitem__(self, index):
                img, _ = self.dataset[index]
                target = self.new_targets[index]
                return img, target
                
            def __len__(self):
                return len(self.dataset)
        
        trainset = NoisyDataset(trainset, noisy_targets)

    # 5. Return DataLoaders
    # Optimize DataLoader with pin_memory=True for faster GPU transfer
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, len(trainset)


def inject_label_noise(targets: list, noise_fraction: float, num_classes: int = 10, seed: int = 42):
    """
    Randomly flip a fraction of labels to different classes.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    n_samples = len(targets)
    n_noise = int(noise_fraction * n_samples)
    
    if n_noise == 0:
        return targets.tolist()
        
    # Choose indices to corrupt
    noise_indices = rng.choice(n_samples, size=n_noise, replace=False)
    
    # Create new random labels
    new_targets = targets.copy()
    
    for idx in noise_indices:
        original_label = targets[idx]
        possible_labels = list(range(num_classes))
        possible_labels.remove(original_label)
        new_targets[idx] = rng.choice(possible_labels)
        
    return new_targets.tolist()


def stratified_subsample_indices(targets: list, fraction: float, seed: int = 42):
    """
    Get indices for stratified subsampling (maintains class balance).
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    indices = []
    
    unique_classes = np.unique(targets)
    
    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        n_cls = len(cls_indices)
        n_keep = max(1, int(fraction * n_cls))
        
        keep_indices = rng.choice(cls_indices, size=n_keep, replace=False)
        indices.extend(keep_indices)
        
    return np.sort(indices)

