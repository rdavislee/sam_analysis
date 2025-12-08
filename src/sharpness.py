"""
Sharpness measurement utilities.
"""
import torch
import numpy as np

def estimate_sharpness(
    model,
    dataloader,
    criterion,
    device,
    rho: float = 0.05,
    n_samples: int = 3
) -> dict:
    """
    Estimate loss landscape sharpness via random perturbation.
    
    Sharpness = E[L(w + ε) - L(w)] where ||ε|| ≈ rho * ||w||
    
    Args:
        model: Neural network
        dataloader: Data to evaluate on (use subset for speed)
        criterion: Loss function
        device: torch device
        rho: Relative perturbation size
        n_samples: Number of random perturbations to average
    
    Returns:
        dict with 'base_loss', 'perturbed_loss', 'sharpness'
    """
    model.eval()
    
    # 1. Compute base loss L(w)
    base_loss = compute_loss(model, dataloader, criterion, device)
    
    perturbed_losses = []
    
    # Save original params
    original_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()
            
    # 2. Random perturbations
    for _ in range(n_samples):
        # Add perturbation
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Random direction
                noise = torch.randn_like(param.data)
                
                # Normalize to length 1
                norm = noise.norm(p=2)
                if norm > 1e-12:
                    noise = noise / norm
                
                # Scale by rho * ||w||
                w_norm = param.data.norm(p=2)
                epsilon = rho * w_norm * noise
                
                param.data = original_params[name] + epsilon
        
        # Compute perturbed loss
        loss = compute_loss(model, dataloader, criterion, device)
        perturbed_losses.append(loss)
        
        # Restore original params
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = original_params[name]
                
    avg_perturbed_loss = np.mean(perturbed_losses)
    sharpness = avg_perturbed_loss - base_loss
    
    return {
        'base_loss': base_loss,
        'perturbed_loss': avg_perturbed_loss,
        'sharpness': sharpness
    }


@torch.no_grad()
def compute_loss(model, dataloader, criterion, device, max_batches: int = 10):
    """Compute average loss over dataloader (or first max_batches)."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= max_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
    return total_loss / total_samples

