import torch
import numpy as np
from .difflevelset import softplus_regularized_heaviside

def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    return intersection / (union + 1e-6)

def compute_connectivity_error(pred, target):
    """
    Compute Connectivity Error for crack segmentation.
    CE = |beta_0_pred - beta_0_gt| / beta_0_gt * 100%
    """
    # Simplified: count connected components
    # For production: use scipy.ndimage.label or Gudhi
    from scipy.ndimage import label
    
    pred_binary = (pred > 0.5).cpu().numpy().astype(np.uint8)
    target_binary = (target > 0.5).cpu().numpy().astype(np.uint8)
    
    pred_components, _ = label(pred_binary[0, 0])
    target_components, _ = label(target_binary[0, 0])
    
    beta_0_pred = pred_components.max()
    beta_0_gt = target_components.max()
    
    if beta_0_gt == 0:
        return 0.0
    
    return abs(beta_0_pred - beta_0_gt) / beta_0_gt * 100

def evaluate(model, dataloader, device, epsilon=1.0, alpha=0.1):
    """
    Evaluate model on validation/test set.
    
    Returns:
        dict with mIoU, BF1, Connectivity Error, etc.
    """
    model.eval()
    
    ious = []
    bf1_scores = []
    connectivity_errors = []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            
            # Forward pass
            phi, _ = model(images)
            
            # Apply SRH to get segmentation
            pred_mask = softplus_regularized_heaviside(phi, epsilon, alpha)
            
            # Compute metrics
            for i in range(len(images)):
                iou = compute_iou(pred_mask[i], targets[i])
                ious.append(iou.item())
                
                # Boundary F1 (simplified)
                # For production: use official BF1 implementation
                bf1 = iou  # Placeholder
                
                # Connectivity Error (for crack datasets)
                ce = compute_connectivity_error(pred_mask[i], targets[i])
                connectivity_errors.append(ce)
    
    return {
        'mIoU': np.mean(ious),
        'BF1': np.mean(bf1_scores) if bf1_scores else 0,
        'ConnectivityError': np.mean(connectivity_errors) if connectivity_errors else 0,
        'val_loss': 1 - np.mean(ious)  # Proxy for validation loss
    }