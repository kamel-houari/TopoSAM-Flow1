import torch
import torch.nn as nn
import torch.nn.functional as F
from .difflevelset import softplus_regularized_heaviside

class WeakSupervisionLoss(nn.Module):
    """
    Weak supervision loss using pseudo-masks from SAM.
    Eq. (5) from the paper.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, phi, pseudo_mask, confidence=None, epsilon=1.0, alpha=0.1):
        """
        Args:
            phi: Signed Distance Function [B, 1, H, W]
            pseudo_mask: Pseudo mask from SAM [B, 1, H, W]
            confidence: Confidence map W_conf [B, 1, H, W] or None
        """
        # Apply SRH to get predicted mask
        pred_mask = softplus_regularized_heaviside(phi, epsilon, alpha)
        
        # Weight pseudo-mask by confidence if provided
        if confidence is not None:
            pseudo_mask = confidence * pseudo_mask
        
        # Binary cross-entropy
        eps = 1e-7
        loss = -pseudo_mask * torch.log(pred_mask + eps) - \
               (1 - pseudo_mask) * torch.log(1 - pred_mask + eps)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class VariationalLoss(nn.Module):
    """
    Variational loss based on Chan-Vese functional.
    Eq. (6) from the paper.
    """
    def __init__(self, mu=0.1, lambda_=1.0, epsilon=1.0):
        super().__init__()
        self.mu = mu  # Length regularization weight
        self.lambda_ = lambda_  # Data fidelity weight
        self.epsilon = epsilon
        
    def forward(self, phi, image):
        """
        Args:
            phi: Signed Distance Function [B, 1, H, W]
            image: Input image [B, C, H, W]
        """
        # Apply SRH
        H_phi = softplus_regularized_heaviside(phi, self.epsilon)
        
        # Compute c_in and c_out (mean intensities)
        # Differentiable estimation
        H_sum = H_phi.sum(dim=[2,3], keepdim=True) + 1e-6
        c_in = (image * H_phi).sum(dim=[2,3], keepdim=True) / H_sum
        c_out = (image * (1 - H_phi)).sum(dim=[2,3], keepdim=True) / \
                ((1 - H_phi).sum(dim=[2,3], keepdim=True) + 1e-6)
        
        # Length term (contour regularization)
        grad_H = torch.sqrt(
            torch.sum(torch.gradient(H_phi, dim=[2,3])**2, dim=1, keepdim=True) + 1e-6
        )
        length_term = grad_H.mean()
        
        # Data fidelity term
        data_term = ((image - c_in)**2 * H_phi + (image - c_out)**2 * (1 - H_phi)).mean()
        
        return self.mu * length_term + self.lambda_ * data_term

class TopologicalLoss(nn.Module):
    """
    Topological loss based on persistent homology.
    Penalizes multiple connected components (beta_0 > 1).
    Simplified implementation - for full version, use Gudhi library.
    """
    def __init__(self, threshold=0.5, weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        
    def forward(self, phi):
        """
        Simplified topological loss.
        For production: integrate with Gudhi for persistent homology.
        """
        # Placeholder: encourage smooth, connected predictions
        # In practice: compute persistence diagram and penalize short-lived components
        
        # Simple proxy: penalize high-frequency variations in SDF
        grad_phi = torch.sqrt(
            torch.sum(torch.gradient(phi, dim=[2,3])**2, dim=1, keepdim=True) + 1e-6
        )
        
        # Penalize excessive variation (proxy for topological complexity)
        topo_loss = grad_phi.mean()
        
        return self.weight * topo_loss