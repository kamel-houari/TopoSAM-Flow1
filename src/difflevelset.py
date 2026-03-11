import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus_regularized_heaviside(phi, epsilon=1.0, alpha=0.1):
    """
    Softplus-Regularized Heaviside (SRH) approximation.
    Eq. (2) from the paper.
    """
    u = phi / epsilon + alpha * F.softplus(-torch.abs(phi) / epsilon)
    return 0.5 + 0.5 * torch.tanh(u)

class DiLevelSetHead(nn.Module):
    """
    DiLevelSet head: predicts Signed Distance Function (SDF).
    
    The SDF is used with SRH approximation for differentiable
    variational segmentation.
    """
    
    def __init__(self, in_channels, num_classes=1, 
                 hidden_channels=64, epsilon=1.0, alpha=0.1):
        super().__init__()
        
        self.epsilon = epsilon
        self.alpha = alpha
        
        # Convolutional layers to predict SDF
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, num_classes, 1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, features):
        """
        Predict Signed Distance Function from backbone features.
        
        Args:
            features: [B, C, H, W] from backbone
            
        Returns:
            phi: [B, 1, H, W] Signed Distance Function
        """
        x = self.relu(self.conv1(features))
        x = self.relu(self.conv2(x))
        phi = self.conv3(x)
        return phi
    
    def apply_srh(self, phi):
        """Apply SRH approximation to get segmentation mask."""
        return softplus_regularized_heaviside(phi, self.epsilon, self.alpha)