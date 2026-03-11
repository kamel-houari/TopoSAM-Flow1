import torch
import torch.nn as nn
from .backbone import MobileNetV3Attention
from .difflevelset import DiLevelSetHead
from .sam_guidance import SAMGuidanceModule

class TopoSAMFlow(nn.Module):
    """
    TopoSAM-Flow: Main architecture for weakly-supervised industrial segmentation.
    
    Combines:
    - MobileNetV3 + Attention backbone for feature extraction
    - SAM guidance module for pseudo-mask generation
    - DiLevelSet head for variational segmentation with SDF output
    """
    
    def __init__(self, num_classes=1, backbone_pretrained=True, 
                 epsilon=1.0, alpha=0.1, lambda_var=0.5, lambda_topo=0.3):
        super(TopoSAMFlow, self).__init__()
        
        # Backbone: MobileNetV3 + Attention blocks
        self.backbone = MobileNetV3Attention(pretrained=backbone_pretrained)
        
        # SAM Guidance module (frozen during training)
        self.sam_guidance = SAMGuidanceModule()
        
        # DiLevelSet head: predicts Signed Distance Function
        self.head = DiLevelSetHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            epsilon=epsilon,
            alpha=alpha
        )
        
        # Loss weights
        self.lambda_var = lambda_var
        self.lambda_topo = lambda_topo
        
    def forward(self, x, boxes=None, sam_model=None):
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, C, H, W]
            boxes: Bounding boxes for weak supervision [B, 4] or None
            sam_model: Pre-loaded SAM model for pseudo-mask generation
            
        Returns:
            phi: Signed Distance Function prediction [B, 1, H, W]
            aux: Dictionary with auxiliary outputs for loss computation
        """
        # Extract features
        features = self.backbone(x)
        
        # SAM guidance (training only, frozen)
        if self.training and boxes is not None and sam_model is not None:
            pseudo_mask, confidence = self.sam_guidance(x, boxes, sam_model)
        else:
            pseudo_mask, confidence = None, None
        
        # Predict SDF
        phi = self.head(features)
        
        aux = {
            'pseudo_mask': pseudo_mask,
            'confidence': confidence,
            'features': features
        }
        
        return phi, aux
    
    def get_loss(self, phi, aux, image, ground_truth=None, boxes=None):
        """
        Compute total loss: L_weak + lambda_var * L_var + lambda_topo * L_topo
        """
        from .losses import WeakSupervisionLoss, VariationalLoss, TopologicalLoss
        
        losses = {}
        
        # Weak supervision loss
        if aux['pseudo_mask'] is not None:
            losses['weak'] = WeakSupervisionLoss()(
                phi, aux['pseudo_mask'], aux['confidence']
            )
        
        # Variational loss (Chan-Vese inspired)
        losses['var'] = VariationalLoss(epsilon=self.head.epsilon)(
            phi, image
        )
        
        # Topological loss (persistent homology)
        if self.training:  # Only during training
            losses['topo'] = TopologicalLoss()(phi)
        
        # Total loss
        total = losses.get('weak', 0) + \
                self.lambda_var * losses.get('var', 0) + \
                self.lambda_topo * losses.get('topo', 0)
        
        return total, losses