import torch
import torch.nn as nn
from torchvision import models

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation attention block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MobileNetV3Attention(nn.Module):
    """MobileNetV3 backbone with attention blocks."""
    
    def __init__(self, pretrained=True, num_attention_blocks=2):
        super().__init__()
        
        # Load MobileNetV3 large
        mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Feature extractor (remove classifier)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:-1])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            SqueezeExcitation(960) for _ in range(num_attention_blocks)
        ])
        
        self.out_channels = 960
        
    def forward(self, x):
        x = self.features(x)
        for attn in self.attention_blocks:
            x = attn(x)
        return x