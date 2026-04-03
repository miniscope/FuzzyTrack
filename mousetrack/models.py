"""Neural network models for FuzzyTrack."""
import torch
import torch.nn as nn
import torchvision.models as models
from .config import HEATMAP_SIZE

# Backbone configurations: (model_func, weights, feature_dim)
BACKBONES = {
    'resnet18': (models.resnet18, 'ResNet18_Weights', 512),
    'resnet50': (models.resnet50, 'ResNet50_Weights', 2048),
}


def get_backbone(name: str):
    """Get backbone model, weights, and feature dimension."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Choose from: {list(BACKBONES.keys())}")
    model_func, weights_name, feature_dim = BACKBONES[name]
    weights = getattr(models, weights_name).IMAGENET1K_V1
    return model_func, weights, feature_dim


class MouseCNN(nn.Module):
    def __init__(self, backbone: str = 'resnet18'):
        super().__init__()
        model_func, weights, feature_dim = get_backbone(backbone)
        self.backbone = model_func(weights=weights)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.coord_head = nn.Linear(feature_dim, 2)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.coord_head(features))


class MouseHeatmapCNN(nn.Module):
    """CNN that predicts a heatmap instead of direct coordinates."""
    def __init__(self, backbone: str = 'resnet18'):
        super().__init__()
        model_func, weights, feature_dim = get_backbone(backbone)
        # Use backbone but remove final pooling to get spatial features
        bb = model_func(weights=weights)
        bb.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove avgpool and fc, keep only conv layers
        self.backbone = nn.Sequential(*list(bb.children())[:-2])  # Remove avgpool and fc

        # Upsample from backbone's final feature map (7x7) to heatmap size (56x56)
        # Added BatchNorm and Dropout for regularization
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
        )

        # Initialize weights
        self._init_heatmap_head()

    def _init_heatmap_head(self):
        """Initialize heatmap head with proper weights."""
        for m in self.heatmap_head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)  # (batch, feature_dim, 7, 7)
        heatmap = self.heatmap_head(features)  # (batch, 1, 56, 56)
        # Return raw heatmap (no softmax normalization)
        return heatmap
