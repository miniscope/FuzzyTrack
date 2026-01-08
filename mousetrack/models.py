"""Neural network models for mouse tracking."""
import torch
import torch.nn as nn
import torchvision.models as models
from .config import HEATMAP_SIZE


class MouseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.coord_head = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.coord_head(features))


class MouseHeatmapCNN(nn.Module):
    """CNN that predicts a heatmap instead of direct coordinates."""
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        # Use ResNet18 but remove final pooling to get spatial features
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove avgpool and fc, keep only conv layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
        
        # Upsample from ResNet18's final feature map (7x7) to heatmap size (56x56)
        # ResNet18 outputs (batch, 512, 7, 7) after removing pooling
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
        )

    def forward(self, x):
        features = self.backbone(x)  # (batch, 512, 7, 7)
        heatmap = self.heatmap_head(features)  # (batch, 1, 56, 56)
        # Apply softmax across spatial dimensions to get probability distribution
        batch_size = heatmap.shape[0]
        heatmap_flat = heatmap.view(batch_size, -1)
        heatmap_flat = torch.softmax(heatmap_flat, dim=1)
        heatmap = heatmap_flat.view(batch_size, 1, HEATMAP_SIZE[0], HEATMAP_SIZE[1])
        return heatmap


class CoordinateLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(lstm_out))
