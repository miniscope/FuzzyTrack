"""Neural network models for mouse tracking."""
import torch
import torch.nn as nn
import torchvision.models as models


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
