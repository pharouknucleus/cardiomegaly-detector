import torch
import torch.nn as nn
from torchvision import models


class CardiomegalyDenseNet(nn.Module):
    """
    DenseNet-121 backbone with a single logit output for cardiomegaly.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = base.classifier.in_features
        base.classifier = nn.Linear(num_features, 1)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # (B, 1) -> (B,)
