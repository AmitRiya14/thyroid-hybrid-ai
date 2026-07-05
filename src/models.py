from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ImageOnlyEfficientNet(nn.Module):
    def __init__(self, pretrained: bool = True, backend: str = "torchvision"):
        super().__init__()
        if backend != "torchvision":
            raise ValueError("ImageOnlyEfficientNet currently supports backend='torchvision'.")
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    @property
    def gradcam_target_layer(self):
        return self.backbone.features[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).squeeze(1)


class HybridEfficientNet(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        pretrained: bool = True,
        tabular_hidden: tuple[int, int] = (32, 16),
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.image_model = models.efficientnet_b0(weights=weights)
        image_feature_dim = self.image_model.classifier[1].in_features
        self.image_model.classifier = nn.Identity()

        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tabular_hidden[0], tabular_hidden[1]),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + tabular_hidden[1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @property
    def gradcam_target_layer(self):
        return self.image_model.features[-1]

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_features = self.image_model(image)
        tabular_features = self.tabular_branch(tabular)
        combined = torch.cat([image_features, tabular_features], dim=1)
        return self.classifier(combined).squeeze(1)


def get_model(model_name: str, tabular_dim: int | None = None, pretrained: bool = True) -> nn.Module:
    if model_name == "image_only":
        return ImageOnlyEfficientNet(pretrained=pretrained)
    if model_name == "hybrid":
        if tabular_dim is None:
            raise ValueError("tabular_dim is required for hybrid models.")
        return HybridEfficientNet(tabular_dim=tabular_dim, pretrained=pretrained)
    raise ValueError(f"Unknown model_name: {model_name}")
