from __future__ import annotations

import torch
import torchvision.models as models
from torch import nn


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = model_name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        "Unsupported model_name. Use one of: resnet18, resnet50, mobilenet_v3_small, efficientnet_b4"
    )


def count_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for p in model.parameters():
        param_size += p.nelement() * p.element_size()
    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.nelement() * b.element_size()
    return (param_size + buffer_size) / 1024 / 1024
