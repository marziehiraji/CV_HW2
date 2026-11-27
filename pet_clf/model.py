"""Model construction for the three transfer-learning approaches."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from . import config


def build_mobilenet(num_classes: int = config.NUM_CLASSES) -> models.MobileNetV3Small:
    """Return a MobileNetV3-Small model with IMAGENET1K_V1 weights and a
    37-output classifier head.

    The default classifier head for MobileNetV3-Small is::

        Sequential(
            Linear(in_features=576, out_features=1024),
            Hardswish(),
            Dropout(0.2),
            Linear(1024, 1000),
        )

    We replace only the final Linear (``classifier[3]``) so the head keeps
    its 576 -> 1024 -> 37 structure.  The 1024-unit hidden layer therefore
    remains trainable in both ``full`` and ``head`` modes.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_trainable_full(model: nn.Module) -> nn.Module:
    """Approach 1: train the whole network (conv + head)."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def set_trainable_head_only(model: nn.Module) -> nn.Module:
    """Approach 2: freeze the conv feature extractor and the pooling layer;
    train only the classifier head (the 576->1024 Linear, Hardswish,
    Dropout, and the new 1024->37 Linear).
    """
    # Freeze features (conv) and the avgpool.
    for p in model.features.parameters():
        p.requires_grad = False
    # The pooling layer has no parameters; nothing to freeze there.
    # Trainable: classifier (Linear(576,1024), Hardswish, Dropout, Linear(1024,37))
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def build_feature_extractor() -> tuple[nn.Module, int]:
    """Return a model that outputs the pooled 576-d feature vector used
    by approach 3 (feature extraction + sklearn classifier).
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    # Drop the classifier; keep features -> avgpool -> flatten.
    feats = model.features
    pool = model.avgpool
    flatten = nn.Flatten()
    extractor = nn.Sequential(feats, pool, flatten)
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()
    return extractor, 576  # MobileNetV3-Small feature dimension
