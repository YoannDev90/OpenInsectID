"""A single taxonomic expert: MobileNetV4-style backbone + linear classifier.

Each node of the cascade instantiates its own expert (one per order at family
level, one per family at genus level, ...), so every classifier only
discriminates between sibling taxa.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import build_backbone


class Expert(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mnv4_small",
        width_mult: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"An expert needs >= 2 classes, got {num_classes}")
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone_name, width_mult=width_mult)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.backbone(x)))
