"""Training utilities: hierarchical losses, distillation, data, trainer."""

from .losses import HierarchicalCrossEntropy, FocalLoss, CombinedTaxonomicLoss
from .distillation import DistillationLoss
from .dataset import InsectDataset, build_dataloaders
from .augment import train_transform, eval_transform, mixup_cutmix_batch
from .trainer import Trainer, TrainConfig

__all__ = [
    "HierarchicalCrossEntropy",
    "FocalLoss",
    "CombinedTaxonomicLoss",
    "DistillationLoss",
    "InsectDataset",
    "build_dataloaders",
    "train_transform",
    "eval_transform",
    "mixup_cutmix_batch",
    "Trainer",
    "TrainConfig",
]
