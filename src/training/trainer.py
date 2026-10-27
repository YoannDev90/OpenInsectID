"""Minimal hierarchical trainer for the Vision MoE.

Designed to be driven from a Kaggle notebook: build the tree from the manifest,
construct the model + loss, and call ``Trainer.fit``. Supports:
- per-level hierarchical loss (CE for order/family, Focal for genus/species)
  plus the taxonomic cost-matrix penalty,
- optional Mixup/CutMix,
- optional BioCLIP knowledge distillation (teacher must be supplied),
- gradient clipping, cosine LR, AMP on GPU,
- accuracy at each level and end-to-end path accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.moe import HierarchicalMoE
from .augment import mixup_cutmix_batch
from .losses import CombinedTaxonomicLoss


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixup_alpha: float = 0.4  # 0 disables
    mixup_prob: float = 0.5
    level_weights: tuple = (8.0, 4.0, 2.0, 1.0)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    lambda_cost: float = 1.0
    use_amp: bool = True
    log_every: int = 50
    device: str = "cuda"


@dataclass
class EpochStats:
    loss: float = 0.0
    acc_order: float = 0.0
    acc_family: float = 0.0
    acc_genus: float = 0.0
    acc_species: float = 0.0
    seconds: float = 0.0
    terms: dict = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        model: HierarchicalMoE,
        config: TrainConfig,
        teacher: nn.Module | None = None,
    ):
        self.model = model
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)
        self.teacher = teacher
        if teacher is not None:
            teacher.to(self.device).eval()
        # species_paths for the cost matrix, aligned with the model label space.
        from .losses import TaxonomicCostMatrix

        self.cost_matrix = TaxonomicCostMatrix(
            torch.zeros(model.n_species, 4, dtype=torch.long)
        ).build_from_tree(model)
        self.loss_fn = CombinedTaxonomicLoss(
            level_weights=config.level_weights,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            lambda_cost=config.lambda_cost,
        )
        # Both loss paths share ONE matrix: the mixup branch calls
        # self.cost_matrix directly, the plain branch goes through loss_fn
        # (whose own matrix would otherwise stay None and silently drop the
        # lambda_cost term). Move it with the model so buffers follow to CUDA.
        self.loss_fn.cost_matrix = self.cost_matrix
        self.loss_fn.to(self.device)
        self.cost_matrix.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            enabled=config.use_amp and self.device.type == "cuda"
        )

    def _targets(self, batch: dict) -> dict[str, torch.Tensor]:
        return {
            lvl: batch[f"target_{lvl}"].to(self.device)
            for lvl in ("order", "family", "genus", "species")
        }

    def _step_loss(self, images, targets):
        logits = self.model(images, targets)
        if self.cfg.mixup_alpha > 0 and self.model.training:
            # Blend targets then run two forward passes weighted by lam.
            images_b, targets_b, lam = mixup_cutmix_batch(
                images, targets, self.cfg.mixup_alpha, self.cfg.mixup_prob
            )
            logits_b = self.model(images_b, targets_b)
            loss = 0.0
            for lvl in ("order", "family", "genus", "species"):
                l_a, _ = self._level_loss(logits[lvl], targets[lvl], lvl)
                l_b, _ = self._level_loss(logits_b[lvl], targets_b[lvl], lvl)
                loss = loss + lam * l_a + (1 - lam) * l_b
            cost = self.cost_matrix(
                logits["species"], targets["species"].clamp(min=0)
            )
            loss = loss + self.cfg.lambda_cost * cost
            return loss, {"total": float(loss.detach())}
        loss, terms = self.loss_fn(logits, targets)
        return loss, terms

    def _level_loss(self, logits, target, level):
        from .losses import FocalLoss
        import torch.nn.functional as F

        if level in ("genus", "species") and self.cfg.focal_gamma > 0:
            return (
                FocalLoss(self.cfg.focal_gamma, self.cfg.focal_alpha)(logits, target),
                None,
            )
        return F.cross_entropy(logits, target, ignore_index=-100), None

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None,
        callbacks: list | None = None,
    ) -> dict[str, list]:
        history: dict[str, list] = {"train": [], "val": []}
        callbacks = callbacks or []
        for epoch in range(self.cfg.epochs):
            self._set_lr(epoch)
            tr = self._run_epoch(train_loader, train=True)
            history["train"].append(tr)
            msg = (f"epoch {epoch+1}/{self.cfg.epochs} "
                   f"loss={tr.loss:.3f} "
                   f"ord={tr.acc_order:.3f} fam={tr.acc_family:.3f} "
                   f"gen={tr.acc_genus:.3f} sp={tr.acc_species:.3f} "
                   f"({tr.seconds:.0f}s)")
            if val_loader is not None:
                va = self._run_epoch(val_loader, train=False)
                history["val"].append(va)
                msg += (f" | val loss={va.loss:.3f} "
                        f"ord={va.acc_order:.3f} fam={va.acc_family:.3f} "
                        f"gen={va.acc_genus:.3f} sp={va.acc_species:.3f}")
            print(msg)
            for cb in callbacks:
                cb(epoch, self.model, history)
        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochStats:
        self.model.train(train)
        stats = EpochStats()
        n, correct, total = 0, {l: 0 for l in ("order", "family", "genus", "species")}, 0
        start = time.time()
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = self._targets(batch)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(
                    self.device.type, enabled=self.cfg.use_amp and self.device.type == "cuda"
                ):
                    loss, terms = self._step_loss(images, targets)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            stats.loss += float(loss.detach())
            stats.terms = terms
            bs = images.size(0)
            n += bs
            total += bs
            with torch.no_grad():
                logits = self.model(images, targets)
                for lvl in correct:
                    correct[lvl] += int((logits[lvl].argmax(-1) == targets[lvl]).sum())
        stats.seconds = time.time() - start
        stats.loss /= max(n, 1)
        for lvl, key in (("order", "acc_order"), ("family", "acc_family"),
                         ("genus", "acc_genus"), ("species", "acc_species")):
            setattr(stats, key, correct[lvl] / max(total, 1))
        return stats

    def _set_lr(self, epoch: int):
        import math

        scale = 0.5 * (1 + math.cos(math.pi * epoch / max(self.cfg.epochs, 1)))
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.lr * scale
