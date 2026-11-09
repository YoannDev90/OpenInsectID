"""Hierarchical Vision MoE: a 4-level taxonomic cascade on a shared trunk.

Architecture (see docs/architecture-shared-trunk.md):

    image -> ONE MobileNetV4 trunk -> features z (B, D)
    z -> 4 global linear heads (order / family / genus / species), each row of
    a head being one class weight; a parent owns the contiguous child range
    [ranges[p], ranges[p+1]) of its head.

Level 1 (order) is the unmasked root head. Levels 2-4 mask each row to the
children of its ground-truth parent (invalid classes filled with a large
negative constant), so the hierarchical loss consumes the same
(B, n_classes_global) tensors as the original per-expert scatter design.

Two entry points:
- ``forward(x, targets)``: training. One trunk pass, masked heads.
- ``forward_mix(x_a, t_a, x_b, t_b)``: mixup/CutMix. One trunk pass over the
  concatenated batch, then the heads on each half.
- ``route(x)``: inference on a single image. Top-K order routing with
  threshold tau_order, per-level acceptance thresholds and early stopping:
  the deepest accepted level is the answer, with an explicit uncertainty flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .backbones import build_backbone
from .heads import CascadeHeads
from .router import prediction_entropy, topk_with_threshold
from .taxonomy import TaxonomyTree


@dataclass
class LevelPrediction:
    level: str  # "order" | "family" | "genus" | "species"
    index: int  # global class index at that level
    name: str
    prob: float
    entropy: float
    accepted: bool

    def __str__(self) -> str:  # pragma: no cover - display helper
        flag = "ok" if self.accepted else "REJECTED"
        return f"{self.level}={self.name} (p={self.prob:.2f}, {flag})"


@dataclass
class CascadeResult:
    """Result of routing one image through the cascade."""

    path: list[LevelPrediction] = field(default_factory=list)
    complete: bool = False  # species reached and accepted
    stop_level: str | None = None  # first rejected level, if any
    ambiguous_orders: bool = False  # Top-K order routing was triggered
    joint_prob: float = 0.0  # product of accepted-level probabilities

    @property
    def validated_path(self) -> list[LevelPrediction]:
        return [p for p in self.path if p.accepted]

    @property
    def last_valid(self) -> LevelPrediction | None:
        v = self.validated_path
        return v[-1] if v else None

    def describe(self) -> str:
        last = self.last_valid
        if last is None:
            return "Uncertain: even the order could not be validated."
        confirmed = " > ".join(p.name for p in self.validated_path)
        if self.complete:
            return f"Identified: {confirmed} (joint p={self.joint_prob:.2f})."
        return (
            f"Confirmed only to {last.level.upper()}: {confirmed}. "
            f"Stopped at level '{self.stop_level}' below the confidence threshold "
            f"(joint p={self.joint_prob:.2f})."
        )


class HierarchicalMoE(nn.Module):
    LEVELS = ("order", "family", "genus", "species")

    def __init__(
        self,
        tree: TaxonomyTree,
        backbone_name: str = "mnv4_small",
        width_mult: float = 1.0,
        tau_order: float = 0.85,
        tau_levels: tuple[float, float, float, float] = (0.85, 0.80, 0.70, 0.50),
        top_k_orders: int = 2,
        head_dim: int | None = None,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.tree = tree
        self.tau_order = tau_order
        self.tau_levels = tuple(tau_levels)
        self.top_k_orders = top_k_orders

        # Global label spaces (order -> family -> genus -> species). Children
        # of one parent are contiguous, matching CascadeHeads' ranges.
        self.order_names = tree.orders
        self.family_names = [f for o in self.order_names for f in tree.families_of(o)]
        self.genus_names = [
            g
            for o in self.order_names
            for f in tree.families_of(o)
            for g in tree.genera_of(f)
        ]
        self.species_names = [
            s
            for o in self.order_names
            for f in tree.families_of(o)
            for g in tree.genera_of(f)
            for s in tree.species_of(g)
        ]
        self.n_orders = len(self.order_names)
        self.n_families = len(self.family_names)
        self.n_genera = len(self.genus_names)
        self.n_species = len(self.species_names)

        # ONE trunk for the whole cascade (was: one full backbone per parent).
        self.trunk = build_backbone(backbone_name, width_mult=width_mult)
        self.heads = CascadeHeads(
            feature_dim=self.trunk.out_channels,
            sizes={
                "order": self.n_orders,
                "family": self.n_families,
                "genus": self.n_genera,
                "species": self.n_species,
            },
            children_per_parent={
                "family": [len(tree.families_of(o)) for o in self.order_names],
                "genus": [len(tree.genera_of(f)) for f in self.family_names],
                "species": [len(tree.species_of(g)) for g in self.genus_names],
            },
            head_dim=head_dim,
            dropout=head_dropout,
        )

        self._level_index = {
            "order": {name: i for i, name in enumerate(self.order_names)},
            "family": {name: i for i, name in enumerate(self.family_names)},
            "genus": {name: i for i, name in enumerate(self.genus_names)},
            "species": {name: i for i, name in enumerate(self.species_names)},
        }

    # --------------------------------------------------------------- training

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk forward: images (B, 3, H, W) -> features (B, D)."""
        return self.trunk(x)

    def forward(
        self, x: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Training forward pass (one trunk pass).

        ``targets`` maps each level to a (B,) LongTensor of *global* class
        indices; the ground-truth parent masks levels 2-4. Returns per-level
        logits (B, n_classes_global).
        """
        return self.heads(self.features(x), targets)

    def forward_mix(
        self,
        x_a: torch.Tensor,
        targets_a: dict[str, torch.Tensor],
        x_b: torch.Tensor,
        targets_b: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Mixup forward: ONE trunk pass over the concatenated batch, then the
        heads applied to each half with its own targets."""
        z = self.features(torch.cat([x_a, x_b], dim=0))
        z_a, z_b = z.chunk(2, dim=0)
        return self.heads(z_a, targets_a), self.heads(z_b, targets_b)

    # --------------------------------------------------------------- inference

    @torch.no_grad()
    def route(
        self,
        x: torch.Tensor,
        tau_order: float | None = None,
        tau_levels: tuple[float, ...] | None = None,
        top_k_orders: int | None = None,
    ) -> CascadeResult:
        """Route a single image (1,3,H,W) through the cascade."""
        if x.dim() == 4:
            if x.size(0) != 1:
                raise ValueError("route() expects a single image; use forward() for batches")
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        tau_order = self.tau_order if tau_order is None else tau_order
        tau = self.tau_levels if tau_levels is None else tuple(tau_levels)
        k = self.top_k_orders if top_k_orders is None else top_k_orders

        # One trunk pass: every level (and every Top-K candidate branch) reuses
        # the same features.
        z = self.features(x)

        if self.heads.order is not None:
            order_probs = torch.softmax(self.heads.order_logits(z), dim=-1)[0]
            dec = topk_with_threshold(order_probs, tau_order, k)
            ambiguous = (not dec.confident) and k > 1
            n_candidates = k if ambiguous else 1
            candidates = list(
                zip(dec.indices[:n_candidates], dec.probs[:n_candidates])
            )
        else:
            # Single order: pass through with certainty (prob 1.0, entropy 0).
            order_probs = torch.ones(1, device=x.device)
            candidates = [(0, 1.0)]
            ambiguous = False

        best: CascadeResult | None = None
        for order_idx, order_p in candidates:
            res = self._descend(z, int(order_idx), float(order_p), order_probs, tau)
            res.ambiguous_orders = ambiguous
            if best is None or res.joint_prob > best.joint_prob:
                best = res
        return best  # type: ignore[return-value]

    def _descend(
        self,
        z: torch.Tensor,
        order_idx: int,
        order_p: float,
        order_probs: torch.Tensor,
        tau: tuple[float, ...],
    ) -> CascadeResult:
        order_name = self.order_names[order_idx]
        res = CascadeResult(joint_prob=order_p)
        res.path.append(
            LevelPrediction(
                "order", order_idx, order_name, order_p,
                prediction_entropy(order_probs), order_p >= tau[0],
            )
        )
        if not res.path[-1].accepted:
            res.stop_level = "order"
            return res

        joint = order_p
        # Descend: at each level the parent is the accepted child's *global*
        # index (heads' ranges are indexed by global parent index).
        parent_idx = order_idx
        for level, tau_level in (
            ("family", tau[1]),
            ("genus", tau[2]),
            ("species", tau[3]),
        ):
            start, end = self.heads.child_range(level, parent_idx)
            if end - start == 1:
                # Single child: deterministic pass-through (prob 1.0).
                child_idx = start
                child_name = getattr(self, f"{level}_names")[child_idx]
                res.path.append(
                    LevelPrediction(level, child_idx, child_name, 1.0, 0.0, True)
                )
                parent_idx = child_idx
                continue
            logits = self.heads.child_logits(z, level, parent_idx)  # (1, n_children)
            probs = torch.softmax(logits, dim=-1)[0]
            child_idx_local = int(probs.argmax())
            child_p = float(probs[child_idx_local])
            child_idx = start + child_idx_local
            child_name = getattr(self, f"{level}_names")[child_idx]
            accepted = child_p >= tau_level
            res.path.append(
                LevelPrediction(
                    level, child_idx, child_name, child_p,
                    prediction_entropy(probs), accepted,
                )
            )
            if not accepted:
                res.stop_level = level
                return res
            joint *= child_p
            parent_idx = child_idx

        res.complete = True
        res.joint_prob = joint
        return res

    # ----------------------------------------------------------------- export

    def export_modules(self):
        """Yield (name, module) pairs for ONNX/TFLite export.

        The cascade cannot be a single static graph (head selection is data
        dependent), but the shared trunk collapses the payload to 5 modules:
        the trunk (image -> features) plus one head per level (features ->
        that level's global logits). The child ranges needed to slice heads
        on-device live in the export manifest.
        """
        yield "trunk", self.trunk
        for level in ("order", "family", "genus", "species"):
            if level == "order" and self.heads.order is None:
                continue  # single-order tree: nothing to route
            yield f"head_{level}", self.heads.export_head(level)
