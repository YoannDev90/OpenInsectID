"""Loss functions: Hierarchical Cross-Entropy, Focal Loss, and the taxonomic
cost matrix.

Implements section 3 of the engineering plan:
- Hierarchical Cross-Entropy: per-level weighted CE (order mistakes cost more
  than species mistakes) plus a taxonomic cost-matrix penalty driven by the
  tree distance between the predicted and true species.
- Focal Loss: down-weights well-classified examples to focus gradients on the
  long-tail (rare species/genera).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = ("order", "family", "genus", "species")


class FocalLoss(nn.Module):
    """Multi-class focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha: float | torch.Tensor = 0.25,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B,) with -100 marking ignore_index
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=-100)
        p_t = torch.exp(-ce)  # probability assigned to the true class
        alpha_t = self.alpha
        if isinstance(alpha_t, torch.Tensor) and alpha_t.dim() > 0:
            alpha_t = alpha_t.to(logits.device)[target.clamp(min=0)]
        else:
            alpha_t = self.alpha
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        loss = loss[target != -100]
        if self.reduction == "mean":
            return loss.mean() if loss.numel() else logits.sum() * 0.0
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HierarchicalCrossEntropy(nn.Module):
    """Weighted sum of per-level cross-entropy.

    ``level_weights`` implements the taxonomic penalty schedule: a mistake at
    the order level is weighted far more heavily than a mistake at the species
    level (w_order >> w_species).
    """

    def __init__(self, level_weights: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0)):
        super().__init__()
        if len(level_weights) != 4:
            raise ValueError("Need exactly 4 level weights (order,family,genus,species)")
        self.register_buffer("weights", torch.tensor(level_weights, dtype=torch.float32))

    def forward(
        self, logits_per_level: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        total = None
        for w, level in zip(self.weights, LEVELS):
            ce = F.cross_entropy(
                logits_per_level[level], targets[level], ignore_index=-100
            )
            term = w * ce
            total = term if total is None else total + term
        return total  # type: ignore[return-value]


class TaxonomicCostMatrix(nn.Module):
    """Expected taxonomic-distance penalty between predicted and true species.

    Precomputes, for every species, its (order, family, genus, species) global
    index path. The cost of a sample is derived from the depth of the lowest
    common ancestor (LCA) between the predicted and true species: predicting a
    species in the wrong order yields the maximum cost, while confusing two
    species of the same genus is cheap.
    """

    # Cost added for each taxonomic level that diverges (order-most to species).
    DEFAULT_LEVEL_COSTS = (1.0, 0.5, 0.25, 0.125)

    def __init__(self, species_paths: torch.Tensor,
                 level_costs: tuple[float, ...] = DEFAULT_LEVEL_COSTS):
        """
        Args:
            species_paths: LongTensor (n_species, 4) of global indices
                (order, family, genus, species) for each species class.
        """
        super().__init__()
        if species_paths.dim() != 2 or species_paths.shape[1] != 4:
            raise ValueError("species_paths must be (n_species, 4)")
        self.register_buffer("paths", species_paths.long())
        self.register_buffer(
            "level_costs", torch.tensor(level_costs, dtype=torch.float32)
        )

    @torch.no_grad()
    def build_from_tree(self, moe) -> "TaxonomicCostMatrix":
        """Fill species_paths from a HierarchicalMoE's global label indices."""
        paths = []
        for name in moe.species_names:
            order, family, genus, _ = moe.tree.path(name)
            paths.append([
                moe._level_index["order"][order],
                moe._level_index["family"][family],
                moe._level_index["genus"][genus],
                moe._level_index["species"][name],
            ])
        self.paths = torch.tensor(paths, dtype=torch.long, device=self.paths.device)
        return self

    def forward(self, species_logits: torch.Tensor,
                target_species: torch.Tensor) -> torch.Tensor:
        """Mean expected cost using the full species distribution.

        E[cost] = sum_s p(s) * cost(s, y) where cost counts divergent levels.
        """
        probs = F.softmax(species_logits, dim=-1)  # (B, n_species)
        true_paths = self.paths[target_species]  # (B, 4)
        # Broadcast: (B, n_species, 4) predicted paths vs (B, 1, 4) true paths.
        pred_paths = self.paths.unsqueeze(0)  # (1, n_species, 4)
        mismatch = pred_paths != true_paths.unsqueeze(1)  # (B, n_species, 4)
        # Cost at each divergent level; LCA-based: charge level costs from the
        # top down to the deepest divergence.
        per_level_cost = self.level_costs.view(1, 1, -1)
        cost = (mismatch.float() * per_level_cost).sum(-1)  # (B, n_species)
        return (probs * cost).sum(-1).mean()


class CombinedTaxonomicLoss(nn.Module):
    """Full v2 loss: per-level CE/Focal + taxonomic cost-matrix penalty.

    L = sum_k w_k * L_k(y_k, yhat_k) + lambda * CostMatrix(y_pred, y_true)

    Args:
        level_weights: weight per taxonomic level.
        focal_gamma: gamma for focal loss on genus/species (0 disables focal).
        focal_alpha: alpha balancing factor for focal loss.
        lambda_cost: strength of the taxonomic cost-matrix term.
        species_paths: (n_species, 4) global index paths (required for cost term).
    """

    def __init__(
        self,
        level_weights: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0),
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_cost: float = 1.0,
        species_paths: torch.Tensor | None = None,
    ):
        super().__init__()
        self.weights = level_weights
        self.focal_gamma = focal_gamma
        self.lambda_cost = lambda_cost
        # Order/family use plain CE; genus/species use focal loss for long-tail.
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.cost_matrix = (
            TaxonomicCostMatrix(species_paths) if species_paths is not None else None
        )

    def build_cost_matrix(self, model) -> "CombinedTaxonomicLoss":
        """Construct the taxonomic cost matrix from a trained ``HierarchicalMoE``.

        Use this when you did not pass ``species_paths`` at construction time:
        ``loss_fn.build_cost_matrix(model)``.
        """
        self.cost_matrix = TaxonomicCostMatrix(
            torch.zeros(model.n_species, 4, dtype=torch.long)
        ).build_from_tree(model)
        return self

    def forward(
        self, logits_per_level: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        terms: dict[str, float] = {}
        use_focal = {"genus": True, "species": True} if self.focal_gamma > 0 else {}
        for w, level in zip(self.weights, LEVELS):
            logits, target = logits_per_level[level], targets[level]
            if use_focal.get(level):
                loss = self.focal(logits, target)
            else:
                loss = self.ce(logits, target)
            term = w * loss
            terms[level] = float(loss.detach())
            total = term if total is None else total + term

        if self.cost_matrix is not None and "species" in logits_per_level:
            # Species targets may be masked (-100); cost matrix needs valid indices.
            cost = self.cost_matrix(
                logits_per_level["species"], targets["species"].clamp(min=0)
            )
            total = total + self.lambda_cost * cost
            terms["cost_matrix"] = float(cost.detach())
        terms["total"] = float(total.detach())
        return total, terms
