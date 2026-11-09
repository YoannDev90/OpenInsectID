"""Per-level linear heads over the shared trunk's feature space.

The cascade keeps its 4 taxonomic levels and its global label spaces, but the
per-parent *backbones* are replaced by rows of four global linear heads: a
class weight is one row of its level's ``Linear(D, n_level)``. Because the
global label space lays out each parent's children contiguously (the same
layout the per-parent experts used), a parent owns the contiguous child range
``[ranges[p], ranges[p+1])``.

Training: rows outside the ground-truth parent's range are filled with
``_NEG`` (``-1e4``), reproducing exactly the scatter the expert dispatch did,
so the hierarchical loss consumes the same ``(B, n_level_global)`` tensors.

Inference: ``child_logits`` computes only the selected parent's slice of the
head (``F.linear`` on sliced weights), i.e. the old per-parent expert at a
fraction of the cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = ("order", "family", "genus", "species")

# Fill for classes that are not valid children of the current parent.
# Large negative constant (not -inf) to stay safe under AMP / logsumexp.
_NEG = -1e4


class CascadeHeads(nn.Module):
    """The 4 level heads (+ optional per-level projections) of the cascade.

    Args:
        feature_dim: trunk output dimension D.
        sizes: global label-space size per level, e.g.
            ``{"order": 30, "family": 1100, "genus": 13000, "species": 40000}``.
            ``sizes["order"] <= 1`` means a single-order tree: no order head is
            built (pass-through, matching the old router-less MoE).
        children_per_parent: for each masked level, the number of children of
            every parent, **in parent global-index order**. The cumulative sums
            of each list become that level's ``ranges`` buffer.
        head_dim: if set, a per-level ``Linear(D, head_dim)`` projection runs
            before the head (shrinks every head by D/head_dim); default None
            keeps full-width D features.
        dropout: dropout on the projected features before the heads (0 = off).
    """

    def __init__(
        self,
        feature_dim: int,
        sizes: dict[str, int],
        children_per_parent: dict[str, list[int]],
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.sizes = {lvl: int(sizes[lvl]) for lvl in LEVELS}
        self.head_dim = int(head_dim) if head_dim else self.feature_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional per-level projections D -> head_dim (absent when head_dim
        # is unset or equals the feature dim, which is the default).
        self.projections = nn.ModuleDict()
        head_in = self.feature_dim
        if head_dim and head_dim != self.feature_dim:
            for lvl in LEVELS:
                if self.sizes[lvl] >= 1:
                    self.projections[lvl] = nn.Linear(self.feature_dim, head_dim)
            head_in = head_dim

        # The order level is unmasked (root): build the head whenever there is
        # more than one order; a single order passes through with certainty.
        self.order = (
            nn.Linear(head_in, self.sizes["order"]) if self.sizes["order"] >= 2 else None
        )
        for lvl in ("family", "genus", "species"):
            setattr(self, lvl, nn.Linear(head_in, self.sizes[lvl]))
            counts = children_per_parent[lvl]
            # ranges[p]:p+1] = children of parent p, as cumulative offsets.
            cum = torch.cumsum(
                torch.tensor([0] + list(counts), dtype=torch.long), dim=0
            )
            assert int(cum[-1]) == self.sizes[lvl], (
                f"{lvl}: children counts sum to {int(cum[-1])}, "
                f"expected {self.sizes[lvl]}"
            )
            self.register_buffer(f"range_{lvl}", cum, persistent=True)

    # ---------------------------------------------------------------- helpers

    def _project(self, z: torch.Tensor, level: str) -> torch.Tensor:
        proj = self.projections[level] if level in self.projections else None
        return self.dropout(proj(z) if proj is not None else z)

    def child_range(self, level: str, parent_idx: int) -> tuple[int, int]:
        """Global [start, end) child range of a parent at ``level``."""
        rng = getattr(self, f"range_{level}")
        return int(rng[parent_idx]), int(rng[parent_idx + 1])

    def order_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Unmasked root logits (B, n_orders); raises if there is no head."""
        if self.order is None:
            raise ValueError("single-order tree: no order head (pass-through)")
        return self.order(self._project(z, "order"))

    def parent_of(self, level: str, child_idx: int) -> int:
        """Global index of the parent owning a child class."""
        rng = getattr(self, f"range_{level}")
        # First boundary strictly greater than child_idx, minus one.
        return int(torch.searchsorted(rng, torch.tensor(child_idx + 1))) - 1

    def child_logits(self, z: torch.Tensor, level: str, parent_idx: int) -> torch.Tensor:
        """Logits over one parent's children only — the inference path.

        Slices the head's weights instead of computing the full level, so
        predicting the species children of a 3-species genus costs 3 rows of
        the 40k-row species head.
        """
        head = getattr(self, level)
        start, end = self.child_range(level, parent_idx)
        w = head.weight[start:end]
        b = head.bias[start:end]
        return F.linear(self._project(z, level), w, b)

    # ---------------------------------------------------------------- forward

    def _mask(self, logits: torch.Tensor, parent_idx: torch.Tensor, level: str) -> torch.Tensor:
        """Fill classes outside each row's parent range with ``_NEG``.

        Vectorized: no data-dependent Python loop, hence no execution-order
        nondeterminism (the property the sorted expert dispatch protected).
        """
        rng = getattr(self, f"range_{level}")
        starts = rng[parent_idx]          # (B,)
        ends = rng[parent_idx + 1]        # (B,)
        cols = torch.arange(logits.size(1), device=logits.device)
        valid = (cols >= starts[:, None]) & (cols < ends[:, None])
        # Cast back to fp32: under AMP the heads run in half precision while
        # the loss consumes fp32 global-label-space tensors (parity with the
        # old scatter buffer, which inherited the input's fp32 dtype).
        return logits.masked_fill(~valid, _NEG).float()

    def forward(
        self, z: torch.Tensor, targets: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Map features (B, D) to per-level logits (B, n_level_global).

        ``targets`` (training): global parent indices per level; masking is
        applied at family/genus/species. Without targets every level returns
        raw logits (the caller slices with :meth:`child_logits` semantics or
        masks externally).
        """
        out: dict[str, torch.Tensor] = {}
        if self.order is not None:
            out["order"] = self.order(self._project(z, "order"))
        else:
            # Single order: constant logit 0 -> probability 1.0 (pass-through).
            out["order"] = z.new_zeros((z.size(0), 1))
        for lvl in ("family", "genus", "species"):
            logits = getattr(self, lvl)(self._project(z, lvl))
            if targets is not None:
                logits = self._mask(logits, targets[lvl_to_parent(lvl)], lvl)
            out[lvl] = logits
        return out

    def export_head(self, level: str) -> nn.Module:
        """Module (projection + dropout + head) to export for one level."""
        head = getattr(self, level) if level != "order" else self.order
        if head is None:
            raise ValueError(f"no head at level '{level}'")
        proj = self.projections[level] if level in self.projections else None
        if proj is None:
            return head
        return nn.Sequential(proj, self.dropout, head)


def lvl_to_parent(level: str) -> str:
    """Level whose *prediction* selects the parent at ``level``."""
    return {"family": "order", "genus": "family", "species": "genus"}[level]
