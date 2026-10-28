"""Hierarchical Vision MoE: a 4-level taxonomic cascade.

Level 1: a global lightweight router (order).
Level 2-4: specialised experts, one per parent node (family experts per order,
genus experts per family, species experts per genus).

Two entry points:
- ``forward(x, targets)``: training mode. Runs the expert matching each
  sample's ground-truth parent and scatters its logits into the global label
  space of each level (invalid classes get a large negative fill), so the
  hierarchical loss can consume (B, n_classes_global) tensors per level.
- ``route(x)``: inference mode on a single image. Top-K order routing with
  threshold tau_order, per-level acceptance thresholds and early stopping:
  the deepest accepted level is the answer, with an explicit uncertainty flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .experts import Expert
from .router import prediction_entropy, topk_with_threshold
from .taxonomy import TaxonomyTree

# Fill for classes that are not valid children of the current parent.
# Large negative constant (not -inf) to stay safe under AMP / logsumexp.
_NEG = -1e4


def _sanitize(key: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", key)


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
    ):
        super().__init__()
        self.tree = tree
        self.tau_order = tau_order
        self.tau_levels = tuple(tau_levels)
        self.top_k_orders = top_k_orders

        # Global label spaces (order -> family -> genus -> species).
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

        backbone_kw = dict(backbone_name=backbone_name, width_mult=width_mult)
        # A single-order tree has nothing to route: deterministic pass-through,
        # the same rule as single-child parents at lower levels.
        self.router = Expert(self.n_orders, **backbone_kw) if self.n_orders >= 2 else None

        # Per-expert key = f"{global_index}_{sanitized_name}" (unique, ModuleDict-safe).
        self._keys = {"family": [], "genus": [], "species": []}
        self._local = {"family": {}, "genus": {}, "species": {}}  # key -> {global: local}
        self._local_to_global = {"family": {}, "genus": {}, "species": {}}
        self._passthrough = {"family": {}, "genus": {}, "species": {}}

        self.family_experts, order_of_family = self._build_level(
            "family",
            [(o, tree.families_of(o)) for o in self.order_names],
            backbone_kw,
        )
        self.genus_experts, family_of_genus = self._build_level(
            "genus",
            [(f, tree.genera_of(f)) for f in self.family_names],
            backbone_kw,
        )
        self.species_experts, genus_of_species = self._build_level(
            "species",
            [(g, tree.species_of(g)) for g in self.genus_names],
            backbone_kw,
        )
        self._parent_of = {
            "family": order_of_family,
            "genus": family_of_genus,
            "species": genus_of_species,
        }
        # For each level, the list that maps a *parent* global index to its name.
        self._PARENT_LIST = {
            "family": self.order_names,
            "genus": self.family_names,
            "species": self.genus_names,
        }
        self._level_index = {
            "order": {name: i for i, name in enumerate(self.order_names)},
            "family": {name: i for i, name in enumerate(self.family_names)},
            "genus": {name: i for i, name in enumerate(self.genus_names)},
            "species": {name: i for i, name in enumerate(self.species_names)},
        }

    # ------------------------------------------------------------------ setup

    def _build_level(self, level: str, groups: list[tuple[str, list[str]]], backbone_kw: dict):
        """Instantiate experts for parents with >= 2 children.

        Single-child parents become deterministic pass-throughs (no expert).
        Returns (ModuleDict, parent_of_child: dict[child_global_idx -> parent_name]).
        Also fills ``self._local_to_global[level][key]`` = {local_idx -> global_idx}.
        """
        experts = nn.ModuleDict()
        parent_of_child: dict[int, str] = {}
        key_lookup: dict[str, str] = {}
        global_pos = 0
        for parent, children in groups:
            for local, child in enumerate(children):
                parent_of_child[global_pos + local] = parent
            if len(children) >= 2:
                key = f"{global_pos}_{_sanitize(parent)}"
                self._keys[level].append(key)
                key_lookup[parent] = key
                self._local[level][key] = {
                    global_pos + local: local for local in range(len(children))
                }
                self._local_to_global[level][key] = {
                    local: global_pos + local for local in range(len(children))
                }
                experts[key] = Expert(len(children), **backbone_kw)
            else:
                # Single child: class of index global_pos deterministically.
                self._passthrough[level][parent] = global_pos
            global_pos += len(children)
        setattr(self, f"_key_lookup_{level}", key_lookup)
        return experts, parent_of_child

    # --------------------------------------------------------------- training

    def _dispatch(
        self,
        x: torch.Tensor,
        parent_global: torch.Tensor,
        experts: nn.ModuleDict,
        level: str,
        n_global: int,
    ) -> torch.Tensor:
        """Run each row's parent expert; scatter its local logits into the
        global label space of this level (invalid classes stay at ``_NEG``)."""
        out = x.new_full((x.size(0), n_global), _NEG)
        parent_list = self._PARENT_LIST[level]  # parent global idx -> name

        # Map each row to its expert key, or None for a pass-through parent.
        row_key: list[str | None] = []
        for p in parent_global.tolist():
            parent = parent_list[int(p)]
            if parent in self._passthrough[level]:
                row_key.append(None)
            else:
                row_key.append(self._key_for_parent(level, parent))

        # sorted() (not raw set iteration): expert execution order must not
        # depend on PYTHONHASHSEED, or runs would stop being bit-identical.
        for key in sorted(set(k for k in row_key if k is not None)):
            rows = [i for i, k in enumerate(row_key) if k == key]
            idx = torch.as_tensor(rows, device=x.device)
            logits = experts[key](x[idx])  # (n, n_local)
            gidx = torch.tensor(
                [self._local_to_global[level][key][j] for j in range(logits.shape[1])],
                device=x.device,
            )
            # Scatter every local logit to its global class position. Under AMP
            # the expert runs in half precision while the buffer follows x's
            # dtype: cast so index_put sees matching dtypes.
            out[idx[:, None], gidx[None, :]] = logits.to(out.dtype)
        # Pass-through parents: logit 0 on their single child.
        for i, p in enumerate(parent_global.tolist()):
            parent = parent_list[int(p)]
            if parent in self._passthrough[level]:
                out[i, self._passthrough[level][parent]] = 0.0
        return out

    def _key_for_parent(self, level: str, parent: str) -> str:
        return getattr(self, f"_key_lookup_{level}")[parent]

    def forward(
        self, x: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        ``targets`` maps each level to a (B,) LongTensor of *global* class
        indices; the ground-truth parent selects the expert for levels 2-4.
        Returns per-level logits (B, n_classes_global).
        """
        out = {
            # Single order: constant logit 0 -> probability 1.0 (pass-through).
            "order": self.router(x) if self.router is not None else x.new_zeros((x.size(0), 1))
        }
        out["family"] = self._dispatch(
            x, targets["order"], self.family_experts, "family", self.n_families
        )
        out["genus"] = self._dispatch(
            x, targets["family"], self.genus_experts, "genus", self.n_genera
        )
        out["species"] = self._dispatch(
            x, targets["genus"], self.species_experts, "species", self.n_species
        )
        return out

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

        if self.router is not None:
            order_probs = torch.softmax(self.router(x), dim=-1)[0]
            dec = topk_with_threshold(order_probs, tau_order, k)
            ambiguous = (not dec.confident) and k > 1
            n_candidates = k if ambiguous else 1
            candidates = list(
                zip(dec.indices[:n_candidates], dec.probs[:n_candidates])
            )
        else:
            # Single order: pass through with certainty (prob 1.0, entropy 0),
            # mirroring the single-child pass-through of lower levels.
            order_probs = torch.ones(1, device=x.device)
            candidates = [(0, 1.0)]
            ambiguous = False

        best: CascadeResult | None = None
        for order_idx, order_p in candidates:
            res = self._descend(x, int(order_idx), float(order_p), order_probs, tau)
            res.ambiguous_orders = ambiguous
            if best is None or res.joint_prob > best.joint_prob:
                best = res
        return best  # type: ignore[return-value]

    def _descend(
        self,
        x: torch.Tensor,
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
        state = (order_name, order_p)
        for level, expert_dict, tau_level in (
            ("family", self.family_experts, tau[1]),
            ("genus", self.genus_experts, tau[2]),
            ("species", self.species_experts, tau[3]),
        ):
            parent = state[0]
            if parent in self._passthrough[level]:
                child_idx = self._passthrough[level][parent]
                child_name = getattr(self, f"{level}_names")[child_idx]
                res.path.append(
                    LevelPrediction(level, child_idx, child_name, 1.0, 0.0, True)
                )
                state = (child_name, 1.0)
                continue
            key = self._key_for_parent(level, parent)
            probs = torch.softmax(expert_dict[key](x), dim=-1)[0]
            child_idx_local = int(probs.argmax())
            child_p = float(probs[child_idx_local])
            child_idx = self._global_index(level, key, child_idx_local)
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
            state = (child_name, child_p)

        res.complete = True
        res.joint_prob = joint
        return res

    def _global_index(self, level: str, key: str, local_idx: int) -> int:
        for g, l in self._local[level][key].items():
            if l == local_idx:
                return g
        raise KeyError(f"local index {local_idx} not found for {level}:{key}")

    # ----------------------------------------------------------------- export

    def export_modules(self):
        """Yield (name, module) pairs for per-module ONNX/TFLite export.

        The cascade cannot be a single static graph (expert selection is data
        dependent), so each router/expert ships as its own file — see the
        export module and the Android loading strategy.
        """
        if self.router is not None:
            yield "router", self.router
        for level, experts in (
            ("family", self.family_experts),
            ("genus", self.genus_experts),
            ("species", self.species_experts),
        ):
            for key, module in experts.items():
                start = int(key.split("_", 1)[0])
                name = f"{level}_{key.split('_', 1)[1]}_{start}"
                yield name, module
