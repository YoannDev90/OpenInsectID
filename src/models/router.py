"""Top-K routing with confidence thresholds and prediction entropy helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TopKDecision:
    """Outcome of a thresholded Top-K selection over a probability vector."""

    indices: torch.Tensor  # (k,) candidate class indices, best first
    probs: torch.Tensor  # (k,) their probabilities
    confident: bool  # top-1 probability >= threshold

    @property
    def top1(self) -> int:
        return int(self.indices[0])

    @property
    def top1_prob(self) -> float:
        return float(self.probs[0])


def topk_with_threshold(
    probs: torch.Tensor, threshold: float, k: int = 2
) -> TopKDecision:
    """Pick the Top-K of a 1-D probability vector and flag low confidence.

    Used at Level 1: if the best order's probability is below ``threshold``,
    the caller keeps ``k`` candidate orders instead of committing to one
    (cascade-error mitigation).
    """
    if probs.dim() != 1:
        raise ValueError(f"Expected a 1-D probability vector, got shape {tuple(probs.shape)}")
    k = min(k, probs.numel())
    top_p, top_i = probs.topk(k)
    return TopKDecision(
        indices=top_i,
        probs=top_p,
        confident=bool(top_p[0] >= threshold),
    )


def prediction_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy (nats) of a probability distribution."""
    p = probs.clamp_min(eps)
    return float(-(p * p.log()).sum())


def entropy_bits_from_logits(logits: torch.Tensor) -> float:
    """Entropy of a softmax distribution given logits, in bits."""
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    return prediction_entropy(probs) / math.log(2)
