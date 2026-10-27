"""Tests for the hierarchical, focal, and cost-matrix losses."""

import torch
import pytest

from src.training.losses import (
    FocalLoss,
    HierarchicalCrossEntropy,
    TaxonomicCostMatrix,
    CombinedTaxonomicLoss,
)


def _random_logits_targets(B, classes, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, classes), torch.randint(0, classes, (B,))


def test_focal_reduces_loss_for_easy_examples():
    logits, target = _random_logits_targets(8, 5)
    # Make predictions very confident & correct -> focal loss << plain CE.
    confident = logits.clone()
    confident[torch.arange(8), target] += 10.0
    focal = FocalLoss(gamma=2.0, alpha=0.25)
    ce = torch.nn.CrossEntropyLoss()
    assert focal(confident, target) < ce(confident, target)


def test_focal_gamma_zero_equals_ce():
    logits, target = _random_logits_targets(16, 7)
    focal = FocalLoss(gamma=0.0, alpha=1.0)
    ce = torch.nn.CrossEntropyLoss()
    assert torch.allclose(focal(logits, target), ce(logits, target), atol=1e-5)


def test_hierarchical_weights_order_mistake_more():
    # Same logits/targets; scale so that flipping the order prediction is what
    # differs. We check that up-weighting the order level raises total loss.
    torch.manual_seed(1)
    B, C = 4, 6
    targets = {
        "order": torch.randint(0, C, (B,)),
        "family": torch.randint(0, C, (B,)),
        "genus": torch.randint(0, C, (B,)),
        "species": torch.randint(0, C, (B,)),
    }
    logits = {k: torch.randn(B, C) for k in targets}

    flat = HierarchicalCrossEntropy((1, 1, 1, 1))
    heavy_order = HierarchicalCrossEntropy((10, 1, 1, 1))
    l_flat = flat(logits, targets)
    l_heavy = heavy_order(logits, targets)
    assert l_heavy > l_flat


def test_cost_matrix_same_species_cheaper_than_cross_order():
    # 4 species: 2 in order A, 2 in order B. Paths encode (order,family,genus,species).
    paths = torch.tensor(
        [
            [0, 0, 0, 0],  # A/.../sp0
            [0, 0, 0, 1],  # A/.../sp1 (same order as sp0)
            [1, 1, 1, 2],  # B/.../sp2
            [1, 1, 1, 3],  # B/.../sp3
        ]
    )
    cm = TaxonomicCostMatrix(paths)
    # One-hot prediction on same-order vs cross-order species, true = sp0.
    same_order = torch.full((1, 4), -10.0)
    same_order[0, 1] = 10.0  # predict sp1 (same order as sp0)
    cross_order = torch.full((1, 4), -10.0)
    cross_order[0, 2] = 10.0  # predict sp2 (different order)
    target = torch.tensor([0])
    assert cm(same_order, target) < cm(cross_order, target)


def test_combined_loss_returns_terms_and_backward():
    B, C = 6, 5
    paths = torch.stack(
        [
            torch.arange(B) % C,
            torch.arange(B) % C,
            torch.arange(B) % C,
            torch.arange(B) % C,
        ],
        dim=1,
    )
    paths = torch.randint(0, C, (C, 4))  # (n_species=C, 4)
    loss_fn = CombinedTaxonomicLoss(
        level_weights=(8, 4, 2, 1), focal_gamma=2.0, lambda_cost=1.0,
        species_paths=paths,
    )
    targets = {
        "order": torch.randint(0, C, (B,)),
        "family": torch.randint(0, C, (B,)),
        "genus": torch.randint(0, C, (B,)),
        "species": torch.randint(0, C, (B,)),
    }
    logits = {k: torch.randn(B, C, requires_grad=True) for k in targets}
    total, terms = loss_fn(logits, targets)
    assert total.item() > 0
    for lvl in ("order", "family", "genus", "species", "cost_matrix", "total"):
        assert lvl in terms
    total.backward()
    assert logits["order"].grad is not None
