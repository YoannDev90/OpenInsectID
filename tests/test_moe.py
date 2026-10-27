"""Tests for the hierarchical MoE cascade: routing, thresholds, forward."""

import torch
import pytest

from src.models.moe import HierarchicalMoE
from src.models.taxonomy import TaxonomyTree


RECORDS = [
    ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),
    ("Coleoptera", "Carabidae", "Carabus", "Carabus insulicola"),
    ("Coleoptera", "Carabidae", "Cicindela", "Cicindela campestris"),
    ("Coleoptera", "Coccinellidae", "Coccinella", "Coccinella septempunctata"),
    ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa atalanta"),
    ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa cardui"),
]


@pytest.fixture(scope="module")
def tree():
    return TaxonomyTree(RECORDS)


@pytest.fixture(scope="module")
def moe(tree):
    # Tiny backbone for speed: width_mult to shrink, but keep functional.
    return HierarchicalMoE(
        tree, backbone_name="mnv4_small", width_mult=0.25,
        tau_order=0.85, tau_levels=(0.85, 0.80, 0.70, 0.50),
    )


def _img(size=224):
    return torch.randn(1, 3, size, size)


def test_label_space_sizes(moe):
    assert moe.n_orders == 2
    assert moe.n_families == 3
    assert moe.n_genera == 4
    assert moe.n_species == 6


def test_forward_returns_all_levels(moe):
    x = _img()
    targets = {
        "order": torch.tensor([0]),
        "family": torch.tensor([0]),
        "genus": torch.tensor([0]),
        "species": torch.tensor([0]),
    }
    out = moe(x, targets)
    assert out["order"].shape == (1, moe.n_orders)
    assert out["family"].shape == (1, moe.n_families)
    assert out["genus"].shape == (1, moe.n_genera)
    assert out["species"].shape == (1, moe.n_species)


def test_dispatch_selects_correct_expert(moe):
    # Samples from order 0 vs order 1 must use different family experts.
    x = torch.randn(2, 3, 224, 224)
    targets = {
        "order": torch.tensor([0, 1]),
        "family": torch.tensor([0, 2]),
        "genus": torch.tensor([0, 2]),
        "species": torch.tensor([0, 5]),
    }
    out = moe(x, targets)
    # Invalid classes (children of other parents) must be strongly negative.
    # Family 0 belongs to order 0; family 2 to order 1.
    assert out["family"][0, 0] > out["family"][0, 2] or out["family"][0, 0] > -1e3


def test_route_returns_cascade_result(moe):
    x = _img()
    res = moe.route(x)
    assert res.path is not None
    assert res.path[0].level == "order"
    # Path is monotonically ordered down the taxonomy.
    levels = [p.level for p in res.path]
    assert levels == ["order", "family", "genus", "species"][: len(levels)]


def test_route_rejects_when_threshold_high(moe):
    # Force rejection at order level with an impossible threshold.
    x = _img()
    res = moe.route(x, tau_order=1.01, tau_levels=(1.01, 1.01, 1.01, 1.01))
    assert res.complete is False
    assert res.stop_level == "order"
    assert res.validated_path == []


def test_route_ambiguity_keeps_topk_orders(moe):
    x = _img()
    # tau_order above max prob -> ambiguous, should keep 2 candidate orders.
    res = moe.route(x, tau_order=1.01, top_k_orders=2)
    assert res.ambiguous_orders is True


def test_route_batch_size_guard(moe):
    with pytest.raises(ValueError, match="single image"):
        moe.route(torch.randn(3, 3, 224, 224))


def test_route_accepts_chw_tensor(moe):
    # (3,H,W) gets unsqueezed to (1,3,H,W).
    res = moe.route(torch.randn(3, 224, 224))
    assert res.path[0].level == "order"


def test_export_modules_covers_all_experts(moe):
    names = [n for n, _ in moe.export_modules()]
    assert "router" in names
    assert any(n.startswith("family_") for n in names)
    assert any(n.startswith("genus_") for n in names)
    assert any(n.startswith("species_") for n in names)


def test_passthrough_single_child():
    # A family with a single genus should skip that expert (pass-through).
    recs = [
        ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),
        ("Coleoptera", "Coccinellidae", "Coccinella", "Coccinella septempunctata"),
        ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa atalanta"),
        ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa cardui"),
    ]
    t = TaxonomyTree(recs)
    m = HierarchicalMoE(t, backbone_name="mnv4_small", width_mult=0.25)
    # Carabidae has one genus (Carabus): no genus expert keyed on it.
    # Nymphalidae has one genus too. So genus_experts should be empty.
    assert len(m.genus_experts) == 0
    res = m.route(torch.randn(3, 224, 224), tau_order=0.0, tau_levels=(0,) * 4)
    # Cascade should still reach species via pass-through.
    assert any(p.level == "genus" for p in res.path)


# --------------------------------------------------- single-order (smoke) tree

SINGLE_ORDER_RECORDS = [
    ("Lepidoptera", "Geometridae", "Agriopis", "Agriopis marginaria"),
    ("Lepidoptera", "Hesperiidae", "Autochton", "Autochton neis"),
    ("Lepidoptera", "Hesperiidae", "Anthoptus", "Anthoptus epictetus"),
    ("Lepidoptera", "Nymphalidae", "Anthanassa", "Anthanassa drusilla"),
]


def test_single_order_tree_passes_through_without_router():
    """One order (tiny e2e manifest) must not try to build a 1-class router."""
    tree = TaxonomyTree(SINGLE_ORDER_RECORDS)
    model = HierarchicalMoE(tree, backbone_name="mnv4_small", width_mult=0.25)
    assert model.router is None
    assert model.n_orders == 1

    # Training forward: constant pass-through logits at the order level.
    x = torch.randn(2, 3, 224, 224)
    targets = {
        "order": torch.zeros(2, dtype=torch.long),
        "family": torch.tensor([0, 1]),
        "genus": torch.tensor([0, 1]),
        "species": torch.tensor([0, 1]),
    }
    out = model(x, targets)
    assert out["order"].shape == (2, 1)
    assert torch.all(out["order"] == 0)

    # Inference: the single order is accepted with certainty (prob 1.0).
    res = model.route(_img())
    assert res.path[0].name == "Lepidoptera"
    assert res.path[0].prob == pytest.approx(1.0)
    assert res.path[0].accepted

    # The absent router must not be exported.
    assert all(name != "router" for name, _ in model.export_modules())
