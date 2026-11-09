"""Tests for the hierarchical cascade: routing, thresholds, forward, masking."""

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


def test_dispatch_masks_to_parent_children(moe):
    """Rows are masked to the children of their ground-truth parent.

    Family 0 belongs to order 0; family 2 to order 1. In the row whose parent
    is order 0, the other order's families must sit at the _NEG fill.
    """
    x = torch.randn(2, 3, 224, 224)
    targets = {
        "order": torch.tensor([0, 1]),
        "family": torch.tensor([0, 2]),
        "genus": torch.tensor([0, 3]),
        "species": torch.tensor([0, 5]),
    }
    out = moe(x, targets)
    # Row 0 (order 0): valid family range is [0, 2) (Carabidae, Coccinellidae).
    assert out["family"][0, 2] <= -1e3  # Nymphalidae is not a child of order 0
    assert out["family"][1, 0] <= -1e3  # Carabidae is not a child of order 1


def test_child_logits_match_masked_forward(moe):
    """Sliced inference logits equal the unmasked rows of the training forward."""
    x = torch.randn(2, 3, 224, 224)
    targets = {
        "order": torch.tensor([0, 1]),
        "family": torch.tensor([0, 2]),
        "genus": torch.tensor([0, 3]),
        "species": torch.tensor([0, 5]),
    }
    z = moe.features(x)
    out = moe.heads(z, targets)
    # For each row, the slice of the full forward under the parent's range
    # must equal child_logits (the inference path).
    for row, (lvl, parent) in enumerate((("family", 0), ("genus", 0))):
        start, end = moe.heads.child_range(lvl, parent)
        sliced = out[lvl][0, start:end]
        ref = moe.heads.child_logits(z[0:1], lvl, parent)[0]
        assert torch.allclose(sliced, ref, atol=1e-5), lvl


def test_forward_mix_matches_single_forwards(moe):
    """forward_mix must give the same logits as two separate forwards."""
    x_a = torch.randn(2, 3, 224, 224)
    x_b = torch.randn(2, 3, 224, 224)
    t_a = {
        "order": torch.tensor([0, 1]),
        "family": torch.tensor([0, 2]),
        "genus": torch.tensor([0, 3]),
        "species": torch.tensor([0, 5]),
    }
    t_b = {
        "order": torch.tensor([1, 0]),
        "family": torch.tensor([2, 1]),
        "genus": torch.tensor([3, 2]),
        "species": torch.tensor([4, 1]),
    }
    moe.eval()
    with torch.no_grad():
        mix_a, mix_b = moe.forward_mix(x_a, t_a, x_b, t_b)
        ref_a = moe(x_a, t_a)
        ref_b = moe(x_b, t_b)
    for lvl in ("order", "family", "genus", "species"):
        assert torch.allclose(mix_a[lvl], ref_a[lvl], atol=1e-5), lvl
        assert torch.allclose(mix_b[lvl], ref_b[lvl], atol=1e-5), lvl


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


def test_export_modules_is_five_modules(moe):
    names = [n for n, _ in moe.export_modules()]
    assert names == ["trunk", "head_order", "head_family", "head_genus", "head_species"]


def test_single_child_parent_is_pass_through():
    # A family with a single genus should pass through deterministically.
    recs = [
        ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),
        ("Coleoptera", "Coccinellidae", "Coccinella", "Coccinella septempunctata"),
        ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa atalanta"),
        ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa cardui"),
    ]
    t = TaxonomyTree(recs)
    m = HierarchicalMoE(t, backbone_name="mnv4_small", width_mult=0.25)
    # Carabidae has one genus (Carabus): its genus range is size 1.
    assert m.heads.child_range("genus", 0) == (0, 1)
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


def test_single_order_tree_passes_through_without_order_head():
    """One order (tiny e2e manifest) must not try to build a 1-class order head."""
    tree = TaxonomyTree(SINGLE_ORDER_RECORDS)
    model = HierarchicalMoE(tree, backbone_name="mnv4_small", width_mult=0.25)
    assert model.heads.order is None
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

    # The absent order head must not be exported.
    assert all(name != "head_order" for name, _ in model.export_modules())


def test_forward_under_autocast_keeps_global_buffers_fp32(moe):
    """AMP makes head logits bf16/fp16; the masked global-label-space output
    must be cast back to fp32 for the loss."""
    x = torch.randn(2, 3, 224, 224)
    targets = {
        "order": torch.tensor([0, 1]),
        "family": torch.tensor([0, 2]),   # Carabidae / Nymphalidae
        "genus": torch.tensor([0, 3]),    # Carabus / Vanessa
        "species": torch.tensor([0, 4]),  # Carabus auratus / Vanessa atalanta
    }
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = moe(x, targets)
    for level in ("family", "genus", "species"):
        assert out[level].dtype == torch.float32, level


# ------------------------------------------------------------- capacity bound

def test_shared_trunk_params_stay_small():
    """At scale the model must be trunk + linear heads, not per-parent backbones.

    Build a tree with many genera (the level that used to explode) and assert
    the parameter count stays within trunk + O(D * classes).
    """
    recs = []
    for gi in range(40):
        for si in range(5):
            recs.append((
                "Coleoptera", f"Family{gi // 4}", f"Genus{gi}",
                f"Genus{gi} species{si}",
            ))
    t = TaxonomyTree(recs)
    m = HierarchicalMoE(t, backbone_name="mnv4_small", width_mult=1.0)
    n = sum(p.numel() for p in m.parameters())
    d = m.heads.feature_dim
    classes = m.n_orders + m.n_families + m.n_genera + m.n_species
    # 40 genera x a full backbone each would be ~175M; heads are ~D*classes.
    assert n < 20e6, f"too many params: {n/1e6:.1f}M"
    assert n > d * classes * 0.9  # heads alone are at least D*classes
