"""Tests for the TaxonomyTree and path-divergence helpers."""

import pytest

from src.models.taxonomy import TaxonomyTree, SpeciesRecord, path_divergence


RECORDS = [
    ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),
    ("Coleoptera", "Carabidae", "Carabus", "Carabus insulicola"),
    ("Coleoptera", "Carabidae", "Cicindela", "Cicindela campestris"),
    ("Coleoptera", "Coccinellidae", "Coccinella", "Coccinella septempunctata"),
    ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa atalanta"),
]


@pytest.fixture
def tree():
    return TaxonomyTree(RECORDS)


def test_orders(tree):
    assert tree.orders == ["Coleoptera", "Lepidoptera"]


def test_families_of(tree):
    assert tree.families_of("Coleoptera") == ["Carabidae", "Coccinellidae"]
    assert tree.families_of("Lepidoptera") == ["Nymphalidae"]


def test_genera_of(tree):
    assert tree.genera_of("Carabidae") == ["Carabus", "Cicindela"]
    assert tree.species_of("Carabus") == ["Carabus auratus", "Carabus insulicola"]


def test_path(tree):
    assert tree.path("Carabus auratus") == (
        "Coleoptera", "Carabidae", "Carabus", "Carabus auratus"
    )


def test_expert_groups(tree):
    fam = tree.expert_groups("family")
    assert fam["Coleoptera"] == ["Carabidae", "Coccinellidae"]
    gen = tree.expert_groups("genus")
    assert gen["Carabidae"] == ["Carabus", "Cicindela"]
    sp = tree.expert_groups("species")
    assert len(sp["Carabus"]) == 2


def test_num_species(tree):
    assert tree.num_species == 5


def test_save_load_roundtrip(tmp_path, tree):
    p = tmp_path / "tree.json"
    tree.save(p)
    loaded = TaxonomyTree.load(p)
    assert loaded.orders == tree.orders
    assert loaded.path("Vanessa atalanta") == tree.path("Vanessa atalanta")


def test_from_manifest():
    import polars as pl

    df = pl.DataFrame(
        {
            "order": ["Coleoptera"] * 2 + ["Lepidoptera"],
            "family": ["Carabidae"] * 2 + ["Nymphalidae"],
            "genus": ["Carabus"] * 2 + ["Vanessa"],
            "species": ["Carabus auratus", "Carabus insulicola", "Vanessa atalanta"],
        }
    )
    tree = TaxonomyTree.from_manifest(df)
    assert tree.num_species == 3


def test_from_manifest_is_row_order_independent():
    """Label spaces must not depend on the manifest's row order.

    polars' ``unique()`` returns rows in an arbitrary (engine-dependent)
    order; if the tree followed it, two otherwise identical runs could build
    different species index assignments — and produce different models from
    the same data and seed. Whatever the input order, the derived tree (and
    hence the label space) must be identical.
    """
    import numpy as np
    import polars as pl

    rows = []
    for family in ("FamA", "FamB"):
        for genus in ("GenusA", "GenusB"):
            for sp in ("SpA", "SpB"):
                rows.append(("Lepidoptera", family, genus, f"{sp} {genus} {family}"))
    reference = TaxonomyTree.from_manifest(
        pl.DataFrame(
            {"order": [r[0] for r in rows], "family": [r[1] for r in rows],
             "genus": [r[2] for r in rows], "species": [r[3] for r in rows]}
        )
    ).to_dict()
    rng = np.random.RandomState(0)
    for _ in range(10):
        perm = rng.permutation(len(rows))
        shuffled = [rows[i] for i in perm]
        df = pl.DataFrame(
            {"order": [r[0] for r in shuffled], "family": [r[1] for r in shuffled],
             "genus": [r[2] for r in shuffled], "species": [r[3] for r in shuffled]}
        )
        assert TaxonomyTree.from_manifest(df).to_dict() == reference


def test_path_divergence():
    a = ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus")
    b = ("Coleoptera", "Carabidae", "Carabus", "Carabus insulicola")
    c = ("Coleoptera", "Carabidae", "Cicindela", "Cicindela campestris")
    d = ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa atalanta")
    assert path_divergence(a, b) == 3  # same genus, differ at species
    assert path_divergence(a, c) == 2  # same family, differ at genus
    assert path_divergence(a, d) == 0  # differ at order


def test_dedupe_and_collision_safety():
    recs = [
        ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),
        ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),  # duplicate
        ("Lepidoptera", "Carabidae", "Carabus", "Carabus altus"),  # name collision across order
    ]
    t = TaxonomyTree(recs)
    assert t.num_species == 2
    # Both Carabus variants resolve without clobbering.
    assert "Carabus auratus" in t.species_of(t.path("Carabus auratus")[2])
