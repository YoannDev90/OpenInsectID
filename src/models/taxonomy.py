"""Taxonomy tree utilities shared by models, training and export.

The tree is built from unique species records (order, family, genus, species)
and provides the child groupings needed by the hierarchical MoE (experts are
instantiated per parent node) plus path-divergence helpers used by the
hierarchical loss.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LEVELS = ("order", "family", "genus", "species")


@dataclass(frozen=True)
class SpeciesRecord:
    order: str
    family: str
    genus: str
    species: str


class TaxonomyTree:
    """Order -> Family -> Genus -> Species tree derived from the dataset.

    Assumptions (true for GBIF Linnean taxonomy): species binomials are unique,
    family and genus names are unique enough to use as expert keys.
    """

    def __init__(self, records: Iterable[SpeciesRecord | tuple[str, str, str, str]]):
        # order -> family -> genus -> [species]
        self._tree: dict[str, dict[str, dict[str, list[str]]]] = {}
        self._family_owner: dict[str, str] = {}  # family -> order
        self._genus_owner: dict[str, tuple[str, str]] = {}  # genus -> (order, family)
        self._species_owner: dict[str, tuple[str, str, str]] = {}  # species -> path

        for rec in records:
            if not isinstance(rec, SpeciesRecord):
                rec = SpeciesRecord(*rec)
            self._insert(rec)

    def _insert(self, rec: SpeciesRecord) -> None:
        if rec.species in self._species_owner:
            return  # dedupe
        families = self._tree.setdefault(rec.order, {})
        if rec.family in self._family_owner and self._family_owner[rec.family] != rec.order:
            # Same family name under two orders: qualify with the order to stay collision-free.
            family_key = f"{rec.order}/{rec.family}"
        else:
            family_key = rec.family
        self._family_owner.setdefault(family_key, rec.order)
        genera = families.setdefault(family_key, {})
        genus_key = (
            f"{family_key}/{rec.genus}"
            if rec.genus in self._genus_owner
            and self._genus_owner[rec.genus][:2] != (rec.order, family_key)
            else rec.genus
        )
        self._genus_owner.setdefault(genus_key, (rec.order, family_key))
        genera.setdefault(genus_key, [])
        if rec.species not in genera[genus_key]:
            genera[genus_key].append(rec.species)
        self._species_owner[rec.species] = (rec.order, family_key, genus_key)

    # -- structure accessors -------------------------------------------------

    @property
    def orders(self) -> list[str]:
        return sorted(self._tree)

    def families_of(self, order: str) -> list[str]:
        return sorted(self._tree[order])

    def genera_of(self, family: str) -> list[str]:
        order = self._family_owner[family]
        return sorted(self._tree[order][family])

    def species_of(self, genus: str) -> list[str]:
        order, family = self._genus_owner[genus]
        return self._tree[order][family][genus]

    def path(self, species: str) -> tuple[str, str, str, str]:
        """Return the (order, family, genus, species) path of a species."""
        order, family, genus = self._species_owner[species]
        return order, family, genus, species

    @property
    def num_species(self) -> int:
        return len(self._species_owner)

    def expert_groups(self, level: str) -> dict[str, list[str]]:
        """Map each parent node to the class list its expert must discriminate.

        - family: order -> families
        - genus: family -> genera
        - species: genus -> species
        """
        if level == "family":
            return {o: self.families_of(o) for o in self.orders}
        if level == "genus":
            return {
                f: self.genera_of(f)
                for o in self.orders
                for f in self.families_of(o)
            }
        if level == "species":
            return {
                g: self.species_of(g)
                for o in self.orders
                for f in self.families_of(o)
                for g in self.genera_of(f)
            }
        raise ValueError(f"expert_groups expects family|genus|species, got {level!r}")

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "orders": {
                o: {
                    f: {g: sp for g, sp in genera.items()}
                    for f, genera in families.items()
                }
                for o, families in self._tree.items()
            }
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TaxonomyTree":
        records = []
        for o, families in payload["orders"].items():
            for f, genera in families.items():
                for g, species in genera.items():
                    for s in species:
                        records.append((o, f, g, s))
        return cls(records)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "TaxonomyTree":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_manifest(cls, df) -> "TaxonomyTree":
        """Build from a Polars DataFrame with order/family/genus/species columns."""
        rows = (
            df.select(["order", "family", "genus", "species"])
            .unique()
            .iter_rows()
        )
        return cls(tuple(r) for r in rows)


def path_divergence(pred_path: tuple[str, ...], true_path: tuple[str, ...]) -> int:
    """Depth of the lowest common ancestor between two taxonomy paths.

    0 = different orders, 1 = same order/different family, ...,
    len(path) = identical paths.
    """
    for i, (p, t) in enumerate(zip(pred_path, true_path)):
        if p != t:
            return i
    return len(true_path)
