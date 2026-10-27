"""Dataset over the curated image manifest (Parquet).

Manifest schema (one row per curated image), produced by the data pipeline:
    image_path : relative path under the image root
    order      : taxonomic order name
    family     : taxonomic family name
    genus      : taxonomic genus name
    species    : taxonomic species (binomial) name
    [optional] h3_cell, lat, lon, occurrence_id, media_url

Targets are global class indices at each level, aligned with the label spaces
of ``HierarchicalMoE`` (order/family/genus/species).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..models.taxonomy import TaxonomyTree


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """HWC uint8 PIL -> CHW float in [0,1]. Normalisation is applied on GPU."""
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class InsectDataset(Dataset):
    """Reads the manifest, loads images, returns per-level global targets."""

    LEVELS = ("order", "family", "genus", "species")

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        tree: TaxonomyTree | None = None,
        transform=None,
        species_names: list[str] | None = None,
    ):
        import polars as pl

        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.transform = transform

        df = pl.read_parquet(self.manifest_path)
        required = {"image_path", "order", "family", "genus", "species"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        self.df = df

        # Taxonomy tree / label spaces.
        self.tree = tree or TaxonomyTree.from_manifest(df)
        self.species_names = species_names or [
            s for o in self.tree.orders for f in self.tree.families_of(o)
            for g in self.tree.genera_of(f) for s in self.tree.species_of(g)
        ]
        self._index = {
            "order": {n: i for i, n in enumerate(self.tree.orders)},
            "family": {n: i for i, n in enumerate(
                [f for o in self.tree.orders for f in self.tree.families_of(o)])},
            "genus": {n: i for i, n in enumerate(
                [g for o in self.tree.orders for f in self.tree.families_of(o)
                 for g in self.tree.genera_of(f)])},
            "species": {n: i for i, n in enumerate(self.species_names)},
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)
        img = Image.open(self.image_root / row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = _pil_to_tensor(img)
        targets = {lvl: self._index[lvl][row[lvl]] for lvl in self.LEVELS}
        return {"image": img, **{f"target_{k}": v for k, v in targets.items()}}

    @property
    def num_classes(self) -> dict[str, int]:
        return {lvl: len(self._index[lvl]) for lvl in self.LEVELS}


def _collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    out = {"image": images}
    for lvl in InsectDataset.LEVELS:
        out[f"target_{lvl}"] = torch.tensor(
            [b[f"target_{lvl}"] for b in batch], dtype=torch.long
        )
    return out


def build_dataloaders(
    manifest_path: str | Path,
    image_root: str | Path,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    transform=None,
    seed: int = 42,
    tree: TaxonomyTree | None = None,
) -> tuple[DataLoader, DataLoader, TaxonomyTree]:
    """Split the manifest into train/val by species-stratified rows.

    Returns (train_loader, val_loader, tree).
    """
    import polars as pl
    from torch.utils.data import random_split

    ds = InsectDataset(manifest_path, image_root, tree=tree, transform=transform)
    n_val = int(len(ds) * val_fraction)
    n_train = len(ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
    )
    return train_loader, val_loader, ds.tree
