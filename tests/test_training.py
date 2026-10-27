"""Dataloader/trainer behaviour on tiny (smoke-test) manifests.

Guards two Kaggle-sized failures: an empty train loader (drop_last discarding
the only batch when samples < batch_size) and a missing val split on small
manifests (8-row e2e batches).
"""

from pathlib import Path

import polars as pl
from PIL import Image


def _write_tiny_manifest(tmp_path: Path, n: int = 8) -> tuple[str, str]:
    """n curated rows; families/genus/species unique, two orders interleaved."""
    root = tmp_path / "images"
    rows = []
    for i in range(n):
        rel = f"Order_{i % 2}/Fam_{i}/Gen_{i}/sp_{i}/{i}.jpg"
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), (i * 25 % 255, 60, 90)).save(p)
        rows.append(
            {
                "image_path": rel,
                "order": f"Order_{i % 2}",
                "family": f"Fam_{i}",
                "genus": f"Gen_{i}",
                "species": f"sp_{i}",
            }
        )
    mpath = tmp_path / "manifest.parquet"
    pl.DataFrame(rows).write_parquet(mpath)
    return str(mpath), str(root)


def test_tiny_manifest_never_yields_empty_loaders(tmp_path):
    from src.training import build_dataloaders, train_transform

    mpath, root = _write_tiny_manifest(tmp_path, n=8)
    train_loader, val_loader, tree = build_dataloaders(
        mpath, root, batch_size=128, val_fraction=0.1,
        num_workers=0, transform=train_transform(image_size=224), seed=42,
    )
    batches = list(train_loader)
    # drop_last must not swallow the only batch (samples < batch_size).
    assert len(batches) == 1
    assert batches[0]["image"].shape[0] == 7  # 1 row held out for validation
    # Tiny manifests must still produce a non-empty val split.
    assert len(val_loader.dataset) == 1
    assert tree.num_species >= 1


def test_trainer_one_epoch_on_tiny_manifest(tmp_path):
    from src.models import HierarchicalMoE
    from src.training import (
        Trainer, TrainConfig, build_dataloaders, train_transform,
    )

    mpath, root = _write_tiny_manifest(tmp_path, n=8)
    train_loader, val_loader, tree = build_dataloaders(
        mpath, root, batch_size=8, val_fraction=0.25,
        num_workers=0, transform=train_transform(image_size=224), seed=0,
    )
    model = HierarchicalMoE(tree, backbone_name="mnv4_small")
    cfg = TrainConfig(
        epochs=1, device="cpu", use_amp=False, mixup_alpha=0.0, log_every=1,
    )
    trainer = Trainer(model, cfg)
    history = trainer.fit(train_loader, val_loader)

    assert history["train"] and history["val"]
    tr = history["train"][0]
    assert tr.loss > 0  # loss averaged over at least one real batch
    assert 0.0 <= tr.acc_species <= 1.0
