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
    # The plain loss path and the mixup path must share one cost matrix
    # (otherwise lambda_cost is silently dropped on non-mixup batches), and
    # its buffers must follow the model to the training device.
    assert trainer.loss_fn.cost_matrix is trainer.cost_matrix
    assert trainer.cost_matrix.paths.device == trainer.device
    history = trainer.fit(train_loader, val_loader)

    assert history["train"] and history["val"]
    tr = history["train"][0]
    assert tr.loss > 0  # loss averaged over at least one real batch
    assert 0.0 <= tr.acc_species <= 1.0


def test_step_loss_mixup_path_costs_and_backprops(tmp_path):
    """The mixup branch calls the cost matrix directly (this is where the
    Kaggle CUDA run crashed); it must share the wired matrix and backprop."""
    import torch

    from src.models import HierarchicalMoE
    from src.training import (
        Trainer, TrainConfig, build_dataloaders, train_transform,
    )

    mpath, root = _write_tiny_manifest(tmp_path, n=8)
    train_loader, _, tree = build_dataloaders(
        mpath, root, batch_size=8, val_fraction=0.25,
        num_workers=0, transform=train_transform(image_size=224), seed=0,
    )
    model = HierarchicalMoE(tree, backbone_name="mnv4_small")
    cfg = TrainConfig(
        epochs=1, device="cpu", use_amp=False,
        mixup_alpha=0.4, mixup_prob=1.0, log_every=1,  # force the mixup branch
    )
    trainer = Trainer(model, cfg)
    assert trainer.loss_fn.cost_matrix is trainer.cost_matrix

    batch = next(iter(train_loader))
    model.train()
    loss, _ = trainer._step_loss(batch["image"], trainer._targets(batch))
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()


def test_checkpoint_roundtrip_and_resume(tmp_path):
    """save_checkpoint -> load_checkpoint restores model + optimizer and
    tells fit() where to continue (chunked Kaggle runs)."""
    import torch

    from src.models import HierarchicalMoE
    from src.training import Trainer, TrainConfig, build_dataloaders, train_transform

    mpath, root = _write_tiny_manifest(tmp_path, n=8)
    train_loader, val_loader, tree = build_dataloaders(
        mpath, root, batch_size=8, val_fraction=0.25,
        num_workers=0, transform=train_transform(image_size=224), seed=0,
    )
    ckpt_dir = tmp_path / "ckpt"
    cfg = TrainConfig(
        epochs=2, device="cpu", use_amp=False, mixup_alpha=0.0, log_every=1,
        ckpt_dir=str(ckpt_dir), ckpt_every=1,
    )
    t1 = Trainer(HierarchicalMoE(tree, backbone_name="mnv4_small"), cfg)
    t1.fit(train_loader, val_loader)
    ckpt = ckpt_dir / "last.pt"
    assert ckpt.exists()

    # A fresh trainer (fresh model + optimizer) resumes with identical state.
    t2 = Trainer(HierarchicalMoE(tree, backbone_name="mnv4_small"), cfg)
    assert t2.load_checkpoint(ckpt) == 2  # both epochs completed
    assert t2.fit(train_loader, val_loader)["train"] == []  # nothing left
    sd1, sd2 = t1.model.state_dict(), t2.model.state_dict()
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    # Optimizer state survived the roundtrip (needed for real continuations).
    assert t2.optimizer.state


def test_time_budget_stops_with_checkpoint(tmp_path):
    """max_minutes stops fit() after a completed epoch — with a checkpoint
    on disk so the next run can resume (quota / kernel-limit guard)."""
    from src.models import HierarchicalMoE
    from src.training import Trainer, TrainConfig, build_dataloaders, train_transform

    mpath, root = _write_tiny_manifest(tmp_path, n=8)
    train_loader, val_loader, tree = build_dataloaders(
        mpath, root, batch_size=8, val_fraction=0.25,
        num_workers=0, transform=train_transform(image_size=224), seed=0,
    )
    cfg = TrainConfig(
        epochs=50, device="cpu", use_amp=False, mixup_alpha=0.0, log_every=1,
        ckpt_dir=str(tmp_path), ckpt_every=1000, max_minutes=1e-9,
    )
    trainer = Trainer(HierarchicalMoE(tree, backbone_name="mnv4_small"), cfg)
    history = trainer.fit(train_loader, val_loader)

    assert len(history["train"]) == 1       # stopped after the first epoch
    assert (tmp_path / "last.pt").exists()  # ...but checkpointed first
    t2 = Trainer(HierarchicalMoE(tree, backbone_name="mnv4_small"), cfg)
    assert t2.load_checkpoint(tmp_path / "last.pt") == 1  # resumes at epoch 2


def test_checkpoint_restores_all_rng_streams(tmp_path):
    """Seamless resume: the checkpoint must capture/restore every RNG stream
    (torch, python random, numpy) so the continuation is bit-identical."""
    import random

    import numpy as np
    import torch

    from src.models import HierarchicalMoE
    from src.training import Trainer, TrainConfig, build_dataloaders, train_transform

    mpath, root = _write_tiny_manifest(tmp_path, n=8)
    train_loader, _, tree = build_dataloaders(
        mpath, root, batch_size=8, val_fraction=0.25,
        num_workers=0, transform=train_transform(image_size=224), seed=0,
    )
    cfg = TrainConfig(epochs=1, device="cpu", mixup_alpha=0.0, log_every=1)
    trainer = Trainer(HierarchicalMoE(tree, backbone_name="mnv4_small"), cfg)
    # Deterministic algorithms are on by default (bit-exact run-to-run).
    assert torch.are_deterministic_algorithms_enabled()

    # Advance every stream so the captured state is non-trivial, then save.
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    torch.rand(4)
    random.random()
    np.random.rand(3)
    ckpt = trainer.save_checkpoint(tmp_path / "last.pt", 42)

    expected = (torch.rand(6), random.random(), np.random.rand(4))

    # Scramble everything, then restore: the next draws must match exactly.
    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    assert trainer.load_checkpoint(ckpt) == 42

    got = (torch.rand(6), random.random(), np.random.rand(4))
    assert torch.equal(expected[0], got[0])
    assert expected[1] == got[1]
    assert np.array_equal(expected[2], got[2])
