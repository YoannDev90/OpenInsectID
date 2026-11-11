"""Generate the Kaggle-oriented Jupyter notebooks for Open Insect ID v2.

Run once: python scripts/make_notebooks.py
"""

import re

import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

# --- Run controls (single source for the generated notebooks) --------------
# Production values: resume enabled, full 5h budget. For the seamless-resume
# verification protocol these are edited LOCALLY per run (the generator edit
# itself is reverted before any commit — only production defaults land in
# git; the generated notebooks pushed to Kaggle may differ in these knobs,
# which don't affect the weights):
#   alpha = RESUME=False, BUDGET_MIN=0.4  (stops mid-training, pushes ckpt)
#   beta  = RESUME=True,  BUDGET_MIN=300  (resumes alpha, prints hash H_beta)
#   gamma = RESUME=False, BUDGET_MIN=300  (fresh full run, prints hash H_gamma)
# Seamless resume holds iff H_beta == H_gamma (same git commit for all three).
RESUME = True
BUDGET_MIN = 300

# Data-campaign phase for notebook 01 — same local-edit workflow as the
# training knobs above (edit here, regenerate, push; never edit the .ipynb):
#   census   -> page the GBIF speciesKey facet      -> meta dataset
#   sample   -> one occurrence page per species     -> meta dataset
#   download -> fill one shard (SHARD) with images  -> data-NN + manifest
#   manifest -> merge the per-shard manifest parts  -> manifest dataset
PHASE = "census"
SHARD = 0            # 0..N_SHARDS-1: shard filled by the download phase
DL_BUDGET_MIN = 540  # wall-clock budget (minutes) for one download run
N_SHARDS = 12        # matches CampaignConfig.n_shards and the data-NN datasets


def nb():
    return nbf.v4.new_notebook()


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


SETUP = """\
# Open Insect ID v2 — data campaign (Kaggle)
#
# CPU notebook driving the 40k-species x 150-images campaign (architecture
# doc section 9): four resumable phases, one per run, each publishing the
# Kaggle dataset consumed by the next phase and by the training notebook
# (02):
#
#   PHASE="census"     page GBIF's speciesKey facet   -> meta dataset
#   PHASE="sample"     one occurrence page per species -> meta dataset
#   PHASE="download"   fill one shard (SHARD) with curated 320-px images
#                      -> data-NN + manifest datasets
#   PHASE="manifest"   merge the per-shard manifest parts -> manifest dataset
#
# PHASE / SHARD / DL_BUDGET_MIN are injected by scripts/make_notebooks.py —
# the single source of truth (edit it, regenerate, push; never edit this
# .ipynb directly). Every phase skips artifacts that already exist, so a
# killed session simply continues on the next run. Set the accelerator to
# **CPU**; only the training notebook (02) needs a **GPU**.
"""

INSTALL = """\
import os, subprocess, sys
from pathlib import Path

# --- Configure this to your repo -------------------------------------------
GITHUB_REPO = os.environ.get("OPENINSECT_REPO", "https://github.com/YoannDev90/OpenInsectID.git")
# ---------------------------------------------------------------------------
IS_KAGGLE = os.path.exists("/kaggle/input")

def _is_our_checkout(path):
    # Identify OUR repo precisely: any folder happening to contain a src/
    # (cwd leftovers, /kaggle, ...) must not be mistaken for the checkout.
    return (os.path.isfile(os.path.join(path, "pyproject.toml"))
            and os.path.isdir(os.path.join(path, "src", "data_pipeline")))

# Walk up from cwd (kernels start in /kaggle/working, Jupyter in the notebook
# folder) to the repo root; anywhere else means: install from GitHub.
repo_root = os.getcwd()
while not _is_our_checkout(repo_root):
    parent = os.path.dirname(repo_root)
    if parent == repo_root:
        break
    repo_root = parent
in_checkout = _is_our_checkout(repo_root)

if in_checkout:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
else:
    # pip needs the `git+` prefix to clone -- a plain https URL makes GitHub
    # serve the HTML page and pip fails with "cannot detect archive format".
    install_url = GITHUB_REPO
    if install_url.startswith(("http://", "https://")) and not install_url.startswith("git+"):
        install_url = "git+" + install_url
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", install_url])

try:
    import src.data_pipeline  # noqa: F401 - fail fast, with context, if shadowed
except ModuleNotFoundError:
    print("DIAG sys.path:", sys.path)
    raise

from pathlib import Path
import yaml

print(f"cwd={os.getcwd()} checkout={in_checkout} repo_root={repo_root}")
print("package ready; importable as `src.*`")
"""


def build_pipeline_nb():
    b = nb()
    controls = (
        '''\
# --- Run controls (injected by scripts/make_notebooks.py) -------------------
PHASE = __PHASE__# census | sample | download | manifest
SHARD = __SHARD__# shard filled by the download phase (0..N_SHARDS-1)
DL_BUDGET_MIN = __DL_BUDGET_MIN__# wall-clock budget (minutes) for one download run
N_SHARDS = __N_SHARDS__# matches CampaignConfig + the data-NN datasets

import json, shutil, subprocess, sys
from pathlib import Path

import polars as pl

from src.data_pipeline import CampaignConfig

cfg = CampaignConfig(
    target_species=40_000,
    n_shards=N_SHARDS,
    # Section-9 defaults: min_occurrences=50, facet_page=1000, sample_limit=300,
    # part_species=500, max_images_per_species=150, image_size=320,
    # jpeg_quality=85, max_shard_bytes=18 GB (working-dir guard).
)
print(f"PHASE={PHASE}"
      + (f" SHARD={SHARD:02d} budget={DL_BUDGET_MIN} min" if PHASE == "download" else ""))

# Both Kaggle mount layouts (classic + BYOD), same probe as notebooks 02/03.
MOUNTS = [Path("/kaggle/input"), Path("/kaggle/input/datasets/yoanndev90")]

def _input(slug):
    """Attached dataset root, or None when it is not mounted."""
    for base in MOUNTS:
        p = base / slug
        if p.exists():
            return p
    return None

def _publish(slug, folder, message, title):
    """Push `folder` as the next version of dataset `slug` (Kaggle runs only)."""
    folder = Path(folder)
    if not IS_KAGGLE:
        print(f"local run: {folder} not pushed -> {slug}")
        return
    # The preinstalled CLI (2.0.2) mangles license/keywords on version
    # pushes — upgrade it first (same guard as notebook 02).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"],
                   check=False)
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title,
        "id": slug,
        "licenses": [{"name": "Attribution 4.0 International (CC BY 4.0)"}],
        # Kaggle keywords are a closed tag list: unknown tags are dropped.
        "keywords": ["classification", "research", "python", "beginner"],
    }))
    # -r tar: directories upload uncompressed (images mount as plain files);
    # -t: never rewrite our parquet files to CSV on upload.
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(folder),
         "-m", message, "-r", "tar", "-t", "-q"],
        capture_output=True, text=True,
    )
    print(f"push {slug} rc:", r.returncode, "|",
          (r.stdout + r.stderr).strip() or "ok")
'''
        .replace("__PHASE__", f"{PHASE!r:<26}")
        .replace("__SHARD__", f"{SHARD!r:<26}")
        .replace("__DL_BUDGET_MIN__", f"{DL_BUDGET_MIN!r:<18}")
        .replace("__N_SHARDS__", f"{N_SHARDS!r:<23}")
    )
    b.cells = [
        md(SETUP),
        md("## 1. Setup & configuration"),
        code(INSTALL),
        code(controls),
        md(
            """## 2. `PHASE="census"` — which species are worth sampling

Page GBIF's `speciesKey` facet (Insecta, still images, PRESENT, human or
preserved records, has coordinate, `>= min_occurrences` eligible
occurrences) until a short page, then merge into `meta/census.parquet` plus
the deterministic `meta/shards.parquet` species→shard assignment (balanced
by estimated images, top `target_species`). Publishes the **meta** dataset.

Existing facet pages are skipped: a killed run resumes at the first missing
page. This phase attaches nothing."""
        ),
        code(
            '''\
if PHASE != "census":
    print(f"skipped: PHASE={PHASE!r} (this cell runs the census)")
else:
    from src.data_pipeline import census_sync

    out = Path("meta")
    census_path = census_sync(cfg, out)   # stored pages are never refetched
    census = pl.read_parquet(census_path)
    plan = pl.read_parquet(out / "shards.parquet")
    print(f"census: {census.height} species with >= {cfg.min_occurrences} "
          f"eligible occurrences")
    print(f"shards: {plan.height} species -> {plan['shard'].n_unique()} shards, "
          f"est. {plan['est_images'].sum():,} images")
    _publish("yoanndev90/open-insect-id-v2-meta", out,
             f"census: {census.height} species", "Open Insect ID - Meta")
'''
        ),
        md(
            """## 3. `PHASE="sample"` — one occurrence page per census species

Read `census.parquet` from the attached **meta** dataset and fetch up to
`sample_limit=300` eligible occurrences per species (lineage + image URLs),
500 species per `part-*.parquet` in census order. Existing parts are
skipped, so a killed run resumes at the first missing part; the merged
**meta** dataset is republished with census + sampling."""
        ),
        code(
            '''\
if PHASE != "sample":
    print(f"skipped: PHASE={PHASE!r} (this cell samples GBIF occurrences)")
else:
    from src.data_pipeline import sample_sync

    work = Path("meta")
    src = _input("open-insect-id-v2-meta")
    if src is not None:
        # Full snapshot: census + every part sampled so far.
        shutil.copytree(src, work, dirs_exist_ok=True)
    census_path = work / "census.parquet"
    if not census_path.exists():
        raise SystemExit("census.parquet not found: run PHASE=census first")
    parts = sample_sync(cfg, census_path, work / "sampling")  # parts are skipped
    part_files = sorted(parts.glob("part-*.parquet"))
    n_rows = sum(pl.read_parquet(p).height for p in part_files)
    print(f"sample: {len(part_files)} parts / {n_rows} candidate rows -> {parts}")
    _publish("yoanndev90/open-insect-id-v2-meta", work,
             f"sample: {len(part_files)} parts / {n_rows} rows",
             "Open Insect ID - Meta")
'''
        ),
        md(
            """## 4. `PHASE="download"` — fill one data shard

Materialise the previous version of `open-insect-id-v2-data-NN` into the
working directory (dataset versions are full snapshots), then download this
shard's species from the sampling index: fetch → curate (224-px floor,
aspect ≤ 3) → resize to 320 px → q85 JPEG, content-hash named under
`order/family/genus/species/`, capped at `max_images_per_species`.

Stops cleanly on `DL_BUDGET_MIN` or the 18 GB working-dir guard; the next
run continues, skipping species that are already full without spending
network. Finishes by writing the shard manifest, pushing the shard, then
this shard's manifest part."""
        ),
        code(
            '''\
if PHASE != "download":
    print(f"skipped: PHASE={PHASE!r} (this cell fills data-{SHARD:02d})")
else:
    from src.data_pipeline import SamplingIndex, download_sync, write_shard_manifest

    # Read-only inputs: census + sampling index (attached meta, or local meta/).
    meta = _input("open-insect-id-v2-meta")
    if meta is None and (Path("meta") / "census.parquet").exists():
        meta = Path("meta")
    if meta is None:
        raise SystemExit("meta not found: attach open-insect-id-v2-meta "
                         "(or run census + sample locally first)")

    shard_id = f"open-insect-id-v2-data-{SHARD:02d}"
    work = Path(f"data-{SHARD:02d}")
    src = _input(shard_id)
    if src is not None:
        # Dataset versions are full snapshots: materialise the previous one so
        # this run continues from exactly the state the last run pushed.
        shutil.copytree(src, work, dirs_exist_ok=True)

    plan = pl.read_parquet(meta / "shards.parquet")
    keys = plan.filter(pl.col("shard") == SHARD)["species_key"].to_list()
    have = sum(p.stat().st_size for p in work.rglob("*.jpg")) if work.exists() else 0
    print(f"shard {SHARD:02d}: {len(keys)} species, {have / 1e9:.2f} GB on disk")

    index = SamplingIndex(meta / "census.parquet", meta / "sampling",
                          part_size=cfg.part_species)
    stats = download_sync(cfg, work, keys, index, budget_min=DL_BUDGET_MIN)
    print("download:", stats)

    manifest = write_shard_manifest(work)
    rows = pl.read_parquet(manifest).height
    print(f"shard manifest: {rows} images -> {manifest}")

    # Shard first: a manifest part that lags behind under-lists images (safe —
    # the next run heals it); the reverse order could list missing files.
    _publish(f"yoanndev90/{shard_id}", work,
             f"shard {SHARD:02d}: +{stats['ok']} images ({rows} total)",
             f"Open Insect ID - Data {SHARD:02d}")
    mwork = Path("manifest")
    msrc = _input("open-insect-id-v2-manifest")
    if msrc is not None:
        shutil.copytree(msrc, mwork, dirs_exist_ok=True)
    mwork.mkdir(exist_ok=True)
    shutil.copy(manifest, mwork / f"data-{SHARD:02d}.parquet")
    _publish("yoanndev90/open-insect-id-v2-manifest", mwork,
             f"part data-{SHARD:02d}: {rows} images", "Open Insect ID - Manifest")
'''
        ),
        md(
            """## 5. `PHASE="manifest"` — merge the shard manifests

Read every `data-NN.parquet` part from the attached **manifest** dataset,
merge into `insectid_manifest.parquet` with a `shard` column (the dataset
slug, for multi-root loading), slug-sorted so train/val splits stay
reproducible, and republish the **manifest** dataset. Run this once all
`N_SHARDS` shards report a manifest part."""
        ),
        code(
            '''\
if PHASE != "manifest":
    print(f"skipped: PHASE={PHASE!r} (this cell merges the shard manifests)")
else:
    from src.data_pipeline import merge_manifests

    work = Path("manifest")
    src = _input("open-insect-id-v2-manifest")
    if src is not None:
        shutil.copytree(src, work, dirs_exist_ok=True)
    pairs, missing = [], []
    for i in range(N_SHARDS):
        name = f"data-{i:02d}"
        part = work / f"{name}.parquet"
        if part.exists():
            pairs.append((name, part))
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"missing manifest parts: {missing} — run the download "
                         f"phase for those shards first")
    df = merge_manifests(pairs, work / "insectid_manifest.parquet")
    print(f"merged {len(pairs)} shard parts -> {df.height} images, "
          f"{df['species'].n_unique()} species, {df['order'].n_unique()} orders")
    _publish("yoanndev90/open-insect-id-v2-manifest", work,
             f"merged: {df.height} images / {df['species'].n_unique()} species",
             "Open Insect ID - Manifest")
'''
        ),
    ]
    return b


def build_train_nb():
    b = nb()
    b.cells = [
        md(
            """# Open Insect ID v2 — Train the hierarchical Vision MoE

Set the accelerator to **GPU**. This notebook:
1. loads the curated manifest + images (attached as a Kaggle input dataset),
2. builds the taxonomy tree,
3. constructs the 4-level MoE, the hierarchical + focal + cost-matrix loss,
4. trains with optional Mixup/CutMix and BioCLIP distillation,
5. saves per-level checkpoints.
"""
        ),
        code(INSTALL),
        md("## Config"),
        code(
            '''\
import random

import numpy as np
import torch

from src.models import HierarchicalMoE, TaxonomyTree
from src.training import (
    Trainer, TrainConfig,
    train_transform, eval_transform, build_dataloaders,
)

# One seed for every RNG before anything is built: with deterministic
# algorithms on, this makes full runs reproducible — and, combined with the
# RNG states captured in checkpoints, makes resumed runs seamless.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Run controls (adjust per execution):
RESUME = __RESUME__       # False ignores existing checkpoints (from-scratch runs)
BUDGET_MIN = __BUDGET_MIN__    # wall-clock budget in minutes (0 = unlimited); on expiry
                    # training stops cleanly with a checkpoint to resume from
cfg = TrainConfig(
    epochs=30, lr=1e-3, weight_decay=1e-4,
    level_weights=(8.0, 4.0, 2.0, 1.0),   # order >> species
    focal_gamma=2.0, focal_alpha=0.25,    # long-tail on genus/species
    lambda_cost=1.0,                      # taxonomic cost-matrix strength
    mixup_alpha=0.4,
    device=device,
    # Chunked training: checkpoint every 5 epochs and stop cleanly once the
    # wall-clock budget is spent (guards the ~9h kernel limit and the GPU
    # quota). The next run resumes from checkpoints/last.pt automatically.
    ckpt_dir="checkpoints", ckpt_every=5, max_minutes=BUDGET_MIN,
)
print("device:", device)
'''.replace("__RESUME__", str(RESUME)).replace("__BUDGET_MIN__", str(BUDGET_MIN))
        ),
        md("## Load manifest + build tree"),
        code(
            '''\
# On Kaggle, point these at your attached input dataset. Kaggle mounts inputs
# at /kaggle/input/<slug> on the classic runtime and at
# /kaggle/input/datasets/<owner>/<slug> on the newer BYOD runtime: probe both.
from pathlib import Path

MOUNTS = [Path("/kaggle/input"), Path("/kaggle/input/datasets/yoanndev90")]
def _input(*rel):
    for base in MOUNTS:
        p = base.joinpath("open-insect-id-v2", *rel)
        if p.exists():
            return p
    return None

MANIFEST, IMAGE_ROOT = _input("insectid_manifest.parquet"), _input("images")
if MANIFEST is None:  # local run
    MANIFEST, IMAGE_ROOT = Path("insectid_manifest.parquet"), Path("data/raw")

train_loader, val_loader, tree = build_dataloaders(
    MANIFEST, IMAGE_ROOT, batch_size=128,
    val_fraction=0.1, num_workers=2,
    transform=train_transform(image_size=224), seed=42,
)
print("orders:", len(tree.orders), "species:", tree.num_species)
'''
        ),
        md("## Build the MoE + loss"),
        code(
            '''\
model = HierarchicalMoE(
    tree, backbone_name="mnv4_small",
    tau_order=0.85, tau_levels=(0.85, 0.80, 0.70, 0.50),
)
n_params = sum(p.numel() for p in model.parameters())
n_trunk = sum(p.numel() for p in model.trunk.parameters())
print(f"cascade parameters: {n_params/1e6:.2f}M "
      f"(trunk {n_trunk/1e6:.2f}M + heads {(n_params-n_trunk)/1e6:.2f}M)")
print(f"classes: {model.n_orders} orders / {model.n_families} families / "
      f"{model.n_genera} genera / {model.n_species} species")
'''
        ),
        md(
            """## Distillation (optional)

Supply a BioCLIP teacher to transfer biologically-informed visual features.
Requires `pip install transformers` and a GPU. Leave `teacher = None` to skip.
"""
        ),
        code(
            '''\
teacher = None  # set to True to attempt loading BioCLIP (needs transformers + GPU)
if teacher:
    from src.training.distillation import load_bioclip_teacher
    teacher = load_bioclip_teacher("imageomics/bioclip", device=device)
else:
    teacher = None
print("distillation:", "enabled" if teacher else "disabled")
'''
        ),
        md(
            """## Train

Resumable: a previous interrupted run continues where it stopped. The state
is looked up in this kernel's working directory first, then in the mounted
checkpoints dataset. Delete `checkpoints/last.pt` (or detach the dataset) to
force a from-scratch run."""
        ),
        code(
            '''\
trainer = Trainer(model, cfg, teacher=teacher)

from pathlib import Path
# Resume source: this kernel's working dir first, then the mounted
# checkpoints dataset (both mount layouts probed, like the manifest above).
candidates = [Path("checkpoints/last.pt")] + [
    base / "open-insect-id-checkpoints" / "last.pt" for base in MOUNTS
]
resume = (
    next((p for p in candidates if p.exists()), None)
    if RESUME
    else None
)
if resume is not None:
    start = trainer.load_checkpoint(resume)
    if start > 0:
        print(f"resumed from {resume} -> will run epochs {start+1}..{cfg.epochs}")
    else:
        print(f"no usable checkpoint at {resume}: training epochs 1..{cfg.epochs}")
else:
    print("no checkpoint found: training from scratch")

history = trainer.fit(train_loader, val_loader)
print("epochs trained this run:", len(history["train"]))
'''
        ),
        md("## Save checkpoints"),
        code(
            '''\
from pathlib import Path
out = Path("checkpoints"); out.mkdir(exist_ok=True)
torch.save(model.state_dict(), out / "moe_state.pt")
tree.save(out / "taxonomy.json")
print("saved:", sorted(p.name for p in out.iterdir()))

# Content digest of the final weights: an interrupted + resumed run must
# print the same sha256 as an uninterrupted one (compare across kernel logs
# — no need to download any artifact).
import hashlib
sd = model.state_dict()
h = hashlib.sha256()
for k in sorted(sd):
    v = sd[k].detach().cpu().contiguous()
    h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(v.numpy().tobytes())
print("model sha256:", h.hexdigest())

# On Kaggle, push the run (final weights + resumable last.pt) to the
# checkpoints dataset consumed by export notebook 03 and by this notebook's
# resume logic on any other machine. `kaggle datasets version` requires the
# metadata file inside the folder, so write it right before pushing. The
# preinstalled CLI (2.0.2) mangles the dataset's license and keywords on
# version pushes — upgrade it first.
import json, subprocess, sys
if IS_KAGGLE and (out / "last.pt").exists():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"],
        check=False,
    )
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Open Insect ID - Checkpoints",
        "id": "yoanndev90/open-insect-id-checkpoints",
        "licenses": [{"name": "Attribution 4.0 International (CC BY 4.0)"}],
        "keywords": ["classification", "research", "python", "beginner"],
    }))
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", "checkpoints",
         "-m", f"02-train-moe: {len(history['train'])} epoch(s) this run", "-q"],
        capture_output=True, text=True,
    )
    print("push rc:", r.returncode, "|", (r.stdout + r.stderr).strip() or "ok")
else:
    print("no local last.pt to push (local run, or resume came from the mounted dataset)")
'''
        ),
    ]
    return b


def build_export_nb():
    b = nb()
    b.cells = [
        md(
            """# Open Insect ID v2 — Export to ONNX / INT8 / TFLite

CPU notebook. Loads the trained cascade, exports the trunk + the 4 heads to
ONNX, applies INT8 post-training quantization to the trunk, and converts to
`.tflite` for the Android app (heads ship as FP16 matrices). Attach the
checkpoint output-dataset from notebook 02.
"""
        ),
        code(INSTALL),
        code(
            '''\
# Export stack. onnx2tf (>=2.x) pins its own fully-py3.13-compatible stack
# (onnx, onnxruntime, onnxscript, ai-edge-litert), so a single install covers
# INT8 PTQ and the ONNX -> TFLite conversion; onnxscript also serves the
# dynamo exporter recent torch builds use by default.
import importlib.util, subprocess, sys

missing = [m for m in ("onnx", "onnxruntime", "onnxscript", "onnx2tf")
           if importlib.util.find_spec(m) is None]
if missing:
    print("installing:", missing)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "onnx2tf"], check=True
    )
else:
    print("export stack already present")
'''
        ),
        md("## Load the trained model"),
        code(
            '''\
import torch
from pathlib import Path
from src.models import HierarchicalMoE, TaxonomyTree

# Probe both Kaggle mount layouts, then fall back to a local run.
MOUNTS = [Path("/kaggle/input"), Path("/kaggle/input/datasets/yoanndev90")]
_ckpt_dir = next(
    (base / "open-insect-id-checkpoints"
     for base in MOUNTS if (base / "open-insect-id-checkpoints" / "moe_state.pt").exists()),
    Path("checkpoints"),
)
CKPT = str(_ckpt_dir / "moe_state.pt")
TAX = str(_ckpt_dir / "taxonomy.json")

tree = TaxonomyTree.load(TAX)
model = HierarchicalMoE(tree, backbone_name="mnv4_small")
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
print("loaded MoE:", model.n_orders, "orders /", model.n_species, "species")
'''
        ),
        md("## Export per-module ONNX"),
        code(
            '''\
from src.export import export_moe
from pathlib import Path

# 5 modules: trunk (image -> features) + one head per level (features ->
# global logits); manifest.json carries labels + child ranges for slicing.
onnx_dir = Path("export/onnx"); onnx_dir.mkdir(parents=True, exist_ok=True)
paths = export_moe(model, onnx_dir, image_size=224)
print(f"exported {len(paths)} modules -> {onnx_dir}")
print("manifest:", (onnx_dir / "manifest.json").exists())
'''
        ),
        md("## INT8 post-training quantization of the trunk (calibration set)"),
        code(
            '''\
from src.export import quantize_static_int8, quantize_dynamic_int8
import torch

# Calibration images: a representative batch from your validation set.
# (Only the trunk consumes images; the heads consume the trunk's features.)
calib = torch.randn(64, 3, 224, 224)

qdir = Path("export/onnx_int8"); qdir.mkdir(parents=True, exist_ok=True)
module_name = "trunk"
module_onnx = onnx_dir / f"{module_name}.onnx"
print("quantizing:", module_name)
q = quantize_static_int8(module_onnx, qdir / f"{module_name}_int8.onnx", calib)
print("quantized ->", q)
'''
        ),
        md(
            """## Convert to TFLite

Uses the `onnx2tf` CLI (installed by the setup cell above: since v2.x it
pins its own py3.13-compatible stack — no tensorflow or tflite-support
needed). If it is absent the converter prints the exact install command
instead of failing.
"""
        ),
        code(
            '''\
from src.export import TFLiteINT8Converter

conv = TFLiteINT8Converter()
tflite_dir = Path("export/tflite"); tflite_dir.mkdir(parents=True, exist_ok=True)
if conv.available():
    # Convert the trunk's INT8 (QDQ) ONNX: onnx2tf maps it to INT8 .tflite.
    tfl = conv.convert(qdir / f"{module_name}_int8.onnx", tflite_dir, int8=True)
    print("tflite ->", tfl)
else:
    print("onnx2tf not installed. Install with:")
    print("  !pip install onnx2tf")
    print("then re-run this cell.")
'''
        ),
        md(
            """## Export the heads as FP16 matrices (option B)

The heads are pure `Linear`s: ship them as raw `(W, b)` matrices instead of
TFLite graphs — 4 tiny matmuls on-device, no head quantization error at 40k
logits, and no extra interpreters. Weights are exported per level as FP16.
"""
        ),
        code(
            '''\
import numpy as np

heads_dir = Path("export/heads"); heads_dir.mkdir(parents=True, exist_ok=True)
for level in ("order", "family", "genus", "species"):
    head = getattr(model.heads, level)
    if head is None:
        continue  # single-order tree: no order head
    np.savez(
        heads_dir / f"head_{level}.npz",
        weight=head.weight.detach().cpu().numpy().astype(np.float16),
        bias=head.bias.detach().cpu().numpy().astype(np.float16),
    )
    print(f"head_{level}: W{tuple(head.weight.shape)} b{tuple(head.bias.shape)} -> fp16")
print("heads ->", heads_dir)
'''
        ),
        md(
            """## Package the artifact

Bundle the INT8 trunk `.tflite`, the FP16 head matrices, `manifest.json`
(labels + child ranges for on-device slicing), and `taxonomy.json` into a
Kaggle dataset (or zip for GitHub Releases). On-device: run the trunk once,
then descend the cascade with 4 head matmuls.
"""
        ),
        code(
            '''\
import shutil, json
bundle = Path("export/open_insect_id_bundle")
bundle.mkdir(parents=True, exist_ok=True)
for src in [Path("export/tflite"), Path("export/heads"), onnx_dir / "manifest.json"]:
    if src.exists():
        if src.is_dir():
            shutil.copytree(src, bundle / src.name, dirs_exist_ok=True)
        else:
            shutil.copy(src, bundle / src.name)
shutil.copy(TAX, bundle / "taxonomy.json")
shutil.make_archive("open_insect_id_tflite", "zip", root_dir=bundle)
print("bundle:", list(p.name for p in bundle.iterdir()))
'''
        ),
    ]
    return b


def _slugify(title: str) -> str:
    """Kaggle derives a kernel's URL slug from its title (lowercase, dashes)."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def _pipeline_inputs(phase: str, shard: int, n_shards: int) -> list:
    """dataset_sources for notebook 01 — one attach set per campaign phase.

    Every referenced dataset must already exist (they are pre-created empty),
    otherwise Kaggle rejects the kernel submission before any cell runs.
    """
    meta = "yoanndev90/open-insect-id-v2-meta"
    manifest = "yoanndev90/open-insect-id-v2-manifest"
    if phase == "census":
        return []  # reads nothing; writes and pushes the meta dataset
    if phase == "sample":
        return [meta]
    if phase == "download":
        return [meta, f"yoanndev90/open-insect-id-v2-data-{shard:02d}", manifest]
    if phase == "manifest":
        return [manifest]
    raise ValueError(f"unknown PHASE {phase!r} (census|sample|download|manifest)")


def _kernel_metadata(title: str, code_file: str, gpu: bool, inputs: list) -> dict:
    """kernel-metadata.json for `kaggle kernels push -p <folder>`.

    enable_internet must be true: the notebooks pip-install this package from
    GitHub in their setup cell. The id must match the title-derived slug,
    otherwise the CLI warns and a re-push risks forking a second kernel.
    Kernels are created private by default.
    """
    return {
        "id": f"yoanndev90/{_slugify(title)}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": inputs,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main():
    if not 0 <= SHARD < N_SHARDS:
        raise ValueError(f"SHARD must be in 0..{N_SHARDS - 1}, got {SHARD}")
    # Fails fast on a typo'd phase, before anything is written.
    pipeline_inputs = _pipeline_inputs(PHASE, SHARD, N_SHARDS)

    # slug -> (builder, human title, needs GPU)
    kernels = {
        "01-data-pipeline": (
            lambda: build_pipeline_nb(),
            "Open Insect ID v2 - Data Pipeline",
            False,
            pipeline_inputs,
        ),
        "02-train-moe": (
            lambda: build_train_nb(),
            "Open Insect ID v2 - Train Vision MoE",
            True,
            # The checkpoints dataset doubles as resume source (mounted
            # last.pt) and as the output consumed by notebook 03.
            [
                "yoanndev90/open-insect-id-v2",
                "yoanndev90/open-insect-id-checkpoints",
            ],
        ),
        "03-export-tflite": (
            lambda: build_export_nb(),
            "Open Insect ID v2 - Export ONNX INT8 TFLite",
            False,
            ["yoanndev90/open-insect-id-checkpoints"],
        ),
    }

    # Drop any stale flat notebooks from the previous layout.
    for old in NB_DIR.glob("*.ipynb"):
        old.unlink()

    for slug, (builder, title, gpu, inputs) in kernels.items():
        folder = NB_DIR / slug
        folder.mkdir(parents=True, exist_ok=True)
        b = builder()
        # Deterministic cell ids: nbformat would mint fresh random ids on
        # every regeneration, dirtying all notebooks in git for no reason.
        for i, cell in enumerate(b.cells):
            cell["id"] = f"c{i:02d}"
        b.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        b.metadata["language_info"] = {"name": "python", "version": "3.10"}

        code_file = f"{slug}.ipynb"
        nbf.write(b, folder / code_file)
        import json

        (folder / "kernel-metadata.json").write_text(
            json.dumps(_kernel_metadata(title, code_file, gpu, inputs), indent=2)
        )
        print("wrote", folder / code_file, "+ kernel-metadata.json")


if __name__ == "__main__":
    main()
