"""Generate the Kaggle-oriented Jupyter notebooks for Open Insect ID v2.

Run once: python scripts/make_notebooks.py
"""

import re

import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def nb():
    return nbf.v4.new_notebook()


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


SETUP = """\
# Open Insect ID v2 — Kaggle pipeline
#
# This notebook is the runnable, on-Kaggle entry point. It installs the
# `src` package from GitHub and exercises the full data pipeline. Set the
# accelerator to **CPU** for the data stages (GPU is not needed) and use the
# training notebook (02) with a **GPU**.
#
# Fill in `GITHUB_REPO` below with your repository URL (or use the local copy
# already in the Kaggle notebook's working directory).
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
    b.cells = [
        md(SETUP),
        md("## 1. Setup & configuration"),
        code(INSTALL),
        code(
            '''\
# Load the data pipeline configuration (filters, H3 resolution, curation).
from src.data_pipeline import GBIFConfig, H3Config, CurationConfig, DownloadConfig

gbif_cfg = GBIFConfig(
    class_key=216,               # Insecta
    media_type="StillImage",
    occurrence_status="PRESENT",
    basis_of_record=("HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"),
    has_coordinate=True,
)
h3_cfg = H3Config(resolution=4, min_occurrences_per_cell=3)
cur_cfg = CurationConfig(min_resolution=224, max_aspect_ratio=3.0)
dl_cfg = DownloadConfig(concurrency=8, per_domain_rps=4.0, raw_root="data/raw")
print("Insecta, world-scale, H3 res", h3_cfg.resolution)
'''
        ),
        md(
            """## 2. Ingest occurrences (GBIF)

Two interchangeable sources:
- **A. Async API** (`GBIFExtractor`) — best for a targeted slice / prototyping.
- **B. DuckDB over a bulk TSV** (`SpatialProcessor.load_occurrences`) — best for
  the full world-scale dump (tens of millions of rows) streamed off disk.
"""
        ),
        code(
            '''\
# --- Option A: async GBIF occurrence search (prototype / bounded slice) ------
from src.data_pipeline import GBIFExtractor

extractor = GBIFExtractor(gbif_cfg)
# Limit records for a quick pass; drop `max_records=None` to page the full set.
records = extractor.extract_sync(max_records=500)
print(f"fetched {len(records)} image-bearing occurrences")
for r in records[:3]:
    print(" ", r.lineage.order, "/", r.lineage.family, "/", r.lineage.genus, "/", r.lineage.species)
'''
        ),
        code(
            '''\
# --- Option B: spatial clean + H3 range mask (world-scale TSV) --------------
# Point `tsv_path` at a GBIF occurrence dump (TSV/CSV) if you have one.
from src.data_pipeline import SpatialProcessor

# sp = SpatialProcessor(h3_cfg)
# occ = sp.load_occurrences("data/gbif_insecta.tsv")   # streams via DuckDB
# clean, stats = sp.clean(occ)                          # builds H3 mask + flags outliers
# print(stats)
print("Uncomment to run DuckDB spatial cleaning on a full GBIF dump.")
'''
        ),
        md("## 3. Download images (async, fault-tolerant)"),
        code(
            '''\
from src.data_pipeline import ImageDownloader

downloader = ImageDownloader(dl_cfg)
# Small test slice: only the first SMALL_N records are downloaded.
# Raise it (or use records directly) for a real run.
SMALL_N = 10
result = downloader.download_sync(records[:SMALL_N])
print(f"from {SMALL_N} records -> ok={result.ok} failed={result.failed} skipped(dup)={result.skipped}")
'''
        ),
        md("## 4. Curation (integrity / resolution / aspect ratio)"),
        code(
            '''\
from src.data_pipeline import curate_directory

verdicts = curate_directory("data/raw", cur_cfg)
ok = sum(v.ok for v in verdicts)
print(f"{ok}/{len(verdicts)} images pass curation")
for v in verdicts:
    if not v.ok:
        print("  drop:", v.path, "->", v.reason)
'''
        ),
        md(
            """## 5. Export the curated manifest

Write a Parquet manifest consumed by the training notebook. Schema:
`image_path, order, family, genus, species` (+ optional `h3_cell, lat, lon`).
Lineage is re-derived from the curated file tree. On Kaggle, push this as a
Dataset so notebook 02 can attach it as input.
"""
        ),
        code(
            '''\
import polars as pl
from pathlib import Path

RAW = Path("data/raw")
schema = ["image_path", "order", "family", "genus", "species"]
rows = []
for v in verdicts:
    if not v.ok:
        continue
    parts = Path(v.path).relative_to(RAW).parts  # order/family/genus/species/hash.jpg
    if len(parts) == 5:
        o, f, g, s = parts[:4]
        rows.append({"image_path": "/".join(parts), "order": o,
                     "family": f, "genus": g, "species": s})

manifest = pl.DataFrame(rows, schema={c: pl.Utf8 for c in schema})
manifest.write_parquet("insectid_manifest.parquet")
print(f"manifest: {manifest.height} curated rows -> insectid_manifest.parquet")
# If running on Kaggle, publish manifest + images into the curated dataset
# (folder must hold dataset-metadata.json with id yoanndev90/open-insect-id-v2):
#   !kaggle datasets version -m "curated run" -p <dataset-folder> --dir-mode tar
# notebook 02 already attaches that dataset as an input.
manifest.head(5)
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
import torch

from src.models import HierarchicalMoE, TaxonomyTree
from src.training import (
    Trainer, TrainConfig,
    train_transform, eval_transform, build_dataloaders,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = TrainConfig(
    epochs=30, lr=1e-3, weight_decay=1e-4,
    level_weights=(8.0, 4.0, 2.0, 1.0),   # order >> species
    focal_gamma=2.0, focal_alpha=0.25,    # long-tail on genus/species
    lambda_cost=1.0,                      # taxonomic cost-matrix strength
    mixup_alpha=0.4,
    device=device,
)
print("device:", device)
'''
        ),
        md("## Load manifest + build tree"),
        code(
            '''\
# On Kaggle, point these at your attached input dataset.
MANIFEST = "/kaggle/input/open-insect-id-v2/insectid_manifest.parquet"
IMAGE_ROOT = "/kaggle/input/open-insect-id-v2/images"

from pathlib import Path
if not Path(MANIFEST).exists():
    MANIFEST = "insectid_manifest.parquet"
    IMAGE_ROOT = "data/raw"

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
print(f"MoE parameters: {n_params/1e6:.2f}M across "
      f"{len(model.family_experts)} family experts, "
      f"{len(model.genus_experts)} genus experts, "
      f"{len(model.species_experts)} species experts")
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
        md("## Train"),
        code(
            '''\
trainer = Trainer(model, cfg, teacher=teacher)
history = trainer.fit(train_loader, val_loader)
'''
        ),
        md("## Save checkpoints"),
        code(
            '''\
from pathlib import Path
out = Path("checkpoints"); out.mkdir(exist_ok=True)
torch.save(model.state_dict(), out / "moe_state.pt")
tree.save(out / "taxonomy.json")
print("saved:", list(p.name for p in out.iterdir()))

# On Kaggle, publish the checkpoint directory as an output dataset so the
# export notebook (03) can attach it:
#   !kaggle datasets create -p checkpoints --dir-mode tar -u
# (slug: yoanndev90/open-insect-id-checkpoints)
'''
        ),
    ]
    return b


def build_export_nb():
    b = nb()
    b.cells = [
        md(
            """# Open Insect ID v2 — Export to ONNX / INT8 / TFLite

CPU notebook. Loads the trained MoE, exports every router + expert to ONNX,
applies INT8 post-training quantization, and converts to `.tflite` for the
Android app. Attach the checkpoint output-dataset from notebook 02.
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
from src.models import HierarchicalMoE, TaxonomyTree

CKPT = "/kaggle/input/open-insect-id-checkpoints/moe_state.pt"
TAX  = "/kaggle/input/open-insect-id-checkpoints/taxonomy.json"
if not Path(CKPT).exists():
    CKPT, TAX = "checkpoints/moe_state.pt", "checkpoints/taxonomy.json"

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

onnx_dir = Path("export/onnx"); onnx_dir.mkdir(parents=True, exist_ok=True)
paths = export_moe(model, onnx_dir, image_size=224)
print(f"exported {len(paths)} modules -> {onnx_dir}")
print("manifest:", (onnx_dir / "manifest.json").exists())
'''
        ),
        md("## INT8 post-training quantization (with calibration set)"),
        code(
            '''\
from src.export import quantize_static_int8, quantize_dynamic_int8
import torch

# Calibration images: a representative batch from your validation set.
calib = torch.randn(64, 3, 224, 224)

qdir = Path("export/onnx_int8"); qdir.mkdir(parents=True, exist_ok=True)
# A single-order smoke tree exports no router: quantize the first exported
# module instead so the INT8 + TFLite steps still exercise the full chain.
module_name = "router" if "router" in paths else sorted(paths)[0]
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
    # Convert the INT8 (QDQ) ONNX: onnx2tf maps it to a full-INT8 .tflite.
    tfl = conv.convert(qdir / f"{module_name}_int8.onnx", tflite_dir, int8=True)
    print("tflite ->", tfl)
else:
    print("onnx2tf not installed. Install with:")
    print("  !pip install onnx2tf")
    print("then re-run this cell.")
'''
        ),
        md(
            """## Package the artifact

Bundle `export/tflite/`, `manifest.json`, and `taxonomy.json` into a Kaggle
dataset (or zip for GitHub Releases). The Android app loads the router first,
then fetches only the experts along the predicted taxonomic path (Option A).
"""
        ),
        code(
            '''\
import shutil, json
bundle = Path("export/open_insect_id_bundle")
bundle.mkdir(parents=True, exist_ok=True)
for src in [Path("export/tflite"), onnx_dir / "manifest.json"]:
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
    # slug -> (builder, human title, needs GPU)
    kernels = {
        "01-data-pipeline": (
            lambda: build_pipeline_nb(),
            "Open Insect ID v2 - Data Pipeline",
            False,
            [],
        ),
        "02-train-moe": (
            lambda: build_train_nb(),
            "Open Insect ID v2 - Train Vision MoE",
            True,
            ["yoanndev90/open-insect-id-v2"],
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
