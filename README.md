# Open Insect ID v2

On-device insect identification built on a **hierarchical vision cascade** —
a shared MobileNetV4 trunk feeding 4 taxonomic heads (Order → Family → Genus →
Species), trained on global GBIF data and exported to **TFLite** for
Android/Kotlin inference.

This repository contains the **model + data + training + export** Python package and
its **Kaggle notebooks**. The Android app lives in a separate repository.

## Why a hierarchical cascade instead of flat classification?

A single softmax over tens of thousands of insect species is dominated by the
long tail and forces every image through one giant classifier. Instead we run
one small backbone and descend a cascade of per-parent heads, each only
discriminating between **siblings**:

```mermaid
flowchart TD
    IMG([Image])

    subgraph TRUNK["Shared trunk — one MobileNetV4 pass"]
        T[Trunk<br/>features z]
    end

    subgraph HEADS["4 level heads (linear)"]
        R[Order head]
        FA[Family head<br/>sliced to the predicted order]
        G[Genus head<br/>sliced to the predicted family]
        S[Species head<br/>sliced to the predicted genus]
    end

    AMB[Two candidate paths<br/>weighted by order probability]
    UNC[Order-level<br/>uncertainty]
    ANSWER([Deepest accepted level = the answer])

    IMG --> T
    T --> R
    R -->|confidence &ge; τ₁ — Top-1| FA
    R -.->|confidence &lt; τ₁ — keep Top-2 orders| AMB
    R -.->|entropy &gt; τ_level| UNC
    AMB -.-> FA
    FA --> G
    G --> S
    S --> ANSWER
```

Each head only discriminates between the children of one parent node, so
sub-problems stay small and edge-friendly, while the shared trunk keeps the
model to ~4M backbone parameters regardless of species count (at 40k species
the whole cascade is ~60M parameters, versus ~185B for a backbone per parent).
A confidence/entropy check at every level lets the pipeline stop early and
return the deepest *certain* taxon instead of guessing — the `-.-` edges above
are the fallback paths (Top-2 order routing and early rejection) that make the
cascade robust to a wrong first guess.

## Repository layout

```
configs/               # data + model YAML (GBIF filters, H3, backbones, τ thresholds)
src/
  data_pipeline/       # GBIF ingestion, DuckDB/H3 spatial cleaning, async image
                       #   download, automated curation
  models/              # MobileNetV4 UIB backbones, taxonomy tree, cascade heads,
                       #   Top-K router, the 4-level HierarchicalMoE cascade
  training/            # hierarchical CE + focal + taxonomic cost-matrix loss,
                       #   BioCLIP distillation, dataset, Mixup/CutMix, Trainer
  export/              # trunk+heads ONNX, INT8 PTQ, TFLite conversion
notebooks/             # runnable Kaggle entry points (data → train → export)
tests/                 # unit tests covering all of the above
scripts/               # notebook generator
```

## Backbones

The default on-device backbone is a **MobileNetV4-style UIB stack**
(`mnv4_small` / `mnv4_medium`) — depthwise-separable, quantization-friendly,
strong accuracy/latency on CPU/NPU.

**On MobileNetV5** (released June 2025): it is a single ~300M-parameter encoder
built for Gemma 3n (768×768 inputs, sparse attention, RMSNorm/GELU). The only
public pretrained weights are under the **Gemma license** and its classification
heads are marked WIP. It is not a drop-in replacement for the compact on-device
trunk, so MobileNetV4 remains the default. A `timm:<model>` seam is provided for
experimentation:

```python
from src.models import build_backbone
backbone = build_backbone("timm:mobilenetv5_base")   # requires `pip install timm`
```

## Quickstart

### Run the tests (CPU)

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q
```

### Run on Kaggle (the intended workflow)

Each folder under `notebooks/` is one pushable Kaggle kernel (notebook +
`kernel-metadata.json`):

| Notebook | Purpose | Accelerator |
|----------|---------|-------------|
| `01-data-pipeline.ipynb` | data campaign: GBIF census → sampling → shard download (curate + 320-px resize) → merged manifest | CPU |
| `02-train-moe.ipynb` | build tree → MoE + loss → train (± distillation) → checkpoints | **GPU** |
| `03-export-tflite.ipynb` | load checkpoints → trunk+heads ONNX → INT8 trunk + FP16 heads → `.tflite` bundle | CPU |

Each notebook installs the package from GitHub in its setup cell
(`GITHUB_REPO`). Artifacts flow through Kaggle datasets: the smoke dataset
[yoanndev90/open-insect-id-v2](https://www.kaggle.com/datasets/yoanndev90/open-insect-id-v2)
(curated manifest + images),
[yoanndev90/open-insect-id-checkpoints](https://www.kaggle.com/datasets/yoanndev90/open-insect-id-checkpoints)
(trained `moe_state.pt` + `taxonomy.json`, published by `02` on its first
run), and — for the 40k-species campaign — `open-insect-id-v2-meta` (census +
sampling), `open-insect-id-v2-data-00..11` (image shards) and
`open-insect-id-v2-manifest` (per-shard manifest parts + the merged training
manifest), all published by `01` depending on its `PHASE`. Notebook `02`
consumes the manifest + shards, `03` the checkpoints.

Training is **resumable**: `02` checkpoints model + optimizer every 5 epochs
(`checkpoints/last.pt`, also pushed to the checkpoints dataset) and stops
cleanly once its wall-clock budget is spent, so a run cut short by the kernel
limit or by GPU quota continues on the next execution instead of restarting.

### Use as a library

```python
from src.models import HierarchicalMoE, TaxonomyTree
from src.training import CombinedTaxonomicLoss, build_dataloaders
from src.export import export_moe

tree = TaxonomyTree.load("taxonomy.json")
model = HierarchicalMoE(tree, backbone_name="mnv4_small",
                        tau_order=0.85, tau_levels=(0.85, 0.80, 0.70, 0.50))
res = model.route(image_tensor)      # inference on one image
print(res.describe())                # deepest certain taxon + confidence
```

## Key ideas from the engineering plan

- **Hierarchical loss** — per-level weighted CE so an *order* mistake costs far
  more than a *species* mistake, plus a **taxonomic cost-matrix** penalty driven
  by tree distance (LCA depth) between predicted and true species.
- **Focal loss** on genus/species to survive the long-tail distribution.
- **Top-K order routing + per-level rejection** to avoid cascade errors and to
  report "Genus certain, species uncertain" style fallbacks.
- **DuckDB + Uber H3** for world-scale, RAM-safe occurrence cleaning and spatial
  outlier (mislabelled / lab specimen) removal.
- **BioCLIP distillation** to transfer biologically-informed features into the
  shared trunk (optional, needs `transformers` + GPU).
- **INT8 PTQ + TFLite** for NNAPI/Hexagon on-device inference.

## Dataset

The curated, world-scale insect image dataset is published on Kaggle — see
[`docs/dataset_card.md`](docs/dataset_card.md).

## License

MIT (see `LICENSE`). Note: any weights distilled from BioCLIP or MobileNetV5
inherit their upstream licenses — check before redistributing.
