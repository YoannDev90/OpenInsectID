# Architecture detail: shared trunk + per-parent heads

Status: **proposal awaiting validation** (user decision "detail the architecture
first"). Target scale: **40,000 species x up to 150 images each**.

## 1. Why the current design cannot reach 40k species

Today `HierarchicalMoE` (src/models/moe.py) instantiates one **full expert per
parent node** — `Expert` = complete MobileNetV4 backbone + linear classifier
(src/models/experts.py:28). Measured cost of one `mnv4_small` expert:
**4,367,480 parameters** (measured locally, feature dim D = 1024).

At GBIF-like taxonomy proportions (~30 orders, ~1,000 families, ~13,000
genera, 40,000 species), parents with >= 2 children ≈ 28 orders-routers +
~950 families + ~11,500 genera + ~30,000 species experts:

| Level | Experts x 4.37 M | Params |
|---|---|---|
| order router | 1 | 4.4 M |
| family | ~950 | ~4.2 B |
| genus | ~11,500 | ~50 B |
| species | ~30,000 | ~131 B |
| **total** | ~42,500 | **~185 B** |

~185 B fp32 parameters = ~740 GB of weights alone; a single training step also
needs ~3 backbone passes per image (router + dispatch at each level, moe.py:261
+ moe.py:263-271), and export would emit ~42,500 ONNX files. Full experts are
only viable at <= ~1k species. Coverage requires a refactor; capacity per class
must come from linear discriminators on a strong shared feature space instead
of from a private backbone per genus.

## 2. Design overview

```
                       ┌─────────────────────────┐
 image (B,3,224,224) ─▶│  ONE MobileNetV4 trunk  │──▶ z (B,1024)
                       │  (mnv4_small, 4.37 M)   │
                       └─────────────────────────┘
                                 │
      ┌──────────────┬───────────┼───────────────┬──────────────┐
      ▼              ▼           ▼               ▼              │
 Linear(D, n_ord) Linear(D,n_fam) Linear(D,n_gen) Linear(D,n_sp)│  heads (~55 M)
      │              │           │               │              │
   order logits   family logits genus logits   species logits   │
                (masked per-row by parent range, fill _NEG)     │
```

- **One shared trunk** (`self.trunk`) runs once per image (or once per
  augmented pair, see 6).
- **Four global linear heads** — one per taxonomic level:
  `Linear(D, n_level)`. A class weight is a *row* of the level head. Because the
  global label space lays children out contiguously per parent (the same layout
  `_build_level` builds today, moe.py:180-198), each parent owns a **contiguous
  child range** `[start, end)` of the head rows.
- **Training masking** reproduces today's `_NEG` fill exactly: per row, classes
  outside the ground-truth parent's range are set to `-1e4` (moe.py:31). The
  hierarchical loss consumes the same `(B, n_level_global)` tensors as before —
  `CombinedTaxonomicLoss` (src/training/losses.py:142) and
  `TaxonomicCostMatrix` need **zero changes**.

Capacity equivalence: a per-parent `Linear(D, k)` is mathematically a
column-submatrix of the global head. The refactor changes *nothing* about
per-class capacity; it removes the redundant per-parent backbones.

## 3. Modules, names, files

| File | Symbol | Role |
|---|---|---|
| src/models/backbones.py | `MobileNetV4`, `build_backbone` | unchanged |
| src/models/heads.py **(new)** | `CascadeHeads(nn.Module)` | holds the 4 level heads + range buffers |
| src/models/moe.py | `HierarchicalMoE` | **same public class name**; reimplemented on `trunk + CascadeHeads` |
| src/models/experts.py | `Expert` | **deleted** (unused after refactor) |

`CascadeHeads` (src/models/heads.py):

```python
class CascadeHeads(nn.Module):
    # Registered parameters: order: Linear(D, n_orders)
    #                        family: Linear(D, n_families)
    #                        genus:   Linear(D, n_genera)
    #                        species: Linear(D, n_species)
    # Registered buffers per level:
    #   ranges: LongTensor (n_parents + 1)  -- parent p's children occupy
    #          rows [ranges[p], ranges[p+1]) of the head AND of the label space
    # Optional: dropout on z before the linears (parity with Expert's dropout)

    def forward(self, z, targets=None) -> dict[str, torch.Tensor]
        # targets given (training): per-row range mask, invalid classes -> _NEG
        # targets None (inference): raw logits, sliced per parent by route()
```

Constructor of `HierarchicalMoE` keeps its signature
(tree, backbone_name, width_mult, tau_order, tau_levels, top_k_orders) plus one
new flag `head_dim: int | None = None` (see 7). All attributes the trainer,
losses and export depend on are preserved verbatim: `tree`, `order_names`,
`family_names`, `genus_names`, `species_names`, `n_orders/families/genera/
species`, `_level_index`, `tau_*`, `top_k_orders`.

## 4. Training forward — shapes and masking

```
x       : (B, 3, 224, 224)
z       = trunk(x)                      : (B, D),  D = 1024 (mnv4_small)
order   = head_order(z)                 : (B, n_orders)          ~30
family  = mask(head_family(z), targets["order"])
        : (B, n_families)               ~1,000
genus   = mask(head_genus(z), targets["family"])
        : (B, n_genera)                 ~13,000
species = mask(head_species(z), targets["genus"])
        : (B, n_species)                ~40,000
```

Masking is vectorized — no Python loop over parents (replaces the
`_dispatch` row grouping, moe.py:204-245):

```python
starts = ranges[targets_level]              # (B,)
ends   = ranges[targets_level + 1]          # (B,)
valid  = (col >= starts[:, None]) & (col < ends[:, None])   # (B, n_level)
logits = logits.masked_fill(~valid, _NEG)
```

Consequences:

- **Determinism improves**: the `sorted(set(...))` expert-order concern
  (moe.py:228) disappears — there is no data-dependent execution order left.
  All previous CUDA-determinism fixes stay (mean pooling backbones.py:164,
  `CUBLAS_WORKSPACE_CONFIG`, RNG checkpointing).
- **Single-child parents need no special case**: their range holds one column;
  a 1-class softmax is identically 1.0 with zero gradient, which is exactly the
  pass-through semantics of today (moe.py:196-197, 240-244) without the
  `logit 0.0` bookkeeping.
- **AMP**: heads run under autocast; `masked_fill` with scalar `_NEG` is
  dtype-safe (test `test_forward_under_autocast_keeps_global_buffers_fp32`
  keeps passing unchanged).
- Single-order trees (smoke dataset): `order` head is `Linear(D, 1)` and forward
  returns zeros `(B, 1)` — preserves `test_single_order_tree_passes_through_*`.

Memory check at B=128, n_species=40k: species logits = 128 x 40,000 fp32 =
20.5 MB (+ fp16 grad copy) — no issue on a T4.

## 5. Inference: `route()` cascade

One trunk pass total, heads sliced by range:

```
z = trunk(x)                             : (1, 1024)
p_order = softmax(head_order(z))         : (1, n_orders)
topk_with_threshold(p_order, tau_order, k)   # unchanged (router.py:28)
for each candidate order o:
    child = argmax softmax(head_family(z)[ranges[o]:ranges[o+1]])   # -> family
    if p < tau_family: stop (rejected)
    child = argmax softmax(head_genus(z)[ranges[family]:...])       # -> genus
    child = argmax softmax(head_species(z)[ranges[genus]:...])      # -> species
    joint = product of accepted probabilities
keep branch with highest joint_prob        # unchanged (moe.py:309-315)
```

Every threshold, rejection flag, entropy and `CascadeResult` field is unchanged
(moe.py:38-82). The **Top-K ambiguous-order fallback becomes cheap**: today it
re-runs a full backbone per candidate order; with a shared trunk both branches
share `z`, so ambiguity costs only two extra row-slices of the heads.

## 6. Mixup & trainer integration

Today the mixup branch runs a **second full forward** (trainer.py:205-208) and
`_run_epoch` runs a **third** no_grad forward per batch for accuracy
(trainer.py:312). With one trunk:

- `HierarchicalMoE.forward_mix(x_a, t_a, x_b, t_b)` (new): concatenate
  `cat([x_a, x_b])` through the trunk once (2B, D), then apply the masked
  heads to each half with its own targets. Loss stays exactly
  `lam * L_a + (1-lam) * L_b + lambda * cost` (trainer.py:209-217).
  Non-mixup path keeps calling `forward(x, targets)`.
- `_run_epoch` reuses the training step's `logits.detach()` for the per-level
  accuracy counters instead of re-running the model (metrics only; values drift
  by one optimizer step — metric-only impact).

Net effect per batch: 1 trunk pass instead of 2-3 (mixup) and 1 eliminated
no_grad pass -> epoch wall time roughly **3-4x faster** than today even before
any hardware consideration, and exactly **1 backbone** shipped on mobile.

All checkpoint/RNG/determinism machinery in src/training/trainer.py is
untouched. **Checkpoint compatibility note**: the refactor changes parameter
names (`backbone.*` per expert -> `trunk.*` + `heads.*`), so the 8-species
protocol checkpoints do not load into the new architecture — expected and
acceptable (they were smoke-scale).

## 7. Parameter math at 40k species (D = 1024)

| Component | Params |
|---|---|
| trunk (mnv4_small) | 4.37 M |
| head_order (1024 x ~30) | 0.03 M |
| head_family (1024 x ~1,000) | 1.03 M |
| head_genus (1024 x ~13,000) | 13.3 M |
| head_species (1024 x ~40,000) | 41.0 M |
| **total** | **~60 M** |

- ~**3,100x smaller** than the full-expert design (~185 B).
- fp32 weights ~240 MB; AdamW checkpoint (weights + 2 moment buffers)
  ~720 MB-1 GB `last.pt` — comfortable under the 20 GB `/kaggle/working` cap.
- Inference: 1 backbone (~0.6 GFLOP @224) + 55 M MACs of heads — well under
  10 ms on a modern phone.

**Size lever — `head_dim`**: optional per-level projection
`Linear(D, head_dim)` (default `head_dim = D`, i.e. off) shrinks every head by
D/head_dim. `head_dim=256` -> heads ~13.8 M params (~14 MB INT8 on device,
~28 MB FP16), at a small capacity cost. Flag only; default keeps full D.

## 8. Export layout (ONNX / TFLite)

Today `export_moe` (src/export/onnx_export.py:56) writes one ONNX **per
expert** — ~42,500 files at 40k species. After the refactor:

| File | Input | Output | Size (INT8) |
|---|---|---|---|
| `trunk.onnx` | image (1,3,224,224) | features (1,1024) | ~4.4 MB |
| `head_order.onnx` | features (1,1024) | (1, n_orders) | <0.1 MB |
| `head_family.onnx` | features (1,1024) | (1, n_families) | ~1 MB |
| `head_genus.onnx` | features (1,1024) | (1, n_genera) | ~13 MB |
| `head_species.onnx` | features (1,1024) | (1, n_species) | ~41 MB |
| `manifest.json` | — | labels, `ranges` per level, feature_dim | KBs |

`export_modules()` yields `("trunk", ...)`, `("head_order", ...)`, ... (5
modules); `export_moe` writes the same `manifest.json` schema plus the per-level
`ranges` so the mobile app can slice child ranges without recomputing the tree.
INT8 PTQ via the existing per-module chain (`quantize_static_int8` ->
`onnx2tf`); calibration for heads uses **cached features** `z` (run the trunk
over the calibration images once, then quantize each head on `(z, logits)`
batches).

Mobile options (app side, decision left to the app work):

- **A — uniform**: all 5 modules as INT8 `.tflite` (existing 03 chain, GPU
  delegate on the trunk).
- **B — lighter** (recommended): trunk as INT8 `.tflite`, heads shipped as raw
  FP16 matrices (`.npz`: W, b) applied with 4 small matmuls in Kotlin. Avoids
  head quantization error at 40k logits and drops 4 interpreters; total device
  footprint ~4.4 MB (INT8 trunk) + ~110 MB FP16 heads (~55 MB INT8 if size
  matters — see `head_dim`).

## 9. Campaign sizing for 40k x 150 (flags & budget)

Data-side consequences of the chosen scale (drives the data-phase flags):

| Quantity | Value | Note |
|---|---|---|
| Images (cap 150/species) | <= 6,000,000 | long tail reduces the realized count |
| Candidates to sample (API `limit=300`/species) | ~8-10 M rows | curation keeps ~60-70% |
| Storage @ 320 px q85 (~28 KB/img) | ~168 GB | 256 px (~18 KB) -> ~108 GB |
| Kaggle dataset cap | 20 GB/run output, 200 GB/dataset | shards `open-insect-id-v2-data-XX`, ~15 GB each -> **8-12 datasets** at 320 px |
| Download wall time (CPU, ~25 img/s aggregate) | ~65-80 h | **~7-10 runs** (12 h session cap), resumable per file (downloader.py already skips existing) |
| GBIF sampling requests | 40k (1/species) | ~2-3 h at polite rate -> 1-2 runs |
| Training (T4, batch 128, ~400-600 img/s measured target) | ~3-4 h/epoch @ 6 M images, ~1.5x with mixup | **10-12 epochs = 45-70 h GPU** vs ~30 h quota (6 x 5 h) |

GPU budget is the binding constraint. Mitigations to validate with a throughput
probe on the first real tranche (actual img/s decides):

1. train **10-12 epochs** instead of 30 (shared-trunk linear heads converge
   fast; cosine schedule adapts),
2. cap the effective epoch at ~2 M images (stream the full set over ~3 epochs)
   — one `TrainConfig` flag (`steps_per_epoch_cap`),
3. store 256 px if 320 px throughput is decode-bound (224 input leaves little
   aug headroom but works with center-crop + light jitter),
4. `head_dim=256` only if mobile footprint demands it (no training benefit).

## 10. Migration plan

1. `src/models/heads.py` + refactor `moe.py` (keep public API), delete
   `experts.py`.
2. Tests: `test_moe.py` mostly survives — label-space, forward-contract,
   route/threshold, single-order, autocast tests unchanged; update
   `test_export_modules_covers_all_experts` (5 fixed names) and the
   passthrough test (`genus_experts` -> range of size 1). New tests:
   masked-vs-sliced equivalence (`forward` logits equal `route` branch logits),
   `forward_mix` == two weighted `forward`s, parameter count bound.
3. `src/export/onnx_export.py`: 5-module export + `ranges` in manifest;
   quantize heads from cached features.
4. `src/training/trainer.py`: `forward_mix` call + reuse detached logits for
   metrics (2 small edits; checkpoint/RNG logic untouched).
5. Notebooks: 02 unchanged (same class/attrs); 03 unchanged (calls `export_moe`).
6. Full test suite green, then a Kaggle smoke run (existing 8-image dataset) to
   re-verify determinism end-to-end before the big campaign.

## 11. Decisions requested

1. **`head_dim`**: default off (full 1024-d heads, ~60 M params) — validate?
2. **Mobile export**: option A (all-INT8 TFLite) or B (INT8 trunk + FP16 head
   matrices)? Recommendation: B.
3. **Epochs**: start at 12 epochs with a throughput probe deciding between the
   options of section 9 — validate?
4. Image storage size: 320 px (~168 GB, more aug headroom) vs 256 px (~108 GB,
   single dataset + shards fewer)? Recommendation: 320 px.
