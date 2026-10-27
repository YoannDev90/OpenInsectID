# Dataset Card — Open Insect ID v2 (Kaggle)

**Target usability: 10/10** — complete description, documented schema, example
notebooks, and reproducible provenance.

## Description

A curated dataset of insect photographs sourced from global **GBIF** occurrence
records, cleaned and balanced for training a hierarchical Vision MoE classifier
(Order → Family → Genus → Species).

## Sourcing

- **Source:** GBIF Occurrence API / bulk occurrence dump.
- **Taxon:** Insecta (`classKey=216`), world-scale.
- **Media:** `StillImage` only.
- **Status:** `PRESENT`.
- **Basis of record:** `HUMAN_OBSERVATION`, `PRESERVED_SPECIMEN`
  (privileges iNaturalist validations and museum specimens).
- **Coordinate required:** yes.
- **Excluded data-quality issues:** `COORDINATE_INVALID`, `ZERO_COORDINATE`,
  `COUNTRY_COORDINATE_MISMATCH`, `RECORDED_DATE_UNLIKELY`.

## Cleaning

1. **DuckDB streaming** over the raw dump (no RAM blow-up), projecting only the
   needed columns and dropping rows with an incomplete species-level lineage.
2. **Uber H3 spatial masking** (resolution 4 by default): a `(species, cell)`
   pair must meet an occurrence threshold to count toward the known range;
   records falling outside a species' mask are flagged as spatial outliers
   (coordinate typos, mislabelled lab specimens) and dropped.
3. **Visual-integrity curation:** reject corrupt files, images below 224×224,
   and extreme aspect ratios (> 3:1).
4. **URL + content de-duplication** during async download.

## Long-tail strategy

- Undersampling cap per species.
- Conditional augmentation for rare classes (Mixup, CutMix, chromatic variation
  simulating undergrowth light, geometric transforms).

## Schema

One row per curated image (Parquet):

| column | type | description |
|--------|------|-------------|
| `image_path` | str | path under the image root, `{order}/{family}/{genus}/{species}/{sha256}.jpg` |
| `order` | str | taxonomic order |
| `family` | str | taxonomic family |
| `genus` | str | taxonomic genus |
| `species` | str | species binomial |
| `h3_cell` | str | *(optional)* H3 cell id at the configured resolution |
| `lat` | float | *(optional)* occurrence latitude |
| `lon` | float | *(optional)* occurrence longitude |
| `occurrence_id` | str | *(optional)* GBIF occurrence id |
| `media_url` | str | *(optional)* source image URL |

## Reproduce

See `notebooks/01_data_pipeline.ipynb`, which performs ingestion, spatial
cleaning, download, curation, and manifest export.

## License & attribution

Images remain under their original GBIF/iNaturalist/museum licenses and creator
terms. Redistribution of the *curated dataset* must preserve per-record
attribution. Include `occurrence_id` and `media_url` when published so provenance
is traceable.
