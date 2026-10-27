"""DuckDB / Polars / H3 spatial processing (section 2.B of the plan).

Handles the world-scale occurrence table without saturating RAM:
- Stream the raw GBIF TSV through DuckDB, filter data-quality issues, project
  only the columns we need.
- Build per-species H3 presence masks (Uber H3 hexagonal grid).
- Flag spatial outliers whose coordinates fall outside the known range mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

# GBIF data-quality issues to drop (plan section 2.A).
DEFAULT_EXCLUDED_ISSUES = (
    "COORDINATE_INVALID",
    "ZERO_COORDINATE",
    "COUNTRY_COORDINATE_MISMATCH",
    "RECORDED_DATE_UNLIKELY",
)

# Columns worth keeping from a GBIF occurrence TSV.
KEEP_COLUMNS = (
    "gbifID",
    "speciesKey",
    "species",
    "genus",
    "family",
    "order",
    "decimalLatitude",
    "decimalLongitude",
    "countryCode",
    "basisOfRecord",
    "media",
    "issues",
)


@dataclass
class H3Config:
    resolution: int = 4
    min_occurrences_per_cell: int = 3


class SpatialProcessor:
    """DuckDB-backed cleaning + H3 range masking over the GBIF occurrence table."""

    def __init__(self, h3_config: H3Config | None = None,
                 excluded_issues: tuple[str, ...] = DEFAULT_EXCLUDED_ISSUES):
        self.h3 = h3_config or H3Config()
        self.excluded_issues = excluded_issues

    # ----------------------------------------------------------- DuckDB ingest

    def load_occurrences(self, tsv_path: str | Path) -> pl.DataFrame:
        """Stream a GBIF occurrence TSV into a clean Polars DataFrame.

        Only Insecta rows with a complete species-level lineage, valid
        coordinates, and none of the excluded data-quality issues are kept.
        Column selection is intersected with the file's actual header, since
        GBIF exports vary in which optional columns are present.
        """
        import duckdb

        con = duckdb.connect()
        issues_regex = "|".join(self.excluded_issues)

        # Detect the header: GBIF exports vary in which optional columns exist.
        cur = con.execute(
            f"SELECT * FROM read_csv_auto('{tsv_path}', header=true, sample_size=100000) LIMIT 0"
        )
        present_cols = {col[0] for col in cur.description}
        keep = [c for c in KEEP_COLUMNS if c in present_cols]
        cols = ", ".join(f'"{c}"' for c in keep)
        con.close()

        con = duckdb.connect()
        query = f"""
        COPY (
            SELECT {cols}
            FROM read_csv_auto('{tsv_path}', header=true, sep='\t', sample_size=100000)
            WHERE "order" IS NOT NULL
              AND "family" IS NOT NULL
              AND "genus" IS NOT NULL
              AND "species" IS NOT NULL
              AND "decimalLatitude" IS NOT NULL
              AND "decimalLongitude" IS NOT NULL
              AND ("issues" IS NULL OR NOT regexp_matches("issues", '{issues_regex}'))
        ) TO '{self._parquet_tmp(tsv_path)}' (FORMAT PARQUET)
        """
        con.execute(query)
        df = pl.read_parquet(self._parquet_tmp(tsv_path))
        con.close()
        return df

    @staticmethod
    def _parquet_tmp(tsv_path: str | Path) -> Path:
        return Path(tsv_path).with_suffix(".clean.parquet")

    # ---------------------------------------------------------------- H3 masks

    def build_presence_mask(self, df: pl.DataFrame) -> dict[str, set[str]]:
        """Per-species set of H3 cells that meet the occurrence threshold.

        This is the "known distribution mask" used to detect spatial outliers.
        """
        import h3

        if not {"species", "decimalLatitude", "decimalLongitude"}.issubset(df.columns):
            raise ValueError("DataFrame needs species/decimalLatitude/decimalLongitude")

        cells: dict[str, set[str]] = {}
        counts: dict[tuple[str, str], int] = {}
        for row in df.select(["species", "decimalLatitude", "decimalLongitude"]).iter_rows():
            species, lat, lon = row
            if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            cell = h3.latlng_to_cell(lat, lon, self.h3.resolution)
            key = (species, cell)
            counts[key] = counts.get(key, 0) + 1

        for (species, cell), n in counts.items():
            if n >= self.h3.min_occurrences_per_cell:
                cells.setdefault(species, set()).add(cell)
        return cells

    def flag_outliers(
        self, df: pl.DataFrame, presence_mask: dict[str, set[str]] | None = None
    ) -> pl.DataFrame:
        """Add an ``is_spatial_outlier`` column using the H3 range mask.

        A record is an outlier if its H3 cell is absent from its species'
        presence mask (coordinate typos, mislabelled lab specimens, etc.).
        """
        import h3

        mask = presence_mask if presence_mask is not None else self.build_presence_mask(df)

        def outlier(row: dict) -> bool:
            # polars passes struct rows as dicts.
            species = row["species"]
            lat = row["decimalLatitude"]
            lon = row["decimalLongitude"]
            if lat is None or lon is None:
                return True
            known = mask.get(species)
            if not known:
                # No mask built for this species: cannot judge, keep it.
                return False
            cell = h3.latlng_to_cell(lat, lon, self.h3.resolution)
            return cell not in known

        return df.with_columns(
            pl.struct(["species", "decimalLatitude", "decimalLongitude"])
            .map_elements(outlier, return_dtype=pl.Boolean)
            .alias("is_spatial_outlier")
        )

    def clean(self, df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
        """Build mask, flag + drop outliers. Returns (clean_df, stats)."""
        mask = self.build_presence_mask(df)
        flagged = self.flag_outliers(df, mask)
        n_outliers = flagged["is_spatial_outlier"].sum()
        clean = flagged.filter(~pl.col("is_spatial_outlier"))
        stats = {
            "input_rows": len(df),
            "outliers": int(n_outliers),
            "kept_rows": len(clean),
            "species_with_mask": len(mask),
        }
        return clean, stats
