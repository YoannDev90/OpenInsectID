"""Tests for data-pipeline curation, spatial masking, and downloader naming."""

import io
from pathlib import Path

import pytest
from PIL import Image

from src.data_pipeline.curation import CurationConfig, curate_image, curate_directory
from src.data_pipeline.downloader import ImageDownloader, DownloadConfig
from src.data_pipeline.gbif import GBIFExtractor, TaxonLineage, OccurrenceRecord
from src.data_pipeline.spatial import SpatialProcessor, H3Config


# ------------------------------------------------------------------ curation

def _write_img(path, size=(300, 300), fmt="JPEG"):
    Image.new("RGB", size, color=(10, 120, 40)).save(path, fmt)


def test_curate_accepts_valid(tmp_path):
    p = tmp_path / "good.jpg"
    _write_img(p, (300, 300))
    v = curate_image(p)
    assert v.ok, v.reason


def test_curate_rejects_low_res(tmp_path):
    p = tmp_path / "small.jpg"
    _write_img(p, (100, 100))
    # Isolate the resolution check (disable the byte-size floor).
    v = curate_image(p, CurationConfig(min_resolution=224, min_bytes=0))
    assert not v.ok and "low_resolution" in v.reason


def test_curate_rejects_extreme_aspect(tmp_path):
    p = tmp_path / "wide.jpg"
    # Both sides >= min_resolution, but ratio 1200/250 = 4.8 > 3.0.
    _write_img(p, (1200, 250))
    v = curate_image(p, CurationConfig(min_resolution=224, max_aspect_ratio=3.0, min_bytes=0))
    assert not v.ok and "extreme_aspect" in v.reason


def test_curate_rejects_corrupt(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"not an image at all")
    v = curate_image(p)
    assert not v.ok


def test_curate_directory_counts(tmp_path):
    _write_img(tmp_path / "a.jpg", (300, 300))
    _write_img(tmp_path / "b.jpg", (100, 100))  # low res
    (tmp_path / "sub").mkdir()
    _write_img(tmp_path / "sub" / "c.png", (400, 400))
    verdicts = curate_directory(tmp_path, CurationConfig(min_resolution=224))
    assert len(verdicts) == 3
    assert sum(v.ok for v in verdicts) == 2


# ------------------------------------------------------------------ spatial

def test_presence_mask_and_outliers():
    import polars as pl

    # Species concentrated around Paris (48.85, 2.35). Give >=3 occurrences in
    # one H3 cell so a mask forms, plus one far-away outlier.
    lat, lon = 48.8566, 2.3522
    rows = []
    for _ in range(5):
        rows.append(("Carabus auratus", lat + 0.0001, lon + 0.0001))
    rows.append(("Carabus auratus", 40.0, -3.7))  # Madrid -> outlier

    df = pl.DataFrame(
        {
            "species": [r[0] for r in rows],
            "decimalLatitude": [r[1] for r in rows],
            "decimalLongitude": [r[2] for r in rows],
        }
    )
    sp = SpatialProcessor(H3Config(resolution=5, min_occurrences_per_cell=3))
    mask = sp.build_presence_mask(df)
    assert "Carabus auratus" in mask
    clean, stats = sp.clean(df)
    assert stats["outliers"] >= 1
    assert len(clean) < len(df)


def test_duckdb_load(tmp_path):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "gbifID": ["1", "2", "3"],
            "species": ["A", "A", None],  # dropped (incomplete lineage)
            "genus": ["A", "A", "A"],
            "family": ["A", "A", "A"],
            "order": ["Coleoptera", "Coleoptera", "Coleoptera"],
            "decimalLatitude": [48.8, 48.8, 48.8],
            "decimalLongitude": [2.3, 2.3, 2.3],
            "countryCode": ["FR", "FR", "FR"],
            "basisOfRecord": ["HUMAN_OBSERVATION"] * 3,
            "media": ["u", "u", "u"],
            "issues": ["", "COORDINATE_INVALID", ""],
        }
    )
    tsv = tmp_path / "occ.tsv"
    df.write_csv(tsv, separator="\t")
    sp = SpatialProcessor()
    out = sp.load_occurrences(tsv)
    # Row with null species dropped; row with COORDINATE_INVALID dropped.
    assert len(out) == 1
    # gbifID may be inferred as int or str depending on the reader.
    assert str(out["gbifID"][0]) == "1"


# ---------------------------------------------------------------- downloader

def test_content_hash_and_dedup():
    d = ImageDownloader(DownloadConfig())
    a = d.content_hash(b"abc")
    b = d.content_hash(b"abc")
    assert a == b
    assert a != d.content_hash(b"abd")


def test_relative_path_sanitizes():
    d = ImageDownloader()
    rec = OccurrenceRecord(
        gbif_id="1",
        lineage=TaxonLineage(
            order="Coleoptera", family="Carabidae", genus="Carabus",
            species="Carabus/auratus",
        ),
        image_urls=["http://x/a.jpg"],
    )
    p = d.relative_path(rec)
    parts = p.parts
    assert parts[0] == "Coleoptera"
    assert parts[1] == "Carabidae"
    # Slash in species must be sanitized into a single path segment.
    assert "/" not in parts[3]
    assert len(parts) == 4


# ---------------------------------------------------------------- gbif parse

def test_gbif_lineage_and_media_parse():
    raw = {
        "key": 123,
        "order": "Coleoptera",
        "family": "Carabidae",
        "genus": "Carabus",
        "species": "Carabus auratus",
        "taxonKey": 999,
        "decimalLatitude": 48.8,
        "decimalLongitude": 2.3,
        "countryCode": "FR",
        "media": [{"type": "StillImage", "identifier": "http://img/1.jpg"}],
    }
    ex = GBIFExtractor()
    rec = ex._parse(raw)
    assert rec is not None
    assert rec.lineage.order == "Coleoptera"
    assert rec.lineage.is_complete
    assert rec.image_urls == ["http://img/1.jpg"]


def test_gbif_parse_skips_medialess():
    raw = {"key": 1, "order": "Coleoptera", "media": []}
    ex = GBIFExtractor()
    assert ex._parse(raw) is None


def test_gbif_params_filters():
    ex = GBIFExtractor()
    params = ex._params(0)
    assert params["classKey"] == 216
    assert params["mediaType"] == "StillImage"
    assert params["occurrenceStatus"] == "PRESENT"
    assert "HUMAN_OBSERVATION" in params["basisOfRecord"]
