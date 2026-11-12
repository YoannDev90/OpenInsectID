"""Tests for the 40k-species data campaign (census / sample / download / merge).

Every phase driver takes an injected ``fetch`` stub, so these tests exercise
the real resumable logic (atomic writes, part/page skipping, caps, guards)
without any network.
"""

from __future__ import annotations

import asyncio
import zlib
from io import BytesIO

import polars as pl
import pytest
from PIL import Image

from src.data_pipeline.campaign import (
    CampaignConfig,
    SamplingIndex,
    census_params,
    finalize_census,
    gbif_fetcher,
    merge_manifests,
    parse_census_page,
    parse_sample_page,
    plan_shards,
    prepare_image,
    run_census,
    run_download,
    run_sample,
    sample_params,
    write_parquet_atomic,
    write_shard_manifest,
)
from src.data_pipeline._compat import run_async
from src.data_pipeline.gbif import GBIFConfig


def _jpeg(w: int = 400, h: int = 400, seed: int = 0) -> bytes:
    """Noise JPEG: distinct seeds -> distinct content hashes, > min_bytes."""
    import random

    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(w * h * 3))
    buf = BytesIO()
    Image.frombytes("RGB", (w, h), data).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _census_df(pairs) -> pl.DataFrame:
    return pl.DataFrame(
        {"species_key": [k for k, _ in pairs], "occurrence_count": [c for _, c in pairs]},
        schema={"species_key": pl.Int64, "occurrence_count": pl.Int64},
    )


def _sample_raw(key: int) -> dict:
    return {
        "key": 10_000 + key,
        "media": [{"identifier": f"http://img.example/{key}.jpg", "type": "StillImage"}],
        "order": "Coleoptera",
        "family": "Carabidae",
        "genus": "Carabus",
        "species": f"Carabus sp{key}",
        "decimalLatitude": 1.0,
        "decimalLongitude": 2.0,
        "countryCode": "FR",
    }


def _cfg(**kwargs) -> CampaignConfig:
    kwargs.setdefault("facet_page", 3)
    kwargs.setdefault("n_shards", 2)
    return CampaignConfig(**kwargs)


# --------------------------------------------------------------------- census


def test_census_params_carry_facet_and_filters():
    q = dict(census_params(_cfg(), offset=3000))
    assert q["facet"] == "speciesKey"
    assert q["facetLimit"] == "3"  # from facet_page
    assert q["facetMincount"] == "50"
    assert q["facetOffset"] == "3000"
    assert q["limit"] == "0"
    assert q["classKey"] == "216"
    # GBIF wants basisOfRecord as repeated params, not comma-joined.
    bor = [v for k, v in census_params(_cfg(), 0) if k == "basisOfRecord"]
    assert len(bor) == 2 and "HUMAN_OBSERVATION" in bor
    assert "speciesKey" not in q  # census must not filter to one species


def test_parse_census_page_ignores_junk():
    payload = {
        "facets": [
            {"field": "SPECIES_KEY", "counts": [
                {"name": "1341976", "count": 783141},
                {"name": "not-a-key", "count": 5},
            ]}
        ]
    }
    assert parse_census_page(payload) == [(1341976, 783141)]
    assert parse_census_page({}) == []
    assert parse_census_page({"facets": []}) == []


def test_run_census_pages_until_short_then_finalizes(tmp_path):
    cfg = _cfg()
    pages = {
        0: [(1, 90), (2, 80), (3, 70)],
        3: [(4, 60), (5, 55), (6, 51)],
        6: [(7, 50)],
    }
    calls: list[int] = []

    async def fetch(query):
        offset = int(dict(query)["facetOffset"])
        calls.append(offset)
        return {"facets": [{"field": "SPECIES_KEY", "counts": [
            {"name": str(k), "count": c} for k, c in pages[offset]
        ]}]}

    out = run_async(run_census(cfg, tmp_path, fetch))
    assert out.exists()
    assert calls == [0, 3, 6]
    df = pl.read_parquet(out)
    assert df.height == 7
    # Deterministic order: count desc, key asc.
    assert df["occurrence_count"].to_list() == [90, 80, 70, 60, 55, 51, 50]
    assert (tmp_path / "shards.parquet").exists()

    # Idempotent: a second run must not touch the network.
    run_async(run_census(cfg, tmp_path, fetch))
    assert calls == [0, 3, 6]


def test_run_census_resumes_from_stored_pages(tmp_path):
    cfg = _cfg()
    write_parquet_atomic(_census_df([(1, 90), (2, 80), (3, 70)]),
                         tmp_path / "page-0000000.parquet")
    calls: list[int] = []

    async def fetch(query):
        offset = int(dict(query)["facetOffset"])
        calls.append(offset)
        return {"facets": [{"field": "SPECIES_KEY", "counts": [
            {"name": "4", "count": 60}
        ]}]}

    out = run_async(run_census(cfg, tmp_path, fetch))
    assert calls == [3]  # resumed, page 0 not refetched
    assert pl.read_parquet(out).height == 4


def test_finalize_census_without_pages_writes_empty(tmp_path):
    cfg = _cfg()
    out = finalize_census(tmp_path, cfg)
    df = pl.read_parquet(out)
    assert df.height == 0
    assert pl.read_parquet(tmp_path / "shards.parquet").height == 0


# -------------------------------------------------------------------- shards


def test_plan_shards_is_balanced_and_deterministic():
    cfg = CampaignConfig(n_shards=3, target_species=100, max_images_per_species=150)
    # 12 species, flat counts -> est = min(count*0.65, 150).
    census = _census_df([(i, 200) for i in range(12)])
    a = plan_shards(census, cfg)
    b = plan_shards(census, cfg)
    assert a.equals(b)
    assert a.height == 12
    assert a["species_key"].unique().len() == 12
    assert a["est_images"].unique().to_list() == [130]  # 200 * 0.65
    # Balanced: no shard strays far from total / 3.
    totals = a.group_by("shard").agg(pl.col("est_images").sum())["est_images"].to_list()
    assert max(totals) - min(totals) <= 130


def test_plan_shards_caps_target_species():
    cfg = CampaignConfig(n_shards=2, target_species=5, max_images_per_species=150)
    census = _census_df([(i, 100) for i in range(10)])
    out = plan_shards(census, cfg)
    assert out.height == 5  # top-5 by census order only


# ------------------------------------------------------------------ sampling


def test_sample_params_pin_one_species_first_page():
    q = dict(sample_params(_cfg(), 4242))
    assert q["speciesKey"] == "4242"
    assert q["limit"] == "300"
    assert q["offset"] == "0"
    assert q["classKey"] == "216"


def test_parse_sample_page_dedups_urls_and_drops_incomplete_lineage():
    payload = {"results": [
        {**_sample_raw(1), "media": [
            {"identifier": "http://a/1.jpg", "type": "StillImage"},
            {"identifier": "http://a/1.jpg", "type": "StillImage"},  # dup
            {"identifier": "http://a/2.jpg", "type": "StillImage"},
        ]},
        {**_sample_raw(2), "genus": None},   # incomplete lineage -> dropped
        {**_sample_raw(3), "media": []},     # no image -> dropped
    ]}
    rows = parse_sample_page(payload, 4242)
    assert [r["url"] for r in rows] == ["http://a/1.jpg", "http://a/2.jpg"]
    assert all(r["species_key"] == 4242 for r in rows)
    assert rows[0]["species"] == "Carabus sp1"


def test_run_sample_writes_parts_and_resumes(tmp_path):
    cfg = _cfg()
    cfg.part_species = 2
    census = _census_df([(k, 60) for k in (10, 11, 12, 13, 14)])
    census_path = write_parquet_atomic(census, tmp_path / "census.parquet")
    calls: list[int] = []

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        calls.append(key)
        return {"results": [_sample_raw(key)]}

    out = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    parts = sorted(p.name for p in out.glob("part-*.parquet"))
    assert parts == ["part-000000.parquet", "part-000002.parquet", "part-000004.parquet"]
    assert sorted(calls) == [10, 11, 12, 13, 14]
    rows = pl.read_parquet(out / "part-000000.parquet")
    assert rows.height == 2
    assert set(rows.columns) == set(pl.read_parquet(
        # schema stability across parts
        out / "part-000004.parquet"
    ).columns)

    # Resume: every part exists -> zero fetches.
    run_async(run_sample(cfg, census_path, out, fetch))
    assert sorted(calls) == [10, 11, 12, 13, 14]


def test_sampling_index_rows_and_errors(tmp_path):
    cfg = _cfg()
    cfg.part_species = 2
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in (10, 11, 12)]), tmp_path / "census.parquet"
    )

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        return {"results": [_sample_raw(key)]}

    parts = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    index = SamplingIndex(census_path, parts, part_size=2)
    rows = index.rows_for(11)
    assert len(rows) == 1 and rows[0]["species_key"] == 11
    with pytest.raises(KeyError):
        index.rows_for(999)

    empty = SamplingIndex(census_path, tmp_path / "nope", part_size=2)
    with pytest.raises(FileNotFoundError, match="sample"):
        empty.rows_for(10)


def test_run_sample_retries_failed_species_within_part(tmp_path):
    """One transient failure must not sink a whole 500-species part."""
    cfg = _cfg()
    cfg.part_species = 3
    cfg.sample_wave_pause = 0.0  # no real storm pause in tests
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in (10, 11, 12)]), tmp_path / "census.parquet"
    )
    attempts: dict[int, int] = {}

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        attempts[key] = attempts.get(key, 0) + 1
        if key == 11 and attempts[key] == 1:
            raise RuntimeError("HTTP 429 (rate limited)")
        return {"results": [_sample_raw(key)]}

    out = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    parts = sorted(out.glob("part-*.parquet"))
    assert len(parts) == 1
    got = pl.read_parquet(parts[0])["species_key"].unique().sort().to_list()
    assert got == [10, 11, 12]
    assert attempts[11] == 2  # retried once inside the part
    assert attempts[10] == 1


def test_run_sample_rescues_species_on_a_later_wave(tmp_path):
    """A storm spanning two waves is still ridden out by wave 2+."""
    cfg = _cfg()
    cfg.part_species = 2
    cfg.sample_wave_pause = 0.0
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in (10, 11)]), tmp_path / "census.parquet"
    )
    attempts: dict[int, int] = {}

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        attempts[key] = attempts.get(key, 0) + 1
        if key == 11 and attempts[key] <= 2:  # fails waves 0 and 1, ok on wave 2
            raise RuntimeError("HTTP 429 (rate limited)")
        return {"results": [_sample_raw(key)]}

    out = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    assert len(list(out.glob("part-*.parquet"))) == 1
    assert attempts[11] == 3


def test_run_sample_raises_without_writing_a_partial_part(tmp_path):
    """A permanently failing species aborts the part; no half-written file."""
    cfg = _cfg()
    cfg.part_species = 3
    cfg.sample_wave_pause = 0.0
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in (10, 11, 12)]), tmp_path / "census.parquet"
    )
    attempts: dict[int, int] = {}

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        attempts[key] = attempts.get(key, 0) + 1
        if key == 11:
            raise RuntimeError("HTTP 429 (rate limited)")
        return {"results": [_sample_raw(key)]}

    with pytest.raises(RuntimeError, match="part-000000"):
        run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    assert not list((tmp_path / "sampling").glob("part-*.parquet"))
    assert attempts[11] == cfg.sample_waves  # every wave was tried
    assert attempts[10] == 1  # neighbours are refetched by the next run


def test_run_sample_budget_cuts_the_retry_pause_cleanly(tmp_path):
    """When the pause would blow the budget, stop without raising."""
    cfg = _cfg()
    cfg.part_species = 2
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in (10, 11)]), tmp_path / "census.parquet"
    )
    clock = [0.0]

    def now():
        return clock[0]

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        if key == 11:
            raise RuntimeError("HTTP 429 (rate limited)")
        return {"results": [_sample_raw(key)]}

    # deadline = 0 + 60 s; the first retry pause (60 s) does not fit -> clean
    # stop with the part left unwritten, no exception.
    out = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch,
                               budget_min=1.0, now=now))
    assert not list(out.glob("part-*.parquet"))


def test_run_sample_budget_stops_between_parts_and_resumes(tmp_path):
    cfg = _cfg()
    cfg.part_species = 2
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in (10, 11, 12, 13)]), tmp_path / "census.parquet"
    )
    clock = [0.0]

    def now():
        return clock[0]

    async def fetch(query):
        clock[0] += 60.0  # one species costs a minute of wall clock
        return {"results": [_sample_raw(int(dict(query)["speciesKey"]))]}

    out = run_async(
        run_sample(cfg, census_path, tmp_path / "sampling", fetch,
                   budget_min=1.5, now=now)
    )
    assert len(list(out.glob("part-*.parquet"))) == 1  # deadline hit before part 2
    # Next run: part 1 is skipped, the remainder completes.
    out = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    assert len(list(out.glob("part-*.parquet"))) == 2


# ------------------------------------------------------------------- prepare


def test_prepare_image_rejects_tiny_corrupt_and_extreme():
    cfg = _cfg()
    assert prepare_image(_jpeg(100, 100), cfg) is None        # below floor
    assert prepare_image(_jpeg(900, 250), cfg) is None        # aspect 3.6 > 3
    assert prepare_image(b"not an image", cfg) is None        # corrupt
    assert prepare_image(_jpeg(400, 400)[:100], cfg) is None   # truncated


def test_prepare_image_downscales_to_keeping_quality_floor():
    cfg = _cfg()
    # Square 400 -> 320 long side (floor not binding).
    out = prepare_image(_jpeg(400, 400), cfg)
    assert Image.open(BytesIO(out)).size == (320, 320)
    # Wide 900x300: 320/900 would put the short side at 107 < 224, so the
    # floor wins: min side 224 (max = 224 x aspect = 672).
    out = prepare_image(_jpeg(900, 300), cfg)
    assert Image.open(BytesIO(out)).size == (672, 224)
    # Already at the floor on the short side: keep the long side as-is.
    out = prepare_image(_jpeg(600, 224), cfg)
    assert Image.open(BytesIO(out)).size == (600, 224)
    # Stored bytes survive as a decodable JPEG above min_bytes.
    img = Image.open(BytesIO(out))
    assert img.format == "JPEG" and img.mode == "RGB"


# ------------------------------------------------------------------ download


def _download_env(tmp_path, n_species=3, rows_per_species=5, **cfg_kwargs):
    """census + sampling parts + index for n synthetic species."""
    cfg = _cfg(max_images_per_species=3, **cfg_kwargs)
    keys = list(range(10, 10 + n_species))
    census_path = write_parquet_atomic(
        _census_df([(k, 60) for k in keys]), tmp_path / "census.parquet"
    )

    async def fetch(query):
        key = int(dict(query)["speciesKey"])
        raw = _sample_raw(key)
        raw["media"] = [
            {"identifier": f"http://img.example/{key}-{i}.jpg", "type": "StillImage"}
            for i in range(rows_per_species)
        ]
        return {"results": [raw]}

    parts = run_async(run_sample(cfg, census_path, tmp_path / "sampling", fetch))
    index = SamplingIndex(census_path, parts, part_size=500)
    return cfg, keys, index


def test_download_caps_per_species_and_skips_when_full(tmp_path):
    cfg, keys, index = _download_env(tmp_path)
    calls: list[int] = []

    async def fetch(url):
        calls.append(url)
        # crc32, not hash(): str hashes are salted per process (flaky tests).
        return _jpeg(seed=zlib.crc32(url.encode()))

    root = tmp_path / "shard-00"
    stats = run_async(run_download(cfg, root, keys, index, fetch))
    assert stats["ok"] == 3 * cfg.max_images_per_species  # cap enforced
    assert stats["species_done"] == 3
    assert stats["stop"] == ""
    for k in keys:
        d = root / "Coleoptera" / "Carabidae" / "Carabus" / f"Carabus sp{k}"
        assert len(list(d.glob("*.jpg"))) == cfg.max_images_per_species

    # Second run: every species already full -> nothing fetched.
    n_calls = len(calls)
    stats = run_async(run_download(cfg, root, keys, index, fetch))
    assert stats["ok"] == 0 and stats["skipped_full"] == 3
    assert len(calls) == n_calls


def test_download_stops_on_byte_guard(tmp_path):
    cfg, keys, index = _download_env(tmp_path, n_species=3)
    # Any single written image trips this (prepare_image guarantees >= min_bytes).
    cfg.max_shard_bytes = 1

    async def fetch(url):
        return _jpeg(seed=zlib.crc32(url.encode()))

    stats = run_async(run_download(cfg, tmp_path / "shard", keys, index, fetch))
    assert stats["stop"] == "bytes"
    assert stats["remaining"] >= 1  # work left for the next run


def test_download_stops_on_wall_clock_budget(tmp_path):
    cfg, keys, index = _download_env(tmp_path, n_species=3)

    async def fetch(url):
        return _jpeg(seed=zlib.crc32(url.encode()))

    # Fake clock: +1 s per call makes the deadline deterministic (a real
    # 1e-6-min budget would race against parquet read latency).
    tick = [1000.0]

    def now():
        tick[0] += 1.0
        return tick[0]

    stats = run_async(
        run_download(
            cfg, tmp_path / "shard", keys, index, fetch, budget_min=1e-3, now=now
        )
    )
    assert stats["stop"] == "budget"


def test_download_is_resumable_mid_species(tmp_path):
    """A killed run keeps its files; the next run only fetches the remainder."""
    cfg, keys, index = _download_env(tmp_path, n_species=1, rows_per_species=6)
    root = tmp_path / "shard"
    # Simulate an interrupted run: 2 of the 3 allowed images already exist.
    d = root / "Coleoptera" / "Carabidae" / "Carabus" / "Carabus sp10"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(_jpeg(seed=1))
    (d / "b.jpg").write_bytes(_jpeg(seed=2))

    async def fetch(url):
        return _jpeg(seed=100 + len(url))

    stats = run_async(run_download(cfg, root, keys, index, fetch))
    assert stats["ok"] == 1  # only the missing slot was fetched
    assert len(list(d.glob("*.jpg"))) == cfg.max_images_per_species


# ------------------------------------------------------ manifest + merge


def _make_shard(root, slug, n_images):
    for i in range(n_images):
        p = (
            root / slug / "Coleoptera" / "Carabidae" / "Carabus"
            / "Carabus auratus" / f"{i}.jpg"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return write_shard_manifest(root / slug)


def test_write_shard_manifest_walks_the_tree(tmp_path):
    path = _make_shard(tmp_path, "data-00", 3)
    df = pl.read_parquet(path)
    assert df.height == 3
    assert set(df.columns) == {"image_path", "order", "family", "genus", "species"}
    assert df["image_path"][0] == "Coleoptera/Carabidae/Carabus/Carabus auratus/0.jpg"
    # Re-walk is idempotent.
    assert pl.read_parquet(write_shard_manifest(tmp_path / "data-00")).height == 3


def test_merge_manifests_adds_shard_column_deterministically(tmp_path):
    m0 = _make_shard(tmp_path, "data-00", 2)
    m1 = _make_shard(tmp_path, "data-01", 3)
    out = tmp_path / "insectid_manifest.parquet"
    df = merge_manifests([("data-01", m1), ("data-00", m0)], out)  # unsorted input
    assert out.exists()
    assert df.height == 5
    assert df["shard"].unique().sort().to_list() == ["data-00", "data-01"]
    # Input order must not matter: slug-sorted, hence stable for splits.
    assert df["shard"].to_list() == ["data-00"] * 2 + ["data-01"] * 3
    df2 = merge_manifests([("data-00", m0), ("data-01", m1)], out)
    assert df.equals(df2)


# ----------------------------------------------------------- gbif fetcher


class _FakeResp:
    def __init__(self, status: int, headers: dict | None = None):
        self.status = status
        self.headers = headers or {}

    async def json(self):
        return {"ok": True}

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _FakeSession:
    """Scripted ``session.get``: one status per call, the last one sticks."""

    def __init__(self, statuses: list[int], headers: dict | None = None):
        self.statuses = statuses
        self.headers = headers or {}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        headers = self.headers

        class _Ctx:
            async def __aenter__(self):
                return _FakeResp(status, headers)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _fake_sleep(monkeypatch) -> list[float]:
    """Replace asyncio.sleep so backoff waits are instant and observable."""
    sleeps: list[float] = []

    async def fake(t):
        sleeps.append(t)

    monkeypatch.setattr(asyncio, "sleep", fake)
    return sleeps


def test_gbif_fetcher_survives_429_beyond_plain_retries(monkeypatch):
    """A throttle storm outlives the old 3-try budget; waits double each time."""
    sleeps = _fake_sleep(monkeypatch)
    sess = _FakeSession([429, 429, 429, 429, 200])
    fetch = gbif_fetcher(sess, GBIFConfig(), interval=0.0)
    assert run_async(fetch([("q", "a")])) == {"ok": True}
    assert sess.calls == 5  # the old fetcher raised after 3 tries
    assert sleeps == [10.0, 20.0, 40.0, 80.0]


def test_gbif_fetcher_honors_retry_after_header(monkeypatch):
    sleeps = _fake_sleep(monkeypatch)
    sess = _FakeSession([429, 200], headers={"Retry-After": "42"})
    fetch = gbif_fetcher(sess, GBIFConfig(), interval=0.0)
    assert run_async(fetch([("q", "a")])) == {"ok": True}
    assert sleeps == [42.0]


def test_gbif_fetcher_gate_adapts_pace_aimd(monkeypatch):
    """429s halve the pace (up to the ceiling), successes give it back."""
    _fake_sleep(monkeypatch)
    sess = _FakeSession([429, 429, 429, 429, 200])
    fetch = gbif_fetcher(sess, GBIFConfig(), interval=0.15)
    assert fetch.gate.interval == 0.15
    run_async(fetch([("q", "a")]))
    # 4 penalties: 0.15 -> 0.3 -> 0.6 -> 1.2 -> 2.4 (below the 6 s ceiling);
    # 1 relax on the final success: /2.
    assert fetch.gate.interval == 1.2
    for _ in range(3):  # plain successes walk the pace back to the base
        run_async(fetch([("q", "a")]))
    assert fetch.gate.interval == 0.15
