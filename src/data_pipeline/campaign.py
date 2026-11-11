"""Data campaign: 40k species x up to 150 images (architecture doc section 9).

Four resumable phases. Every artifact is written atomically (temp file +
rename) and existing artifacts are skipped, so a killed Kaggle session simply
resumes on the next run:

1. ``census``    - page GBIF's ``speciesKey`` facet for species with at least
                   ``min_occurrences`` eligible occurrences ->
                   ``census/page-*.parquet``, merged into ``census.parquet``
                   plus the deterministic ``shards.parquet`` assignment.
2. ``sample``    - one occurrence page (``limit=300``) per species ->
                   ``sampling/part-*.parquet`` (500 species per part, in
                   census order: stable across runs).
3. ``download``  - per shard: fetch -> curate -> resize -> q85 JPEG into the
                   taxonomic tree, capped per species; writes the shard's
                   ``manifest.parquet``.
4. ``manifest``  - merge the per-shard manifests into the training manifest
                   with a ``shard`` column (dataset slug) for multi-root
                   loading.

HTTP is injected (``fetch`` callables) so every phase driver is unit-testable
without network; the ``*_sync`` wrappers open a real aiohttp session.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ._compat import run_async
from .curation import CurationConfig
from .downloader import DownloadConfig, ImageDownloader
from .gbif import GBIF_SEARCH_URL, GBIFConfig, GBIFExtractor

log = logging.getLogger(__name__)

# Shared manifest schema (one row per curated image file).
MANIFEST_SCHEMA = {
    "image_path": pl.Utf8,
    "order": pl.Utf8,
    "family": pl.Utf8,
    "genus": pl.Utf8,
    "species": pl.Utf8,
}

# One row per sampled candidate image URL.
SAMPLE_SCHEMA = {
    "species_key": pl.Int64,
    "gbif_id": pl.Utf8,
    "url": pl.Utf8,
    "order": pl.Utf8,
    "family": pl.Utf8,
    "genus": pl.Utf8,
    "species": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "country": pl.Utf8,
}


@dataclass
class CampaignConfig:
    """Sizing knobs for the 40k x 150 campaign (docs section 9)."""

    gbif: GBIFConfig = field(default_factory=GBIFConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    # Census: which species are worth sampling.
    min_occurrences: int = 50
    facet_page: int = 1000
    target_species: int = 40_000
    n_shards: int = 12
    curation_yield: float = 0.65  # sampled candidates that survive curation
    # Sampling: candidates gathered per species.
    sample_limit: int = 300
    part_species: int = 500
    # Download: what lands on disk.
    max_images_per_species: int = 150
    image_size: int = 320  # stored long side; the 224-px quality floor wins
    jpeg_quality: int = 85
    download_concurrency: int = 32  # image fetches in flight (global)
    species_concurrency: int = 16  # species worked on in parallel
    sample_concurrency: int = 8  # GBIF sample requests in flight
    request_interval: float = 0.15  # min seconds between GBIF request starts
    max_shard_bytes: int = 18 * 1024**3  # /kaggle/working guard (20 GB cap)


# --------------------------------------------------------------------- atomic


def write_parquet_atomic(df: pl.DataFrame, path: str | Path) -> Path:
    """Write a parquet file via temp + rename (crash-safe resume)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return path


# GBIF throttles "depending on server load": a 429 is normal, not an error.
# It gets a patient budget of its own (doubling waits, Retry-After honoured)
# on top of the regular transient retries, and halves the request pace.
GBIF_429_TRIES = 5
GBIF_429_BACKOFF = 10.0  # s before the first 429 retry; doubles each time
GBIF_MAX_INTERVAL = 2.0  # AIMD ceiling for the gate (base interval 0.15 s)


class RequestGate:
    """Spaces request starts by >= interval while allowing overlap.

    ``penalize``/``relax`` implement AIMD pacing: a 429 doubles the interval
    (up to ``ceiling``), each success halves it back toward the base — GBIF's
    limit moves with server load, so the pace must adapt both ways.
    """

    def __init__(self, interval: float, ceiling: float = GBIF_MAX_INTERVAL):
        self.base = interval
        self.interval = interval
        self.ceiling = ceiling
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)

    def penalize(self) -> None:
        """On HTTP 429: halve throughput by doubling the pacing interval."""
        self.interval = min(self.interval * 2, self.ceiling)

    def relax(self) -> None:
        """On success: undo half of the throttling, never below the base."""
        if self.interval > self.base:
            self.interval = max(self.interval / 2, self.base)


def _retry_after(resp) -> float | None:
    """Seconds from a sane ``Retry-After`` header, if the response carries one."""
    try:
        value = float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None
    return value if 0 < value <= 300 else None


def gbif_fetcher(session, cfg: GBIFConfig, interval: float = 0.15):
    """Async ``fetch(query) -> dict`` with pacing, retries and 429 backoff.

    Load-dependent 429s are not plain retries: each one halves the request
    pace (AIMD, restored on success) and waits a patient doubling delay
    (honouring ``Retry-After``) out of a dedicated ``GBIF_429_TRIES`` budget,
    so a throttle storm can't sink a whole part after 3 quick attempts.
    """
    gate = RequestGate(interval)

    async def fetch(query: list[tuple[str, str]]) -> dict:
        import aiohttp

        last: Exception | None = None
        n429 = 0
        total = cfg.max_retries + GBIF_429_TRIES
        for attempt in range(total):
            await gate.wait()
            delay = cfg.backoff ** attempt
            try:
                async with session.get(
                    GBIF_SEARCH_URL,
                    params=query,
                    timeout=aiohttp.ClientTimeout(total=cfg.timeout),
                ) as resp:
                    if resp.status == 429:
                        n429 += 1
                        gate.penalize()
                        delay = _retry_after(resp) or min(
                            GBIF_429_BACKOFF * 2 ** (n429 - 1),
                            GBIF_429_BACKOFF * 2 ** (GBIF_429_TRIES - 1),
                        )
                        last = RuntimeError("HTTP 429 (rate limited)")
                    elif resp.status >= 500:
                        last = RuntimeError(f"HTTP {resp.status}")
                    else:
                        resp.raise_for_status()
                        gate.relax()
                        return await resp.json()
            except Exception as e:  # noqa: BLE001 - retry any transient error
                last = e
            if attempt + 1 < total:  # never sleep after the final try
                await asyncio.sleep(delay)
        raise RuntimeError(f"GBIF request failed after {total} tries: {last}")

    fetch.gate = gate  # exposed for tests: observe the AIMD pace
    return fetch


# --------------------------------------------------------------------- census


def census_params(cfg: CampaignConfig, offset: int) -> list[tuple[str, str]]:
    """Occurrence-search query for one page of the speciesKey facet."""
    query = [
        p
        for p in GBIFExtractor(cfg.gbif)._query(0)
        if p[0] not in ("limit", "offset")
    ]
    query += [
        ("limit", "0"),
        ("facet", "speciesKey"),
        ("facetLimit", str(cfg.facet_page)),
        ("facetMincount", str(cfg.min_occurrences)),
        ("facetOffset", str(offset)),
    ]
    return query


def parse_census_page(payload: dict) -> list[tuple[int, int]]:
    """[(species_key, eligible_occurrence_count)] from one facet page."""
    facets = payload.get("facets") or []
    if not facets:
        return []
    return [
        (int(c["name"]), int(c["count"]))
        for c in facets[0].get("counts") or []
        if str(c.get("name", "")).lstrip("-").isdigit()
    ]


def plan_shards(census: pl.DataFrame, cfg: CampaignConfig) -> pl.DataFrame:
    """Deterministic species -> shard assignment, balanced by est. images.

    Species are taken in census order (count desc, key asc) and packed into
    ``n_shards`` contiguous blocks whose estimated image totals stay close to
    ``total / n_shards``. Estimated images = occurrences x curation yield,
    capped at ``max_images_per_species``.
    """
    top = census.head(cfg.target_species)
    keys = top["species_key"].to_list()
    counts = top["occurrence_count"].to_list()
    est = [
        min(int(round(c * cfg.curation_yield)), cfg.max_images_per_species)
        for c in counts
    ]
    total = sum(est)
    target = total / max(cfg.n_shards, 1) if total else 0.0
    shard, acc, shards = 0, 0.0, []
    for e in est:
        if acc > 0 and acc + e > target and shard < cfg.n_shards - 1:
            shard += 1
            acc = 0.0
        shards.append(shard)
        acc += e
    return pl.DataFrame(
        {"species_key": keys, "shard": shards, "est_images": est},
        schema={"species_key": pl.Int64, "shard": pl.Int32, "est_images": pl.Int32},
    )


def finalize_census(out_dir: str | Path, cfg: CampaignConfig) -> Path:
    """Merge the facet pages into census.parquet + shards.parquet."""
    out = Path(out_dir)
    pages = sorted(out.glob("page-*.parquet"))
    if pages:
        df = pl.concat([pl.read_parquet(p) for p in pages])
        df = df.unique(subset=["species_key"], keep="first")
        df = df.sort(["occurrence_count", "species_key"], descending=[True, False])
    else:
        df = pl.DataFrame(
            schema={"species_key": pl.Int64, "occurrence_count": pl.Int64}
        )
    write_parquet_atomic(df, out / "census.parquet")
    write_parquet_atomic(plan_shards(df, cfg), out / "shards.parquet")
    log.info("census finalized: %d species", df.height)
    return out / "census.parquet"


async def run_census(cfg: CampaignConfig, out_dir: str | Path, fetch) -> Path:
    """Page the facet until a short page, writing one parquet per page.

    Resumes from the last stored page; a short last page means the facet is
    exhausted (only the final merge is missing).
    """
    out = Path(out_dir)
    final = out / "census.parquet"
    if final.exists():
        return final
    pages = sorted(out.glob("page-*.parquet"))
    if pages:
        last_offset = int(pages[-1].name[len("page-") : -len(".parquet")])
        if pl.read_parquet(pages[-1]).height < cfg.facet_page:
            return finalize_census(out, cfg)
        offset = last_offset + cfg.facet_page
    else:
        offset = 0
    while True:
        payload = await fetch(census_params(cfg, offset))
        rows = parse_census_page(payload)
        if rows:
            write_parquet_atomic(
                pl.DataFrame(
                    rows,
                    schema={"species_key": pl.Int64, "occurrence_count": pl.Int64},
                    orient="row",
                ),
                out / f"page-{offset:07d}.parquet",
            )
        if len(rows) < cfg.facet_page:
            break
        offset += cfg.facet_page
    return finalize_census(out, cfg)


def census_sync(cfg: CampaignConfig, out_dir: str | Path) -> Path:
    async def go():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            fetch = gbif_fetcher(session, cfg.gbif, cfg.request_interval)
            return await run_census(cfg, out_dir, fetch)

    return run_async(go())


# -------------------------------------------------------------------- sampling


def sample_params(cfg: CampaignConfig, species_key: int) -> list[tuple[str, str]]:
    """One occurrence page (all campaign filters + the speciesKey)."""
    query = [
        p
        for p in GBIFExtractor(cfg.gbif)._query(0)
        if p[0] not in ("limit", "offset")
    ]
    query += [
        ("limit", str(cfg.sample_limit)),
        ("offset", "0"),
        ("speciesKey", str(species_key)),
    ]
    return query


def parse_sample_page(payload: dict, species_key: int) -> list[dict]:
    """Flat candidate rows (one per image URL) from one species' page."""
    rows: list[dict] = []
    seen: set[str] = set()
    for raw in payload.get("results") or []:
        urls = GBIFExtractor._image_urls(raw)
        if not urls:
            continue
        lin = GBIFExtractor._lineage_from(raw)
        if not lin.is_complete:
            continue  # image path needs the full order/family/genus/species
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            rows.append(
                {
                    "species_key": species_key,
                    "gbif_id": str(raw.get("key") or ""),
                    "url": url,
                    "order": lin.order,
                    "family": lin.family,
                    "genus": lin.genus,
                    "species": lin.species,
                    "lat": raw.get("decimalLatitude"),
                    "lon": raw.get("decimalLongitude"),
                    "country": raw.get("countryCode"),
                }
            )
    return rows


async def run_sample(
    cfg: CampaignConfig,
    census_path: str | Path,
    out_dir: str | Path,
    fetch,
    *,
    budget_min: float = 0.0,
    now=None,
) -> Path:
    """Sample every census species into deterministic part files.

    Parts that already exist are skipped, so an interrupted run resumes at the
    first missing part (500 species per part, census order). A species whose
    request fails is retried once inside the part; a part that still has
    failures raises WITHOUT being written (the next run refetches only that
    part). ``budget_min`` stops cleanly between parts, leaving the rest queued.
    """
    now = now or time.monotonic
    deadline = now() + budget_min * 60 if budget_min > 0 else None
    keys = pl.read_parquet(census_path)["species_key"].to_list()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(cfg.sample_concurrency)
    for start in range(0, len(keys), cfg.part_species):
        part = out / f"part-{start:06d}.parquet"
        if part.exists():
            continue
        if deadline is not None and now() >= deadline:
            log.info(
                "sample budget reached: %d/%d species done, next run resumes here",
                start, len(keys),
            )
            break
        chunk = keys[start : start + cfg.part_species]

        async def one(key: int) -> list[dict]:
            async with sem:
                payload = await fetch(sample_params(cfg, key))
            return parse_sample_page(payload, key)

        # Two waves: a transient failure (429, timeout) must not sink a whole
        # part — the species is retried once before the part is abandoned.
        by_key: dict[int, list[dict]] = {}
        pending = list(chunk)
        for wave in range(2):
            got = await asyncio.gather(
                *(one(k) for k in pending), return_exceptions=True
            )
            for k, r in zip(pending, got):
                if isinstance(r, BaseException):
                    log.warning("sample species %s failed (wave %d): %r", k, wave, r)
                else:
                    by_key[k] = r
            pending = [k for k in pending if k not in by_key]
            if not pending:
                break
        if pending:
            raise RuntimeError(
                f"part {part.name}: {len(pending)} species still failing "
                f"(first: {pending[0]}); part left unwritten for the next run"
            )
        rows = [r for k in chunk for r in by_key[k]]
        write_parquet_atomic(pl.DataFrame(rows, schema=SAMPLE_SCHEMA), part)
        log.info("sampled part %s: %d species, %d candidate rows", part.name, len(chunk), len(rows))
    return out


def sample_sync(
    cfg: CampaignConfig,
    census_path: str | Path,
    out_dir: str | Path,
    budget_min: float = 0.0,
) -> Path:
    async def go():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            fetch = gbif_fetcher(session, cfg.gbif, cfg.request_interval)
            return await run_sample(
                cfg, census_path, out_dir, fetch, budget_min=budget_min
            )

    return run_async(go())


class SamplingIndex:
    """Lazy access to sampled candidate rows, one part file cached at a time.

    A shard covers a contiguous census range, so part access is sequential
    and the cache never needs more than the current part.
    """

    def __init__(self, census_path: str | Path, parts_dir: str | Path, part_size: int = 500):
        self.keys = pl.read_parquet(census_path)["species_key"].to_list()
        self.pos = {k: i for i, k in enumerate(self.keys)}
        self.parts_dir = Path(parts_dir)
        self.part_size = part_size
        self._cache: dict[int, dict[int, list[dict]]] = {}

    def rows_for(self, species_key: int) -> list[dict]:
        if species_key not in self.pos:
            raise KeyError(f"species {species_key} not in census")
        start = (self.pos[species_key] // self.part_size) * self.part_size
        if start not in self._cache:
            path = self.parts_dir / f"part-{start:06d}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"sampling part {path} missing: run the 'sample' phase first"
                )
            grouped: dict[int, list[dict]] = defaultdict(list)
            for row in pl.read_parquet(path).to_dicts():
                grouped[row["species_key"]].append(row)
            self._cache = {start: grouped}  # single-part cache: shards are contiguous
        return self._cache[start].get(species_key, [])


# ------------------------------------------------------------------- download


def prepare_image(data: bytes, cfg: CampaignConfig) -> bytes | None:
    """Curate + resize + re-encode one image; None = reject.

    The long side is downscaled to ``image_size`` unless that would push the
    short side under the curation floor: the training transform upscales the
    short side to 224 px, so the floor protects real detail (wide images then
    keep min side = floor, max side <= floor x max_aspect_ratio).
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None
    w, h = img.size
    cur = cfg.curation
    if min(w, h) < cur.min_resolution:
        return None
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > cur.max_aspect_ratio or ratio < cur.min_aspect_ratio:
        return None
    if cfg.image_size and max(w, h) > cfg.image_size:
        s = cfg.image_size / max(w, h)
        if min(w, h) * s < cur.min_resolution:
            s = cur.min_resolution / min(w, h)  # curation guarantees <= 1
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    if img.mode != "RGB":
        try:
            img = img.convert("RGB")
        except ValueError:
            return None
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=cfg.jpeg_quality)
    out = buf.getvalue()
    return out if len(out) >= cur.min_bytes else None


class _Guard:
    """Wall-clock + byte budget shared by every in-flight download task."""

    def __init__(self, now, deadline: float | None, byte_limit: int):
        self._now = now
        self._deadline = deadline
        self._byte_limit = byte_limit
        self.bytes = 0
        self.reason = "bytes" if byte_limit <= 0 else ""

    def expired(self) -> bool:
        if self.reason:
            return True
        if self._deadline is not None and self._now() >= self._deadline:
            self.reason = "budget"
            return True
        return False

    def add_bytes(self, n: int) -> bool:
        """Account written bytes; False once the working-dir guard trips."""
        self.bytes += n
        if self.bytes >= self._byte_limit:
            self.reason = "bytes"
            return False
        return True


_LINEAGE_KEYS = ("order", "family", "genus", "species")


async def _download_species(
    cfg: CampaignConfig,
    root: Path,
    species_key: int,
    index: SamplingIndex,
    fetch,
    sem: asyncio.Semaphore,
    stop: asyncio.Event,
    guard: _Guard,
    stats: dict,
    now,
) -> bool:
    """Download one species up to the cap. True = species is complete."""
    rows = index.rows_for(species_key)
    if not rows:
        return True
    # Canonical taxonomic dir from the first complete lineage; drop rows whose
    # lineage disagrees (synonym records would land in the wrong folder).
    base = None
    for r in rows:
        if all(r.get(k) for k in _LINEAGE_KEYS):
            base = tuple(ImageDownloader.safe_segment(r[k]) for k in _LINEAGE_KEYS)
            break
    if base is None:
        return True
    rows = [
        r
        for r in rows
        if tuple(ImageDownloader.safe_segment(r[k]) for k in _LINEAGE_KEYS) == base
    ]
    species_dir = root.joinpath(*base)
    existing = len(list(species_dir.glob("*.jpg"))) if species_dir.exists() else 0
    need = cfg.max_images_per_species - existing
    if need <= 0:
        stats["skipped_full"] += 1
        return True
    saved = 0

    async def one(url: str) -> None:
        nonlocal saved
        if stop.is_set() or guard.expired() or saved >= need:
            return
        async with sem:
            if stop.is_set() or saved >= need:
                return  # cap reached while queued: no network spent
            data = await fetch(url)
        if data is None:
            stats["failed"] += 1
            return
        out = prepare_image(data, cfg)
        if out is None:
            stats["rejected"] += 1
            return
        digest = hashlib.sha256(out).hexdigest()
        dest = species_dir / f"{digest}.jpg"
        if dest.exists():
            stats["duplicate"] += 1
            return
        if saved >= need:
            return  # the cap filled meanwhile (no await below this line)
        species_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(out)
        tmp.replace(dest)
        saved += 1
        stats["ok"] += 1
        if not guard.add_bytes(len(out)):
            stop.set()

    await asyncio.gather(*(one(r["url"]) for r in rows))
    return not guard.expired()


async def run_download(
    cfg: CampaignConfig,
    shard_dir: str | Path,
    species_keys: list[int],
    index: SamplingIndex,
    fetch,
    *,
    budget_min: float = 0.0,
    now=None,
) -> dict:
    """Fill one shard: every species up to its cap, resumably.

    ``fetch(url) -> bytes | None`` is already rate-limited/retried. Stops
    cleanly on the wall-clock budget or the working-dir byte guard; unvisited
    species stay queued for the next run.
    """
    now = now or time.monotonic
    root = Path(shard_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing_bytes = sum(p.stat().st_size for p in root.rglob("*.jpg"))
    deadline = now() + budget_min * 60 if budget_min > 0 else None
    guard = _Guard(now, deadline, max(cfg.max_shard_bytes - existing_bytes, 0))
    sem = asyncio.Semaphore(cfg.download_concurrency)
    stop = asyncio.Event()
    stats = {
        "ok": 0, "failed": 0, "rejected": 0, "duplicate": 0,
        "skipped_full": 0, "species_done": 0, "bytes": 0, "stop": "",
    }
    queue: asyncio.Queue[int] = asyncio.Queue()
    for k in species_keys:
        queue.put_nowait(k)
    incomplete = 0  # popped but unfinished: work for the next run too

    async def worker() -> None:
        nonlocal incomplete
        while not stop.is_set():
            try:
                key = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            done = await _download_species(
                cfg, root, key, index, fetch, sem, stop, guard, stats, now
            )
            if done:
                stats["species_done"] += 1
            else:
                incomplete += 1
                if guard.reason:
                    stop.set()

    await asyncio.gather(*(worker() for _ in range(cfg.species_concurrency)))
    stats["bytes"] = guard.bytes
    stats["stop"] = guard.reason
    stats["remaining"] = queue.qsize() + incomplete
    log.info("download shard %s: %s", root, stats)
    return stats


def download_sync(
    cfg: CampaignConfig,
    shard_dir: str | Path,
    species_keys: list[int],
    index: SamplingIndex,
    budget_min: float = 0.0,
) -> dict:
    async def go():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            dl = ImageDownloader(
                DownloadConfig(
                    concurrency=cfg.download_concurrency,
                    min_bytes=cfg.curation.min_bytes,
                    raw_root=str(shard_dir),
                )
            )

            async def fetch(url: str) -> bytes | None:
                return await dl.fetch(session, url)

            return await run_download(
                cfg, shard_dir, species_keys, index, fetch, budget_min=budget_min
            )

    return run_async(go())


def write_shard_manifest(shard_dir: str | Path) -> Path:
    """Walk the shard's image tree into ``manifest.parquet`` (atomic)."""
    root = Path(shard_dir)
    rows = []
    for p in sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root)
        if len(rel.parts) != 5:
            continue  # order/family/genus/species/hash.jpg only
        o, f, g, s, _name = rel.parts
        rows.append(
            {
                "image_path": rel.as_posix(),
                "order": o,
                "family": f,
                "genus": g,
                "species": s,
            }
        )
    df = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
    return write_parquet_atomic(df, root / "manifest.parquet")


# -------------------------------------------------------------------- manifest


def merge_manifests(
    shard_manifests: list[tuple[str, Path]], out_path: str | Path
) -> pl.DataFrame:
    """Concat per-shard manifests into the training manifest (+ ``shard``).

    ``shard_manifests`` is [(slug, manifest.parquet path)] in a deterministic
    order (sort by slug first): the merged row order must be stable across
    runs so train/val splits stay reproducible.
    """
    frames = []
    for slug, path in sorted(shard_manifests, key=lambda t: t[0]):
        df = pl.read_parquet(path)
        frames.append(df.with_columns(pl.lit(slug, dtype=pl.Utf8).alias("shard")))
    if frames:
        out = pl.concat(frames, how="vertical_relaxed")
    else:
        out = pl.DataFrame(schema={**MANIFEST_SCHEMA, "shard": pl.Utf8})
    write_parquet_atomic(out, out_path)
    return out
