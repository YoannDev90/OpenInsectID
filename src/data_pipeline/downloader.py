"""Async, fault-tolerant image downloader (sections 1.2-1.3 of the plan).

Downloads occurrence images concurrently with ``asyncio`` + ``aiohttp``:
per-domain rate limiting, retry/backoff, URL + content de-duplication, and a
content-hash naming scheme under ``data/raw/{order}/{family}/{genus}/{species}/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    concurrency: int = 8
    per_domain_rps: float = 4.0
    timeout: float = 30.0
    max_retries: int = 3
    backoff: float = 1.5
    min_bytes: int = 1024  # discard tiny / placeholder images
    raw_root: str = "data/raw"


class _DomainLimiter:
    """Token-bucket rate limiter, one per host."""

    def __init__(self, rps: float):
        self.interval = 1.0 / max(rps, 1e-6)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


@dataclass
class DownloadResult:
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    seen_hashes: set[str] = field(default_factory=set)


class ImageDownloader:
    def __init__(self, config: DownloadConfig | None = None):
        self.config = config or DownloadConfig()
        self._limiters: dict[str, _DomainLimiter] = {}
        self._results = DownloadResult()

    # ------------------------------------------------------------- naming

    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def safe_segment(name: str | None, fallback: str = "unknown") -> str:
        if not name:
            return fallback
        return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or fallback

    def relative_path(self, record) -> Path:
        """``{order}/{family}/{genus}/{species}/{sha256}.jpg`` under raw_root."""
        lin = record.lineage
        seg = [self.safe_segment(getattr(lin, k)) for k in ("order", "family", "genus", "species")]
        return Path(*seg)

    # ------------------------------------------------------------- download

    def _limiter(self, url: str) -> _DomainLimiter:
        from urllib.parse import urlparse

        host = urlparse(url).netloc
        if host not in self._limiters:
            self._limiters[host] = _DomainLimiter(self.config.per_domain_rps)
        return self._limiters[host]

    async def _fetch(self, session, url: str) -> bytes | None:
        import aiohttp

        limiter = self._limiter(url)
        for attempt in range(self.config.max_retries):
            await limiter.acquire()
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status != 200:
                        log.debug("HTTP %s for %s", resp.status, url)
                        await asyncio.sleep(self.config.backoff ** attempt)
                        continue
                    data = await resp.read()
                    if len(data) < self.config.min_bytes:
                        return None
                    return data
            except Exception as e:  # noqa: BLE001 - transient network errors
                log.debug("retry %s (%s) after error: %s", url, attempt, e)
                await asyncio.sleep(self.config.backoff ** attempt)
        return None

    async def download_records(self, records, session) -> DownloadResult:
        """Download every image URL of each record into its taxonomic folder."""
        import aiohttp

        sem = asyncio.Semaphore(self.config.concurrency)
        root = Path(self.config.raw_root)
        self._results = DownloadResult()

        async with aiohttp.ClientSession() as session:
            async def one(record) -> None:
                base = root / self.relative_path(record)
                async with sem:
                    for url in record.image_urls:
                        data = await self._fetch(session, url)
                        if data is None:
                            self._results.failed += 1
                            continue
                        h = self.content_hash(data)
                        if h in self._results.seen_hashes:
                            self._results.skipped += 1
                            continue
                        self._results.seen_hashes.add(h)
                        dest = base / f"{h}.jpg"
                        if dest.exists():
                            self._results.skipped += 1
                            continue
                        base.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                        self._results.ok += 1

            tasks = [asyncio.create_task(one(r)) for r in records]
            await asyncio.gather(*tasks)
        return self._results

    def download_sync(self, records) -> DownloadResult:
        from ._compat import run_async

        return run_async(self.download_records(records, None))
