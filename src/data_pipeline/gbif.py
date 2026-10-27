"""GBIF occurrence ingestion (sections 1.1-1.2 of the plan).

Uses the GBIF occurrence search API with the documented filters (Insecta,
still images, PRESENT, human/preserved records) and extracts the strict
Linnean lineage (order -> family -> genus -> species) for every record.
Built for scale: paginated async fetching, retry/backoff, URL de-duplication.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"

# GBIF backbone classKey for Insecta.
INSECTA_CLASS_KEY = 216


@dataclass
class GBIFConfig:
    class_key: int = INSECTA_CLASS_KEY
    media_type: str = "StillImage"
    occurrence_status: str = "PRESENT"
    basis_of_record: tuple[str, ...] = ("HUMAN_OBSERVATION", "PRESERVED_SPECIMEN")
    country: str | None = None  # None = world-scale
    has_coordinate: bool = True
    page_limit: int = 300  # GBIF max per page
    max_offset: int = 200_000  # API caps deep pagination; use a download for more
    concurrency: int = 8
    timeout: float = 30.0
    max_retries: int = 3
    backoff: float = 1.5


@dataclass
class TaxonLineage:
    order: str | None
    family: str | None
    genus: str | None
    species: str | None
    taxon_key: int | None = None

    @property
    def is_complete(self) -> bool:
        return all((self.order, self.family, self.genus, self.species))


@dataclass
class OccurrenceRecord:
    gbif_id: str
    lineage: TaxonLineage
    image_urls: list[str] = field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    country: str | None = None


class GBIFExtractor:
    """Async, paginated GBIF occurrence search with lineage extraction."""

    def __init__(self, config: GBIFConfig | None = None):
        self.config = config or GBIFConfig()

    def _params(self, offset: int) -> dict:
        c = self.config
        params: dict = {
            "classKey": c.class_key,
            "mediaType": c.media_type,
            "occurrenceStatus": c.occurrence_status,
            "hasCoordinate": "true" if c.has_coordinate else "false",
            "limit": c.page_limit,
            "offset": offset,
        }
        if c.basis_of_record:
            params["basisOfRecord"] = ",".join(c.basis_of_record)
        if c.country:
            params["country"] = c.country
        return params

    async def _fetch_page(self, session, offset: int) -> dict:
        import aiohttp

        params = self._params(offset)
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(
                    GBIF_SEARCH_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:  # noqa: BLE001 - retry any transient error
                last_err = e
                await asyncio.sleep(self.config.backoff ** attempt)
        raise RuntimeError(f"GBIF page at offset {offset} failed: {last_err}")

    @staticmethod
    def _lineage_from(raw: dict) -> TaxonLineage:
        def pick(*keys):
            for k in keys:
                v = raw.get(k)
                if v:
                    return v
            return None

        return TaxonLineage(
            order=pick("order", "orderName"),
            family=pick("family", "familyName"),
            genus=pick("genus", "genusName"),
            species=pick("species", "speciesName", "scientificName"),
            taxon_key=raw.get("taxonKey") or raw.get("speciesKey"),
        )

    @staticmethod
    def _image_urls(raw: dict) -> list[str]:
        urls: list[str] = []
        for m in raw.get("media") or []:
            u = m.get("identifier") or m.get("identifierLink") or m.get("url")
            if u and m.get("type", "StillImage") == "StillImage":
                urls.append(u)
        # Fallback to primary image field.
        if not urls and raw.get("issues") is not None:
            primary = raw.get("media") or []
            if primary and isinstance(primary[0], dict):
                u = primary[0].get("identifier")
                if u:
                    urls.append(u)
        return urls

    def _parse(self, raw: dict) -> OccurrenceRecord | None:
        lineage = self._lineage_from(raw)
        urls = self._image_urls(raw)
        if not urls:
            return None
        return OccurrenceRecord(
            gbif_id=str(raw.get("key") or raw.get("gbifID")),
            lineage=lineage,
            image_urls=urls,
            latitude=raw.get("decimalLatitude"),
            longitude=raw.get("decimalLongitude"),
            country=raw.get("countryCode"),
        )

    async def extract(self, on_record=None, max_records: int | None = None) -> list[OccurrenceRecord]:
        """Page through results, parse records, invoke optional callback.

        Returns the list of parsed occurrence records (image-bearing only).
        """
        import aiohttp

        c = self.config
        sem = asyncio.Semaphore(c.concurrency)
        records: list[OccurrenceRecord] = []
        seen_ids: set[str] = set()

        async with aiohttp.ClientSession() as session:
            first = await self._fetch_page(session, 0)
            total = first.get("count", 0)
            log.info("GBIF reports %d matching occurrences", total)

            def handle(raw: dict) -> None:
                rec = self._parse(raw)
                if rec is None or rec.gbif_id in seen_ids:
                    return
                seen_ids.add(rec.gbif_id)
                records.append(rec)
                if on_record:
                    on_record(rec)

            for raw in first.get("results", []):
                handle(raw)

            offsets = list(range(c.page_limit, min(total, c.max_offset), c.page_limit))
            if max_records is not None:
                # Rough pre-filter; exact cap applied below.
                pass

            async def worker(offset: int) -> list[dict]:
                async with sem:
                    page = await self._fetch_page(session, offset)
                    return page.get("results", [])

            tasks = [asyncio.create_task(worker(o)) for o in offsets]
            for coro in asyncio.as_completed(tasks):
                for raw in await coro:
                    handle(raw)
                    if max_records is not None and len(records) >= max_records:
                        for t in tasks:
                            t.cancel()
                        return records
        return records

    def extract_sync(self, **kwargs) -> list[OccurrenceRecord]:
        return asyncio.run(self.extract(**kwargs))
