"""Data pipeline: GBIF ingestion, DuckDB/H3 spatial processing, async image
download, and automated curation (sections 1 and 2 of the plan).
"""

from .gbif import GBIFExtractor, GBIFConfig, TaxonLineage
from .spatial import SpatialProcessor, H3Config
from .downloader import ImageDownloader, DownloadConfig
from .curation import CurationConfig, curate_image, curate_directory

__all__ = [
    "GBIFExtractor",
    "GBIFConfig",
    "TaxonLineage",
    "SpatialProcessor",
    "H3Config",
    "ImageDownloader",
    "DownloadConfig",
    "CurationConfig",
    "curate_image",
    "curate_directory",
]
