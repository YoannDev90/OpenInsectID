"""Automated curation (section 1.3 of the plan).

Validates visual integrity: drop corrupt files, images below a minimum
resolution, and extreme aspect ratios. A full pass writes a manifest with a
per-image status so nothing is silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurationConfig:
    min_resolution: int = 224
    max_aspect_ratio: float = 3.0
    min_aspect_ratio: float = 1 / 3.0
    min_bytes: int = 1024
    expected_modes: tuple[str, ...] = ("RGB", "RGBA", "L", "P")


@dataclass
class CurationVerdict:
    path: str
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0


def curate_image(path: str | Path, config: CurationConfig | None = None) -> CurationVerdict:
    """Validate a single image file. Never raises on a bad file."""
    from PIL import Image, UnidentifiedImageError

    cfg = config or CurationConfig()
    path = Path(path)
    try:
        size_bytes = path.stat().st_size
    except OSError as e:
        return CurationVerdict(str(path), False, f"stat_failed: {e}")
    if size_bytes < cfg.min_bytes:
        return CurationVerdict(str(path), False, "too_small_bytes")

    try:
        with Image.open(path) as img:
            img.verify()  # integrity check (corrupt JPEG/PNG detection)
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return CurationVerdict(str(path), False, f"decode_failed: {e}")

    if mode not in cfg.expected_modes:
        return CurationVerdict(str(path), False, f"bad_mode: {mode}", width, height)

    if min(width, height) < cfg.min_resolution:
        return CurationVerdict(
            str(path), False, f"low_resolution: {min(width,height)}", width, height
        )

    ratio = max(width, height) / max(min(width, height), 1)
    if ratio > cfg.max_aspect_ratio or ratio < cfg.min_aspect_ratio:
        return CurationVerdict(
            str(path), False, f"extreme_aspect: {ratio:.2f}", width, height
        )

    return CurationVerdict(str(path), True, "", width, height)


def curate_directory(
    root: str | Path,
    config: CurationConfig | None = None,
    delete_invalid: bool = False,
    suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> list[CurationVerdict]:
    """Walk an image tree and validate every file.

    Args:
        delete_invalid: when True, remove files that fail validation.
    """
    root = Path(root)
    verdicts: list[CurationVerdict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in suffixes or not path.is_file():
            continue
        v = curate_image(path, config)
        verdicts.append(v)
        if delete_invalid and not v.ok:
            try:
                path.unlink()
            except OSError:
                pass
    return verdicts
