"""Generate the Kaggle dataset cover image (1080x1080) — no network, no data.

Usage: .venv/bin/python scripts/make_cover.py
Output: docs/cover-open-insect-id-v2-meta.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = H = 1080
BG = (14, 18, 23)  # near-black slate
ACCENT = (120, 220, 140)  # insect-green
MUTED = (150, 160, 172)
WHITE = (240, 244, 248)

OUT = Path(__file__).resolve().parent.parent / "docs" / "cover-open-insect-id-v2-meta.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Header rule
    d.rectangle([60, 60, W - 60, 68], fill=ACCENT)
    d.text((60, 96), "OPEN INSECT ID v2", font=font(54, bold=True), fill=WHITE)
    d.text(
        (60, 168),
        "Insect Species Census & Image Index",
        font=font(36),
        fill=ACCENT,
    )

    # Big stats block
    stats = [
        ("33,434", "insect species (≥50 GBIF occurrences)"),
        ("3.6M", "candidate images estimated from the census"),
        ("12", "image shards planned · 320 px q85 · ~168 GB"),
    ]
    y = 300
    for value, label in stats:
        d.text((60, y), value, font=font(96, bold=True), fill=WHITE)
        d.text((60, y + 116), label, font=font(30), fill=MUTED)
        y += 210

    # Footer
    d.rectangle([60, H - 150, W - 60, H - 144], fill=(40, 50, 60))
    d.text(
        (60, H - 124),
        "Metadata only · no images · GBIF-sourced",
        font=font(28),
        fill=MUTED,
    )
    d.text(
        (60, H - 80),
        "Data: CC BY 4.0 · Occurrences: GBIF.org",
        font=font(28),
        fill=MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
