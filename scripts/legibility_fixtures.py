"""Images with text of a known size, for the legibility gate. All made here; nothing real.

Two kinds: hotel paperwork pages (after the AI Lab generator of 25 Nov 2026, credited in README) rendered at 200 dpi
and then degraded to a 150 dpi phone photo and a 100 dpi 1-bit fax; and listing photos with a caption stamped on
at three glyph heights. Writes to the folder given (default results/legibility-fixtures, not committed)."""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "data" / "photos" / "set").glob("p??.jpg"))
CAPTION = "From £129 a night · call 01632 960123"  # 01632 is the UK's reserved fiction range
CAP_HEIGHTS = (36, 20, 10)


def font(px: int) -> ImageFont.FreeTypeFont:
    """A font whose capital letters are about px tall (Pillow's bundled font, so the same on every machine)."""
    f = ImageFont.load_default(size=px)
    _, top, _, bottom = f.getbbox("H")
    return ImageFont.load_default(size=round(px * px / max(bottom - top, 1)))


def page(seed: int) -> Image.Image:
    """A made-up rate card at A4 200 dpi with 30 px body text (about 21 px capitals), like the AI Lab pages."""
    r = random.Random(seed)
    img = Image.new("RGB", (1654, 2339), "white")
    d = ImageDraw.Draw(img)
    big, body = font(34), font(21)
    d.text((120, 140), r.choice(["Harbour Light Hotel", "Ashdown Park House", "Castle Bay Resort"]), font=big, fill="black")
    d.text((120, 230), "RATE CARD", font=big, fill="black")
    y = 340
    for room in ("Classic Double", "Superior King", "Junior Suite", "Family Room", "Garden Twin", "Penthouse"):
        d.text((120, y), room, font=body, fill="black")
        for i in range(3):
            d.text((760 + i * 260, y), f"£{r.uniform(79, 560):,.2f}", font=body, fill="black")
        y += 52
    d.text((120, y + 40), f"Target occupancy {r.randint(55, 96)}%   VAT {r.randint(5, 20)}% included", font=body, fill="black")
    return img


def degrade(img: Image.Image, cond: str, seed: int) -> Image.Image:
    r = random.Random(seed)
    w, h = img.size
    if cond == "clean":
        return img
    if cond == "phone":  # 150 dpi, soft, slightly tilted, uneven light
        out = img.filter(ImageFilter.GaussianBlur(1.1)).rotate(r.uniform(-2.5, 2.5), fillcolor="white")
        shade = Image.linear_gradient("L").resize((w, h)).rotate(r.choice([0, 90]))
        out = Image.composite(out, Image.new("RGB", (w, h), (200, 200, 200)), shade.point(lambda v: 120 + v // 2))
        return out.resize((int(w * 0.75), int(h * 0.75)), Image.BILINEAR)
    if cond == "fax":  # 100 dpi, 1-bit
        return img.resize((w // 2, h // 2), Image.BILINEAR).convert("L").point(lambda v: 255 if v > 150 else 0).convert("RGB")
    raise ValueError(cond)


def caption(src: Path, cap_px: int) -> Image.Image:
    img = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(img)
    f = font(cap_px)
    x, y = cap_px, img.height - 2 * cap_px - 10
    d.text((x, y), CAPTION, font=f, fill="white", stroke_width=max(1, cap_px // 12), stroke_fill="black")
    return img


def make(out: Path, seeds: int = 3, photos: int = 10) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(seeds):
        base = page(s)
        for cond in ("clean", "phone", "fax"):
            p = out / f"page-{s}-{cond}.png"
            degrade(base, cond, s).save(p)
            rows.append({"file": p.name, "kind": "page", "cond": cond, "cap_px": {"clean": 21, "phone": 16, "fax": 11}[cond]})
    for src in SOURCES[:photos]:
        for cap in CAP_HEIGHTS:
            p = out / f"{src.stem}-cap{cap}.jpg"
            caption(src, cap).save(p, quality=90)
            rows.append({"file": p.name, "kind": "photo", "cond": f"cap{cap}", "cap_px": cap})
    return rows


if __name__ == "__main__":
    rows = make(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "legibility-fixtures")
    print(len(rows), "fixtures")
