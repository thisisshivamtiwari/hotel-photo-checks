"""Build the photo set: each source photo resized to 800 px on the long side, plus three edited copies with a known
defect. Labels are known by construction (edits) or by hand (data/amenities.yaml), and written to data/labels.json.

Each source gets three of the six edits, alternating between two patterns so every edit is used 20 times:
  even sources: blur, dark, tilt        odd sources: motion, bright, near-duplicate
Severities are drawn from a seeded generator per photo, inside ranges where the defect is plainly visible at 800 px.
"""

import hashlib
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SET = DATA / "photos" / "set"
LONG_SIDE = 800
AMENITIES = ["pool", "gym", "bathtub", "sea_view", "fireplace"]
PATTERNS = (["blur", "dark", "tilt"], ["motion", "bright", "dup"])


def normalise(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    scale = LONG_SIDE / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def motion_blur(img: Image.Image, length: int, angle: float) -> Image.Image:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return Image.fromarray(cv2.filter2D(np.asarray(img), -1, k))


def exposure(img: Image.Image, factor: float) -> Image.Image:
    """Scale linear light by `factor` (sRGB decoded with gamma 2.2), so 'dark' and 'bright' look like a wrong
    exposure rather than a contrast change; highlights clip above 1."""
    lin = (np.asarray(img, np.float32) / 255.0) ** 2.2
    out = np.clip(lin * factor, 0, 1) ** (1 / 2.2)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


def tilt(img: Image.Image, degrees: float) -> Image.Image:
    """Rotate, then crop the largest centred rectangle with no empty corners and scale back, so the only cue is the
    slant (no black wedges for a rule to latch onto)."""
    w, h = img.size
    rot = img.rotate(degrees, resample=Image.BICUBIC, expand=False)
    a = np.radians(abs(degrees))
    # largest axis-aligned rectangle inside a w x h rectangle rotated by a, same aspect ratio
    s = min(w / (w * np.cos(a) + h * np.sin(a)), h / (w * np.sin(a) + h * np.cos(a)))
    cw, ch = int(w * s), int(h * s)
    left, top = (w - cw) // 2, (h - ch) // 2
    return rot.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def near_duplicate(img: Image.Image) -> Image.Image:
    """The same shot re-exported: 3% trimmed off each edge, slightly brighter, a low-quality JPEG round trip."""
    w, h = img.size
    dx, dy = int(w * 0.03), int(h * 0.03)
    out = img.crop((dx, dy, w - dx, h - dy))
    out = ImageEnhance.Brightness(out).enhance(1.05)
    small = out.resize((int(out.width * 0.7), int(out.height * 0.7)), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=65)
    return Image.open(io.BytesIO(buf.getvalue()), formats=["JPEG"]).convert("RGB").resize(img.size, Image.LANCZOS)


def edit(img: Image.Image, kind: str, rng: random.Random) -> tuple[Image.Image, dict]:
    if kind == "blur":
        r = round(rng.uniform(3.0, 6.0), 2)
        return img.filter(ImageFilter.GaussianBlur(r)), {"radius_px": r}
    if kind == "motion":
        n, ang = rng.randint(17, 31), round(rng.uniform(0, 180), 1)
        return motion_blur(img, n, ang), {"length_px": n, "angle": ang}
    if kind == "dark":
        f = round(rng.uniform(0.06, 0.14), 3)
        return exposure(img, f), {"factor": f}
    if kind == "bright":
        f = round(rng.uniform(3.0, 5.0), 2)
        return exposure(img, f), {"factor": f}
    if kind == "tilt":
        d = round(rng.uniform(5.0, 10.0), 1) * rng.choice([-1, 1])
        return tilt(img, d), {"degrees": d}
    if kind == "dup":
        return near_duplicate(img), {}
    raise ValueError(kind)


def label(source: str, kind: str, hotel: str, split: str, amen: list[str], params: dict) -> dict:
    return {
        "source": source,
        "edit": kind,
        "params": params,
        "hotel": hotel,
        "split": split,
        "blur": "problem" if kind in ("blur", "motion") else "ok",
        "exposure": {"dark": "dark", "bright": "bright"}.get(kind, "ok"),
        "framing": "tilted" if kind == "tilt" else "ok",
        "dup_of": source if kind == "dup" else None,
        "amenities": {a: a in amen for a in AMENITIES},
    }


def build(src_dir: Path = ROOT / ".cache" / "original") -> dict:
    hotels = yaml.safe_load((DATA / "hotels.yaml").read_text())
    amen = yaml.safe_load((DATA / "amenities.yaml").read_text())
    where = {p: (h, v["split"]) for h, v in hotels.items() for p in v["photos"]}
    SET.mkdir(parents=True, exist_ok=True)
    labels = {}
    for i, source in enumerate(sorted(where)):
        hotel, split = where[source]
        img = normalise(Image.open(src_dir / f"{source}.jpg", formats=["JPEG"]))
        img.save(SET / f"{source}.jpg", quality=90)
        labels[source] = label(source, "none", hotel, split, amen[source], {})
        rng = random.Random(f"{source}-edits")
        for kind in PATTERNS[i % 2]:
            out, params = edit(img, kind, rng)
            name = f"{source}-{kind}"
            out.save(SET / f"{name}.jpg", quality=90)
            labels[name] = label(source, kind, hotel, split, amen[source], params)
    for name in labels:
        labels[name]["sha256"] = hashlib.sha256((SET / f"{name}.jpg").read_bytes()).hexdigest()[:16]
    (DATA / "labels.json").write_text(json.dumps(labels, indent=1) + "\n")
    return labels
