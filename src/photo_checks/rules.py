"""The cheap checks: an input gate, then classic image measures with thresholds tuned on the dev hotel only.

Every check answers "ok", a problem, or "unknown". A measure inside the band around its threshold is "unknown", and
so is any photo the gate can't read: an unknown is never folded into a pass.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "results" / "thresholds.json"
MIN_SIDE = 200  # below this a listing photo can't be judged for focus or framing
WORK_SIDE = 800


@dataclass
class Loaded:
    img: Image.Image | None
    reason: str | None = None  # why the gate refused it
    info: dict = field(default_factory=dict)


def load(path: Path) -> Loaded:
    """Open a photo for checking, or say why it can't be. Handles CMYK, greyscale, 16-bit, alpha and EXIF rotation."""
    path = Path(path)
    if not path.exists():
        return Loaded(None, "missing file")
    if path.stat().st_size == 0:
        return Loaded(None, "empty file")
    try:
        with Image.open(path) as im:
            im.load()  # a cut-off JPEG raises here
            img = ImageOps.exif_transpose(im)
            info = {"format": im.format, "mode": im.mode, "size": list(im.size)}
            if img.mode in ("I;16", "I;16B", "I;16L", "I"):
                arr = np.asarray(img, np.float32)
                img = Image.fromarray((arr / max(arr.max(), 1) * 255).astype(np.uint8))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return Loaded(None, f"unreadable ({type(e).__name__})")
    if min(img.size) < MIN_SIDE:
        return Loaded(None, f"too small to judge ({img.width}x{img.height})", info)
    scale = WORK_SIDE / max(img.size)
    if scale != 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return Loaded(img, None, info)


def measures(img: Image.Image) -> dict:
    """Raw numbers the rules threshold. Kept separate so tuning and the README can show them."""
    gray = np.asarray(img.convert("L"))
    lum = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "laplacian_var": float(lap.var()),
        "mean_lum": float(lum.mean()),
        "dark_frac": float((lum < 0.08).mean()),
        "clip_frac": float((lum > 0.97).mean()),
        **tilt_measure(gray),
    }


def tilt_measure(gray: np.ndarray) -> dict:
    """Length-weighted median slant of near-vertical straight lines (walls, door frames, poles). A rotated photo
    slants them all one way; looking up or down (keystone) slants the left and right ones in opposite directions,
    so the median stays near zero. Fewer than 4 such lines: no reading."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    side = min(gray.shape)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=int(side * 0.15), maxLineGap=4)
    devs, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4) if lines is not None else []:
        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            continue
        dev = np.degrees(np.arctan2(dx, dy))  # 0 = vertical
        dev = (dev + 90) % 180 - 90
        if abs(dev) <= 20:
            devs.append(dev)
            weights.append(np.hypot(dx, dy))
    if len(devs) < 4:
        return {"tilt_deg": None, "vertical_lines": len(devs)}
    order = np.argsort(devs)
    cum = np.cumsum(np.asarray(weights)[order])
    median = float(np.asarray(devs)[order][np.searchsorted(cum, cum[-1] / 2)])
    return {"tilt_deg": median, "vertical_lines": len(devs)}


def phash(img: Image.Image) -> str:
    return str(imagehash.phash(img))


def hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def thresholds() -> dict:
    return json.loads(THRESHOLDS.read_text())


def _band(value: float, t: float, band: float, low_is_problem: bool) -> str:
    """'problem' / 'ok' / 'unknown' for a value against threshold t with a relative unknown band."""
    lo, hi = t * (1 - band), t * (1 + band)
    if lo <= value <= hi:
        return "unknown"
    return "problem" if (value < lo) == low_is_problem else "ok"


def judge(m: dict, th: dict) -> dict:
    blur = _band(m["laplacian_var"], th["blur_laplacian"], th["band"], low_is_problem=True)
    dark = _band(m["mean_lum"], th["dark_mean"], th["band"], low_is_problem=True)
    bright = _band(m["clip_frac"], th["bright_clip"], th["band"], low_is_problem=False)
    if dark == "problem":
        exposure = "dark"
    elif bright == "problem":
        exposure = "bright"
    elif "unknown" in (dark, bright):
        exposure = "unknown"
    else:
        exposure = "ok"
    if m["tilt_deg"] is None:
        framing = "unknown"
    else:
        f = _band(abs(m["tilt_deg"]), th["tilt_deg"], th["band"], low_is_problem=False)
        framing = {"problem": "tilted"}.get(f, f)
    return {"blur": blur, "exposure": exposure, "framing": framing}
