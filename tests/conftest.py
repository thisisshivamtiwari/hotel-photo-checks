import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "data" / "photos" / "set"


def write_broken(d: Path) -> dict[str, Path]:
    """Inputs a real photo batch contains sooner or later. All are made here, none are committed."""
    good = SET / "p19.jpg"
    raw = good.read_bytes()
    files = {
        "empty": d / "empty.jpg",
        "truncated": d / "truncated.jpg",
        "not_an_image": d / "notes.jpg",
        "tiny": d / "tiny.jpg",
        "missing": d / "not-there.jpg",
        "cmyk": d / "cmyk.jpg",
        "grey16": d / "grey16.png",
        "rgba": d / "rgba.png",
        "webp": d / "photo.webp",
    }
    files["empty"].write_bytes(b"")
    files["truncated"].write_bytes(raw[: len(raw) // 3])
    files["not_an_image"].write_text("room 12 has a sea view, honest")
    Image.open(good).resize((64, 43)).save(files["tiny"])
    Image.open(good).convert("CMYK").save(files["cmyk"])
    arr = (np.asarray(Image.open(good).convert("L"), np.uint16) * 257).astype(np.uint16)
    Image.fromarray(arr).save(files["grey16"])
    Image.open(good).convert("RGBA").save(files["rgba"])
    buf = io.BytesIO()
    Image.open(good).save(buf, "WEBP")
    files["webp"].write_bytes(buf.getvalue())
    return files


@pytest.fixture
def broken(tmp_path):
    return write_broken(tmp_path)
