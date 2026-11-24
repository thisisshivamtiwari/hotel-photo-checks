"""List candidate photos from Wikimedia Commons for one search per scene, keeping only open licences and dropping
anything that names a hotel. Writes .cache/candidates.json and contact sheets for a hand review; the photos picked
from it are pinned in data/sources.yaml (scripts/fetch_photos.py downloads exactly those)."""

import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
UA = "hotel-photo-checks/0.1 (https://github.com/thisisshivamtiwari/hotel-photo-checks)"
API = "https://commons.wikimedia.org/w/api.php"
SEARCHES = {
    "pool": "outdoor swimming pool filetype:bitmap",
    "gym": "fitness gym treadmill filetype:bitmap",
    "bathtub": "bathtub bathroom filetype:bitmap",
    "shower": "shower bathroom filetype:bitmap",
    "bedroom": "bedroom bed interior filetype:bitmap",
    "fireplace": "fireplace living room filetype:bitmap",
    "beach": "beach sea view filetype:bitmap",
    "dining": "dining room interior filetype:bitmap",
}
OPEN = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? \d(\.\d)?)", re.I)
BANNED = re.compile(r"hotel|motel|resort|hostel|\binn\b|lodge|marriott|hilton|hyatt|ibis|airbnb|booking", re.I)


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def strip(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def candidates(scene: str, query: str, n: int = 24) -> list[dict]:
    data = get({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": 50,
        "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1024,
    })  # fmt: skip
    out = []
    for page in data.get("query", {}).get("pages", {}).values():
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        lic = strip(meta.get("LicenseShortName", {}).get("value", ""))
        desc = strip(meta.get("ImageDescription", {}).get("value", ""))
        cats = meta.get("Categories", {}).get("value", "")
        if info.get("mime") != "image/jpeg" or info["width"] < 1000 or not OPEN.match(lic):
            continue
        if BANNED.search(page["title"] + " " + desc + " " + cats):
            continue
        out.append({
            "scene": scene, "title": page["title"], "license": lic, "artist": strip(meta.get("Artist", {}).get("value", "")),
            "page": info["descriptionurl"], "thumb": info["thumburl"], "width": info["width"], "height": info["height"],
        })  # fmt: skip
    return out[:n]


def sheet(items: list[dict], path: Path) -> None:
    if not items:
        return
    cols, w, h = 6, 220, 165
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, it in enumerate(items):
        req = urllib.request.Request(it["thumb"], headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                im = Image.open(io.BytesIO(r.read()), formats=["JPEG", "PNG"]).convert("RGB")  # never Ghostscript on web bytes
        except Exception as e:  # noqa: BLE001
            print("skip", it["title"], e, file=sys.stderr)
            continue
        im.thumbnail((w, h))
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        canvas.paste(im, (x, y))
        draw.text((x + 2, y + h + 2), f"{i}", fill="black")
        time.sleep(0.2)
    canvas.save(path, quality=80)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    allc = {}
    for scene, q in SEARCHES.items():
        allc[scene] = candidates(scene, q)
        print(scene, len(allc[scene]))
        sheet(allc[scene], CACHE / f"sheet-{scene}.jpg")
    (CACHE / "candidates.json").write_text(json.dumps(allc, indent=1))
