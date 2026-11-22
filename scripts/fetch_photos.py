"""Download the pinned source photos (data/sources.yaml) as 1024 px JPEGs into .cache/original/ (not committed) and write
data/photos/LICENSES.csv with title, author, licence and source page for each one."""

import csv
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
UA = "hotel-photo-checks/0.1 (https://github.com/thisisshivamtiwari/hotel-photo-checks)"

if __name__ == "__main__":
    rows = yaml.safe_load((ROOT / "data/sources.yaml").read_text())
    out = ROOT / ".cache/original"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        path = out / f"{r['id']}.jpg"
        if not path.exists():
            req = urllib.request.Request(r["thumb"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                path.write_bytes(resp.read())
            time.sleep(0.5)
    with (ROOT / "data/photos/LICENSES.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "author", "license", "source"])
        for r in rows:
            w.writerow([r["id"], r["title"].removeprefix("File:"), r["artist"], r["license"], r["page"]])
    print(len(rows), "photos in", out)
