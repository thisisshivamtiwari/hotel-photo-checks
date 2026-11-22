"""A short terminal walk-through of recorded results (no model calls), used for the demo video."""

import json
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import yaml

from . import combine, rules

ROOT = Path(__file__).resolve().parents[2]
SHOW = ["p27-bright", "p07-tilt", "p30-tilt", "p11", "p05"]
SHORT = {"blur": "blur", "exposure": "light", "framing": "tilt"}
WORD = {"ok": "ok", "problem": "BLURRY", "dark": "TOO DARK", "bright": "TOO BRIGHT", "tilted": "TILTED", "unknown": "can't tell"}


def run(pause: float = 0.6) -> None:
    per = json.loads((ROOT / "results" / "per_photo.json").read_text())
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    hotels = yaml.safe_load((ROOT / "data" / "hotels.yaml").read_text())
    s = json.loads((ROOT / "results" / "summary.json").read_text())

    def say(line=""):
        print(line, flush=True)
        time.sleep(pause)

    say("hotel-photo-checks: recorded results, no model calls")
    say()
    for n in SHOW:
        lab = labels[n]
        hotel = lab["hotel"]
        truth = lab["edit"] if lab["edit"] not in ("none", "dup") else "no defect"
        say(f"PHOTO {n:11} {hotel:18} truth: {truth}")
        for arm, name in (("rules", "rules"), ("sonnet-5.5", "sonnet-5.5"), ("both-sonnet", "rules+sonnet")):
            p = per[n][arm]
            checks = " ".join(f"{SHORT[c]}:{WORD[p[c]]:10}" for c in combine.TECH)
            say(f"  {name:12} {checks} -> {p['verdict'].upper()}")
        note = per[n]["sonnet-5.5"].get("note", "").split(". ")[0]
        for i, part in enumerate(textwrap.wrap(note, 62)[:2]):
            say(f"  {'sonnet note:' if i == 0 else '':12} {part}")
        facts = hotels[hotel]["facts"]
        final = per[n]["both-sonnet"]["verdict"]
        for a in [a for a, v in lab["amenities"].items() if v]:
            seen = per[n]["both-sonnet"]["amenities"].get(a)
            res = combine.contradiction(seen, facts[a])
            fact = {True: "yes", False: "NO", None: "blank"}[facts[a]]
            say(f"  fact sheet:  {a} = {fact:5} photo shows it: {res.upper() if res else 'no claim'}")
            if res == "contradiction":
                final = "fix"
            elif res in ("unverifiable", "cant_tell") and final == "ready" and facts[a] is not True:
                final = "person"
        say(f"  VERDICT (rules+sonnet): {final.upper()}")
        say()
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "upload-7.jpg"
        bad.write_bytes(b"\xff\xd8\xff\xe0 cut off mid-upload")
        gate = rules.load(bad)
    say(f"PHOTO upload-7.jpg UNREADABLE: {gate.reason}, no model called")
    say()
    say("TEST SET (120 photos)  bad passed as ready  good ready")
    for arm, name in (
        ("rules", "rules + CLIP"),
        ("haiku-4.5", "haiku-4.5"),
        ("sonnet-5.5", "sonnet-5.5"),
        ("both-sonnet", "rules + sonnet"),
    ):
        g = s["arms"][arm]["gate"]
        say(f"  {name:20} {g['bad_let_through']:>3}/{g['bad_photos']:<17} {g['good_ready']}/{g['good_photos']}")


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.6)
