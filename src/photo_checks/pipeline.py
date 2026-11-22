"""Check a batch of photos for one hotel: the tool the evaluation measures, usable on any folder.

Each photo gets a verdict: "ready" (every check ok and nothing contradicts the fact sheet), "fix" (a problem was
found), "person" (nothing wrong found, but something couldn't be judged) or "unreadable" (the gate refused the file).
The batch summary counts each verdict separately. There is no single score, so a batch of unreadable or unjudged
photos can't come out looking perfect.
"""

import subprocess
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

from . import combine, rules, vlm
from .dataset import AMENITIES

LlmFn = Callable[[Path], dict]  # returns vlm.parse() output


def live_llm(alias: str) -> LlmFn:
    """Call a vision LLM through the Claude Code CLI. Any failure, or a reply that isn't the expected JSON, comes
    back as unparsed, which every check reads as unknown."""

    def fn(path: Path) -> dict:
        try:
            res = vlm._run_cli(alias, Path(path).read_bytes())
        except (subprocess.SubprocessError, OSError, ValueError):
            return vlm.parse("")
        return vlm.parse("" if res.get("is_error") else res.get("result", ""))

    return fn


def check_photo(path: Path, facts: dict | None, th: dict, llm: LlmFn | None = None, clip_scores: dict | None = None) -> dict:
    loaded = rules.load(path)
    if loaded.img is None:
        return {"photo": Path(path).name, "verdict": "unreadable", "reason": loaded.reason}
    r = rules.judge(rules.measures(loaded.img), th)
    r["amenities"] = combine.clip_amenities(clip_scores, th) if clip_scores else {a: None for a in AMENITIES}
    pred = r
    if llm is not None:
        p = llm(path)
        pred = {c: combine.merge(r[c], p[c]) for c in combine.TECH}
        pred["amenities"] = combine.merge_amenities(r["amenities"], p["amenities"]) if clip_scores else p["amenities"]
        pred["llm_parsed"] = p["parsed"]
    facts = facts or {}  # no fact sheet: every amenity is unverifiable, never "consistent"
    pred["facts"] = {a: combine.contradiction(pred["amenities"].get(a), facts.get(a)) for a in AMENITIES}
    problems = [c for c in combine.TECH if pred[c] not in ("ok", "unknown")]
    problems += [f"shows {a} but the fact sheet says no" for a, v in pred["facts"].items() if v == "contradiction"]
    unknowns = [c for c in combine.TECH if pred[c] == "unknown"]
    unknowns += [f"{a}: {v}" for a, v in pred["facts"].items() if v in ("unverifiable", "cant_tell") and facts.get(a) is not True]
    verdict = "fix" if problems else "person" if unknowns else "ready"
    return {"photo": Path(path).name, "verdict": verdict, "problems": problems, "unknowns": unknowns, "checks": pred,
            "phash": rules.phash(loaded.img)}  # fmt: skip


def check_batch(paths: list[Path], facts: dict | None, th: dict, llm: LlmFn | None = None, clip: dict | None = None) -> dict:
    photos = [check_photo(p, facts, th, llm, (clip or {}).get(Path(p).stem)) for p in paths]
    dups = []
    readable = [p for p in photos if p["verdict"] != "unreadable"]
    for a, b in combinations(readable, 2):
        d = rules.hamming(a["phash"], b["phash"])
        if d <= th["dup_max_bits"] + th["dup_unknown_bits"]:
            dups.append(
                {"photos": [a["photo"], b["photo"]], "bits": d, "call": "duplicate" if d <= th["dup_max_bits"] else "possible"}
            )
    counts = {v: sum(p["verdict"] == v for p in photos) for v in ("ready", "fix", "person", "unreadable")}
    return {"counts": counts, "photos": photos, "duplicates": dups, "fact_sheet": "missing" if facts is None else "given"}
