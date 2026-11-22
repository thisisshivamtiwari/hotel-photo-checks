"""Apply the pre-registered rules in evals/gate.yaml to results/summary.json."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def known_failures() -> set[str]:
    path = ROOT / "evals" / "known_failures.yaml"
    return {k["id"] for k in yaml.safe_load(path.read_text()) or []} if path.exists() else set()


def check(summary: dict) -> list[str]:
    spec = yaml.safe_load((ROOT / "evals" / "gate.yaml").read_text())
    arm = summary["arms"][spec["arm"]]
    g, c = arm["gate"], arm["contradictions"]
    rules_through = summary["arms"]["rules"]["gate"]["bad_let_through"]
    broken = []
    for rule in spec["rules"]:
        rid = rule["id"]
        ok = {
            "fewer-bad-through-than-rules": lambda r: g["bad_let_through"] < rules_through,
            "bad-through-at-most-10pct": lambda r: g["bad_let_through"] <= r["max_share"] * g["bad_photos"],
            "contradictions-caught-80pct": lambda r: c["caught"] >= r["min_share"] * c["expected"],
            "false-contradictions-at-most-3": lambda r: c["false_contradictions"] <= r["max_count"],
            "good-ready-at-least-half": lambda r: g["good_ready"] >= r["min_share"] * g["good_photos"],
        }[rid](rule)
        if not ok:
            broken.append(f"{rid}: {rule['what']}")
    return broken


def ci_problems(summary: dict) -> list[str]:
    """Broken rules that aren't recorded as known failures, plus known failures that now pass (update the record)."""
    broken = {b.split(":")[0]: b for b in check(summary)}
    known = known_failures()
    out = [b for rid, b in broken.items() if rid not in known]
    out += [f"{rid}: listed in known_failures.yaml but now passes" for rid in known - set(broken)]
    return out
