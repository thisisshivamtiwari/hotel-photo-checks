"""Score every arm on the test hotels and write results/summary.json (the README tables are generated from it).

Arms: rules (image measures + CLIP for amenities + pHash for duplicates), haiku-4.5, sonnet-5.5, and both
(rules + each LLM, merged by combine.merge). Duplicates are a pairwise check, so only the hash rule answers them.
"""

import json
import time
from itertools import combinations
from math import comb
from pathlib import Path

import yaml

from . import clip_scores, combine, rules, tune, vlm
from .dataset import AMENITIES

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "data" / "photos" / "set"
RESULTS = ROOT / "results"
LLMS = ["haiku-4.5", "sonnet-5.5"]
ARMS = ["rules", "haiku-4.5", "sonnet-5.5", "both-haiku", "both-sonnet", "strict-haiku", "strict-sonnet"]  # strict-*: post hoc


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def predictions(labels: dict, retune: bool = False) -> tuple[dict, dict, dict]:
    t0 = time.perf_counter()
    meas = {n: rules.measures(rules.load(SET / f"{n}.jpg").img) for n in labels}
    rule_secs = (time.perf_counter() - t0) / len(labels)
    clip = clip_scores.load(labels)
    th = tune.tune(labels, meas, clip) if retune or not rules.THRESHOLDS.exists() else rules.thresholds()
    hashes = {n: rules.phash(rules.load(SET / f"{n}.jpg").img) for n in labels}
    preds = {arm: {} for arm in ARMS}
    cost = {"rules": {"usd": 0.0, "seconds": round(rule_secs + clip["seconds_per_photo"], 2)}}
    for n in labels:
        preds["rules"][n] = {**rules.judge(meas[n], th), "amenities": combine.clip_amenities(clip["photos"][n], th)}
    for m in LLMS:
        usd = ms = 0.0
        unparsed = 0
        for n in labels:
            rec = vlm.call(m, n, SET / f"{n}.jpg")
            p = vlm.parse("" if rec.get("is_error") else rec["text"])
            unparsed += not p["parsed"]
            usd, ms = usd + rec["cost_usd"], ms + rec["ms"]
            preds[m][n] = p
            r = preds["rules"][n]
            short = m.split("-")[0]
            preds[f"both-{short}"][n] = {
                **{c: combine.merge(r[c], p[c]) for c in combine.TECH},
                "amenities": combine.merge_amenities(r["amenities"], p["amenities"]),
            }
            preds[f"strict-{short}"][n] = {
                **{c: combine.merge_strict(r[c], p[c]) for c in combine.TECH},
                "amenities": combine.merge_amenities(r["amenities"], p["amenities"], combine.merge_strict),
            }
        cost[m] = {"usd": round(usd / len(labels), 5), "seconds": round(ms / len(labels) / 1000, 2), "unparsed": unparsed}
        for kind in ("both", "strict"):
            cost[f"{kind}-{m.split('-')[0]}"] = {
                "usd": cost[m]["usd"],
                "seconds": round(cost[m]["seconds"] + cost["rules"]["seconds"], 1),
            }
    return preds, {"thresholds": th, "measures": meas, "hashes": hashes}, cost


def tech_scores(labels, preds, names):
    out = {}
    for c in combine.TECH:
        s = {"problems": 0, "caught": 0, "missed": 0, "wrong_kind": 0, "unknown_on_problem": 0,
             "fine": 0, "false_alarm": 0, "unknown_on_fine": 0}  # fmt: skip
        for n in names:
            g, p = labels[n][c], preds[n][c]
            if g == "ok":
                s["fine"] += 1
                s["false_alarm"] += p not in ("ok", "unknown")
                s["unknown_on_fine"] += p == "unknown"
            else:
                s["problems"] += 1
                s["caught"] += p == g
                s["missed"] += p == "ok"
                s["unknown_on_problem"] += p == "unknown"
                s["wrong_kind"] += p not in (g, "ok", "unknown")
        out[c] = s
    return out


def photo_verdict(pred):
    """ready (every check ok), fix (a problem found), or person (nothing wrong found, but something unknown)."""
    vals = [pred[c] for c in combine.TECH]
    if all(v == "ok" for v in vals):
        return "ready"
    if any(v not in ("ok", "unknown") for v in vals):
        return "fix"
    return "person"


def gate_scores(labels, preds, names):
    bad = [n for n in names if any(labels[n][c] != "ok" for c in combine.TECH)]
    good = [n for n in names if n not in bad]
    v = {n: photo_verdict(preds[n]) for n in names}
    return {
        "bad_photos": len(bad),
        "bad_let_through": sum(v[n] == "ready" for n in bad),
        "bad_to_person": sum(v[n] == "person" for n in bad),
        "bad_flagged": sum(v[n] == "fix" for n in bad),
        "good_photos": len(good),
        "good_ready": sum(v[n] == "ready" for n in good),
        "good_to_person": sum(v[n] == "person" for n in good),
        "good_flagged": sum(v[n] == "fix" for n in good),
    }, v


def amenity_scores(labels, preds, names):
    out = {}
    for a in AMENITIES:
        s = {"shown": 0, "absent": 0, "correct": 0, "wrong": 0, "unknown": 0, "false_yes": 0, "false_no": 0}
        for n in names:
            g, p = labels[n]["amenities"][a], preds[n]["amenities"].get(a)
            s["shown" if g else "absent"] += 1
            if p is None:
                s["unknown"] += 1
            elif p == g:
                s["correct"] += 1
            else:
                s["wrong"] += 1
                s["false_yes" if p else "false_no"] += 1
        out[a] = s
    return out


def contradiction_scores(labels, preds, names, hotels):
    s = {"expected": 0, "caught": 0, "missed": 0, "cant_tell": 0, "false_contradictions": 0,
         "unverifiable_expected": 0, "unverifiable_given": 0, "naive_confirmed": 0,
         "examples_missed": [], "examples_false": []}  # fmt: skip
    for n in names:
        facts = hotels[labels[n]["hotel"]]["facts"]
        for a in AMENITIES:
            truth = combine.contradiction(labels[n]["amenities"][a], facts[a])
            got = combine.contradiction(preds[n]["amenities"].get(a), facts[a])
            if truth == "contradiction":
                s["expected"] += 1
                s["caught"] += got == "contradiction"
                s["missed"] += got is None or got == "consistent"
                s["cant_tell"] += got == "cant_tell"
                if got is None:
                    s["examples_missed"].append(f"{n}:{a}")
            elif got == "contradiction":
                s["false_contradictions"] += 1
                s["examples_false"].append(f"{n}:{a}")
            if truth == "unverifiable":
                s["unverifiable_expected"] += 1
                s["unverifiable_given"] += got == "unverifiable"
            # a scorer that reads a blank fact as "has it" would have confirmed these
            if facts[a] is None and preds[n]["amenities"].get(a):
                s["naive_confirmed"] += 1
    s["examples_missed"], s["examples_false"] = s["examples_missed"][:12], s["examples_false"][:12]
    return s


def duplicate_scores(labels, hashes, th, names):
    s = {"pairs": 0, "true_pairs": 0, "caught": 0, "possible": 0, "missed": 0, "false_pairs": 0, "false_possible": 0,
         "excluded_same_shot_edits": 0, "hard_pairs": {}}  # fmt: skip
    hard = {("p26", "p27"), ("p03", "p04")}
    by_hotel = {}
    for n in names:
        by_hotel.setdefault(labels[n]["hotel"], []).append(n)
    for group in by_hotel.values():
        for x, y in combinations(sorted(group), 2):
            lx, ly = labels[x], labels[y]
            same = lx["source"] == ly["source"]
            is_dup = same and {lx["edit"], ly["edit"]} == {"none", "dup"}
            if same and not is_dup:
                s["excluded_same_shot_edits"] += 1
                continue
            d = rules.hamming(hashes[x], hashes[y])
            call = "dup" if d <= th["dup_max_bits"] else "possible" if d <= th["dup_max_bits"] + th["dup_unknown_bits"] else "no"
            s["pairs"] += 1
            if is_dup:
                s["true_pairs"] += 1
                s["caught" if call == "dup" else "possible" if call == "possible" else "missed"] += 1
            else:
                s["false_pairs"] += call == "dup"
                s["false_possible"] += call == "possible"
            if (x, y) in hard:
                s["hard_pairs"][f"{x}~{y}"] = {"bits": d, "call": call}
    return s


def run(retune: bool = False) -> dict:
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    hotels = yaml.safe_load((ROOT / "data" / "hotels.yaml").read_text())
    preds, aux, cost = predictions(labels, retune)
    test = [n for n in labels if labels[n]["split"] == "test"]
    summary = {"test_photos": len(test), "test_hotels": [h for h, v in hotels.items() if v["split"] == "test"],
               "thresholds": aux["thresholds"], "cost": cost, "arms": {}}  # fmt: skip
    verdicts = {}
    for arm in ARMS:
        gate, verdicts[arm] = gate_scores(labels, preds[arm], test)
        summary["arms"][arm] = {
            "gate": gate,
            "checks": tech_scores(labels, preds[arm], test),
            "amenities": amenity_scores(labels, preds[arm], test),
            "contradictions": contradiction_scores(labels, preds[arm], test, hotels),
        }
    summary["duplicates"] = duplicate_scores(labels, aux["hashes"], aux["thresholds"], test)
    # paired test on "bad photo let through" between arms (same photos)
    bad = [n for n in test if any(labels[n][c] != "ok" for c in combine.TECH)]
    pairs = {}
    for a, b in [("rules", "sonnet-5.5"), ("rules", "haiku-4.5"), ("sonnet-5.5", "both-sonnet"), ("haiku-4.5", "both-haiku"),
                 ("rules", "both-sonnet"), ("both-sonnet", "strict-sonnet"), ("rules", "strict-sonnet")]:  # fmt: skip
        only_a = sum(verdicts[a][n] == "ready" and verdicts[b][n] != "ready" for n in bad)
        only_b = sum(verdicts[b][n] == "ready" and verdicts[a][n] != "ready" for n in bad)
        pairs[f"{a} vs {b}"] = {
            "only_first_lets_through": only_a,
            "only_second_lets_through": only_b,
            "p": round(mcnemar_p(only_a, only_b), 4),
        }
    summary["paired_bad_let_through"] = pairs
    by_edit = {}
    for arm in ARMS:
        by_edit[arm] = {}
        for n in test:
            e = labels[n]["edit"]
            d = by_edit[arm].setdefault(e, {"n": 0, "ready": 0, "person": 0, "fix": 0})
            d["n"] += 1
            d[verdicts[arm][n]] += 1
    summary["verdict_by_edit"] = by_edit
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    per_photo = {n: {arm: {**preds[arm][n], "verdict": photo_verdict(preds[arm][n])} for arm in ARMS} for n in labels}
    (RESULTS / "per_photo.json").write_text(json.dumps(per_photo, indent=1, default=str) + "\n")
    return summary
