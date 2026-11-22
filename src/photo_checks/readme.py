"""Regenerate the results tables in README.md (between the results markers) from results/summary.json."""

from pathlib import Path

from .chart import NAMES
from .dataset import AMENITIES

ROOT = Path(__file__).resolve().parents[2]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def tables(s: dict) -> str:
    arms = list(NAMES)
    A = s["arms"]
    g0 = A["rules"]["gate"]
    out = [f"Test set: {s['test_photos']} photos from the three test hotels ({g0['bad_photos']} with a blur, exposure or tilt "
           f"defect, {g0['good_photos']} without; the 15 near-duplicates count as good here and are scored in the duplicate table).",
           "", "**What happens to each photo** (ready = every check ok; person = nothing wrong found but something couldn't be judged)", "",
           "| Checker | Bad photos passed as ready | Bad sent to a person | Bad flagged | Good ready | Good sent to a person | Good flagged | Cost per photo | Seconds per photo |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in arms:
        g, c = A[a]["gate"], s["cost"][a]
        out.append(f"| {NAMES[a]} | {g['bad_let_through']}/{g['bad_photos']} | {g['bad_to_person']} | {g['bad_flagged']} | "
                   f"{g['good_ready']}/{g['good_photos']} | {g['good_to_person']} | {g['good_flagged']} | ${c['usd']:.4f} | {c['seconds']} |")  # fmt: skip
    out += ["", "**Each check** (caught / missed = called ok / can't tell, out of the photos with that defect; false alarms out of photos without it)", "",
            "| Checker | Blur caught | Blur missed | Blur can't tell | Exposure caught | Exposure missed | Exposure can't tell | Tilt caught | Tilt missed | Tilt can't tell | False alarms (blur / exposure / tilt) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in arms:
        ch = A[a]["checks"]
        cells = []
        for c in ("blur", "exposure", "framing"):
            x = ch[c]
            cells += [f"{x['caught']}/{x['problems']}", str(x["missed"]), str(x["unknown_on_problem"])]
        fa = " / ".join(f"{ch[c]['false_alarm']}/{ch[c]['fine']}" for c in ("blur", "exposure", "framing"))
        out.append(f"| {NAMES[a]} | " + " | ".join(cells) + f" | {fa} |")
    out += ["", "**Photos that contradict the hotel's own fact sheet** (e.g. a pool photo from a hotel that says it has no pool)", "",
            "| Checker | Contradictions caught | Missed | Can't tell | False contradictions | Blank fact left unverified | Blank fact a naive scorer would confirm |",
            "|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in arms:
        c = A[a]["contradictions"]
        out.append(f"| {NAMES[a]} | {c['caught']}/{c['expected']} | {c['missed']} | {c['cant_tell']} | {c['false_contradictions']} | "
                   f"{c['unverifiable_given']}/{c['unverifiable_expected']} | {c['naive_confirmed']} |")  # fmt: skip
    out += ["", "**Amenity detection, per amenity** (right / wrong / can't tell over the 120 test photos)", "",
            "| Checker | " + " | ".join(AMENITIES) + " |", "|---|" + "---|" * len(AMENITIES)]  # fmt: skip
    for a in arms:
        am = A[a]["amenities"]
        out.append(
            f"| {NAMES[a]} | "
            + " | ".join(f"{am[x]['correct']} / {am[x]['wrong']} / {am[x]['unknown']}" for x in AMENITIES)
            + " |"
        )
    d = s["duplicates"]
    hard = ", ".join(f"{k} ({v['bits']} bits: {v['call']})" for k, v in d["hard_pairs"].items())
    out += ["", f"**Near-duplicates** (pHash, at most {s['thresholds']['dup_max_bits']} bits apart = duplicate, up to "
            f"{s['thresholds']['dup_max_bits'] + s['thresholds']['dup_unknown_bits']} = possible): {d['caught']}/{d['true_pairs']} "
            f"re-exported copies caught, {d['possible']} marked possible, {d['missed']} missed; {d['false_pairs']} false duplicates and "
            f"{d['false_possible']} false 'possible' among {d['pairs'] - d['true_pairs']} pairs of different shots. "
            f"Same room, different shot: {hard}. ({d['excluded_same_shot_edits']} pairs of edits of the same shot are not scored.)"]  # fmt: skip
    out += ["", "**Paired comparison, bad photos passed as ready** (exact McNemar on the same photos)", "",
            "| Pair | Only the first passes it | Only the second passes it | p |", "|---|---|---|---|"]  # fmt: skip
    for k, v in s["paired_bad_let_through"].items():
        a, b = k.split(" vs ")
        out.append(f"| {NAMES[a]} vs {NAMES[b]} | {v['only_first_lets_through']} | {v['only_second_lets_through']} | {v['p']} |")
    out += ["", "**Verdicts by edit** (ready / person / fix)", "", "| Edit | " + " | ".join(NAMES[a] for a in arms) + " |",
            "|---|" + "---|" * len(arms)]  # fmt: skip
    for e in ["none", "dup", "blur", "motion", "dark", "bright", "tilt"]:
        row = [s["verdict_by_edit"][a].get(e) for a in arms]
        out.append(f"| {e} (n={row[0]['n']}) | " + " | ".join(f"{r['ready']} / {r['person']} / {r['fix']}" for r in row) + " |")
    return "\n".join(out)


def write(summary: dict, path: Path = ROOT / "README.md") -> None:
    text = path.read_text()
    if START not in text:
        return
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(head + START + "\n" + tables(summary) + "\n" + END + tail)
