"""python -m photo_checks <command>

build-data   resize the downloaded sources and make the edited copies (needs scripts/fetch_photos.py first)
clip         recompute the CLIP amenity scores (needs `uv sync --extra clip`)
record       record vision-LLM checks for any photo not yet in cassettes/ (calls Claude via the CLI)
eval         score every arm from recordings -> results/summary.json, per_photo.json, README tables, chart
ci           replay: fail if the run no longer reproduces results/summary.json or breaks evals/gate.yaml
check DIR [--facts FILE.yaml] [--llm sonnet-5.5]   check a folder of photos (rules only unless --llm)
"""

import json
import sys
from pathlib import Path

from . import dataset, evaluate, gate, rules


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    if cmd == "build-data":
        print(len(dataset.build()), "images")
    elif cmd == "clip":
        from . import clip_scores

        labels = json.loads((dataset.DATA / "labels.json").read_text())
        print(clip_scores.compute(labels, dataset.SET)["seconds_per_photo"], "s/photo")
    elif cmd == "record":
        from . import record

        failed = record.record()
        print("failed:", failed or "none")
        return 1 if failed else 0
    elif cmd == "eval":
        summary = evaluate.run(retune="--retune" in argv)
        from . import chart, readme

        chart.draw(summary)
        readme.write(summary)
        broken = gate.check(summary)
        print(json.dumps({a: v["gate"] for a, v in summary["arms"].items()}, indent=1))
        print("gate (pre-registered):", "pass" if not broken else broken)
        print("ci:", gate.ci_problems(summary) or "pass")
    elif cmd == "ci":
        saved = (evaluate.RESULTS / "summary.json").read_text()
        summary = evaluate.run()
        now = (evaluate.RESULTS / "summary.json").read_text()
        problems = gate.ci_problems(summary)
        if json.loads(now) != json.loads(saved) | {"cost": json.loads(now)["cost"]}:
            problems.append("replay no longer reproduces results/summary.json")
        print("ci:", "pass" if not problems else problems)
        return 1 if problems else 0
    elif cmd == "check":
        import yaml

        from . import pipeline

        folder = Path(argv[1])
        facts = yaml.safe_load(Path(argv[argv.index("--facts") + 1]).read_text()) if "--facts" in argv else None
        llm = pipeline.live_llm(argv[argv.index("--llm") + 1]) if "--llm" in argv else None
        paths = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
        report = pipeline.check_batch(paths, facts, rules.thresholds(), llm)
        for p in report["photos"]:
            why = "; ".join(p.get("problems", []) + p.get("unknowns", [])) or p.get("reason", "")
            print(f"{p['verdict']:10} {p['photo']:28} {why}")
        print(json.dumps(report["counts"]))
        for d in report["duplicates"]:
            print(f"{d['call']:10} {' + '.join(d['photos'])} ({d['bits']} bits)")
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
