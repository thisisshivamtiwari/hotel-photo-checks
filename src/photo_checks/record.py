"""Record the vision-LLM checks for every photo in data/labels.json (PHC_MODE=record), 4 calls at a time.
A call that errors is retried once; if it fails again it is left unrecorded and reported, so the run can be
repeated and only the missing keys are called."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import vlm

ROOT = Path(__file__).resolve().parents[2]


def record(models=("haiku-4.5", "sonnet-5.5"), workers: int = 4) -> list[str]:
    os.environ["PHC_MODE"] = "record"
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    jobs = [(m, n) for m in models for n in labels]
    failed = []

    def one(job):
        m, n = job
        for _ in range(2):
            try:
                return vlm.call(m, n, ROOT / "data" / "photos" / "set" / f"{n}.jpg")
            except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
                err = e
        failed.append(f"{m} {n}: {type(err).__name__}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    return failed
