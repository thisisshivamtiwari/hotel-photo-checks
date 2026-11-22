"""Pick the rule thresholds on the dev hotel (Saltmarrow House, 40 photos) only, then freeze them in
results/thresholds.json (rules and the CLIP amenity cut-off). Each threshold maximises balanced accuracy on dev;
ties go to the middle of the best range.
The unknown band (25% either side) is fixed, not tuned."""

import json

import numpy as np

from .dataset import AMENITIES
from .rules import THRESHOLDS

BAND = 0.25
DUP_MAX_BITS = 6  # pHash distance at or below this = duplicate; up to +4 bits more = "possible duplicate" (unknown)


def balanced_acc(values, positive, t, low_is_problem):
    v, p = np.asarray(values, float), np.asarray(positive, bool)
    pred = v < t if low_is_problem else v > t
    tpr = (pred & p).sum() / max(p.sum(), 1)
    tnr = (~pred & ~p).sum() / max((~p).sum(), 1)
    return (tpr + tnr) / 2


def best_threshold(values, positive, low_is_problem, grid):
    scores = [balanced_acc(values, positive, t, low_is_problem) for t in grid]
    best = max(scores)
    idx = [i for i, s in enumerate(scores) if s == best]
    return float(grid[idx[len(idx) // 2]]), float(best)


def tune(labels: dict, measures: dict, clip: dict) -> dict:
    dev = [n for n, lab in labels.items() if lab["split"] == "dev"]
    m = {n: measures[n] for n in dev}
    t_blur, a_blur = best_threshold([m[n]["laplacian_var"] for n in dev], [labels[n]["blur"] == "problem" for n in dev],
                                    True, np.geomspace(5, 2000, 400))  # fmt: skip
    t_dark, a_dark = best_threshold([m[n]["mean_lum"] for n in dev], [labels[n]["exposure"] == "dark" for n in dev],
                                    True, np.linspace(0.05, 0.6, 221))  # fmt: skip
    t_clip, a_clip = best_threshold([m[n]["clip_frac"] for n in dev], [labels[n]["exposure"] == "bright" for n in dev],
                                    False, np.linspace(0.005, 0.5, 199))  # fmt: skip
    tilt = [n for n in dev if m[n]["tilt_deg"] is not None]
    t_tilt, a_tilt = best_threshold([abs(m[n]["tilt_deg"]) for n in tilt], [labels[n]["framing"] == "tilted" for n in tilt],
                                    False, np.linspace(0.5, 10, 96))  # fmt: skip
    # one shared CLIP threshold over all five amenities (dev has pool, bathtub and sea-view photos, no gym or fireplace)
    cv = [clip["photos"][n][a] for n in dev for a in AMENITIES]
    cp = [labels[n]["amenities"][a] for n in dev for a in AMENITIES]
    t_clip_yes, a_clip_yes = best_threshold(cv, cp, False, np.linspace(0.02, 0.9, 177))
    th = {
        "blur_laplacian": round(t_blur, 2),
        "dark_mean": round(t_dark, 4),
        "bright_clip": round(t_clip, 4),
        "tilt_deg": round(t_tilt, 2),
        "clip_yes": round(t_clip_yes, 3),
        "band": BAND,
        "dup_max_bits": DUP_MAX_BITS,
        "dup_unknown_bits": 4,
        "dev_balanced_accuracy": {
            "blur": round(a_blur, 3),
            "dark": round(a_dark, 3),
            "bright": round(a_clip, 3),
            "tilt": round(a_tilt, 3),
            "clip_yes": round(a_clip_yes, 3),
        },  # fmt: skip
        "dev_images": len(dev),
        "dev_images_with_tilt_reading": len(tilt),
    }
    THRESHOLDS.parent.mkdir(exist_ok=True)
    THRESHOLDS.write_text(json.dumps(th, indent=1) + "\n")
    return th
