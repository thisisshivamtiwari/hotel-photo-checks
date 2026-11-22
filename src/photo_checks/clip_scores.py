"""Zero-shot amenity scores from a small open image model (OpenCLIP ViT-B-32, LAION-2B weights, ~150M params, runs
on a laptop CPU). Each photo is scored against ten scene prompts (five amenities, five ordinary rooms or places);
the softmax probability of each amenity prompt is saved to results/clip_scores.json with the photo's hash.

CI doesn't install torch: it reads the saved scores and checks every hash still matches the photo on disk."""

import json
import time
from pathlib import Path

from .dataset import AMENITIES

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results" / "clip_scores.json"
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
PROMPTS = {
    "pool": "a photo of a swimming pool",
    "gym": "a photo of a gym with exercise machines",
    "bathtub": "a photo of a bathroom with a bathtub",
    "sea_view": "a photo of the sea and a beach",
    "fireplace": "a photo of a fireplace",
    "_bedroom": "a photo of a bedroom",
    "_shower": "a photo of a bathroom with a shower",
    "_dining": "a photo of a dining room",
    "_living": "a photo of a living room",
    "_outside": "a photo of a building from outside",
}


def compute(labels: dict, set_dir: Path) -> dict:
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval()
    keys = list(PROMPTS)
    out = {"model": f"{MODEL}/{PRETRAINED}", "prompts": PROMPTS, "photos": {}}
    t0 = time.time()
    with torch.no_grad():
        text = model.encode_text(tok([PROMPTS[k] for k in keys]))
        text /= text.norm(dim=-1, keepdim=True)
        for name, lab in labels.items():
            img = preprocess(Image.open(set_dir / f"{name}.jpg").convert("RGB")).unsqueeze(0)
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            probs = (100.0 * feat @ text.T).softmax(dim=-1)[0].tolist()
            out["photos"][name] = {"sha256": lab["sha256"], **{a: round(probs[keys.index(a)], 4) for a in AMENITIES}}
    out["seconds_per_photo"] = round((time.time() - t0) / len(labels), 3)
    SCORES.parent.mkdir(exist_ok=True)
    SCORES.write_text(json.dumps(out, indent=1) + "\n")
    return out


def load(labels: dict) -> dict:
    data = json.loads(SCORES.read_text())
    for name, lab in labels.items():
        if data["photos"][name]["sha256"] != lab["sha256"]:
            raise ValueError(f"{name}: saved CLIP score is for a different image; rerun `python -m photo_checks clip`")
    return data
