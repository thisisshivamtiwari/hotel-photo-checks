"""Vision-LLM checks through the Claude Code CLI (`claude -p` with the photo as an image block), recorded to
cassettes so CI can replay them without a key.

A cassette entry is keyed by a hash of (model, system prompt, image hash, sample). `PHC_MODE=record` calls the model
for any key not yet on disk; the default, `replay`, never calls a model and fails on a missing key. Each entry keeps
the CLI's own cost (`total_cost_usd`, at API list prices) and duration: the cost and speed figures come from there.
A call that fails, or a reply that isn't the expected JSON, becomes "unknown" for every check, never "ok".
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASSETTES = ROOT / "cassettes"
PROMPT = (ROOT / "prompts" / "photo_check.md").read_text()
MODELS = {"haiku-4.5": "claude-haiku-4-5-20251001", "sonnet-5.5": "claude-sonnet-5-5"}
EFFORT = {"haiku-4.5": None, "sonnet-5.5": "low"}  # Haiku 4.5 has no effort setting
CHECKS = {
    "blur": {"ok": "ok", "blurry": "problem", "cant_tell": "unknown"},
    "exposure": {"ok": "ok", "too_dark": "dark", "too_bright": "bright", "cant_tell": "unknown"},
    "framing": {"ok": "ok", "tilted": "tilted", "cant_tell": "unknown"},
}
AMENITY = {"yes": True, "no": False, "cant_tell": None}
_lock = threading.Lock()
_cache: dict[str, dict[str, dict]] = {}


class MissingCassette(KeyError):
    pass


def _key(model: str, image_sha: str, sample: int) -> str:
    raw = json.dumps([model, PROMPT, image_sha, sample])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load(alias: str) -> dict[str, dict]:
    if alias not in _cache:
        path = CASSETTES / f"{alias}.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        _cache[alias] = {e["key"]: e for e in map(json.loads, lines)}
    return _cache[alias]


def _run_cli(alias: str, image: bytes) -> dict:
    msg = {"type": "user", "message": {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image).decode()}},
        {"type": "text", "text": "Check this photo."},
    ]}}  # fmt: skip
    cmd = ["claude", "-p", "--model", MODELS[alias], "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--system-prompt", PROMPT]  # fmt: skip
    if EFFORT[alias]:
        cmd += ["--effort", EFFORT[alias]]
    out = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True, check=True, timeout=300, cwd="/tmp")
    return json.loads(out.stdout.strip().splitlines()[-1])


def call(alias: str, name: str, image_path: Path, sample: int = 0) -> dict:
    image = Path(image_path).read_bytes()
    sha = hashlib.sha256(image).hexdigest()[:16]
    key = _key(MODELS[alias], sha, sample)
    with _lock:
        hit = _load(alias).get(key)
    if hit:
        return {**hit, "cached": True}
    if os.environ.get("PHC_MODE", "replay") != "record":
        raise MissingCassette(f"{alias} {name}: no recording for key {key}; run with PHC_MODE=record")
    res = _run_cli(alias, image)
    entry = {
        "key": key,
        "photo": name,
        "image_sha256": sha,
        "sample": sample,
        "text": res.get("result", ""),
        "is_error": bool(res.get("is_error")),
        "cost_usd": res.get("total_cost_usd", 0.0),
        "ms": res.get("duration_ms", 0),
        "input_tokens": res.get("usage", {}).get("input_tokens", 0),
        "output_tokens": res.get("usage", {}).get("output_tokens", 0),
    }
    with _lock:
        CASSETTES.mkdir(exist_ok=True)
        with (CASSETTES / f"{alias}.jsonl").open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _load(alias)[key] = entry
    return {**entry, "cached": False}


def parse(text: str) -> dict:
    """Map a reply to the shared vocabulary. Anything missing or out of vocabulary is 'unknown' (None for
    amenities), and `parsed` says whether the reply was usable at all."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.S)
    body = fenced.group(1) if fenced else (text or "")
    obj = None
    start = body.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(body[start:])
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return {"parsed": False, **{c: "unknown" for c in CHECKS}, "amenities": {}, "note": ""}
    out = {"parsed": True, "note": str(obj.get("note", ""))[:300]}
    for check, vocab in CHECKS.items():
        out[check] = vocab.get(str(obj.get(check, "")).strip().lower(), "unknown")
    am = obj.get("amenities") if isinstance(obj.get("amenities"), dict) else {}
    out["amenities"] = {a: AMENITY.get(str(v).strip().lower()) for a, v in am.items()}
    return out
