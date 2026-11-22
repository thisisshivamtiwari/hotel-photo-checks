import json

import pytest

from photo_checks import pipeline, rules, vlm

TH = json.loads(rules.THRESHOLDS.read_text())
REFUSED = ["empty", "truncated", "not_an_image", "tiny", "missing"]
READABLE = ["cmyk", "grey16", "rgba", "webp"]


@pytest.mark.parametrize("kind", REFUSED)
def test_gate_refuses_broken_files_with_a_reason(broken, kind):
    loaded = rules.load(broken[kind])
    assert loaded.img is None and loaded.reason


@pytest.mark.parametrize("kind", READABLE)
def test_gate_reads_odd_but_valid_formats(broken, kind):
    loaded = rules.load(broken[kind])
    assert loaded.img is not None and loaded.img.mode == "RGB" and max(loaded.img.size) == rules.WORK_SIDE


def test_broken_files_are_unreadable_never_ready(broken):
    report = pipeline.check_batch([broken[k] for k in REFUSED], {"pool": False}, TH)
    assert report["counts"] == {"ready": 0, "fix": 0, "person": 0, "unreadable": len(REFUSED)}


def test_missing_fact_sheet_is_never_consistent():
    def sees_pool(_path):
        return {**vlm.parse('{"blur":"ok","exposure":"ok","framing":"ok","amenities":{"pool":"yes"}}')}

    r = pipeline.check_photo(rules.ROOT / "data/photos/set/p03.jpg", None, TH, llm=sees_pool)
    assert r["checks"]["facts"]["pool"] == "unverifiable"
    assert r["verdict"] != "ready"


def test_llm_failure_is_unknown_not_ok():
    def broken_llm(_path):
        return vlm.parse("Sorry, I can't help with that.")

    r = pipeline.check_photo(rules.ROOT / "data/photos/set/p01-dark.jpg", {"pool": True}, TH, llm=broken_llm)
    assert r["checks"]["exposure"] == "dark"  # the rules still answer on their own
    assert r["checks"]["llm_parsed"] is False
    assert r["verdict"] == "fix"


def test_contradiction_flagged():
    def sees_pool(_path):
        return vlm.parse('{"blur":"ok","exposure":"ok","framing":"ok","amenities":{"pool":"yes","gym":"no"}}')

    r = pipeline.check_photo(rules.ROOT / "data/photos/set/p03.jpg", {"pool": False}, TH, llm=sees_pool)
    assert r["verdict"] == "fix" and any("pool" in p for p in r["problems"])
