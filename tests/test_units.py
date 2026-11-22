from photo_checks import combine, evaluate, vlm


def test_merge_never_turns_disagreement_into_a_pass():
    assert combine.merge("ok", "problem") == "unknown"
    assert combine.merge("ok", "unknown") == "ok"
    assert combine.merge("unknown", "unknown") == "unknown"
    assert combine.merge(True, False, unknown=None) is None


def test_contradiction_table():
    assert combine.contradiction(True, False) == "contradiction"
    assert combine.contradiction(True, True) == "consistent"
    assert combine.contradiction(True, None) == "unverifiable"
    assert combine.contradiction(None, False) == "cant_tell"
    assert combine.contradiction(False, False) is None


def test_parse_handles_fences_junk_and_vocab():
    p = vlm.parse(
        '```json\n{"blur":"blurry","exposure":"TOO_DARK","framing":"wonky","amenities":{"pool":"yes","gym":"maybe"}}\n```'
    )
    assert p["parsed"] and p["blur"] == "problem" and p["exposure"] == "dark" and p["framing"] == "unknown"
    assert p["amenities"]["pool"] is True and p["amenities"]["gym"] is None
    q = vlm.parse("no json here")
    assert not q["parsed"] and q["blur"] == "unknown"


def test_photo_verdict():
    assert evaluate.photo_verdict({"blur": "ok", "exposure": "ok", "framing": "ok"}) == "ready"
    assert evaluate.photo_verdict({"blur": "ok", "exposure": "unknown", "framing": "ok"}) == "person"
    assert evaluate.photo_verdict({"blur": "problem", "exposure": "unknown", "framing": "ok"}) == "fix"


def test_mcnemar():
    assert evaluate.mcnemar_p(0, 0) == 1.0
    assert evaluate.mcnemar_p(0, 10) < 0.01
