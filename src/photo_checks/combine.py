"""How the arms answer each check, and how 'both' merges two answers.

'both' rule: if the two agree, that answer; if one can't tell, the other's answer; if they disagree, unknown (a
person looks). It never turns a disagreement into a pass."""

from .dataset import AMENITIES

TECH = ["blur", "exposure", "framing"]


def merge(a, b, unknown="unknown"):
    if a == b:
        return a
    if a == unknown:
        return b
    if b == unknown:
        return a
    return unknown


def merge_strict(a, b, unknown="unknown"):
    """Post-hoc variant (added after the first test run, see README): only an agreed answer counts; if either side
    can't tell, the photo goes to a person."""
    return a if a == b else unknown


def clip_amenities(scores: dict, th: dict) -> dict:
    """yes / no / unknown (True / False / None) from a CLIP probability, with the shared relative band."""
    out = {}
    t, band = th["clip_yes"], th["band"]
    for a in AMENITIES:
        p = scores[a]
        out[a] = None if t * (1 - band) <= p <= t * (1 + band) else p > t
    return out


def merge_amenities(a: dict, b: dict, how=merge) -> dict:
    return {k: how(a.get(k), b.get(k), unknown=None) for k in AMENITIES}


def contradiction(seen, fact):
    """What a photo means for one amenity given the fact sheet: 'contradiction' (shows what the hotel says it doesn't
    have), 'consistent', 'unverifiable' (the fact sheet doesn't say), 'cant_tell' (the checker couldn't see), or
    None (the photo doesn't show it, so it claims nothing). A missing fact is never read as yes or no."""
    if seen is None:
        return "cant_tell"
    if not seen:
        return None
    if fact is None:
        return "unverifiable"
    return "consistent" if fact else "contradiction"
