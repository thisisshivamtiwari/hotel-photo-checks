"""Is the text in this image big enough to read by machine? A gate for any check that reads text.

Nothing in this repo reads text yet, but a listing photo can carry a price, a phone number, a watermark or another
business's name, and a hotel's paperwork (rate cards, folios) is all numbers. The AI Lab test of 25 Nov 2026
(github.com/thisisshivamtiwari/ai-lab, 2026-11-25-ocria-nano-numbers) found that two OCR engines read 99% of the
numbers on pages whose body text was about 16 px tall (150 dpi) and lost most of them, inventing amount-shaped
strings, at about 11 px (100 dpi, 1-bit). So text read below a floor is not a result: it is "unknown".

This module uses classic image measures only (MSER glyph candidates, grouped into lines) and never reads the text.
It reports glyph heights in the image's own pixels. Four answers:

- "none":       document-like image, no text-like line found;
- "legible":    lines found, median glyph height at or above the floor;
- "too_small":  lines found but below the floor: anything read from this image is unknown, not a result;
- "unmeasured": a photo rather than a document. Photo texture (tiles, railings, windows) looks like glyph lines to
                these rules (see results/legibility.json), so the height is not measured and a text reader must
                treat the photo as unknown until a real scene-text detector is added.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

FLOOR_PX = 14  # median glyph height; between the 11 px that failed and the 16 px that read 99% in the AI Lab test
MIN_GLYPHS = 4  # a line needs this many glyph-sized regions in a row
MAX_SIDE = 2400  # bigger images are measured at this size and heights scaled back
DOC_BG_SHARE = 0.8  # a document: this share of pixels within 25 levels of one background shade (pages 0.95+, photos 0.69-)
ACTION = {"none": "nothing to read", "legible": "read", "too_small": "unknown", "unmeasured": "unknown"}


@dataclass
class TextLines:
    lines: int
    heights: list[int]  # one median glyph height per line, in the image's own pixels

    @property
    def median_height(self) -> float | None:
        return float(np.median(self.heights)) if self.heights else None

    @property
    def status(self) -> str:
        if not self.lines:
            return "none"
        return "legible" if self.median_height >= FLOOR_PX else "too_small"


def document_like(gray: np.ndarray) -> tuple[bool, float]:
    """A page has one background shade over most of its area; a photo doesn't."""
    hist = np.bincount(gray.ravel(), minlength=256)
    mode = int(hist.argmax())
    share = float(hist[max(0, mode - 25) : mode + 26].sum() / gray.size)
    return share >= DOC_BG_SHARE, round(share, 3)


def glyph_boxes(gray: np.ndarray) -> dict[tuple[int, int, int, int], float]:
    """Regions shaped like single letters or digits, with their stroke width. MSER finds stable blobs; the filters
    keep the ones a glyph could be: 5 px to a fifth of the image tall, not too wide or thin, not a solid block,
    strokes thinner than half the height, and clearly darker or lighter than the pixels around them."""
    h, w = gray.shape
    mser = cv2.MSER.create(delta=5, min_area=12, max_area=int(0.02 * h * w))
    boxes: dict[tuple[int, int, int, int], float] = {}
    for polarity in (gray, 255 - gray):  # dark text on light, and light text on dark
        regions, _ = mser.detectRegions(polarity)
        for pts in regions:
            x, y, bw, bh = cv2.boundingRect(pts)
            if bh < 5 or bh > 0.2 * h or bw > 2.2 * bh or bw < 0.1 * bh:
                continue
            if not 0.15 <= len(pts) / (bw * bh) <= 0.9:
                continue
            mask = np.zeros((bh + 2, bw + 2), np.uint8)
            mask[pts[:, 1] - y + 1, pts[:, 0] - x + 1] = 255
            stroke = 2 * float(cv2.distanceTransform(mask, cv2.DIST_L2, 3)[mask > 0].mean())
            if stroke / bh > 0.35:
                continue
            y0, y1, x0, x1 = max(0, y - 3), min(h, y + bh + 3), max(0, x - 3), min(w, x + bw + 3)
            patch = polarity[y0:y1, x0:x1]
            ring = (patch.sum() - polarity[y : y + bh, x : x + bw].sum()) / max(1, patch.size - bw * bh)
            if polarity[pts[:, 1], pts[:, 0]].mean() - ring < 50:
                continue
            boxes[(x, y, bw, bh)] = stroke
    return boxes


def group_lines(boxes: dict[tuple[int, int, int, int], float]) -> list[list[tuple[int, int, int, int]]]:
    """Chain glyph boxes left to right when they share a baseline, height and spacing. A line of real text has
    glyphs of varied width (i, m) and even strokes; tiles and window panes are all the same width, so they are
    dropped."""
    lines: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda b: (b[0], b[1])):
        x, y, bw, bh = box
        cy = y + bh / 2
        for line in lines:
            lx, ly, lw, lh = line[-1]
            same_row = abs(cy - (ly + lh / 2)) < 0.5 * max(bh, lh) and max(bh, lh) / min(bh, lh) < 1.7
            if same_row and -0.3 * lh < x - (lx + lw) < 2.0 * lh:
                line.append(box)
                break
        else:
            lines.append([box])
    kept = []
    for ln in lines:
        if len(ln) < MIN_GLYPHS:
            continue
        ws, hs = np.array([b[2] for b in ln], float), np.array([b[3] for b in ln], float)
        sw = np.array([boxes[b] for b in ln])
        if ws.std() / ws.mean() < 0.3 or hs.std() / hs.mean() > 0.35 or sw.std() / sw.mean() > 0.6:
            continue
        kept.append(ln)
    return kept


def measure(img: Image.Image) -> TextLines:
    """Text-like lines in the image and their glyph heights, in the image's own pixels. Measures any image; see
    gate() for when the answer can be trusted."""
    scale = 1.0
    if max(img.size) > MAX_SIDE:
        scale = max(img.size) / MAX_SIDE
        img = img.resize((round(img.width / scale), round(img.height / scale)), Image.LANCZOS)
    gray = np.asarray(img.convert("L"))
    lines = group_lines(glyph_boxes(gray))
    heights = [int(round(float(np.median([b[3] for b in ln])) * scale)) for ln in lines]
    return TextLines(len(lines), sorted(heights))


def gate(img: Image.Image) -> dict:
    """What a text reader should do with this image: 'read' only for a document whose lines are tall enough."""
    gray = np.asarray(img.convert("L"))
    is_doc, bg = document_like(gray)
    out = {"status": "unmeasured", "document": is_doc, "background_share": bg, "floor_px": FLOOR_PX}
    if is_doc:
        t = measure(img)
        out.update(status=t.status, lines=t.lines, median_height_px=t.median_height)
    out["action"] = ACTION[out["status"]]
    return out
