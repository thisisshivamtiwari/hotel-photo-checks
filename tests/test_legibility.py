"""The text-legibility gate: pages of known text size, photos, and the floor it inherits from the AI Lab test."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from photo_checks import legibility, rules

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import legibility_fixtures as fx  # noqa: E402


def test_page_at_200_dpi_is_legible_and_fax_at_100_dpi_is_not():
    page = fx.page(0)
    clean, fax = legibility.gate(page), legibility.gate(fx.degrade(page, "fax", 0))
    assert clean["status"] == "legible" and clean["action"] == "read"
    assert clean["median_height_px"] >= legibility.FLOOR_PX
    assert fax["status"] == "too_small" and fax["action"] == "unknown"
    assert fax["median_height_px"] < legibility.FLOOR_PX


def test_phone_photo_of_a_page_at_150_dpi_still_reads():
    assert legibility.gate(fx.degrade(fx.page(1), "phone", 1))["status"] == "legible"


def test_a_photo_is_unmeasured_and_never_read():
    g = legibility.gate(rules.load(fx.SOURCES[0]).full)
    assert g["status"] == "unmeasured" and g["document"] is False and g["action"] == "unknown"


def test_blank_page_has_nothing_to_read():
    g = legibility.gate(Image.new("RGB", (800, 600), "white"))
    assert g["status"] == "none" and g["action"] == "nothing to read" and g["lines"] == 0


def test_heights_are_in_the_images_own_pixels_when_downscaled():
    small = legibility.measure(fx.page(2)).median_height  # 200 dpi: about 20 px
    big = legibility.measure(fx.page(2).resize((3308, 4678))).median_height  # the same page at 400 dpi, measured at 2400 px
    assert big is not None and 1.4 * small <= big <= 2.2 * small  # scaled back, if roughly: upscaled glyphs split more


def test_grid_texture_is_not_text():
    img = np.full((600, 800), 255, np.uint8)
    img[::20, :] = 0  # tiles: all the same width, so no line of text
    img[:, ::20] = 0
    assert legibility.measure(Image.fromarray(img)).lines == 0


def test_check_photo_reports_the_text_gate():
    from photo_checks import pipeline

    r = pipeline.check_photo(fx.SOURCES[0], None, rules.thresholds())
    assert r["text"]["status"] == "unmeasured" and r["verdict"] in ("ready", "fix", "person")
