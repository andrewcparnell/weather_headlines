"""Integration tests verifying generated site/data assets, gazetteer, and editorial style rules."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeline.config import DATA_DIR, LAYER_ORDER, SITE_DIR
from pipeline.validator import validate_and_clean_bulletin


def test_gazetteer_has_countries_and_cities():
    gaz_path = SITE_DIR / "assets" / "gazetteer.json"
    assert gaz_path.exists()
    entries = json.loads(gaz_path.read_text())
    assert len(entries) >= 150
    names = {e["name"] for e in entries}
    for must_have in ("Ireland", "United Kingdom", "United States", "Australia", "New Zealand", "Dublin"):
        assert must_have in names


def test_generated_manifest_and_webp_layers():
    manifest_path = DATA_DIR / "manifest.json"
    assert manifest_path.exists(), "manifest.json missing - run `python -m pipeline.build` first"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["grid"]["nx"] == 720
    assert manifest["grid"]["ny"] == 361
    assert [l["id"] for l in manifest["layers"]] == LAYER_ORDER

    for layer_id in LAYER_ORDER:
        webp = DATA_DIR / "layers" / f"{layer_id}.webp"
        assert webp.exists(), f"Missing {webp}"
        with Image.open(webp) as img:
            assert img.size == (720, 361)
            assert img.mode in ("L", "RGB", "RGBA")


def test_generated_headlines_bulletin_is_valid_and_style_compliant():
    headlines_path = DATA_DIR / "headlines.json"
    assert headlines_path.exists(), "headlines.json missing - run `python -m pipeline.build` first"
    raw_text = headlines_path.read_text()
    assert "\u2014" not in raw_text, "Em dashes are forbidden in headlines.json"
    assert "utilise" not in raw_text.lower()
    assert "utilize" not in raw_text.lower()

    payload = json.loads(raw_text)
    cleaned, errors = validate_and_clean_bulletin(payload)
    assert errors == []
    assert len(cleaned["stories"]) == 5


def test_no_em_dashes_in_frontend_or_readme():
    files_to_check = [
        Path(__file__).resolve().parent.parent / "README.md",
        SITE_DIR / "index.html",
        SITE_DIR / "js" / "main.js",
        SITE_DIR / "js" / "globe.js",
        SITE_DIR / "js" / "tour.js",
        SITE_DIR / "js" / "search.js",
    ]
    for path in files_to_check:
        txt = path.read_text()
        assert "\u2014" not in txt, f"Found forbidden em dash in {path}"
