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
    for idx, st in enumerate(cleaned["stories"], start=1):
        assert "**" in st["ticker"], f"Story {st['id']} ticker missing **keyword** highlights"
        mp3_path = SITE_DIR / f"data/audio/story-{idx}.mp3"
        assert mp3_path.exists() and mp3_path.stat().st_size > 15_000, f"Missing or empty MP3 {mp3_path}"


def test_no_em_dashes_in_frontend_or_readme():
    files_to_check = [
        Path(__file__).resolve().parent.parent / "README.md",
        SITE_DIR / "index.html",
        SITE_DIR / "style.css",
        SITE_DIR / "js" / "main.js",
        SITE_DIR / "js" / "globe.js",
        SITE_DIR / "js" / "tour.js",
        SITE_DIR / "js" / "search.js",
    ]
    for path in files_to_check:
        txt = path.read_text()
        assert "\u2014" not in txt, f"Found forbidden em dash in {path}"
        assert "utilise" not in txt.lower(), f"Found forbidden word in {path}"
        assert "utilize" not in txt.lower(), f"Found forbidden word in {path}"


def test_broadcast_chyron_hidden_and_dual_speed_tickers():
    import re

    html = (SITE_DIR / "index.html").read_text()
    css = (SITE_DIR / "style.css").read_text()
    tour_js = (SITE_DIR / "js" / "tour.js").read_text()

    assert "Loading global weather headlines" not in html
    assert "Voice Off" in html
    assert "[hidden]" in css and "display: none !important" in css
    assert "this.voiceEnabled = false" in tour_js

    sting_wav = SITE_DIR / "assets" / "bulletin-sting.wav"
    assert sting_wav.exists() and sting_wav.stat().st_size > 50_000
    assert "bulletin-sting.wav" in tour_js

    story_match = re.search(r"STORY_TICKER_SPEED\s*=\s*(\d+)", tour_js)
    links_match = re.search(r"LINKS_TICKER_SPEED\s*=\s*(\d+)", tour_js)
    assert story_match and links_match, "Both ticker speeds must be defined in tour.js"
    assert int(story_match.group(1)) != int(links_match.group(1)), "Tickers must scroll at different speeds"


def test_every_story_has_multi_layer_multi_location_synced_cues():
    from pipeline.scanner import haversine_km

    payload = json.loads((DATA_DIR / "headlines.json").read_text())
    for st in payload["stories"]:
        cues = st.get("cues") or []
        assert len(cues) >= 5, f"{st['id']} must have >= 5 sentence cues"
        layers_used = {c["layer"] for c in cues}
        assert len(layers_used) >= 3, f"{st['id']} cues should switch across >= 3 variables, got {layers_used}"

        prev_end = 0.0
        path_km = 0.0
        for idx, c in enumerate(cues):
            assert c["end_sec"] > c["start_sec"] >= prev_end - 0.05
            prev_end = c["end_sec"]
            if idx > 0:
                path_km += haversine_km(cues[idx - 1]["lat"], cues[idx - 1]["lon"], c["lat"], c["lon"])
        assert path_km >= 500.0, f"{st['id']} cue locations only move {path_km:.0f} km"


def test_dom_ids_and_module_paths_match_index_html():
    import re

    html = (SITE_DIR / "index.html").read_text()
    html_ids = set(re.findall(r'\bid="([^"]+)"', html))

    main_js = (SITE_DIR / "js" / "main.js").read_text()
    referenced_ids = set(re.findall(r"\$\(\s*'([^']+)'\s*\)", main_js))
    missing_ids = referenced_ids - html_ids
    assert not missing_ids, f"IDs referenced in main.js missing from index.html: {missing_ids}"

    # Verify importmap and vendor files exist
    for rel in (
        "vendor/three.module.js",
        "vendor/three.core.min.js",
        "vendor/OrbitControls.js",
        "assets/basemap.json",
        "assets/gazetteer.json",
        "data/manifest.json",
        "data/headlines.json",
    ):
        assert (SITE_DIR / rel).exists(), f"Missing static asset {rel}"

