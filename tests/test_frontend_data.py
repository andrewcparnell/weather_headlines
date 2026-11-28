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
        audio_rel = st.get("audio_url") or ""
        assert audio_rel.startswith(f"data/audio/story-{idx}-") and audio_rel.endswith(".mp3"), (
            f"Story {idx} audio_url must be cache-busted with generated_utc stamp, got {audio_rel!r}"
        )
        mp3_path = SITE_DIR / audio_rel
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


def test_section5_simplified_tour_ui_legend_accessibility_and_performance():
    import re

    from pipeline.config import LAYERS
    from pipeline.stamp_assets import compute_frontend_asset_version

    html = (SITE_DIR / "index.html").read_text()
    css = (SITE_DIR / "style.css").read_text()
    main_js = (SITE_DIR / "js" / "main.js").read_text()
    tour_js = (SITE_DIR / "js" / "tour.js").read_text()
    globe_js = (SITE_DIR / "js" / "globe.js").read_text()

    # 1. Dual-speed scrolling tickers (STORY + IN THE NEWS, stopping on hover) & inline SVGs (no emoji)
    assert "tourTextBtn" not in html and "Text On" not in html
    assert 'id="tickerViewport"' in html and 'id="tickerTrack"' in html and ">STORY<" in html
    assert 'id="linksViewport"' in html and 'id="tourNewsLinks"' in html and ">IN THE NEWS<" in html
    assert "STORY_TICKER_SPEED" in tour_js and "LINKS_TICKER_SPEED" in tour_js
    # Both tickers pause on hover only (never stuck after a link click)
    assert "_isTickerPaused(this.dom.linksBar" in tour_js
    assert "_isTickerPaused(this.dom.tickerTrack.closest" in tour_js
    assert "link.blur()" in tour_js and "linksHovered" not in tour_js
    assert 'aria-hidden="true" inert' in tour_js
    assert 'id="voiceScriptOverlay"' in html
    for forbidden_emoji in ("🔊", "🔇", "📝", "◀", "▶", "✕"):
        assert forbidden_emoji not in html and forbidden_emoji not in tour_js

    # 2. End after Story 5, restore previous layer, and show "Replay headlines"
    assert "index >= this.stories.length" in tour_js
    assert "Replay headlines" in tour_js
    assert "this._preTourLayer" in tour_js

    # 3. Mobile (< 600 px) single "Layers" button, legend visible on all screens, >= 44x44 px touch targets
    assert 'id="mobileLayersToggle"' in html and ">Layers<" in html
    assert "@media (max-width: 599px)" in css
    assert not re.search(r"\.legend-card[^{]*\{[^}]*display:\s*none", css), "Legend must remain visible on phones"
    assert "min-height: 44px" in css and "min-width: 44px" in css

    # 4. Fixed top-anchored layer rail (no translateY(-50%) centring jump), solid contrast, >= 13px font sizes
    assert "translateY(-50%)" not in css
    px_sizes = [float(m.group(1)) for m in re.finditer(r"font-size:\s*(\d+(?:\.\d+)?)px", css)]
    assert px_sizes and min(px_sizes) >= 13.0, f"Found font-size below 13px in style.css: {min(px_sizes)}"

    # 5. Legend & scale fixes: anomaly [-28, 28], sqrt rain gradient, wind '0 km/h'
    assert (LAYERS["anomaly"].lo, LAYERS["anomaly"].hi) == (-28.0, 28.0)
    assert "spec.scale === 'sqrt' ? p * p : p" in main_js
    assert "'0 km/h'" in main_js

    # 6. prefers-reduced-motion honoured across CSS, Globe, and Tour
    assert "prefers-reduced-motion: reduce" in css
    assert "prefers-reduced-motion: reduce" in globe_js
    assert "prefers-reduced-motion: reduce" in tour_js
    assert "-dt * 0.062" in globe_js

    # 7. Front-end bug fixes & accessibility
    assert "Promise.allSettled" in globe_js
    assert ".dispose()" in globe_js
    assert "Headlines unavailable, please try again shortly" in tour_js
    assert "_clearPendingMetaListener" in tour_js
    assert "_queuedBulletin" in tour_js
    assert "safeHttpUrl" in tour_js and "parsed.protocol === 'http:'" in tour_js
    assert 'id="liveBadge" aria-live' not in html
    assert "visibility = 'hidden'" in globe_js
    assert "Story ${idx + 1}:" in globe_js
    assert "setSearchMarker" in globe_js and "setSearchMarker" in main_js
    assert "this.dom.nextBtn?.focus()" in tour_js and "this.dom.playTourBtn?.focus()" in tour_js

    # 8. Performance: lazy Opus sting (<25 KB, 120 KB WAV gone), basemap.webp, modulepreload, no-cache polling, setDrawRange, _windScratch, translate3d, mobile DPR <= 1.5, auto ?v= hash
    sting_opus = SITE_DIR / "assets" / "bulletin-sting.mp4"
    assert sting_opus.exists() and 1_000 < sting_opus.stat().st_size < 25_000
    assert not (SITE_DIR / "assets" / "bulletin-sting.wav").exists()
    assert "this.stingAudio = null" in tour_js and "this.audio.preload = 'none'" in tour_js

    basemap_webp = SITE_DIR / "assets" / "basemap.webp"
    assert basemap_webp.exists() and 20_000 < basemap_webp.stat().st_size < 200_000
    assert "assets/basemap.webp" in globe_js

    assert 'rel="modulepreload" href="./vendor/three.module.js"' in html
    assert "cache: 'no-cache'" in main_js and "no-store" not in main_js
    assert "document.hidden" in main_js and "isNearScheduledReportWindow" in main_js
    assert "_windScratch" in globe_js
    assert "setDrawRange" in globe_js
    assert "translate3d(" in globe_js and "style.left" not in globe_js
    assert "window.innerWidth < 768 ? 1.5 : 2" in globe_js

    expected_ver = compute_frontend_asset_version(SITE_DIR)
    assert f"style.css?v={expected_ver}" in html
    assert f"js/main.js?v={expected_ver}" in html
    assert f"./globe.js?v={expected_ver}" in main_js
    assert f"./globe.js?v={expected_ver}" in tour_js


def test_every_story_has_multi_layer_multi_location_synced_cues():
    from pipeline.scanner import haversine_km
    from pipeline.validator import opening_mentions_location, validate_story_geography

    payload = json.loads((DATA_DIR / "headlines.json").read_text())
    for idx, st in enumerate(payload["stories"]):
        cues = st.get("cues") or []
        assert len(cues) >= 3, f"{st['id']} must have >= 3 sentence cues"
        assert opening_mentions_location(cues[0]["spoken"], st.get("country", ""), st.get("region", "")), (
            f"{st['id']} first cue must clearly name country/region, got: {cues[0]['spoken']!r}"
        )
        geo_errs = validate_story_geography(
            f"stories[{idx}]",
            float(st["centre"]["lat"]),
            float(st["centre"]["lon"]),
            cues,
            headline=st.get("headline", ""),
        )
        assert geo_errs == [], f"{st['id']} failed geographic sanity check: {geo_errs}"

        prev_end = 0.0
        for c in cues:
            assert c["end_sec"] > c["start_sec"] >= prev_end - 0.05
            prev_end = c["end_sec"]
            assert haversine_km(c["lat"], c["lon"], st["centre"]["lat"], st["centre"]["lon"]) <= 500.0
        expected_voice = " ".join(c["spoken"].strip() for c in cues if c.get("spoken", "").strip())
        assert st.get("voice_script", "").strip() == expected_voice, (
            f"{st['id']} voice_script must match cues[*].spoken word-for-word"
        )
        if st.get("duration_sec") is not None:
            assert abs(cues[-1]["end_sec"] - float(st["duration_sec"])) <= 0.35, (
                f"{st['id']} final cue end_sec ({cues[-1]['end_sec']}) must match duration_sec ({st['duration_sec']})"
            )


def test_dom_ids_and_module_paths_match_index_html():
    import re

    html = (SITE_DIR / "index.html").read_text()
    html_ids = set(re.findall(r'\bid="([^"]+)"', html))

    main_js = (SITE_DIR / "js" / "main.js").read_text()
    referenced_ids = set(re.findall(r"\$\(\s*'([^']+)'\s*\)", main_js))
    missing_ids = referenced_ids - html_ids
    assert not missing_ids, f"IDs referenced in main.js missing from index.html: {missing_ids}"

    # Verify importmap, custom domain CNAME, and vendor files exist
    assert (SITE_DIR / "CNAME").read_text().strip() == "headlines.aimsir.org"
    for rel in (
        "vendor/three.module.js",
        "vendor/three.core.min.js",
        "vendor/OrbitControls.js",
        "assets/basemap.json",
        "assets/gazetteer.json",
        "data/manifest.json",
        "data/headlines.json",
        "favicon.svg",
        "favicon.ico",
        "favicon-32x32.png",
        "apple-touch-icon.png",
        "icon-192.png",
        "icon-512.png",
        "site.webmanifest",
    ):
        assert (SITE_DIR / rel).exists(), f"Missing static asset {rel}"


def test_app_icons_and_web_manifest():
    html = (SITE_DIR / "index.html").read_text()
    for href in ("/favicon.svg", "/favicon.ico", "/favicon-32x32.png", "/apple-touch-icon.png", "/site.webmanifest"):
        assert href in html, f"Missing icon/manifest link {href} in index.html <head>"

    svg_txt = (SITE_DIR / "favicon.svg").read_text()
    assert "<svg" in svg_txt and 'viewBox="0 0 512 512"' in svg_txt

    with Image.open(SITE_DIR / "apple-touch-icon.png") as img:
        assert img.size == (180, 180)
        assert img.mode == "RGB", "apple-touch-icon.png must have an opaque background (RGB, no alpha)"
        # Ensure corners are opaque navy rather than pre-rounded white/transparent
        tl = img.getpixel((0, 0))
        br = img.getpixel((179, 179))
        assert max(tl) < 40 and max(br) < 40

    with Image.open(SITE_DIR / "icon-192.png") as img192:
        assert img192.size == (192, 192)
    with Image.open(SITE_DIR / "icon-512.png") as img512:
        assert img512.size == (512, 512)
    with Image.open(SITE_DIR / "favicon-32x32.png") as img32:
        assert img32.size == (32, 32)
    with Image.open(SITE_DIR / "favicon.ico") as ico:
        assert ico.format == "ICO"

    manifest = json.loads((SITE_DIR / "site.webmanifest").read_text())
    assert manifest["name"] == "Weather Headlines · Aimsir"
    assert manifest["short_name"] == "Headlines"
    assert manifest["start_url"] == "/"
    assert manifest["display"] == "standalone"
    icon_srcs = {i["src"] for i in manifest["icons"]}
    assert {"/favicon.svg", "/icon-192.png", "/icon-512.png"}.issubset(icon_srcs)
    for icon_entry in manifest["icons"]:
        rel_path = icon_entry["src"].lstrip("/")
        assert (SITE_DIR / rel_path).exists(), f"Manifest icon missing on disk: {rel_path}"


def test_mobile_camera_framing_and_responsive_controls():
    import math

    html = (SITE_DIR / "index.html").read_text()
    css = (SITE_DIR / "style.css").read_text()
    globe_js = (SITE_DIR / "js" / "globe.js").read_text()

    assert "</script></script>" not in html
    assert "computeViewportCameraParams" in globe_js
    assert "returnToHomeView" in globe_js
    assert "flex-wrap: wrap;" in css

    # Verify the exact camera framing formula across iPhone SE, iPhone 15, Pro Max, iPad, and Desktop
    viewports = [
        ("iPhone SE", 375, 667),
        ("iPhone 15", 390, 844),
        ("iPhone Pro Max", 430, 932),
        ("iPad Portrait", 768, 1024),
        ("Desktop", 1440, 900),
    ]
    for label, w, h in viewports:
        aspect = w / h
        fov = min(54.0, 38.0 + (1.0 - aspect) * 28.0) if aspect < 1.0 else 38.0
        tan_half_x = aspect * math.tan(math.radians(fov * 0.5))
        target_frac = 0.76 if aspect < 1.0 else 0.70
        ft = max(0.12, target_frac * tan_half_x)
        exact_dist = math.sqrt(1.0 + ft * ft) / ft
        init_dist = max(3.4, min(6.0, exact_dist)) if aspect < 1.0 else 2.75

        # Projected horizontal screen fraction occupied by the 3D Earth silhouette: tan(arcsin(1/d)) / tan_half_x
        alpha = math.asin(1.0 / init_dist)
        screen_width_fraction = math.tan(alpha) / tan_half_x
        assert 0.55 <= screen_width_fraction <= 0.90, (
            f"{label} ({w}x{h}) globe occupies {screen_width_fraction:.2%} of width (expected 55%-90%)"
        )

