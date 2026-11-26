"""Unit tests for the Weather Headlines pipeline, WebP encoding, scanner, and validator."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from pipeline.opendata import LiveWeatherFields, to_half_degree
from pipeline.pack_globe import dequantise_u8, quantise_u8
from pipeline.scanner import detect_global_candidates, extract_pressure_features, haversine_km
from pipeline.validator import (
    ensure_dynamic_waypoints,
    sanitise_prose,
    validate_and_clean_bulletin,
)


def test_quantise_roundtrip_linear_and_sqrt():
    arr = np.array([[-30.0, 0.0, 25.5, np.nan]], dtype=np.float32)
    enc = quantise_u8(arr, -45.0, 50.0, "linear")
    assert enc[0, 3] == 255
    dec = dequantise_u8(enc, -45.0, 50.0, "linear")
    assert np.isnan(dec[0, 3])
    assert np.allclose(dec[0, :3], arr[0, :3], atol=0.4)

    rain = np.array([[0.0, 2.0, 25.0, 150.0]], dtype=np.float32)
    r_enc = quantise_u8(rain, 0.0, 200.0, "sqrt")
    r_dec = dequantise_u8(r_enc, 0.0, 200.0, "sqrt")
    assert np.allclose(r_dec, rain, atol=1.2)


def test_to_half_degree_shape():
    grid_025 = np.ones((721, 1440), dtype=np.float32) * 12.0
    grid_05 = to_half_degree(grid_025)
    assert grid_05.shape == (361, 720)
    assert np.allclose(grid_05, 12.0)


def test_sanitise_prose_rules():
    raw = "A storm\u2014unprecedented in scale \u2013 utilises warm water and Utilizing moisture."
    cleaned = sanitise_prose(raw)
    assert "\u2014" not in cleaned
    assert "utilis" not in cleaned.lower()
    assert "utiliz" not in cleaned.lower()
    assert "uses warm water" in cleaned
    assert "Using moisture" in cleaned


def test_ensure_dynamic_waypoints_enforces_motion():
    # Static waypoints should be replaced with a >= 450 km sweeping path
    wps = ensure_dynamic_waypoints(53.3, -6.2, [{"lat": 53.3, "lon": -6.2, "distance": 1.8}])
    assert len(wps) == 4
    dist_km = sum(
        haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in zip(wps, wps[1:])
    )
    assert dist_km >= 450.0


def test_scanner_and_pressure_extraction_on_synthetic_globe():
    lats = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lons = (np.arange(1440, dtype=np.float32) * 0.25) - 180.0
    climo_mean = np.zeros((721, 1440), dtype=np.float32) + 10.0
    climo_std = np.full((721, 1440), 2.5, dtype=np.float32)

    t2m = climo_mean.copy()
    # Inject a strong heat dome around (45N, 10E) -> iy=180, ix=760
    yy, xx = np.ogrid[:721, :1440]
    heat_blob = np.exp(-(((yy - 180) ** 2 + (xx - 760) ** 2) / 180.0))
    t2m += 16.0 * heat_blob

    rain = np.zeros((721, 1440), dtype=np.float32)
    rain_blob = np.exp(-(((yy - 300) ** 2 + (xx - 1200) ** 2) / 120.0))
    rain += 140.0 * rain_blob

    mslp = 1013.0 - 45.0 * np.exp(-(((yy - 150) ** 2 + (xx - 680) ** 2) / 400.0))
    mslp += 28.0 * np.exp(-(((yy - 220) ** 2 + (xx - 960) ** 2) / 500.0))
    u10 = np.full((721, 1440), 5.0, dtype=np.float32)
    v10 = np.zeros((721, 1440), dtype=np.float32)
    wspd = np.hypot(u10, v10)
    gust = wspd + 28.0 * np.exp(-(((yy - 150) ** 2 + (xx - 680) ** 2) / 200.0))

    now = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
    wf = LiveWeatherFields(
        run_time=now,
        valid_time=now,
        step_hours=6,
        rain_window_hours=(0, 24),
        lats=lats,
        lons=lons,
        t2m_c=t2m,
        u10=u10,
        v10=v10,
        wind_speed=wspd,
        gust_max=gust,
        rain_24h_mm=rain,
        mslp_hpa=mslp.astype(np.float32),
        lsm=np.ones((721, 1440), dtype=np.float32),
    )

    cands, anom, _ = detect_global_candidates(wf, climo_mean, climo_std)
    kinds = {c["kind"] for c in cands}
    assert "heat" in kinds
    assert "rain" in kinds
    assert "cyclone" in kinds
    assert float(np.max(anom)) > 14.0

    centres, isobars = extract_pressure_features(to_half_degree(wf.mslp_hpa))
    assert any(c["kind"] == "L" for c in centres)
    assert any(c["kind"] == "H" for c in centres)
    assert len(isobars) > 0


def test_validate_and_clean_bulletin():
    sample_story = {
        "id": "test-storm",
        "kicker": "ATLANTIC CYCLONE",
        "headline": "Explosive Atlantic storm\u2014bringing severe gales to western coasts",
        "region": "North Atlantic & Ireland",
        "country": "Ireland",
        "hazard": "cyclone",
        "primary_layer": "wind",
        "centre": {"lat": 53.3, "lon": -8.0},
        "stats": {
            "headline_metric": "36 m/s",
            "metric_label": "Peak coastal gusts",
            "era5_rank": "#3 since 1940",
            "persistence": "4th storm in 3 weeks",
        },
        "camera_waypoints": [],
        "beats": [
            {"tag": "LIVE", "title": "Now", "layer": "wind", "text": "Fierce Atlantic gales are sweeping inland across western coasts right now."},
            {"tag": "PERSISTENCE", "title": "Weeks", "layer": "pressure", "text": "A locked North Atlantic jet stream has steered four deep lows here in two weeks."},
            {"tag": "RANK", "title": "History", "layer": "wind", "text": "ERA5 records back to 1940 place these late-November gusts in the top 3%."},
            {"tag": "CLIMATE", "title": "Climate", "layer": "anomaly", "text": "Warmer North Atlantic sea surfaces add extra latent heat energy to autumn storms."},
        ],
        "news_links": [
            {"source": "BBC News", "title": "Atlantic storm warnings", "url": "https://www.bbc.com/news/uk"}
        ],
    }
    draft = {"edition_title": "Bulletin", "stories": [dict(sample_story, id=f"s-{i}") for i in range(5)]}
    cleaned, errors = validate_and_clean_bulletin(draft)
    assert errors == []
    assert len(cleaned["stories"]) == 5
    assert "\u2014" not in cleaned["stories"][0]["headline"]
    assert len(cleaned["stories"][0]["camera_waypoints"]) == 4
