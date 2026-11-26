"""Unit tests for the Weather Headlines pipeline, WebP encoding, scanner, and validator."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from pipeline.opendata import LiveWeatherFields, to_half_degree
from pipeline.pack_globe import dequantise_u8, quantise_u8
from pipeline.scanner import detect_global_candidates, extract_pressure_features, haversine_km
from pipeline.tts import normalise_for_newsreader
from pipeline.validator import (
    ensure_dynamic_waypoints,
    is_specific_event_url,
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
            {
                "source": "The Guardian",
                "title": "Atlantic storm brings gales and travel disruption",
                "url": "https://www.theguardian.com/uk-news/2026/nov/24/atlantic-storm-warnings-gales",
            }
        ],
    }
    draft = {"edition_title": "Bulletin", "stories": [dict(sample_story, id=f"s-{i}") for i in range(5)]}
    cleaned, errors = validate_and_clean_bulletin(draft)
    assert errors == []
    assert len(cleaned["stories"]) == 5
    assert "\u2014" not in cleaned["stories"][0]["headline"]
    assert "**" in cleaned["stories"][0]["ticker"]
    assert len(cleaned["stories"][0]["camera_waypoints"]) == 4


def test_generic_hub_urls_rejected_and_specific_articles_accepted():
    assert not is_specific_event_url("https://www.bbc.com/weather")
    assert not is_specific_event_url("https://www.theguardian.com/world/extreme-weather")
    assert not is_specific_event_url("https://www.irishtimes.com/tags/weather-events/")
    assert not is_specific_event_url("https://climate.copernicus.eu/climate-bulletins")
    assert is_specific_event_url(
        "https://www.theguardian.com/us-news/2026/nov/25/heavy-rain-high-winds-thanksgiving-east-coast"
    )
    assert is_specific_event_url("https://www.gdacs.org/report.aspx?eventid=1018332&episodeid=28&eventtype=DR")


def test_newsreader_speech_normalisation():
    raw = "Met Eireann warns of **139 mm / 24 h** and +16.4 °C (#1 in 1940-1979) during El Nino."
    spoken = normalise_for_newsreader(raw)
    assert "**" not in spoken
    assert "Met Air-uhn" in spoken
    assert "El Neen-yo" in spoken
    assert "139 millimetres in 24 hours" in spoken
    assert "16.4 degrees Celsius" in spoken
    assert "number 1" in spoken
    assert "1940 to 1979" in spoken


def test_deterministic_bulletin_satisfies_all_cue_and_layer_invariants():
    from pipeline.news_agent import build_deterministic_bulletin

    now = datetime(2026, 11, 26, 21, tzinfo=timezone.utc)
    lats = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lons = (np.arange(1440, dtype=np.float32) * 0.25) - 180.0
    zeros = np.zeros((721, 1440), dtype=np.float32)
    wf = LiveWeatherFields(
        run_time=now,
        valid_time=now,
        step_hours=9,
        rain_window_hours=(0, 24),
        lats=lats,
        lons=lons,
        t2m_c=zeros + 12.0,
        u10=zeros + 5.0,
        v10=zeros,
        wind_speed=zeros + 5.0,
        gust_max=zeros + 25.0,
        rain_24h_mm=zeros,
        mslp_hpa=zeros + 1005.0,
    )
    enriched = [
        {
            "id": f"cand-{i}",
            "kind": kind,
            "primary_layer": "wind" if kind == "cyclone" else ("rain" if kind == "rain" else "anomaly"),
            "lat": lat,
            "lon": lon,
            "t2m_c": 24.0,
            "anom_c": 9.5 if kind != "cold" else -10.5,
            "rain_24h_mm": 115.0,
            "wind_ms": 22.0,
            "gust_ms": 33.0,
            "mslp_hpa": 962.0,
            "era5": {
                "rank_since_1940": 1,
                "years_compared": 87,
                "percentile": 99.4,
                "persistence": {"anomalous_days_last_14": 8, "anomalous_days_last_28": 15, "rain_14d_mm": 140.0},
                "climate_shift": {
                    "local_seasonal_warming_c_since_1940_1979": 1.6,
                    "p90_exceedances_1940_1979": 2,
                    "p90_exceedances_1986_2025": 7,
                },
            },
        }
        for i, (kind, lat, lon) in enumerate(
            [
                ("cyclone", 58.0, -18.0),
                ("rain", 36.5, -75.5),
                ("heat", -25.0, 135.0),
                ("cold", 61.0, 95.0),
                ("rain", 14.0, 121.0),
            ],
            start=1,
        )
    ]
    bulletin = build_deterministic_bulletin(wf, enriched, warnings=[])
    assert len(bulletin["stories"]) == 5
    for st in bulletin["stories"]:
        cues = st["cues"]
        assert len(cues) >= 5
        assert len({c["layer"] for c in cues}) >= 3
        assert "**" in st["ticker"]
        prev_end = 0.0
        path_km = 0.0
        for idx, c in enumerate(cues):
            assert c["end_sec"] > c["start_sec"] >= prev_end - 0.05
            prev_end = c["end_sec"]
            if idx > 0:
                path_km += haversine_km(cues[idx - 1]["lat"], cues[idx - 1]["lon"], c["lat"], c["lon"])
        assert path_km >= 500.0


def test_validator_filters_secondary_hub_link_when_valid_event_link_exists():
    base_beat = "Heavy Atlantic gales are sweeping inland across western coastal counties right now."
    story = {
        "id": "s",
        "headline": "Atlantic storm brings severe coastal gales across Ireland",
        "primary_layer": "wind",
        "hazard": "cyclone",
        "centre": {"lat": 53.3, "lon": -8.0},
        "beats": [{"tag": f"B{j}", "title": "Beat", "layer": "wind", "text": base_beat} for j in range(4)],
        "news_links": [
            {"source": "BBC Hub", "title": "Weather", "url": "https://www.bbc.com/weather"},
            {
                "source": "The Irish Times",
                "title": "Storm warnings issued for western counties",
                "url": "https://www.irishtimes.com/ireland/2026/11/26/storm-warnings-western-counties/",
            },
        ],
    }
    cleaned, errors = validate_and_clean_bulletin(
        {"stories": [dict(story, id=f"s-{i}") for i in range(5)]}
    )
    assert errors == []
    assert len(cleaned["stories"][0]["news_links"]) == 1
    assert "irishtimes.com" in cleaned["stories"][0]["news_links"][0]["url"]


