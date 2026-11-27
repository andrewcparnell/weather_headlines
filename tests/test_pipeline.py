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


def test_ensure_dynamic_waypoints_stays_within_local_radius():
    # Static single waypoint should expand into 4 local waypoints all within 500 km (never forced >550 km into the ocean)
    wps = ensure_dynamic_waypoints(53.3, -6.2, [{"lat": 53.3, "lon": -6.2, "distance": 1.8}])
    assert len(wps) == 4
    for wp in wps:
        assert haversine_km(wp["lat"], wp["lon"], 53.3, -6.2) <= 150.0


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
        # 2.5: No hardcoded fallback GDACS drought link when warnings=[]
        assert st["news_links"] == []
        cues = st["cues"]
        assert len(cues) >= 5
        assert "**" in st["ticker"]
        prev_end = 0.0
        for c in cues:
            assert c["end_sec"] > c["start_sec"] >= prev_end - 0.05
            prev_end = c["end_sec"]
            assert haversine_km(c["lat"], c["lon"], st["centre"]["lat"], st["centre"]["lon"]) <= 500.0


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


def test_opening_cue_automatically_anchors_country_when_omitted():
    base_beat = "Central pressure has plunged to 958 hPa across the storm centre right now."
    story = {
        "id": "iceland-storm",
        "headline": "Relentless 958 hPa cyclone lashes Iceland with blizzard chaos",
        "region": "Iceland and North Atlantic",
        "country": "Iceland",
        "primary_layer": "pressure",
        "hazard": "cyclone",
        "centre": {"lat": 63.5, "lon": -20.5},
        "cues": [
            {
                "layer": "pressure",
                "lat": 63.5,
                "lon": -20.5,
                "spoken": "At the heart of this relentless North Atlantic cyclone, central pressure has plunged to 958 hPa.",
                "ticker": "**958 hPa** low centre",
            },
            {"layer": "wind", "lat": 65.8, "lon": -18.1, "spoken": "Gales are sweeping North Iceland tonight.", "ticker": "Gales **160 km/h**"},
            {"layer": "rain", "lat": 64.0, "lon": -16.2, "spoken": "Heavy rain is drenching the Eastfjords.", "ticker": "Rain **140 mm**"},
            {"layer": "wind", "lat": 63.2, "lon": -20.8, "spoken": "Storm-force winds have persisted for two weeks.", "ticker": "Storms **13 days**"},
            {"layer": "anomaly", "lat": 62.0, "lon": -17.0, "spoken": "Warmer Atlantic waters are intensifying winter lows.", "ticker": "**+4.2 °C** anomaly"},
        ],
        "beats": [{"tag": f"B{j}", "title": "Beat", "layer": "wind", "text": base_beat} for j in range(4)],
        "news_links": [
            {
                "source": "Iceland Review",
                "title": "Blizzard warnings across North Iceland",
                "url": "https://www.icelandreview.com/news/iceland-orange-weather-warnings-blizzard-november-2026/",
            }
        ],
    }
    cleaned, errors = validate_and_clean_bulletin(
        {"stories": [dict(story, id=f"s-{i}") for i in range(5)]}
    )
    assert errors == []
    assert cleaned["stories"][0]["cues"][0]["spoken"].startswith(
        "Iceland: at the heart of this relentless North Atlantic cyclone,"
    )


def test_section1_rain_normals_era_frequency_orography_and_gazetteer():
    from pipeline.climatology import (
        LAPSE_RATE_C_PER_M,
        compute_era_frequency_metrics,
        compute_rain_persistence_metrics,
        nearest_gazetteer_place,
    )

    # 1.1 & 1.4: 14-day and 28-day normals must multiply daily rate by 14 and 28 (never * 0.5 or 5 mm floor)
    daily_obs = [5.0] * 20 + [15.0] * 8  # 28 days: 20*5 + 8*15 = 220 mm; 8 days >= 10 mm
    rp = compute_rain_persistence_metrics(daily_obs, climo_daily_rain_mm=5.0)
    assert rp["expected_rain_14d_mm"] == 70.0
    assert rp["expected_rain_28d_mm"] == 140.0
    assert rp["rain_28d_mm"] == 220.0
    assert rp["rain_28d_pct_of_normal"] == 157.0
    assert rp["wet_days_ge_10mm_last_28"] == 8
    assert "wet days" in rp["wet_days_label"]

    # Low desert 28-day normal (<5 mm) suppresses misleading percentage multiples
    rp_arid = compute_rain_persistence_metrics([0.0] * 27 + [12.0], climo_daily_rain_mm=0.1)
    assert rp_arid["expected_rain_28d_mm"] == 2.8
    assert rp_arid["rain_28d_pct_of_normal"] is None

    # 1.5: 40-year era exceedance counts - 3 vs 5 must NOT allow a frequency claim; 5 vs 9 (>=1.5x) DOES allow it
    years = np.array([yr for yr in range(1940, 2026) for _ in range(3)], dtype=np.int16)
    doys = np.array([330, 331, 332] * 86, dtype=np.int16)
    rng = np.random.default_rng(42)
    vals_small = rng.uniform(5.0, 15.0, size=len(years)).astype(np.float32)
    for yr in [1950, 1960, 1970]:
        vals_small[years == yr] = 50.0
    for yr in [1990, 1995, 2000, 2005, 2010, 2020]:
        vals_small[years == yr] = 50.0
    fm_blocked = compute_era_frequency_metrics(years, doys, vals_small, target_doy=331, tail="high")
    assert fm_blocked["p90_years_count_1940_1979_out_of_40"] == 3
    assert fm_blocked["p90_years_count_1986_2025_out_of_40"] == 6
    assert fm_blocked["frequency_claim_allowed"] is False
    assert "1979" in fm_blocked["era5_pre_1979_note"]

    vals_allowed = rng.uniform(5.0, 15.0, size=len(years)).astype(np.float32)
    for yr in [1945, 1952, 1960, 1968, 1975]:
        vals_allowed[years == yr] = 55.0
    for yr in [1988, 1992, 1996, 2001, 2006, 2012, 2018, 2022]:
        vals_allowed[years == yr] = 55.0
    # Lower non-spike years so p90 sits below 55 (13 spikes / 86 yrs = 15% > 10%)
    fm_ok = compute_era_frequency_metrics(years, doys, vals_allowed, target_doy=331, tail="high")
    assert fm_ok["p90_years_count_1940_1979_out_of_40"] >= 5
    assert fm_ok["frequency_claim_allowed"] is True

    # 1.3: Mountain orography mismatch (>300 m) excludes spurious high-terrain temperature anomalies
    lats = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lons = (np.arange(1440, dtype=np.float32) * 0.25) - 180.0
    climo_mean = np.full((721, 1440), 5.0, dtype=np.float32)
    climo_std = np.full((721, 1440), 2.0, dtype=np.float32)
    t2m = climo_mean.copy()
    # Inject +15 C anomaly ONLY over a mountain cell with 800 m IFS-vs-ERA5 elevation mismatch
    t2m[208, 1018] += 15.0
    ifs_elev = np.zeros((721, 1440), dtype=np.float32)
    era5_elev = np.zeros((721, 1440), dtype=np.float32)
    ifs_elev[206:211, 1016:1021] = 3200.0
    era5_elev[206:211, 1016:1021] = 4000.0  # 800 m mismatch > 300 m threshold
    assert abs(LAPSE_RATE_C_PER_M - 0.0065) < 1e-6

    now = datetime(2026, 11, 27, 0, tzinfo=timezone.utc)
    zeros = np.zeros((721, 1440), dtype=np.float32)
    wf = LiveWeatherFields(
        run_time=now,
        valid_time=now,
        step_hours=6,
        rain_window_hours=(0, 24),
        lats=lats,
        lons=lons,
        t2m_c=t2m,
        u10=zeros,
        v10=zeros,
        wind_speed=zeros,
        gust_max=zeros,
        rain_24h_mm=zeros,
        mslp_hpa=zeros + 1013.0,
        lsm=np.ones((721, 1440), dtype=np.float32),
        ifs_elev_m=ifs_elev,
    )
    cands, _, _ = detect_global_candidates(wf, climo_mean, climo_std, era5_elev_025_m=era5_elev)
    assert all(c["kind"] != "heat" for c in cands), "Spurious >300 m mountain mismatch cell should be excluded"

    # 1.6: Nearest gazetteer place attachment
    dub = nearest_gazetteer_place(53.35, -6.26)
    assert dub["nearest_place"] == "Dublin"
    assert dub["distance_km"] < 25.0


def test_section2_validator_geography_numbers_contradictions_and_urls(monkeypatch):
    import urllib.error
    import urllib.request

    from pipeline.validator import (
        check_url_resolves,
        validate_and_clean_bulletin,
        validate_internal_consistency,
        validate_story_geography,
        validate_story_numbers_against_candidates,
    )

    # 2.1 & 2.2: Explicit cue coordinates and explicit cue layers are NEVER overwritten even on a compact 35 km path
    compact_story = {
        "id": "dublin-rain",
        "headline": "Heavy rain expected across Dublin and eastern Leinster today",
        "region": "Dublin and Eastern Ireland",
        "country": "Ireland",
        "hazard": "rain",
        "primary_layer": "rain",
        "centre": {"lat": 53.35, "lon": -6.26},
        "stats": {
            "headline_metric": "54 mm / 24 h",
            "metric_label": "Forecast 24-hour rainfall",
            "era5_rank": "#2 for time of year since 1940",
            "persistence": "6 of 28 days wet (>=10 mm)",
        },
        "cues": [
            {"layer": "rain", "lat": 53.35, "lon": -6.26, "label": "DUBLIN", "spoken": "Ireland: up to 54 mm of rain is forecast across Dublin over the next 24 hours."},
            {"layer": "rain", "lat": 53.20, "lon": -6.10, "label": "WICKLOW COAST", "spoken": "Eastern coastal catchments face surface flooding during the evening commute."},
            {"layer": "temperature", "lat": 53.45, "lon": -6.22, "label": "NORTH COUNTY", "spoken": "Temperatures remain mild at 11.5 °C across the Dublin bay area."},
        ],
        "beats": [
            {"tag": "LIVE", "title": "Forecast", "layer": "rain", "text": "Up to 54 mm of rain is forecast across Dublin in the 24-hour window."},
            {"tag": "PERSISTENCE", "title": "Wet Days", "layer": "rain", "text": "The grid point has seen 6 of 28 days with at least 10 mm of rain."},
            {"tag": "RANK", "title": "Rank", "layer": "rain", "text": "Forecast 24-hour rainfall ranks #2 for this time of year since 1940."},
            {"tag": "CLIMATE", "title": "Context", "layer": "anomaly", "text": "Local late-autumn temperatures have warmed by +1.1 °C since 1940-1979."},
        ],
        "news_links": [
            {"source": "RTÉ News", "title": "Rain warning for Dublin", "url": "https://www.rte.ie/news/ireland/2026/1127/1489999-weather-rain-dublin/"}
        ],
    }
    cleaned, errs = validate_and_clean_bulletin({"stories": [dict(compact_story, id=f"s-{i}") for i in range(5)]})
    assert errs == []
    cues_clean = cleaned["stories"][0]["cues"]
    assert (cues_clean[0]["lat"], cues_clean[0]["lon"]) == (53.35, -6.26)
    assert (cues_clean[2]["lat"], cues_clean[2]["lon"]) == (53.45, -6.22)
    assert [c["layer"] for c in cues_clean] == ["rain", "rain", "temperature"]

    # 2.3a: Guayaquil cue placed out in the Pacific (-6.5, -88.51) is rejected (>500 km from centre & >150 km from Guayaquil)
    guayaquil_errs = validate_story_geography(
        "stories[1]",
        -2.0,
        -79.5,
        [{"lat": -6.5, "lon": -88.51, "label": "GUAYAQUIL PORT", "spoken": "Ecuador: heavy rain in Guayaquil."}],
        headline="Heavy rain forecast for Guayaquil",
    )
    assert len(guayaquil_errs) >= 2
    assert any("500 km" in e for e in guayaquil_errs)
    assert any("Guayaquil" in e and "150 km" in e for e in guayaquil_errs)

    # 2.3b: Naming Novosibirsk in a story centred at 65N, 87E (~1,135 km away) is rejected
    novosibirsk_errs = validate_story_geography(
        "stories[2]",
        65.0,
        87.0,
        [{"lat": 65.0, "lon": 87.0, "label": "NOVOSIBIRSK", "spoken": "Russia: mild air in Novosibirsk."}],
        headline="Mild air reaches Novosibirsk",
    )
    assert any("Novosibirsk" in e for e in novosibirsk_errs)

    # 2.4a: Internal contradictions (Pamir -7.7 C vs normal -7.6 C with +14.4 C anomaly; Iceland 34.3 m/s vs 44 m/s)
    pamir_contradiction = {
        "headline": "Pamir plateau sees +14.4 °C anomaly",
        "stats": {"headline_metric": "+14.4 °C anomaly", "metric_label": "Actual -7.7 °C vs normal -7.6 °C"},
        "cues": [],
        "beats": [],
    }
    assert validate_internal_consistency("stories[4]", pamir_contradiction) != []

    iceland_contradiction = {
        "headline": "Deep low nears Iceland",
        "stats": {"headline_metric": "34.3 m/s gusts", "metric_label": "Gusts to 160 km/h in Eastfjords"},
        "cues": [],
        "beats": [],
    }
    assert validate_internal_consistency("stories[0]", iceland_contradiction) != []

    # 2.4b: Candidate numeric fact-check rejects ungrounded figures & unsupported multiples, accepts news_facts claims
    cand = {
        "lat": 53.35,
        "lon": -6.26,
        "t2m_c": 11.5,
        "anom_c": 1.8,
        "rain_24h_mm": 54.0,
        "wind_ms": 12.0,
        "gust_ms": 20.0,
        "mslp_hpa": 996.0,
        "era5": {
            "units": "mm",
            "rank_since_1940": 2,
            "seasonal_normal_1991_2020": 3.1,
            "persistence": {
                "daily_rain_normal_mm": 3.1,
                "rain_14d_mm": 48.0,
                "expected_rain_14d_mm": 43.4,
                "rain_28d_mm": 92.0,
                "expected_rain_28d_mm": 86.8,
                "rain_28d_pct_of_normal": 106.0,
                "rain_28d_multiple_of_normal": 1.06,
                "wet_days_ge_10mm_last_28": 6,
            },
            "climate_shift": {"local_seasonal_warming_c_since_1940_1979": 1.1, "frequency_claim_allowed": False},
        },
    }
    assert validate_story_numbers_against_candidates("stories[0]", cleaned["stories"][0], [cand]) == []

    bad_story = dict(cleaned["stories"][0], headline="Dublin braced for 190 mm rain, 15 times the seasonal average")
    bad_num_errs = validate_story_numbers_against_candidates("stories[0]", bad_story, [cand])
    assert any("190 mm" in e for e in bad_num_errs)
    assert any("15x" in e for e in bad_num_errs)

    attributed_story = dict(
        cleaned["stories"][0],
        headline="Dublin braced for 54 mm forecast as agency warns of 120 mm 4-day total",
        news_facts=[
            {
                "claim": "Met Éireann warned of up to 120 mm over four days on high ground.",
                "source_url": "https://www.rte.ie/news/ireland/2026/1127/1489999-weather-rain-dublin/",
            }
        ],
    )
    assert validate_story_numbers_against_candidates("stories[0]", attributed_story, [cand]) == []

    # 2.5: HTTP HEAD/GET check accepts 200 and paywall 403, rejects 404
    class _DummyResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=8):
        u = req.full_url
        if "404-missing" in u:
            raise urllib.error.HTTPError(u, 404, "Not Found", {}, None)
        if "reuters.com" in u:
            raise urllib.error.HTTPError(u, 403, "Forbidden", {}, None)
        return _DummyResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert check_url_resolves("https://www.rte.ie/news/2026/1127/ok-story/") is True
    assert check_url_resolves("https://www.reuters.com/world/europe/storm-hits-2026-11-27/") is True
    assert check_url_resolves("https://www.rte.ie/news/2026/1127/404-missing-story/") is False


def test_section3_writer_prompt_edition_title_and_calm_accuracy():
    from pipeline.news_agent import SYSTEM_PROMPT, format_edition_title
    from pipeline.validator import validate_and_clean_bulletin, validate_editorial_tone_and_framing

    # 3.1: Hard-coded neutral edition_title format & zero positive BBC/Sky styling instructions
    dt = datetime(2026, 11, 27, 7, 30, tzinfo=timezone.utc)
    assert format_edition_title(dt) == "World Weather Headlines - 07:30 UTC, 27 November 2026"
    assert "clear, calm, accurate public-service tone; never exaggerate; when in doubt, understate" in SYSTEM_PROMPT
    assert "rain_window_start_utc" in SYSTEM_PROMPT
    assert "rain_window_end_utc" in SYSTEM_PROMPT
    assert "news_facts" in SYSTEM_PROMPT
    assert "preliminary" in SYSTEM_PROMPT
    assert "BBC News / Sky News" not in SYSTEM_PROMPT
    assert "600 km" not in SYSTEM_PROMPT

    # 3.2: Validator rejects BBC/Sky edition_title, banned hype phrases, sub-zero 'record heat', and past-tense model rain
    hype_story = {
        "id": "siberia-mild",
        "kicker": "CLIMATE EMERGENCY · SIBERIA",
        "headline": "West Siberia shatters historic record heat in baking heat dome",
        "ticker": "An **unprecedented** anomaly **ever recorded**",
        "region": "West Siberia",
        "country": "Russia",
        "hazard": "heat",
        "primary_layer": "anomaly",
        "centre": {"lat": 65.0, "lon": 87.0},
        "stats": {"headline_metric": "+18 °C", "metric_label": "Actual temp -7.0 °C", "era5_rank": "#1", "persistence": "9 of 14 days"},
        "cues": [
            {"layer": "anomaly", "lat": 65.0, "lon": 87.0, "label": "WEST SIBERIA", "spoken": "Russia: 40 mm of rain fell at -7.0 °C in record heat."}
        ],
        "beats": [
            {"tag": "FORECAST", "title": "Forecast", "layer": "anomaly", "text": "Forecast temperatures remain near -7.0 °C across central West Siberia."}
            for _ in range(4)
        ],
        "news_links": [{"source": "Example", "title": "Mild weather in Siberia", "url": "https://example.org/news/2026/11/27/siberia-mild-weather"}],
    }
    tone_errs = validate_editorial_tone_and_framing("stories[0]", hype_story, matched_cand={"t2m_c": -7.0})
    for expected in ("unprecedented", "ever recorded", "heat dome", "baking", "historic", "climate emergency", "shatters", "exceptionally mild", "forecast"):
        assert any(expected in e for e in tone_errs), f"Expected {expected!r} in {tone_errs}"

    _, ed_errs = validate_and_clean_bulletin(
        {"edition_title": "BBC/Sky World Weather Bulletin", "stories": [dict(hype_story, id=f"h-{i}") for i in range(5)]}
    )
    assert any("edition_title" in e for e in ed_errs)


def test_section4_gdacs_iscurrent_filter_workflow_and_stamped_audio():
    from pathlib import Path

    from pipeline.tts import format_audio_stamp
    from pipeline.warnings_feed import fetch_global_warnings, is_gdacs_event_current

    # 4.1: GDACS iscurrent filter handles both boolean False and string 'false'
    assert is_gdacs_event_current({"iscurrent": True}) is True
    assert is_gdacs_event_current({"iscurrent": "true"}) is True
    assert is_gdacs_event_current({"iscurrent": False}) is False
    assert is_gdacs_event_current({"iscurrent": "false"}) is False

    sample_gdacs = {
        "features": [
            {
                "properties": {
                    "eventtype": "TC",
                    "name": "Ended Cyclone",
                    "alertlevel": "Orange",
                    "iscurrent": "false",
                    "url": {"report": "https://www.gdacs.org/report.aspx?eventid=1&episodeid=1&eventtype=TC"},
                },
                "geometry": {"type": "Point", "coordinates": [120.0, 15.0]},
            },
            {
                "properties": {
                    "eventtype": "FL",
                    "name": "Active Flood",
                    "alertlevel": "Red",
                    "iscurrent": "true",
                    "url": {"report": "https://www.gdacs.org/report.aspx?eventid=2&episodeid=1&eventtype=FL"},
                },
                "geometry": {"type": "Point", "coordinates": [108.0, 16.0]},
            },
        ]
    }
    active = fetch_global_warnings(raw_payload=sample_gdacs)
    assert len(active) == 1
    assert active[0]["name"] == "Active Flood"

    # 4.2: Stamped audio filenames
    assert format_audio_stamp("2026-11-27T07:30:12Z") == "20261127T073012Z"

    # 4.3: Workflow never commits site/data to main and never puts github.run_id in cache key
    wf_txt = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "refresh.yml").read_text()
    assert "git add site/data" not in wf_txt
    assert "HEAD:main" not in wf_txt
    assert "github.run_id" not in wf_txt





