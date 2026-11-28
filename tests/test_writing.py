"""Tests for place naming, the template fallback writer, and the Claude writer brief."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np

from pipeline.opendata import LiveWeatherFields


def _weather(now: datetime, mslp: np.ndarray | None = None) -> LiveWeatherFields:
    zeros = np.zeros((721, 1440), dtype=np.float32)
    return LiveWeatherFields(
        run_time=now - timedelta(hours=9),
        valid_time=now,
        step_hours=9,
        rain_window_hours=(0, 24),
        lats=np.linspace(90.0, -90.0, 721, dtype=np.float32),
        lons=(np.arange(1440, dtype=np.float32) * 0.25) - 180.0,
        t2m_c=zeros,
        u10=zeros,
        v10=zeros,
        wind_speed=zeros,
        gust_max=zeros,
        rain_24h_mm=zeros,
        mslp_hpa=zeros + 1013.0 if mslp is None else mslp,
        rain_window_start_utc=now - timedelta(hours=9),
        rain_window_end_utc=now + timedelta(hours=15),
    )


def _cand(i: int, kind: str, lat: float, lon: float, **era_over) -> dict:
    era = {
        "rank_since_1940": None,
        "rank_available": False,
        "wmo_tmax_normal_c": 20.0,
        "wmo_tmin_normal_c": 8.0,
        "persistence": {
            "warm_days_above_p90_tmax_last_14": 6,
            "cold_days_below_p10_tmin_last_14": 1,
            "wet_days_ge_10mm_last_28": 7,
            "days_gust_over_65kmh_last_28": 8,
            "rain_28d_pct_of_normal": 170.0,
            "daily_rain_normal_mm": 4.0,
            "expected_rain_28d_mm": 112.0,
            "local_gust_p90_kmh": 60.0,
        },
        "climate_shift": {"local_seasonal_warming_c_since_1940_1979": None},
    }
    era.update(era_over)
    return {
        "id": f"cand-{i}",
        "kind": kind,
        "lat": lat,
        "lon": lon,
        "peak_value": 1.0,
        "units": "",
        "t2m_c": 31.0 if kind == "heat" else (-12.0 if kind == "cold" else 15.0),
        "t2m_max_24h_c": 33.0 if kind == "heat" else 5.0,
        "t2m_min_24h_c": -14.0 if kind == "cold" else 10.0,
        "anom_c": 9.0 if kind != "cold" else -9.0,
        "rain_24h_mm": 130.0 if kind == "rain" else 5.0,
        "wind_ms": 15.0,
        "gust_ms": 33.0,
        "mslp_hpa": 1004.0,
        "era5": era,
    }


def test_describe_location_names_states_for_large_countries_and_handles_offshore():
    from pipeline.places import describe_location

    ok = describe_location(35.5, -97.5)
    assert ok["headline_prefix"] == "Oklahoma, USA"
    assert ok["country"] == "United States" and not ok["offshore"]

    vn = describe_location(15.5, 108.5)
    assert vn["headline_prefix"] == "Vietnam"

    # 82 km off Peru must be Peru (the old nearest-centroid lookup said Ecuador)
    pe = describe_location(-5.0, -82.0)
    assert pe["country"] == "Peru" and pe["offshore"]

    ru = describe_location(65.0, 87.0)
    assert ru["headline_prefix"] == "Krasnoyarsk Krai, Russia"
    assert ru["nearest_town"] == "Turukhansk"

    # Open ocean: no country at all rather than a guess
    assert describe_location(0.0, -140.0)["country"] == ""


def test_writer_payload_is_json_serialisable_with_datetime_rain_window():
    """Regression: datetime fields crashed json.dumps, so every run fell back to template text."""
    from pipeline.news_agent import build_writer_payload

    now = datetime(2026, 11, 28, 3, tzinfo=timezone.utc)
    payload = build_writer_payload(_weather(now), [_cand(1, "rain", 15.5, 108.5)], [], [], fetch_news=False)
    text = json.dumps(payload)
    assert payload["rain_window_start_utc"] == "2026-11-27T18:00:00Z"
    assert "2026-11-28T18:00:00Z" in text


def test_fallback_writer_uses_state_prefix_varied_wording_and_no_invented_rank():
    from pipeline.news_agent import build_deterministic_bulletin
    from pipeline.places import attach_location_context

    now = datetime(2026, 11, 28, 3, tzinfo=timezone.utc)
    weather = _weather(now)
    cands = attach_location_context(
        [
            _cand(1, "heat", 35.5, -97.5),  # Oklahoma
            _cand(2, "heat", -25.0, 135.0),  # Northern Territory
            _cand(3, "rain", 15.5, 108.5),  # Vietnam
            _cand(4, "rain", 14.0, 121.0),  # Philippines
            _cand(5, "cold", 61.0, 95.0),  # Krasnoyarsk Krai
        ],
        weather,
    )
    bulletin = build_deterministic_bulletin(weather, cands, warnings=[], fetch_live_rss=False)
    stories = bulletin["stories"]
    heads = [st["headline"] for st in stories]
    assert any(h.startswith("Oklahoma, USA:") for h in heads)
    assert any(h.startswith("Krasnoyarsk Krai, Russia:") for h in heads)

    # Two stories of the same hazard must not share a headline or opening line template
    by_kind: dict[str, list[dict]] = {}
    for st in stories:
        by_kind.setdefault(st["hazard"], []).append(st)
    for group in by_kind.values():
        if len(group) == 2:
            a, b = group
            assert a["headline"].split(":", 1)[1][:25] != b["headline"].split(":", 1)[1][:25]
            assert a["cues"][0]["spoken"].split(":", 1)[1][:25] != b["cues"][0]["spoken"].split(":", 1)[1][:25]

    blob = json.dumps(stories).lower()
    # No rank or percentile was available, so none may be claimed
    assert "since 1940" not in blob and "top 1%" not in blob and "wettest 1%" not in blob
    # And no invented long-term warming figure either
    assert "mid-twentieth century" not in blob


def test_storm_story_uses_nearby_low_centre_pressure():
    from pipeline.news_agent import build_deterministic_bulletin
    from pipeline.places import attach_location_context

    now = datetime(2026, 11, 28, 3, tzinfo=timezone.utc)
    lat = np.linspace(90, -90, 721)[:, None]
    lon = (np.arange(1440) * 0.25 - 180)[None, :]
    mslp = (1013.0 - 45.0 * np.exp(-((lat - 64.0) ** 2 + ((lon + 22.0) * 0.45) ** 2) / 8.0)).astype(np.float32)
    weather = _weather(now, mslp=mslp)
    cands = attach_location_context(
        [
            _cand(1, "cyclone", 62.5, -15.5),
            _cand(2, "rain", 15.5, 108.5),
            _cand(3, "heat", -25.0, 135.0),
            _cand(4, "cold", 61.0, 95.0),
            _cand(5, "heat", 35.5, -97.5),
        ],
        weather,
    )
    assert cands[0]["low_centre"]["mslp_hpa"] < 975
    bulletin = build_deterministic_bulletin(weather, cands, warnings=[], fetch_live_rss=False)
    storm = next(st for st in bulletin["stories"] if st["hazard"] == "cyclone")
    assert "968 hPa" in storm["cues"][0]["spoken"] and "deep" in storm["cues"][0]["spoken"]


def test_salvage_replaces_only_failing_claude_stories():
    from pipeline.news_agent import build_deterministic_bulletin, salvage_bulletin
    from pipeline.places import attach_location_context

    now = datetime(2026, 11, 28, 3, tzinfo=timezone.utc)
    weather = _weather(now)
    cands = attach_location_context(
        [
            _cand(1, "heat", 35.5, -97.5),
            _cand(2, "heat", -25.0, 135.0),
            _cand(3, "rain", 15.5, 108.5),
            _cand(4, "rain", 14.0, 121.0),
            _cand(5, "cold", 61.0, 95.0),
            _cand(6, "rain", 45.0, 10.0),
        ],
        weather,
    )
    good = build_deterministic_bulletin(weather, cands[:5], warnings=[], fetch_live_rss=False)
    good["generator"] = "anthropic:test"
    for i, st in enumerate(good["stories"]):
        st["id"] = f"claude-story-{i}"
        st["news_links"] = [{"source": "Example", "title": "Story", "url": "https://example.org/news/2026/11/28/weather-story"}]
    fixed = salvage_bulletin(good, ["stories[2]: rainfall figure '999 mm' does not match"], weather, cands)
    assert fixed is not None
    assert fixed["stories"][0]["headline"] == good["stories"][0]["headline"]
    # The failing story is rewritten from template text (here for the same, still-unused hotspot)
    assert fixed["stories"][2]["id"].startswith("auto-")
    assert [st["id"] for i, st in enumerate(fixed["stories"]) if i != 2] == [f"claude-story-{i}" for i in (0, 1, 3, 4)]

    # Bulletin-level problems cannot be salvaged
    assert salvage_bulletin(good, ["Expected 5 stories, got 3."], weather, cands) is None
