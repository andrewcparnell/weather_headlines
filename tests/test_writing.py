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


def test_era5_seasonal_windows_cost_one_call_per_location_per_year(monkeypatch, tmp_path):
    """The ranking history is fetched as 14-day windows (1 API call per location per year), all years 1940-2025."""
    from datetime import date

    import pipeline.climatology as clim

    monkeypatch.setattr(clim, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(clim, "OPEN_METEO_BUDGET_PER_MIN", 1e9)
    calls: list[str] = []

    def fake_get(url, timeout=45):
        calls.append(url)
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(url).query)
        start, end = date.fromisoformat(q["start_date"][0]), date.fromisoformat(q["end_date"][0])
        assert (end - start).days == 13  # 14 days -> one call per location
        n = len(q["latitude"][0].split(","))
        days = [(start + timedelta(days=i)).isoformat() for i in range(14)]
        year = start.year if start.month == 11 else end.year
        return [
            {
                "daily": {
                    "time": days,
                    "temperature_2m_max": [20.0 + (year - 1940) * 0.01] * 14,
                    "temperature_2m_min": [10.0] * 14,
                    "precipitation_sum": [1.0] * 14,
                    "wind_gusts_10m_max": [12.0] * 14,
                }
            }
            for _ in range(n)
        ]

    monkeypatch.setattr(clim, "_get_json_with_rate_limit", fake_get)
    cands = [_cand(1, "heat", 11.0, 18.5), _cand(2, "rain", 15.5, 108.5)]
    vt = datetime(2026, 11, 28, 9, tzinfo=timezone.utc)
    hist = clim.fetch_era5_seasonal_windows(cands, vt)
    assert len(calls) == 86  # one request per year, both locations in each request
    for h in hist:
        assert h is not None and len(np.unique(h["years"])) == 86 and len(h["tmax"]) == 86 * 14

    # Second run is served from the cache without any requests
    calls.clear()
    assert all(h is not None for h in clim.fetch_era5_seasonal_windows(cands, vt)) and calls == []


def test_fact_check_accepts_spelled_out_units_and_supported_hedges():
    from pipeline.validator import _speed_to_ms, extract_numbers_by_unit, remove_supported_hedged_figures

    # "kilometres per hour" written out must not be read as metres per second
    assert abs(_speed_to_ms(139.0, "kilometres per hour") - 38.6) < 0.1
    (v, _), = extract_numbers_by_unit("gusts near 139 kilometres per hour")["speed_ms"]
    assert abs(v - 38.6) < 0.1

    pools = {"temp": [12.4, 0.9], "rain": [320.0], "speed": [38.7]}
    assert remove_supported_hedged_figures("more than 10 °C above normal", pools).strip() == "above normal"
    assert remove_supported_hedged_figures("10 mm or more", pools).strip() == ""
    assert remove_supported_hedged_figures("nearly 140 km/h", pools).strip() == ""
    # Unsupported claims stay in the text and are still checked
    assert "400 mm" in remove_supported_hedged_figures("more than 400 mm", pools)
    assert "15 °C" in remove_supported_hedged_figures("15 °C warmer", pools)


def test_claude_retry_keeps_same_tools_and_has_room_to_finish(monkeypatch):
    """A truncated first reply is retried in the same conversation with an identical tool list
    (newer models reject thinking blocks replayed under a different tool list) and a large max_tokens."""
    import types

    import anthropic

    from pipeline import news_agent
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
        ],
        weather,
    )
    good = news_agent.build_deterministic_bulletin(weather, cands, warnings=[], fetch_live_rss=False)
    for st in good["stories"]:
        st["news_links"] = [{"source": "Example", "title": "x", "url": "https://example.org/news/2026/11/28/weather-story"}]
    replies = [
        ("max_tokens", '{"stories": [{"id": "trunc'),
        ("end_turn", json.dumps({"stories": good["stories"]})),
    ]
    calls: list[dict] = []

    class FakeStream:
        def __init__(self, kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            stop, text = replies[len(calls) - 1]
            return types.SimpleNamespace(stop_reason=stop, content=[types.SimpleNamespace(type="text", text=text)])

    class FakeMessages:
        def stream(self, **kw):
            calls.append(kw)
            return FakeStream(kw)

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: types.SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(news_agent, "_safe_news", lambda c: [])
    monkeypatch.setattr("pipeline.validator.check_url_resolves", lambda u, timeout=8: True)

    monkeypatch.setattr(news_agent, "WEB_SEARCHES_PER_REQUEST", 2)
    bulletin = news_agent.generate_headlines_with_claude(weather, cands, [], None)
    assert len(calls) == 2
    assert calls[0]["tools"] == calls[1]["tools"]
    assert calls[0]["max_tokens"] >= 32000
    assert calls[0]["output_config"] == {"effort": "low"}
    assert calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert bulletin["generator"].startswith("anthropic:")

    # Default: no web search at all (the cheap mode), so no tools are sent
    calls.clear()
    monkeypatch.setattr(news_agent, "WEB_SEARCHES_PER_REQUEST", 0)
    news_agent.generate_headlines_with_claude(weather, cands, [], None)
    assert "tools" not in calls[0]
    assert calls[0]["max_tokens"] >= 32000


def test_cheap_mode_brief_is_small_and_news_is_limited_to_leads():
    from pipeline import news_agent
    from pipeline.places import attach_location_context

    now = datetime(2026, 11, 28, 3, tzinfo=timezone.utc)
    weather = _weather(now)
    cands = attach_location_context([_cand(i, k, la, lo) for i, (k, la, lo) in enumerate(
        [("heat", 35.5, -97.5), ("heat", -25.0, 135.0), ("rain", 15.5, 108.5), ("rain", 14.0, 121.0),
         ("cold", 61.0, 95.0), ("rain", 45.0, 10.0), ("cyclone", 62.5, -15.5), ("heat", 11.0, 18.5)])], weather)
    payload = news_agent.build_writer_payload(weather, cands, [], [], fetch_news=False)
    text = json.dumps(payload, separators=(",", ":"))
    assert len(payload["stories_to_write"]) == 5
    assert len(text) < 6000  # roughly 2k tokens for 5 stories without news
    first = payload["stories_to_write"][0]
    assert first["location"]["headline_prefix"] == "Oklahoma, USA"
    assert "anomaly_c" not in json.dumps(first)  # the one-hour anomaly is never offered for comparisons
    assert first["needs_more_news"] == 3

    draft = {"stories": [{
        "cues": [{"spoken": f"Sentence {i}.", "ticker": f"**{i}**", "layer": "rain"} for i in range(5)],
        "news_links": [{"url": "https://ok.example/a-1"}, {"url": "https://made-up.example/b-2"}],
        "news_facts": [{"claim": "x", "source_url": "https://made-up.example/b-2"}],
    }]}
    news_agent.complete_story_fields(draft)
    st = draft["stories"][0]
    assert len(st["beats"]) == 4 and "**0**" in st["ticker"]
    news_agent._restrict_news_to_leads(draft, {"https://ok.example/a-1"})
    assert [lk["url"] for lk in st["news_links"]] == ["https://ok.example/a-1"] and st["news_facts"] == []


def test_usage_meter_estimates_cost():
    import types

    from pipeline.news_agent import UsageMeter

    m = UsageMeter()
    usage = types.SimpleNamespace(input_tokens=1000, output_tokens=6000, cache_creation_input_tokens=4000,
                                  cache_read_input_tokens=0, server_tool_use=None)
    m.add("claude-sonnet-5-5", types.SimpleNamespace(usage=usage))
    # 1k x $2 + 6k x $10 + 4k x $2.5 per million = $0.072
    assert abs(m.cost_usd() - 0.072) < 1e-9
    assert "about $0.072" in m.summary()


def test_negative_temperatures_may_be_written_as_magnitudes():
    """'8 °C below freezing' for a typical high of -8.2 °C must pass (seen in a live run)."""
    from pipeline.validator import validate_story_numbers_against_candidates

    cand = _cand(1, "heat", 53.5, -94.5)
    cand["era5"]["wmo_tmax_normal_c"] = -8.2
    story = {
        "centre": {"lat": 53.5, "lon": -94.5},
        "headline": "Manitoba, Canada: winter on hold",
        "stats": {},
        "cues": [{"spoken": "Normally, afternoons here barely climb past 8 °C below freezing at this time of year."}],
        "beats": [],
        "news_facts": [],
    }
    assert validate_story_numbers_against_candidates("stories[0]", story, [cand]) == []


def _fake_claude(monkeypatch, replies):
    import types

    import anthropic

    from pipeline import news_agent

    calls: list[dict] = []

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            text = replies[min(len(calls), len(replies)) - 1]
            return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text=text)])

    class FakeMessages:
        def stream(self, **kw):
            calls.append(kw)
            return FakeStream()

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: types.SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(news_agent, "_safe_news", lambda c: [])
    monkeypatch.setattr("pipeline.validator.check_url_resolves", lambda u, timeout=8: True)
    return calls


def _five_cands():
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
    return weather, cands


def test_failing_story_is_fixed_by_a_targeted_retry_not_template_text(monkeypatch):
    from pipeline import news_agent

    weather, cands = _five_cands()
    stories = news_agent.build_deterministic_bulletin(weather, cands[:5], warnings=[], fetch_live_rss=False)["stories"]
    for i, st in enumerate(stories):
        st["id"] = f"claude-{i}"
        st["news_links"] = []
    good_story_4 = json.loads(json.dumps(stories[4]))
    stories[4]["headline"] = stories[4]["headline"] + " and 999 mm of rain"
    calls = _fake_claude(
        monkeypatch,
        [json.dumps({"stories": stories}), json.dumps({"stories": {"4": good_story_4}})],
    )
    bulletin = news_agent.generate_headlines_with_claude(weather, cands[:5], [], None)
    assert len(calls) == 2 and {c["model"] for c in calls} == {"claude-sonnet-5-5"}
    retry_request = calls[1]["messages"][-1]["content"]
    assert "Include just these stories: 4" in retry_request and "999 mm" in retry_request
    assert [st["id"] for st in bulletin["stories"]] == [f"claude-{i}" for i in range(5)]
    assert "999" not in bulletin["stories"][4]["headline"]

def test_unusable_bulletin_falls_back_to_template_without_trying_another_model(monkeypatch):
    import pytest

    from pipeline import news_agent

    weather, cands = _five_cands()
    calls = _fake_claude(monkeypatch, [json.dumps({"stories": []})])
    with pytest.raises(news_agent.WriterValidationError):
        news_agent.generate_headlines_with_claude(weather, cands, [], None)
    assert {c["model"] for c in calls} == {"claude-sonnet-5-5"}


def test_temperature_comparisons_use_daily_highs_not_the_hourly_anomaly():
    """Live bug: Chad 40 °C vs typical high 36 °C was described as 'around seven degrees warmer'."""
    from pipeline.validator import validate_story_numbers_against_candidates

    cand = _cand(1, "heat", 11.0, 18.5)
    cand.update({"t2m_max_24h_c": 39.9, "anom_c": 7.9})
    cand["era5"]["wmo_tmax_normal_c"] = 36.2
    story = lambda text: {  # noqa: E731
        "centre": {"lat": 11.0, "lon": 18.5}, "headline": "Chad: hot", "stats": {}, "beats": [], "news_facts": [],
        "cues": [{"spoken": text}],
    }
    bad = validate_story_numbers_against_candidates("s", story("That is around seven degrees warmer than usual for the date."), [cand])
    assert bad and "+3.7" in bad[0]
    assert validate_story_numbers_against_candidates("s", story("Highs near 40 °C, about 4 degrees above the usual high."), [cand]) == []


def test_choose_stories_prefers_news_land_and_variety():
    from pipeline.news_leads import choose_stories

    def c(i, kind, lat, lon, leads, country="X", offshore_km=0):
        return {"id": f"c{i}", "kind": kind, "lat": lat, "lon": lon, "news_leads": [{}] * leads,
                "location": {"country": country, "offshore_km": offshore_km}}

    cands = [
        c(1, "heat", 11.0, 18.5, 0, "Chad"),
        c(2, "heat", 12.0, 36.0, 0, "Ethiopia"),  # near-duplicate of Chad
        c(3, "rain", 17.5, -104.0, 0, "Mexico", offshore_km=174),  # rain far out at sea
        c(4, "rain", 15.5, 108.5, 3, "Vietnam"),
        c(5, "cyclone", 62.5, -15.5, 3, "Iceland", offshore_km=60),
        c(6, "cold", 50.5, 2.5, 2, "France"),
        c(7, "heat", 53.5, -94.5, 1, "Canada"),
        c(8, "rain", -30.0, 150.0, 0, ""),  # open ocean
    ]
    names = [x["location"]["country"] for x in choose_stories(cands)]
    assert names[:2] == ["Vietnam", "Iceland"] and "France" in names
    assert not {"Chad", "Ethiopia"} <= set(names)  # no near-identical heat stories side by side
    assert "Mexico" not in names and "" not in names  # rain far out at sea, open ocean


def test_gather_news_keeps_recent_relevant_reputable_local_articles(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from pipeline import news_leads

    now = datetime.now(timezone.utc)
    articles = [
        {"title": "Heavy rain floods homes in central Vietnam", "url": "https://news.tuoitre.vn/heavy-rain-floods-homes-123.htm",
         "summary": "Heavy rain has isolated nearly 1,000 households in Quang Ngai.", "source": "news.tuoitre", "published": now},
        {"title": "Vietnam floods leave 90 dead", "url": "https://www.theguardian.com/world/2025/nov/20/vietnam-floods-90",
         "summary": "", "source": "The Guardian", "published": now - timedelta(days=370)},  # too old
        {"title": "Vietnam floods: what we know", "url": "https://uk.news.yahoo.com/vietnam-floods-what-we-know-1.html",
         "summary": "", "source": "Yahoo News UK", "published": now},  # unreliable aggregator
        {"title": "Viettel launches satellite internet in Vietnam", "url": "https://e.vnexpress.net/news/viettel-starlink-1.html",
         "summary": "", "source": "VnExpress", "published": now},  # not about the weather
    ]
    monkeypatch.setattr(news_leads, "search_bing_news", lambda q: articles)
    monkeypatch.setattr(news_leads, "search_google_news", lambda q: [])
    monkeypatch.setattr("pipeline.validator.check_url_resolves", lambda u, timeout=8: True)
    cand = {"kind": "rain", "location": {"country": "Vietnam", "country_domain": "vn"}}
    leads = news_leads.gather_news(cand)
    assert [n["source"] for n in leads] == ["news.tuoitre"]
    assert leads[0]["summary"].startswith("Heavy rain has isolated")


def test_build_fails_instead_of_publishing_template_text(monkeypatch, tmp_path):
    from pipeline import build

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not build._is_recent_valid_bulletin({"generator": "era5-deterministic", "stories": []})
    # Writer failure must surface as a build failure, never as template text
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def boom(*a, **k):
        raise RuntimeError("writer exploded")

    monkeypatch.setattr(build, "generate_headlines_with_claude", boom)
    monkeypatch.setattr(build, "build_deterministic_bulletin", lambda *a, **k: (_ for _ in ()).throw(AssertionError("template used")))
    now = datetime(2026, 11, 28, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(build, "fetch_live_weather", lambda: _weather(now))
    monkeypatch.setattr(build, "load_era5_historical_context", lambda *a, **k: (None, None, "x", None))
    monkeypatch.setattr(build, "detect_global_candidates", lambda *a, **k: ([], None, None))
    monkeypatch.setattr(build, "package_globe_layers", lambda *a, **k: {"isobars": [], "pressure_centres": []})
    monkeypatch.setattr(build, "attach_location_context", lambda c, *a, **k: c)
    monkeypatch.setattr(build, "attach_news", lambda c: c)
    monkeypatch.setattr(build, "choose_stories", lambda c, n=5: c)
    monkeypatch.setattr(build, "enrich_all_candidates", lambda *a, **k: [])
    monkeypatch.setattr(build, "fetch_global_warnings", lambda **k: [])
    import pytest

    with pytest.raises(build.BulletinError):
        build.run_pipeline(out_dir=tmp_path)
    assert not (tmp_path / "headlines.json").exists()


def test_news_citation_rule_needs_three_working_links_not_every_lead():
    """Live failure: the writer rightly skipped a lead whose link did not load, and the old rule
    ('cite every lead') rejected the whole bulletin three times."""
    from pipeline.news_agent import check_story_news

    payload = {"stories_to_write": [
        {"news_leads": [{"url": f"https://a.example/news/{i}"} for i in range(4)]},
        {"news_leads": [{"url": "https://b.example/news/1"}]},
    ]}
    three = [{"url": f"https://a.example/news/{i}"} for i in (0, 1, 3)]
    assert check_story_news([{"news_links": three}, {"news_links": [{"url": "https://b.example/news/1"}]}], payload) == []
    errs = check_story_news([{"news_links": three[:2]}, {"news_links": []}], payload)
    assert len(errs) == 2 and "at least 3" in errs[0] and "at least 1" in errs[1]


def test_rain_brief_gives_the_record_year_but_not_the_incomparable_record_total():
    from pipeline.news_agent import slim_candidate

    cand = _cand(1, "rain", 15.5, 108.5, rank_since_1940=1, rank_available=True, clear_record=False,
                 previous_record_value=190.0, previous_record_year=1971, units="mm")
    hist = slim_candidate(cand)["history"]
    assert hist["previous_record_year"] == 1971 and "previous_record_value" not in hist


def test_hourly_freshness_check_only_runs_when_a_new_forecast_is_out():
    from pipeline.check_fresh import bulletin_due

    now = datetime(2026, 11, 28, 14, 25, tzinfo=timezone.utc)
    run06, run12 = datetime(2026, 11, 28, 6, tzinfo=timezone.utc), datetime(2026, 11, 28, 12, tzinfo=timezone.utc)
    fresh = {"stories": [{}], "generator": "anthropic:x", "generated_utc": "2026-11-28T13:30:00Z",
             "stories_model_run_utc": "2026-11-28T06:00:00Z"}
    assert bulletin_due(fresh, run06, now)[0] is False  # nothing new yet: skip, no cost
    assert bulletin_due(fresh, run12, now)[0] is True  # new ECMWF run: write a new bulletin
    assert bulletin_due(dict(fresh, generated_utc="2026-11-28T05:00:00Z"), run06, now)[0] is True  # too old
    assert bulletin_due(None, run06, now)[0] is True
    assert bulletin_due(dict(fresh, generator="era5-deterministic"), run06, now)[0] is True


def test_surface_fields_never_pick_a_pressure_level():
    """Live bug: the index lists `z` on pressure levels and at the surface, in varying order. Taking the
    first `z` returned the 300 hPa height (~9 km) as terrain and turned the whole anomaly map red."""
    from pipeline.opendata import select_index_row

    rows = [
        {"param": "z", "levtype": "pl", "levelist": "300", "_offset": 1},
        {"param": "z", "levtype": "pl", "levelist": "500", "_offset": 2},
        {"param": "z", "levtype": "sfc", "_offset": 3},
    ]
    assert select_index_row(rows, "z")["_offset"] == 3
    assert select_index_row(rows, "z", levelist=500)["_offset"] == 2
    assert select_index_row(rows[:2], "z") is None


def test_implausible_elevation_field_is_never_used_for_the_lapse_rate():
    from pipeline.climatology import orography_is_plausible

    era5 = np.zeros((10, 10), dtype=np.float32)
    era5[2:4, 2:4] = 3000.0
    good = era5 + np.random.default_rng(0).normal(0, 80, era5.shape).astype(np.float32)
    assert orography_is_plausible(good, era5)
    assert not orography_is_plausible(era5 + 9100.0, era5)  # a 300 hPa height field


def test_ranking_years_must_come_from_the_data():
    from pipeline.validator import validate_history_years

    cand = {"era5": {"previous_record_year": 1978, "last_exceeded_year": 1976,
                     "top_years": [{"year": 1950, "value": 38.4}, {"year": 1976, "value": 38.0}]}}
    ok = "The windiest late-November day here since 1976; only 1950 has been windier. Records go back to 1940."
    assert validate_history_years("s", ok, cand, "") == []
    errs = validate_history_years("s", "The strongest since 1962.", cand, "")
    assert errs and "1962" in errs[0] and "1976" in errs[0]
    # A year quoted from a cited news fact is fine
    assert validate_history_years("s", "The worst floods since 1999.", cand, "Worst floods since 1999, officials say") == []


def test_nearby_cities_for_map_labels():
    from pipeline.places import attach_nearby_cities, nearby_cities

    names = [c["name"] for c in nearby_cities([(39.3, 22.9)])]  # Volos, Greece
    assert names[0] == "Volos"  # the nearest sizeable town always comes first
    assert "Athens" in names and len(names) <= 6
    b = attach_nearby_cities({"stories": [{"centre": {"lat": 39.3, "lon": 22.9}, "cues": [{"lat": 38.0, "lon": 23.7}]}]})
    assert b["stories"][0]["nearby_cities"][0]["name"] == "Volos"
    assert nearby_cities([(0.0, -140.0)]) == []  # open ocean: nothing to label
