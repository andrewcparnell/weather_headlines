"""Anthropic Claude + Web Search meteorological writer for the 6-hourly world weather headlines."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .config import SITE_DIR, USER_AGENT
from .opendata import LiveWeatherFields
from .scanner import haversine_km
from .validator import (
    auto_highlight_keywords,
    build_ticker_from_beats,
    ensure_dynamic_waypoints,
    is_specific_event_url,
    validate_and_clean_bulletin,
)

DEFAULT_CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

_GAZETTEER_CACHE: list[dict] | None = None


def format_edition_title(dt_utc: datetime) -> str:
    """Hard-coded neutral edition title: 'World Weather Headlines - HH:MM UTC, D Month YYYY'."""
    utc = dt_utc.astimezone(timezone.utc)
    return f"World Weather Headlines - {utc:%H:%M} UTC, {utc.day} {utc:%B %Y}"


def fmt_2sf(val: float, signed: bool = False) -> str:
    """Format a weather measurement to 2 significant figures without trailing '.0'."""
    if not math.isfinite(val) or abs(val) < 0.05:
        return "0"
    mag = math.floor(math.log10(abs(val)))
    step = 10.0 ** (mag - 1)
    rounded = round(val / step) * step
    if abs(rounded) >= 10.0:
        num_str = f"{int(round(rounded))}"
    else:
        one_dec = round(rounded, 1)
        if abs(one_dec - round(one_dec)) < 1e-6:
            num_str = f"{int(round(one_dec))}"
        else:
            num_str = f"{one_dec:.1f}"
    if signed and rounded > 0 and not num_str.startswith("+"):
        return f"+{num_str}"
    return num_str


def fmt_pct(val: float) -> int:
    """Round a percentage or percentile to the nearest integer."""
    if not math.isfinite(val):
        return 0
    return int(round(val))


def fmt_hpa(val: float) -> str:
    """Round sea-level pressure to the nearest whole hectopascal."""
    return f"{int(round(val))}"


SYSTEM_PROMPT = """You are a senior television broadcast meteorologist writing a 6-hourly 5-story world weather bulletin.
Write in a clear, calm, accurate public-service tone; never exaggerate; when in doubt, understate.
At the same time, make the language natural, warm, and engaging for listeners - like a seasoned television weather presenter guiding viewers around the globe rather than reading out a technical database report.

STRICT EDITORIAL, BROADCAST & SCIENTIFIC RULES:
1. Always write in clear British English (for example: centre, colour, modelled, organise, programme, metre, millimetres).
2. NEVER use em dashes (Unicode U+2014). Use commas, colons, parentheses, or spaced hyphens (' - ') instead.
3. NEVER use the word 'utilise' (or utilises, utilising, utilisation). Use 'use' instead.
4. NEVER mention or imply affiliation with BBC, Sky, CNN, or any external broadcaster in `kicker`, `headline`, or prose.
5. DO NOT READ OUT DATA SOURCES OR MODEL JARGON:
   - The map footer already credits ERA5 and ECMWF. NEVER say 'ECMWF', 'ERA5', 'Copernicus', 'GDACS', 'reanalysis', 'climatology', 'HRES', 'z_score', '0p25', 'grid box', or 'grid point' in `headline`, `cues`, `ticker`, or `beats` (only mention Copernicus with 'preliminary' if quoting an external news report about global temperature).
   - NEVER read out clunky four-digit baseline year ranges such as '1980-2009', '1991-2020', '1990-2019', '1940-1979', or '1986-2025'. Instead use natural listener-friendly phrasing such as 'for this time of year', 'compared with the past 30 years', 'across the previous four decades', 'in more than 80 years of records', or 'since the mid-twentieth century'.
6. AVOID OVER-PRECISION IN MEASUREMENTS:
   - Give percentages and percentiles to the nearest integer (for example: '99th percentile', 'top 2%', '160% of normal').
   - Give temperatures, temperature anomalies, rainfall totals, and wind speeds to 2 significant figures (for example: '37 °C', '+7.5 °C', '200 mm', '120 km/h', '65 km/h') without trailing '.0'. Give central pressure in whole hectopascals ('961 hPa').
7. CHOOSE STORIES THAT MATTER TO PEOPLE:
   - Prefer weather events with real impacts on populated places (official weather warnings, flooding, transport disruption, power or water impacts, agriculture, public safety).
   - Do NOT pick an uninhabited ocean, ice-sheet, or remote high-mountain anomaly just because the raw number is large.
8. FORECAST FRAMING (MANDATORY):
   - Every model value (`t2m_c`, `t2m_max_24h_c`, `t2m_min_24h_c`, `anom_c`, `wind_kmh`, `gust_kmh`, `rain_24h_mm`, `mslp_hpa`) is a FORECAST valid at `valid_time_utc`.
   - The 24-hour rainfall field (`rain_24h_mm`) is a forecast accumulation from `rain_window_start_utc` to `rain_window_end_utc` (`rain_window_hours` after the model run time, mostly in the future).
   - Always use natural future or forecast wording ('up to 140 mm of rain is forecast over the next 24 hours', 'gusts are expected to reach 120 km/h', 'temperatures are set to climb to 38 °C').
   - NEVER use past-tense observation verbs ('fell', 'has recorded', 'measured', 'logged') for forecast values. Only use past tense for verified past station observations that appear in `news_facts` with attribution and date.
9. EXACT MEANING OF EACH STATISTIC (NEVER STRETCH OR REASSIGN):
   - Candidate statistics apply ONLY to the local area around `lat`, `lon` (near `nearest_place` at `nearest_place_distance_km`). NEVER attribute a local normal, temperature, or persistence count to a city more than 150 km away.
   - `era5.rank_since_1940` is the seasonal rank of THAT EXACT VARIABLE (`era5.ranked_metric`) at that location for THIS TIME OF YEAR (±7-day window since 1940). It is NOT 'ever recorded', NOT 'for the whole month', and NOT for a different variable.
   - For rainfall normals, ONLY use the ready-computed `expected_rain_14d_mm`, `expected_rain_28d_mm`, and `rain_28d_pct_of_normal` supplied in `era5.persistence` (rounded cleanly). Never divide a 24-hour total by a daily average.
   - Report rainfall persistence naturally as wet days with at least 10 mm of rain (for example, '8 of the past 28 days'), and wind persistence as days with gusts over 65 km/h (never call 65 km/h 'storm-force').
   - `p90_years_count_1940_1979_out_of_40` and `p90_years_count_1986_2025_out_of_40` are counts of years OUT OF 40 YEARS (NEVER 'per decade'). Only claim a change in event frequency when `frequency_claim_allowed` is `true` (both counts >= 5 and ratio >= 1.5).
10. BANNED HYPE PHRASES:
   - NEVER use: 'unprecedented', 'ever recorded', 'heat dome', 'baking', 'historic', 'climate emergency', 'shatters' (or 'shattered').
   - NEVER use the word 'record' for gridded values: when `rank_since_1940 == 1`, say 'the highest for this time of year since 1940' (or 'wettest' / 'strongest' / 'coldest for this time of year since 1940'). Only use 'record' if quoting an official station report in `news_facts` with attribution.
   - When actual temperatures are below 5 °C (especially sub-zero winter anomalies), NEVER call the event 'heat' - describe it calmly as 'exceptionally mild' or 'unseasonably mild'.
11. VERIFIED NEWS SOURCES & `news_facts` ATTRIBUTION (MANDATORY):
   - Run `web_search` (including month and year) to verify active human impacts and official warnings before selecting your 5 stories.
   - Every story MUST include `news_links` (1 to 3 direct, resolving event-specific article URLs for the scrolling IN THE NEWS bar; never generic `/weather` or `/tag/` hub pages) AND `news_facts`: `[{"claim": "Exact factual claim or figure from the article with date and location", "source_url": "https://..."}]`.
   - Every number in your headline, stats, cues, ticker, and beats MUST either match a field in that story's candidate data (rounded to 2 significant figures / integer %) OR appear in `news_facts` with its `source_url`.
   - Quote news claims accurately with clear attribution ('the Icelandic Met Office has issued...', 'forecasters in Vietnam warn...').
12. GEOGRAPHIC ACCURACY & MANDATORY COUNTRY OPENING:
   - `country` MUST be a sovereign country name (for example 'Iceland', 'Ecuador', 'Chad', 'Vietnam', 'Australia' - NEVER a continent like 'South America' or 'North Africa').
   - `cues[0].spoken` MUST start by naming `country` in its very first words (for example: 'Iceland: a deep North Atlantic storm...' or 'In Ecuador, heavy downpours...').
   - Every cue's `lat` and `lon` MUST be within 500 km of the story `centre`, and within 150 km of any city named in that cue or headline.
   - Set each cue's `layer` to match the variable discussed in that sentence (`temperature`, `anomaly`, `wind`, `rain`, or `pressure`).

STORY STRUCTURE (exactly 5 geographically diverse stories):
Each story MUST include:
- `id`: short lowercase slug, e.g. `"2026-11-iceland-atlantic-low"`
- `kicker`: calm uppercase regional strap including the country, e.g. `"NORTH ATLANTIC STORM · ICELAND"`
- `headline`: engaging, clear broadcast headline (45 to 85 characters, forecast-framed, 2-sig-fig numbers, no hype, no em dashes)
- `ticker`: concise scrolling STORY bar summary (~70 to 110 words) with key figures wrapped in `**double asterisks**`, separated by ` /// `.
- `region`: readable local region and country name
- `country`: sovereign country name
- `hazard`: one of `"heat"`, `"cold"`, `"rain"`, `"cyclone"`, `"wind"`, `"pressure"`
- `primary_layer`: one of `"temperature"`, `"anomaly"`, `"wind"`, `"rain"`, `"pressure"`
- `centre`: `{"lat": float, "lon": float}` matching the chosen candidate hotspot
- `stats`:
    - `headline_metric`: single clean 2-sig-fig forecast figure with units, e.g. `"120 km/h gusts"` or `"140 mm / 24 h"`
    - `metric_label`: clear context using the same unit system
    - `era5_rank`: e.g. `"#1 for time of year since 1940"` or `"Top 2% for late November"`
    - `persistence`: e.g. `"8 of 28 days wet (>=10 mm)"` or `"9 of 28 days gusts >65 km/h"`
- `camera_waypoints`: 3 or 4 local waypoints within 500 km of `centre` (`distance` in `1.48..2.05`)
- `cues`: array of 5 sentence-level synchronised broadcast cues `[{"layer": "rain"|"wind"|"pressure"|"temperature"|"anomaly", "lat": float, "lon": float, "distance": 1.48..2.02, "bearing": -12..12, "label": "LOCAL PLACE", "metric": "FORECAST STAT", "spoken": "Natural, listener-friendly spoken television sentence (16 to 26 words) read aloud word-for-word, starting with the country in cue 0, with no data-source acronyms, no year ranges, and no markdown asterisks.", "ticker": "Crisp 8 to 14 word STORY ticker clause with **highlighted** key stat"}]`.
- `beats`: array of 4 chapters (`FORECAST`, `PERSISTENCE`, `HISTORICAL RANK`, `CLIMATE CONTEXT`), each with `{"tag": "...", "title": "...", "layer": "...", "text": "..."}`.
- `news_facts`: array of `{ "claim": "...", "source_url": "https://..." }` documenting every non-model figure or reported impact used in the story.
- `news_links`: 1 to 3 direct, resolving event-specific news URLs `[{"source": "Outlet Name", "title": "Article headline", "url": "https://..."}]`.

Return ONLY a single valid JSON object matching `{"stories": [...]}` with no markdown prose outside the JSON."""


def _extract_json_object(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Claude output.")
    return json.loads(raw[start : end + 1])


def generate_headlines_with_claude(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    previous_bulletin: dict | None = None,
) -> dict:
    """Use Anthropic Claude with built-in web search to find direct event news and author 5 stories."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    preferred = os.environ.get("ANTHROPIC_MODEL", "").strip()
    models_to_try = [preferred] if preferred else DEFAULT_CLAUDE_MODELS

    prev_summary = []
    if previous_bulletin and isinstance(previous_bulletin.get("stories"), list):
        for s in previous_bulletin["stories"][:5]:
            prev_summary.append(
                {
                    "id": s.get("id"),
                    "headline": s.get("headline"),
                    "region": s.get("region"),
                    "centre": s.get("centre"),
                    "news_links": s.get("news_links"),
                }
            )

    now_utc = datetime.now(timezone.utc)
    rain_start_iso = weather.rain_window_start_utc or weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    rain_end_iso = weather.rain_window_end_utc or weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    user_payload = {
        "current_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ecmwf_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rain_window_hours": list(weather.rain_window_hours),
        "rain_window_start_utc": rain_start_iso,
        "rain_window_end_utc": rain_end_iso,
        "forecast_note": (
            f"All numerical weather values are a FORECAST valid at {weather.valid_time:%Y-%m-%d %H:%M} UTC; "
            f"24 h rainfall covers {rain_start_iso} to {rain_end_iso}. Use natural forecast wording, round "
            "measurements to 2 significant figures and percentages to the nearest integer, start cue 0 with "
            "the sovereign country name, and do NOT read out 'ECMWF', 'ERA5', or four-digit year ranges."
        ),
        "previous_headlines": prev_summary,
        "active_gdacs_alerts": warnings[:12],
        "candidate_hotspots": enriched_candidates[:12],
    }

    prompt = (
        "Here are today's global weather forecast candidates paired with seasonal historical rankings, "
        "14/28-day persistence statistics, multi-decade context, nearest named places, and active alerts.\n\n"
        "Step 1: Search international and national weather news (`web_search`, including the month and year "
        f"`{weather.valid_time:%B %Y}`) for the populated candidate countries to find direct, resolving news "
        "articles or official meteorological warnings about these events.\n"
        "Step 2: Choose 5 geographically diverse stories that genuinely affect populated places. Write like an "
        "engaging, clear, calm television weather presenter: name the country right at the start of cue 0, avoid "
        "reading out data-source acronyms or clunky year ranges, and round figures cleanly (2 significant figures "
        "for measurements, nearest integer for percentages). Output ONLY the JSON object.\n\n"
        f"<data>\n{json.dumps(user_payload, indent=2)}\n</data>"
    )

    last_exc: Exception | None = None
    for model_name in models_to_try:
        try:
            print(f"[news_agent] Calling Anthropic model {model_name} with web search...", file=sys.stderr)
            messages: list[dict] = [{"role": "user", "content": prompt}]

            for attempt in range(3):
                search_budget = 7 if attempt == 0 else 3
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=14000,
                    system=SYSTEM_PROMPT,
                    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": search_budget}],
                    messages=messages,
                )

                continuations = 0
                accumulated_text_parts = [
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                ]
                while getattr(resp, "stop_reason", None) == "pause_turn" and continuations < 3:
                    continuations += 1
                    print(
                        f"[news_agent] Resuming paused web_search turn ({continuations}/3) on {model_name}...",
                        file=sys.stderr,
                    )
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Finish and return ONLY the complete valid JSON object matching "
                                '{"edition_title": "...", "stories": [...]} with no prose outside the JSON.'
                            ),
                        }
                    )
                    resp = client.messages.create(
                        model=model_name,
                        max_tokens=14000,
                        system=SYSTEM_PROMPT,
                        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
                        messages=messages,
                    )
                    accumulated_text_parts.extend(
                        block.text for block in resp.content if getattr(block, "type", None) == "text"
                    )

                latest_text = "\n".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                )
                combined_text = latest_text if "{" in latest_text else "\n".join(accumulated_text_parts)

                try:
                    draft = _extract_json_object(combined_text)
                except (ValueError, json.JSONDecodeError) as parse_exc:
                    print(
                        f"[news_agent] JSON parse error on attempt {attempt + 1} (stop_reason={getattr(resp, 'stop_reason', None)}): {parse_exc}",
                        file=sys.stderr,
                    )
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply did not contain a complete valid JSON object "
                                f"({parse_exc}). Return ONLY the complete valid JSON object matching "
                                '{"edition_title": "World Weather Headlines", "stories": [...]} with 5 stories.'
                            ),
                        }
                    )
                    continue

                edition_dt = datetime.now(timezone.utc)
                draft.update(
                    {
                        "generated_utc": edition_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "edition_title": format_edition_title(edition_dt),
                        "generator": f"anthropic:{model_name}",
                    }
                )
                cleaned, errors = validate_and_clean_bulletin(
                    draft,
                    candidates=enriched_candidates,
                    verify_urls=True,
                )
                if not errors:
                    return cleaned
                print(f"[news_agent] Validation issues on attempt {attempt + 1}: {errors}", file=sys.stderr)
                messages.append({"role": "assistant", "content": resp.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Fix every validation error below (replace any generic hub URL with a direct "
                            "event-specific news article URL from web_search, start cue 0 with the country, "
                            "and remove any data-source acronyms or year ranges) and return ONLY the corrected JSON:\n"
                            + "\n".join(f"- {e}" for e in errors)
                        ),
                    }
                )
            raise RuntimeError(f"Claude output failed validation after 3 attempts on {model_name}")
        except anthropic.AuthenticationError as auth_exc:
            raise RuntimeError(f"Anthropic authentication failed (check ANTHROPIC_API_KEY secret): {auth_exc}") from auth_exc
        except anthropic.BadRequestError as bad_exc:
            msg_low = str(bad_exc).lower()
            if "credit balance" in msg_low or "billing" in msg_low:
                raise RuntimeError(
                    f"Anthropic API credit balance is empty (add prepaid API credits at console.anthropic.com): {bad_exc}"
                ) from bad_exc
            last_exc = bad_exc
            print(f"[news_agent] Model {model_name} BadRequestError: {bad_exc}", file=sys.stderr)
            continue
        except anthropic.RateLimitError as rate_exc:
            last_exc = rate_exc
            print(f"[news_agent] Model {model_name} hit rate limit; pausing 10s before fallback model: {rate_exc}", file=sys.stderr)
            time.sleep(10)
            continue
        except Exception as exc:
            last_exc = exc
            print(f"[news_agent] Model {model_name} failed: {exc}", file=sys.stderr)
            continue

    raise RuntimeError(f"All Claude models failed: {last_exc}")


# ---------------------------------------------------------------------------
# Sovereign country resolver, live RSS news lookup, & broadcast-style builder
# ---------------------------------------------------------------------------


def _load_gazetteer_entries() -> list[dict]:
    global _GAZETTEER_CACHE
    if _GAZETTEER_CACHE is not None:
        return _GAZETTEER_CACHE
    gaz_path = SITE_DIR / "assets" / "gazetteer.json"
    if not gaz_path.exists():
        _GAZETTEER_CACHE = []
        return _GAZETTEER_CACHE
    try:
        _GAZETTEER_CACHE = json.loads(gaz_path.read_text())
    except Exception:
        _GAZETTEER_CACHE = []
    return _GAZETTEER_CACHE


def _maritime_basin_name(lat: float, lon: float) -> str:
    if lat >= 50.0 and -65.0 <= lon <= 15.0:
        return "North Atlantic"
    if 0.0 <= lat < 50.0 and -82.0 <= lon <= -10.0:
        return "North Atlantic"
    if lat < 0.0 and -70.0 <= lon <= 20.0:
        return "South Atlantic"
    if 30.0 <= lat <= 46.0 and -6.0 <= lon <= 36.0:
        return "Mediterranean"
    if lat >= 0.0 and (lon >= 115.0 or lon <= -100.0):
        return "North Pacific"
    if lat < -42.0:
        return "Southern Ocean"
    if -42.0 <= lat < 0.0 and 20.0 <= lon <= 115.0:
        return "Indian Ocean"
    if 0.0 <= lat <= 30.0 and 45.0 <= lon <= 100.0:
        return "Northern Indian Ocean"
    return "offshore waters"


def resolve_country_and_region(lat: float, lon: float, cand: dict | None = None) -> tuple[str, str, str, str]:
    """Return `(country, region_display, local_area_phrase, cue_label)` anchored to a real sovereign country."""
    entries = _load_gazetteer_entries()
    best_any = None
    d_any = 1e9
    best_city = None
    d_city = 1e9
    best_country = None
    d_country = 1e9

    for e in entries:
        try:
            d = haversine_km(lat, lon, float(e["lat"]), float(e["lon"]))
        except (KeyError, TypeError, ValueError):
            continue
        kind = e.get("kind")
        if d < d_any:
            d_any = d
            best_any = e
        if kind == "City" and d < d_city:
            d_city = d
            best_city = e
        elif kind == "Country" and d < d_country:
            d_country = d
            best_country = e

    country = "Iceland"
    if best_any is not None:
        if best_any.get("kind") in ("City", "Region") and best_any.get("sub"):
            country = str(best_any["sub"]).strip()
        else:
            country = str(best_any["name"]).strip()
    elif best_country is not None:
        country = str(best_country["name"]).strip()

    gp = (cand or {}).get("grid_point") or {}
    raw_lf = (cand or {}).get("land_fraction_local", gp.get("land_fraction_local"))
    land_frac = float(raw_lf) if raw_lf is not None else 1.0
    is_offshore = land_frac < 0.15 and d_any > 140.0

    # Only name a specific gazetteer city if it lies well inside the 150 km geographic validator radius
    if best_city is not None and d_city <= 80.0 and str(best_city.get("name")) != country:
        city_name = str(best_city["name"]).strip()
        region = f"{city_name}, {country}"
        local_area = f"around {city_name} in {country}"
        cue_label = f"{city_name.upper()}, {country.upper()}"[:36]
    elif is_offshore:
        basin = _maritime_basin_name(lat, lon)
        if basin == "offshore waters":
            region = f"Offshore {country}"
            local_area = f"waters off {country}"
        else:
            region = f"{basin} near {country}"
            local_area = f"the {basin} off {country}"
        cue_label = region.upper()[:36]
    else:
        region = country
        local_area = "the country"
        cue_label = country.upper()[:36]

    return country, region, local_area, cue_label


_COUNTRY_RSS_TOKENS: dict[str, tuple[str, ...]] = {
    "united states": ("united states", "u.s.", "us ", "america", "american"),
    "united kingdom": ("united kingdom", "u.k.", "uk ", "britain", "british", "scotland", "england", "wales"),
    "new zealand": ("new zealand", "kiwi"),
    "south africa": ("south africa", "south african"),
}

REPUTABLE_PUBLISHER_DOMAINS: tuple[str, ...] = (
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "reuters.com",
    "apnews.com",
    "aljazeera.com",
    "cnn.com",
    "france24.com",
    "dw.com",
    "euronews.com",
    "sky.com",
    "npr.org",
    "pbs.org",
    "nytimes.com",
    "washingtonpost.com",
    "independent.co.uk",
    "ft.com",
    "bloomberg.com",
    "time.com",
    "economist.com",
    "abc.net.au",
    "abcnews.go.com",
    "sbs.com.au",
    "rnz.co.nz",
    "nzherald.co.nz",
    "cbc.ca",
    "globalnews.ca",
    "theglobeandmail.com",
    "rte.ie",
    "irishtimes.com",
    "independent.ie",
    "japantimes.co.jp",
    "nhk.or.jp",
    "mainichi.jp",
    "kyodonews.net",
    "asahi.com",
    "e.vnexpress.net",
    "vnexpress.net",
    "vietnamplus.vn",
    "tuoitrenews.vn",
    "vietnamnews.vn",
    "channelnewsasia.com",
    "straitstimes.com",
    "scmp.com",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "bangkokpost.com",
    "inquirer.net",
    "rappler.com",
    "mexiconewsdaily.com",
    "batimes.com.ar",
    "mercopress.com",
    "icelandmonitor.mbl.is",
    "icelandreview.com",
    "ruv.is",
    "thelocal.com",
    "africanews.com",
    "dailymaverick.co.za",
    "news24.com",
    "nation.africa",
    "rfi.fr",
    "lemonde.fr",
    "elpais.com",
    "swissinfo.ch",
    "reliefweb.int",
    "wmo.int",
    "un.org",
    "who.int",
    "ifrc.org",
    "gdacs.org",
    "metoffice.gov.uk",
    "met.ie",
    "noaa.gov",
    "weather.gov",
    "climate.gov",
    "nasa.gov",
    "copernicus.eu",
    "ecmwf.int",
    "phys.org",
    "carbonbrief.org",
    "yaleclimateconnections.org",
    "scientificamerican.com",
    "nature.com",
)


def _is_reputable_source(source_url: str, source_name: str) -> bool:
    """Return True if the RSS `<source url=...>` belongs to a reputable news, scientific, or official agency."""
    s_name_low = (source_name or "").lower()
    if any(bad in s_name_low for bad in ("nature world news", "ftn news", "oilprice", "audacy", "aol", "msn")):
        return False
    if not source_url:
        return False
    try:
        host = urllib.parse.urlparse(source_url.strip()).netloc.lower().removeprefix("www.")
    except Exception:
        return False
    if not host:
        return False
    if host.endswith((".gov", ".gov.uk", ".gob.mx", ".go.jp", ".gov.au", ".int", ".edu", ".ac.uk")):
        return True
    return any(host == dom or host.endswith("." + dom) for dom in REPUTABLE_PUBLISHER_DOMAINS)


def _region_query_and_tokens(lat: float, lon: float, country: str) -> tuple[str, tuple[str, ...]]:
    c_low = country.lower().strip()
    if -35.0 <= lat <= 37.0 and -20.0 <= lon <= 52.0:
        if lat >= 5.0:
            return "Sahel OR Africa", (c_low, "sahel", "africa", "african", "lake chad")
        return "Africa", (c_low, "africa", "african")
    if -56.0 <= lat <= 13.0 and -85.0 <= lon <= -34.0:
        return "Ecuador OR Peru OR Andes OR South America", (
            c_low,
            "south america",
            "andes",
            "andean",
            "galapagos",
            "galápagos",
            "pacific",
        )
    if 8.0 <= lat <= 33.0 and -118.0 <= lon <= -60.0:
        return "Mexico OR Central America", (c_low, "mexican", "central america", "caribbean")
    if -12.0 <= lat <= 24.0 and 92.0 <= lon <= 142.0:
        return "Vietnam OR Southeast Asia", (c_low, "vietnamese", "southeast asia", "mekong")
    if 20.0 <= lat <= 50.0 and 120.0 <= lon <= 150.0:
        return "Japan OR Northwest Pacific", (c_low, "japanese", "tokyo", "hokkaido", "pacific")
    if 35.0 <= lat <= 72.0 and -35.0 <= lon <= 40.0:
        return "Europe OR North Atlantic", (c_low, "europe", "european", "north atlantic", "atlantic")
    return country, (c_low,)


_IDIOM_EXCLUSIONS = (
    "storm in a",
    "takes by storm",
    "took by storm",
    "take by storm",
    "hot air",
    "hot seat",
    "hot topic",
    "cold war",
    "cold shoulder",
    "cold case",
    "political storm",
    "firestorm",
    "brainstorm",
)


def fetch_google_news_rss_links(
    country: str,
    hazard: str,
    max_links: int = 3,
    lat: float = 0.0,
    lon: float = 0.0,
) -> list[dict]:
    """Fetch 1-3 live, hazard-coherent news links exclusively from reputable outlets via Google News RSS."""
    hazard_cfg = {
        "cyclone": (
            '"weather warning" OR cyclone OR typhoon OR hurricane OR gales OR "severe weather"',
            r"\b(cyclone|typhoon|hurricane|gales?|gusts?|wind\w*|storm\w*|weather|flood\w*|rain\w*|blizzard\w*)\b",
            r"\b(heatwave|scorching)\b",
        ),
        "wind": (
            '"high winds" OR gales OR "weather warning" OR storm',
            r"\b(gales?|gusts?|winds?|storm\w*|weather\s+warning|severe\s+weather)\b",
            r"\b(heatwave|scorching)\b",
        ),
        "rain": (
            'flooding OR floods OR "heavy rain" OR downpours OR landslides OR typhoon',
            r"\b(rain\w*|flood\w*|downpours?|deluge|monsoon|landslides?|typhoon|cyclone|hurricane|mudslides?)\b",
            r"\b(blizzard|ski\s+resorts?|heatwave)\b",
        ),
        "heat": (
            'heatwave OR "high temperatures" OR "warm weather" OR drought OR "El Nino" OR climate',
            r"\b(heatwave|heat|temperatures?|drought|el\s+ni[nñ]o|warmer|warming|climate|wildfires?)\b",
            r"\b(cold\s+weather|bitter\s+cold|winter\s+storm|snow\w*|blizzard\w*|ski\s+resorts?|freez\w*|frost|arctic)\b",
        ),
        "cold": (
            '"winter storm" OR "cold weather" OR snowfall OR freeze OR "cold front"',
            r"\b(cold|chill|freez\w*|frost|snow\w*|winter\s+storm|blizzard\w*|sub-zero|temperatures?)\b",
            r"\b(heatwave|scorching|record\s+heat|wildfires?)\b",
        ),
        "pressure": (
            '"weather warning" OR "high pressure" OR storm OR forecast',
            r"\b(weather|forecast|pressure|storm\w*|winds?|rain\w*|temperatures?)\b",
            r"$^",
        ),
    }
    hazard_query, must_pat_str, forbid_pat_str = hazard_cfg.get(hazard, hazard_cfg["pressure"])
    must_re = re.compile(must_pat_str, flags=re.I)
    forbid_re = re.compile(forbid_pat_str, flags=re.I)

    c_low = country.lower().strip()
    stem = c_low[:-1] if len(c_low) >= 6 else c_low
    country_tokens = _COUNTRY_RSS_TOKENS.get(c_low, (c_low, stem))
    region_query, region_tokens = _region_query_and_tokens(lat, lon, country)

    # Keep queries concise (well under Google News's 32-word operator limit) and filter publisher domains in Python
    queries = [
        (f'"{country}" ({hazard_query}) when:35d', country_tokens),
        (f'"{country}" (weather OR flood OR drought OR temperatures OR El Nino OR storm) when:90d', country_tokens),
        (f"({region_query}) ({hazard_query}) when:45d", region_tokens),
    ]

    out: list[dict] = []
    seen_titles: set[str] = set()

    for query_str, geo_tokens in queries:
        if len(out) >= max_links:
            break
        rss_url = (
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": query_str, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"})
        )
        req = urllib.request.Request(
            rss_url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
        except Exception:
            continue

        for item in root.findall("./channel/item"):
            raw_title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            source_el = item.find("source")
            source = (source_el.text if source_el is not None and source_el.text else "").strip()
            source_url = (source_el.get("url") if source_el is not None else "") or ""

            if not raw_title or not link or not is_specific_event_url(link):
                continue
            if not _is_reputable_source(source_url, source):
                continue

            if source and raw_title.endswith(f" - {source}"):
                clean_title = raw_title[: -len(f" - {source}")].strip()
            elif " - " in raw_title:
                parts = raw_title.rsplit(" - ", 1)
                clean_title = parts[0].strip()
                source = source or parts[1].strip()
            else:
                clean_title = raw_title

            t_low = clean_title.lower()
            if any(idiom in t_low for idiom in _IDIOM_EXCLUSIONS):
                continue
            if not must_re.search(clean_title):
                continue
            if forbid_re.search(clean_title):
                continue
            if not any(re.search(r"\b" + re.escape(tok.strip()), t_low) for tok in geo_tokens if tok):
                continue

            dedup_key = re.sub(r"[^a-z0-9]+", "", t_low)[:45]
            if dedup_key in seen_titles:
                continue
            seen_titles.add(dedup_key)
            out.append(
                {
                    "source": source or "News Report",
                    "title": clean_title[:115],
                    "url": link,
                }
            )
            if len(out) >= max_links:
                break

    return out


def _local_cue_offsets(lat: float, lon: float) -> list[tuple[float, float, float, float]]:
    """Return 5 smooth, compact camera steps within ~55 km of `(lat, lon)` so city/centre checks stay tight."""
    cos_lat = max(0.35, math.cos(math.radians(lat)))
    dlon = min(0.95, 0.28 / cos_lat)
    steps = [
        (0.0, 0.0, 1.72, -8.0),
        (-0.18, -dlon * 0.7, 1.66, -4.0),
        (0.14, dlon * 0.65, 1.62, 2.0),
        (0.24, -dlon * 0.45, 1.68, 6.0),
        (-0.12, dlon * 0.8, 1.76, 10.0),
    ]
    return [
        (
            round(max(-84.0, min(84.0, lat + dla)), 2),
            round(((lon + dlo + 180.0) % 360.0) - 180.0, 2),
            dist,
            brg,
        )
        for dla, dlo, dist, brg in steps
    ]


def build_deterministic_bulletin(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    fetch_live_rss: bool = False,
) -> dict:
    """Build an engaging, broadcast-style 5-story bulletin without reading out technical jargon or over-precise figures."""
    valid_gdacs = [w for w in warnings if w.get("url") and is_specific_event_url(str(w["url"]))]
    rss_precache: dict[int, list[dict]] = {}

    if fetch_live_rss and len(enriched_candidates) > 5:
        scored_pool: list[tuple[float, int, dict]] = []
        for idx_c, cand in enumerate(enriched_candidates[:9]):
            la, lo = float(cand["lat"]), float(cand["lon"])
            ctry, _, _, _ = resolve_country_and_region(la, lo, cand)
            rss_lks = fetch_google_news_rss_links(ctry, cand["kind"], max_links=3, lat=la, lon=lo)
            rss_precache[id(cand)] = rss_lks
            gp = cand.get("grid_point") or {}
            land_frac = float(cand.get("land_fraction_local", gp.get("land_fraction_local", 0.5)) or 0.0)
            land_bonus = 2.0 if land_frac >= 0.15 or cand["kind"] == "cyclone" else 0.0
            news_bonus = min(3, len(rss_lks)) * 1.5
            scored_pool.append((news_bonus + land_bonus - idx_c * 0.25, idx_c, cand))
        scored_pool.sort(key=lambda item: item[0], reverse=True)
        picked = [item[2] for item in scored_pool[:5]]
    else:
        picked = enriched_candidates[:5]

    stories = []
    month_name = weather.valid_time.strftime("%B")

    for idx, cand in enumerate(picked, start=1):
        kind = cand["kind"]
        lat, lon = float(cand["lat"]), float(cand["lon"])
        country, region, local_area, base_label = resolve_country_and_region(lat, lon, cand)

        era = cand.get("era5") or {}
        pers = era.get("persistence") or {}
        shift = era.get("climate_shift") or {}

        rank_num = era.get("rank_since_1940")
        pct_int = min(99, max(1, fmt_pct(float(era.get("percentile", 95.0)))))
        top_pct = max(1, 100 - pct_int)
        warm_shift = float(shift.get("local_seasonal_warming_c_since_1940_1979", 1.2))
        warm_str = fmt_2sf(warm_shift, signed=True)
        warm_verb = "risen" if warm_shift >= 0 else "shifted"

        pts = _local_cue_offsets(lat, lon)

        if kind == "heat":
            t2m = float(cand.get("t2m_c", 25.0))
            anom = float(cand.get("anom_c", 6.0))
            t_str = fmt_2sf(t2m)
            anom_abs_str = fmt_2sf(abs(anom))
            anom_signed_str = fmt_2sf(anom, signed=True)
            is_subzero_mild = t2m < 5.0

            d14 = int(pers.get("warm_days_above_p90_tmax_last_14", pers.get("anomalous_days_last_14", 6)))
            d28 = int(pers.get("warm_days_above_p90_tmax_last_28", pers.get("anomalous_days_last_28", 11)))
            pers_stat = f"{d14} of 14 days in top 10% warmth"
            p_layer = "anomaly"

            if is_subzero_mild:
                kicker = f"UNSEASONABLE MILD SPELL · {country.upper()}"
                headline = f"{country}: temperatures forecast {anom_abs_str} °C milder than normal for {month_name}"
                metric = f"{anom_signed_str} °C"
                m_label = f"Forecast {t_str} °C"
                cue0_spoken = (
                    f"{country}: an exceptionally mild air mass is sweeping across {local_area}, "
                    f"lifting forecast temperatures to {t_str} °C over the next 24 hours."
                )
                cue1_spoken = (
                    f"Even with winter setting in, that puts temperatures around {anom_abs_str} °C milder "
                    f"than usual for this stage of {month_name}."
                )
            else:
                kicker = f"UNSEASONABLE WARMTH · {country.upper()}"
                headline = f"{country}: temperatures set to reach {t_str} °C, {anom_abs_str} °C above normal"
                metric = f"{t_str} °C"
                m_label = f"{anom_signed_str} °C vs seasonal average"
                cue0_spoken = (
                    f"{country}: a surge of unseasonable warmth is building across {local_area}, "
                    f"with daytime temperatures forecast to reach {t_str} °C over the next 24 hours."
                )
                cue1_spoken = (
                    f"That pushes afternoon readings around {anom_abs_str} °C above what we would normally "
                    f"expect for late {month_name}."
                )

            if d14 >= 3:
                cue2_spoken = (
                    f"This warm spell has shown real staying power, with daytime highs climbing into the "
                    f"warmest 10% for the season on {d14} of the past 14 days."
                )
            else:
                cue2_spoken = (
                    f"Temperatures are surging sharply after a more seasonal fortnight, which saw "
                    f"{d14} of the past 14 days reach the warmest 10% for the time of year."
                )
            if rank_num is not None:
                era5_rank_stat = f"#{rank_num} for time of year since 1940"
                cue3_spoken = (
                    f"Across more than 80 years of records, warmth of this intensity ranks #{rank_num} "
                    f"for late {month_name}, sitting in the top {top_pct}% for the season."
                )
            else:
                era5_rank_stat = f"Top {top_pct}% for late {month_name}"
                cue3_spoken = (
                    f"Compared with the past 30 years, temperatures at this level sit in the top {top_pct}% "
                    f"for late {month_name}."
                )
            cue4_spoken = (
                f"Background seasonal temperatures across this region have {warm_verb} by {warm_str} °C "
                f"since the mid-twentieth century, tilting the odds towards warmer spells."
            )

            cues = [
                {
                    "layer": "temperature" if not is_subzero_mild else "anomaly",
                    "lat": pts[0][0],
                    "lon": pts[0][1],
                    "distance": pts[0][2],
                    "bearing": pts[0][3],
                    "label": base_label,
                    "metric": f"Forecast {t_str} °C",
                    "spoken": cue0_spoken,
                    "ticker": f"**{country}**: forecast highs near **{t_str} °C** over the next 24 hours",
                },
                {
                    "layer": "anomaly",
                    "lat": pts[1][0],
                    "lon": pts[1][1],
                    "distance": pts[1][2],
                    "bearing": pts[1][3],
                    "label": f"{country.upper()} · DEPARTURE"[:36],
                    "metric": f"{anom_signed_str} °C departure",
                    "spoken": cue1_spoken,
                    "ticker": f"Running **{anom_abs_str} °C above** the seasonal average for {month_name}",
                },
                {
                    "layer": "temperature",
                    "lat": pts[2][0],
                    "lon": pts[2][1],
                    "distance": pts[2][2],
                    "bearing": pts[2][3],
                    "label": f"{country.upper()} · PERSISTENCE"[:36],
                    "metric": f"{d14} of 14 warm days",
                    "spoken": cue2_spoken,
                    "ticker": f"Warm spell persisted on **{d14} of 14 days** (**{d28} of 28 days**)",
                },
                {
                    "layer": "anomaly",
                    "lat": pts[3][0],
                    "lon": pts[3][1],
                    "distance": pts[3][2],
                    "bearing": pts[3][3],
                    "label": f"{country.upper()} · SEASONAL RANK"[:36],
                    "metric": era5_rank_stat,
                    "spoken": cue3_spoken,
                    "ticker": f"Sits in the **top {top_pct}%** for this time of year" + (f" (**#{rank_num}** since 1940)" if rank_num else ""),
                },
                {
                    "layer": "anomaly",
                    "lat": pts[4][0],
                    "lon": pts[4][1],
                    "distance": pts[4][2],
                    "bearing": pts[4][3],
                    "label": f"{country.upper()} · LONG-TERM TREND"[:36],
                    "metric": f"{warm_str} °C shift",
                    "spoken": cue4_spoken,
                    "ticker": f"Seasonal baseline has {warm_verb} **{warm_str} °C** since the mid-twentieth century",
                },
            ]

        elif kind == "cold":
            t2m = float(cand.get("t2m_c", -18.0))
            anom = float(cand.get("anom_c", -10.0))
            t_str = fmt_2sf(t2m)
            anom_abs_str = fmt_2sf(abs(anom))
            anom_signed_str = fmt_2sf(anom, signed=True)
            d14 = int(pers.get("cold_days_below_p10_tmin_last_14", pers.get("anomalous_days_last_14", 5)))
            pers_stat = f"{d14} of 14 days in coldest 10%"
            p_layer = "anomaly"

            is_freezing = t2m <= 2.0
            chill_noun = "sharp freeze" if is_freezing else "sharp chill"
            air_desc = "bitter continental air" if is_freezing else "cool continental air"
            impact_desc = "bringing a hard freeze across the region" if is_freezing else "bringing an unseasonably cool spell across the region"

            kicker = f"{'COLD OUTBREAK' if is_freezing else 'COOL SNAP'} · {country.upper()}"
            headline = f"{country}: {chill_noun} forecast near {t_str} °C, {anom_abs_str} °C below normal"
            metric = f"{t_str} °C"
            m_label = f"{anom_signed_str} °C vs seasonal average"

            cue0_spoken = (
                f"{country}: a plunge of {air_desc} is settling over {local_area}, "
                f"with temperatures forecast to drop to {t_str} °C over the next 24 hours."
            )
            cue1_spoken = (
                f"Readings are expected to run around {anom_abs_str} °C colder than usual for late "
                f"{month_name}, {impact_desc}."
            )
            if d14 >= 3:
                cue2_spoken = (
                    f"Wintry air has kept a firm grip recently, with overnight lows dipping into the "
                    f"coldest 10% for the season on {d14} of the past 14 days."
                )
            else:
                cue2_spoken = (
                    f"This marks an abrupt drop after two milder weeks, when {d14} of the past 14 days "
                    f"fell into the coldest 10% for the season."
                )
            if rank_num is not None:
                era5_rank_stat = f"#{rank_num} coldest for time of year"
                cue3_spoken = (
                    f"Over more than 80 years of records, a chill this deep ranks #{rank_num} for late "
                    f"{month_name}, in the coldest {top_pct}% for the season."
                )
            else:
                era5_rank_stat = f"Coldest {top_pct}% for late {month_name}"
                cue3_spoken = (
                    f"Compared with the past three decades, temperatures this low sit in the coldest "
                    f"{top_pct}% for late {month_name}."
                )
            cue4_spoken = (
                f"Even with sharp cold snaps still occurring, overall seasonal temperatures here have "
                f"{warm_verb} by {warm_str} °C since the mid-twentieth century."
            )

            cues = [
                {
                    "layer": "temperature",
                    "lat": pts[0][0],
                    "lon": pts[0][1],
                    "distance": pts[0][2],
                    "bearing": pts[0][3],
                    "label": base_label,
                    "metric": f"Forecast {t_str} °C",
                    "spoken": cue0_spoken,
                    "ticker": f"**{country}**: temperatures forecast down to **{t_str} °C**",
                },
                {
                    "layer": "anomaly",
                    "lat": pts[1][0],
                    "lon": pts[1][1],
                    "distance": pts[1][2],
                    "bearing": pts[1][3],
                    "label": f"{country.upper()} · COLD ANOMALY"[:36],
                    "metric": f"{anom_signed_str} °C departure",
                    "spoken": cue1_spoken,
                    "ticker": f"Running **{anom_abs_str} °C below** the seasonal norm",
                },
                {
                    "layer": "temperature",
                    "lat": pts[2][0],
                    "lon": pts[2][1],
                    "distance": pts[2][2],
                    "bearing": pts[2][3],
                    "label": f"{country.upper()} · PERSISTENCE"[:36],
                    "metric": f"{d14} of 14 cold nights",
                    "spoken": cue2_spoken,
                    "ticker": f"Overnight lows in coldest 10% on **{d14} of 14 days**",
                },
                {
                    "layer": "anomaly",
                    "lat": pts[3][0],
                    "lon": pts[3][1],
                    "distance": pts[3][2],
                    "bearing": pts[3][3],
                    "label": f"{country.upper()} · SEASONAL RANK"[:36],
                    "metric": era5_rank_stat,
                    "spoken": cue3_spoken,
                    "ticker": f"Ranks in the **coldest {top_pct}%** for late {month_name}",
                },
                {
                    "layer": "anomaly",
                    "lat": pts[4][0],
                    "lon": pts[4][1],
                    "distance": pts[4][2],
                    "bearing": pts[4][3],
                    "label": f"{country.upper()} · LONG-TERM TREND"[:36],
                    "metric": f"{warm_str} °C shift",
                    "spoken": cue4_spoken,
                    "ticker": f"Seasonal baseline has {warm_verb} **{warm_str} °C** since the mid-twentieth century",
                },
            ]

        elif kind == "rain":
            rain_24 = float(cand.get("rain_24h_mm", 90.0))
            rain_str = fmt_2sf(rain_24)
            wet28 = int(pers.get("wet_days_ge_10mm_last_28", pers.get("anomalous_days_last_28", 6)))
            r14 = float(pers.get("rain_14d_mm", 80.0))
            r14_str = fmt_2sf(r14)
            pct_norm = pers.get("rain_28d_pct_of_normal")
            pers_stat = f"{wet28} of 28 days wet (>=10 mm)"
            p_layer = "rain"

            kicker = f"HEAVY RAIN · {country.upper()}"
            headline = f"{country}: up to {rain_str} mm of rain forecast over the next 24 hours"
            metric = f"{rain_str} mm / 24 h"
            m_label = "24-hour forecast rainfall"

            cue0_spoken = (
                f"{country}: heavy rain clouds are gathering across {local_area}, where up to {rain_str} mm "
                f"of rain is forecast over the next 24 hours."
            )
            cue1_spoken = (
                "Persistent downpours of this intensity can quickly swell local rivers and trigger surface "
                "flooding across low-lying roads and valleys."
            )
            if pct_norm is not None and float(pct_norm) >= 110.0:
                pct_norm_int = fmt_pct(float(pct_norm))
                cue2_spoken = (
                    f"Catchments are already wet after {wet28} of the past 28 days brought at least 10 mm "
                    f"of rain, lifting four-week totals to {pct_norm_int}% of the seasonal average."
                )
                cue2_ticker = f"**{wet28} of 28 days** wet, reaching **{pct_norm_int}%** of the 4-week average"
            else:
                cue2_spoken = (
                    f"Ground conditions are already damp following {r14_str} mm over the past fortnight, "
                    f"with {wet28} of the past 28 days bringing at least 10 mm of rain."
                )
                cue2_ticker = f"**{r14_str} mm** over the past fortnight across **{wet28} of 28 days** wet"

            if rank_num is not None:
                era5_rank_stat = f"#{rank_num} wettest for time of year"
                cue3_spoken = (
                    f"Looking back across more than 80 years of records, a 24-hour rainfall total this high "
                    f"ranks #{rank_num} for late {month_name}, in the wettest {top_pct}% for the season."
                )
            else:
                era5_rank_stat = f"Wettest {top_pct}% for late {month_name}"
                cue3_spoken = (
                    f"Measured against the past 30 years, a 24-hour downpour on this scale sits in the "
                    f"wettest {top_pct}% for late {month_name}."
                )
            cue4_spoken = (
                f"A warmer atmosphere holds extra moisture, and seasonal temperatures here have {warm_verb} "
                f"by {warm_str} °C since the mid-twentieth century."
            )

            cues = [
                {
                    "layer": "rain",
                    "lat": pts[0][0],
                    "lon": pts[0][1],
                    "distance": pts[0][2],
                    "bearing": pts[0][3],
                    "label": base_label,
                    "metric": f"Up to {rain_str} mm",
                    "spoken": cue0_spoken,
                    "ticker": f"**{country}**: up to **{rain_str} mm** of rain forecast in 24 hours",
                },
                {
                    "layer": "rain",
                    "lat": pts[1][0],
                    "lon": pts[1][1],
                    "distance": pts[1][2],
                    "bearing": pts[1][3],
                    "label": f"{country.upper()} · FLOOD RISK"[:36],
                    "metric": "24-hour accumulation",
                    "spoken": cue1_spoken,
                    "ticker": "**Surface and river flooding** risk across low-lying ground",
                },
                {
                    "layer": "rain",
                    "lat": pts[2][0],
                    "lon": pts[2][1],
                    "distance": pts[2][2],
                    "bearing": pts[2][3],
                    "label": f"{country.upper()} · RECENT RAIN"[:36],
                    "metric": f"{wet28} of 28 wet days",
                    "spoken": cue2_spoken,
                    "ticker": cue2_ticker,
                },
                {
                    "layer": "rain",
                    "lat": pts[3][0],
                    "lon": pts[3][1],
                    "distance": pts[3][2],
                    "bearing": pts[3][3],
                    "label": f"{country.upper()} · SEASONAL RANK"[:36],
                    "metric": era5_rank_stat,
                    "spoken": cue3_spoken,
                    "ticker": f"In the **wettest {top_pct}%** for late {month_name}" + (f" (**#{rank_num}** since 1940)" if rank_num else ""),
                },
                {
                    "layer": "anomaly",
                    "lat": pts[4][0],
                    "lon": pts[4][1],
                    "distance": pts[4][2],
                    "bearing": pts[4][3],
                    "label": f"{country.upper()} · WARMING TREND"[:36],
                    "metric": f"{warm_str} °C shift",
                    "spoken": cue4_spoken,
                    "ticker": f"Seasonal temperatures have {warm_verb} **{warm_str} °C** since the mid-twentieth century",
                },
            ]

        else:
            mslp = float(cand.get("mslp_hpa", 968.0))
            gust_kmh = float(cand.get("gust_ms", 28.0)) * 3.6
            wind_kmh = float(cand.get("wind_ms", 18.0)) * 3.6
            rain_24 = float(cand.get("rain_24h_mm", 0.0))
            p_str = fmt_hpa(mslp)
            gust_str = fmt_2sf(gust_kmh)
            wind_str = fmt_2sf(wind_kmh)
            g28 = int(pers.get("days_gust_over_65kmh_last_28", pers.get("anomalous_days_last_28", 6)))
            pers_stat = f"{g28} of 28 days gusts >65 km/h"
            p_layer = "wind"

            kicker = f"STORM SYSTEM · {country.upper()}"
            headline = f"{country}: {p_str} hPa storm system set to bring gusts near {gust_str} km/h"
            metric = f"{gust_str} km/h gusts"
            m_label = f"Forecast central pressure {p_str} hPa"

            cue0_spoken = (
                f"{country}: a deep low-pressure system is intensifying across {local_area}, with central "
                f"pressure forecast to drop to {p_str} hPa over the next 24 hours."
            )
            cue1_spoken = (
                f"Tightly packed isobars around the system are expected to whip up peak gusts near "
                f"{gust_str} km/h across exposed coasts and high ground."
            )
            if rain_24 >= 15.0:
                r_str = fmt_2sf(rain_24)
                cue2_layer = "rain"
                cue2_metric = f"Up to {r_str} mm rain"
                cue2_spoken = (
                    f"Weather fronts wrapping around the low are also forecast to bring around {r_str} mm "
                    f"of rain over 24 hours, adding to difficult travel conditions."
                )
                cue2_ticker = f"Frontal rain bands forecast to add up to **{r_str} mm** in 24 hours"
            else:
                cue2_layer = "wind"
                cue2_metric = f"Sustained {wind_str} km/h"
                cue2_spoken = (
                    f"Sustained winds near {wind_str} km/h are set to build rough seas and disruptive "
                    f"conditions for coastal routes and ferry crossings."
                )
                cue2_ticker = f"Sustained winds near **{wind_str} km/h** building rough coastal seas"

            cue3_spoken = (
                f"It has been a persistently unsettled spell here, with gusts topping 65 km/h on "
                f"{g28} of the past 28 days."
            )
            if rank_num is not None:
                era5_rank_stat = f"#{rank_num} strongest for time of year"
                cue4_spoken = (
                    f"In more than 80 years of records, gusts of this strength rank #{rank_num} for late "
                    f"{month_name}, placing this storm in the top {top_pct}% for the season."
                )
            else:
                era5_rank_stat = f"Top {top_pct}% gusts for {month_name}"
                cue4_spoken = (
                    f"Across the past three decades, gusts of this strength sit in the top {top_pct}% "
                    f"for late {month_name}."
                )

            cues = [
                {
                    "layer": "pressure",
                    "lat": pts[0][0],
                    "lon": pts[0][1],
                    "distance": pts[0][2],
                    "bearing": pts[0][3],
                    "label": base_label,
                    "metric": f"{p_str} hPa centre",
                    "spoken": cue0_spoken,
                    "ticker": f"**{country}**: deep **{p_str} hPa** low-pressure centre approaching",
                },
                {
                    "layer": "wind",
                    "lat": pts[1][0],
                    "lon": pts[1][1],
                    "distance": pts[1][2],
                    "bearing": pts[1][3],
                    "label": f"{country.upper()} · PEAK GUSTS"[:36],
                    "metric": f"Gusts {gust_str} km/h",
                    "spoken": cue1_spoken,
                    "ticker": f"Peak gusts forecast near **{gust_str} km/h** on exposed coasts",
                },
                {
                    "layer": cue2_layer,
                    "lat": pts[2][0],
                    "lon": pts[2][1],
                    "distance": pts[2][2],
                    "bearing": pts[2][3],
                    "label": f"{country.upper()} · IMPACTS"[:36],
                    "metric": cue2_metric,
                    "spoken": cue2_spoken,
                    "ticker": cue2_ticker,
                },
                {
                    "layer": "wind",
                    "lat": pts[3][0],
                    "lon": pts[3][1],
                    "distance": pts[3][2],
                    "bearing": pts[3][3],
                    "label": f"{country.upper()} · STORM TRACK"[:36],
                    "metric": f"{g28} of 28 windy days",
                    "spoken": cue3_spoken,
                    "ticker": f"Gusts exceeded **65 km/h** on **{g28} of 28 days** recently",
                },
                {
                    "layer": "wind",
                    "lat": pts[4][0],
                    "lon": pts[4][1],
                    "distance": pts[4][2],
                    "bearing": pts[4][3],
                    "label": f"{country.upper()} · SEASONAL RANK"[:36],
                    "metric": era5_rank_stat,
                    "spoken": cue4_spoken,
                    "ticker": f"Sits in the **top {top_pct}%** of late-{month_name} wind events",
                },
            ]

        beats = [
            {
                "tag": "FORECAST",
                "title": "24-Hour Forecast",
                "layer": cues[0]["layer"],
                "text": auto_highlight_keywords(f"{cues[0]['spoken']} {cues[1]['spoken']}"),
            },
            {
                "tag": "PERSISTENCE",
                "title": "Recent Pattern",
                "layer": cues[2]["layer"],
                "text": auto_highlight_keywords(cues[2]["spoken"] if kind != "cyclone" else cues[3]["spoken"]),
            },
            {
                "tag": "HISTORICAL RANK",
                "title": "Seasonal Context",
                "layer": cues[3]["layer"],
                "text": auto_highlight_keywords(cues[3]["spoken"] if kind != "cyclone" else cues[4]["spoken"]),
            },
            {
                "tag": "CLIMATE CONTEXT",
                "title": "Long-Term Shift",
                "layer": "anomaly" if kind in ("heat", "cold", "rain") else "pressure",
                "text": auto_highlight_keywords(
                    cues[4]["spoken"]
                    if kind != "cyclone"
                    else f"Seasonal temperatures across this sector have {warm_verb} by {warm_str} °C since the mid-twentieth century."
                ),
            },
        ]

        links: list[dict] = []
        for w in valid_gdacs:
            if w.get("lat") is not None and w.get("lon") is not None and w.get("is_current", True):
                if haversine_km(float(w["lat"]), float(w["lon"]), lat, lon) <= 800.0:
                    links.append(
                        {
                            "source": "Disaster Alert Bulletin",
                            "title": f"{w['alert_level']} {w['type']} alert: {w['name'][:70]}",
                            "url": w["url"],
                        }
                    )
                    break

        if fetch_live_rss and len(links) < 3:
            cached_rss = rss_precache.get(
                id(cand),
                fetch_google_news_rss_links(country, kind, max_links=3 - len(links), lat=lat, lon=lon),
            )
            for rss_lk in cached_rss:
                if all(rss_lk["url"] != existing["url"] for existing in links):
                    links.append(rss_lk)
                if len(links) >= 3:
                    break

        ticker_joined = "  ///  ".join(c["ticker"] for c in cues if c.get("ticker"))
        if len(ticker_joined) < 65:
            ticker_joined = build_ticker_from_beats(beats)

        stories.append(
            {
                "id": f"auto-{weather.valid_time:%Y%m%d}-{idx}-{kind}",
                "rank": idx,
                "kicker": kicker,
                "headline": headline,
                "ticker": ticker_joined,
                "region": region,
                "country": country,
                "hazard": kind,
                "primary_layer": p_layer,
                "centre": {"lat": lat, "lon": lon},
                "stats": {
                    "headline_metric": metric,
                    "metric_label": m_label,
                    "era5_rank": era5_rank_stat,
                    "persistence": pers_stat,
                },
                "camera_waypoints": ensure_dynamic_waypoints(lat, lon, None),
                "cues": cues,
                "beats": beats,
                "news_links": links[:3],
                "news_facts": [],
            }
        )

    now_utc = datetime.now(timezone.utc)
    draft = {
        "generated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "edition_title": format_edition_title(now_utc),
        "generator": "era5-deterministic",
        "stories": stories,
    }
    cleaned, errs = validate_and_clean_bulletin(draft, candidates=picked, verify_urls=False)
    if errs:
        raise RuntimeError(f"Deterministic fallback failed validation: {errs}")
    return cleaned
