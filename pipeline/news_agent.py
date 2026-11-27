"""Anthropic Claude + Web Search meteorological writer for the 6-hourly world weather headlines."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from .opendata import LiveWeatherFields
from .scanner import haversine_km
from .validator import (
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


def format_edition_title(dt_utc: datetime) -> str:
    """Hard-coded neutral edition title: 'World Weather Headlines - HH:MM UTC, D Month YYYY'."""
    utc = dt_utc.astimezone(timezone.utc)
    return f"World Weather Headlines - {utc:%H:%M} UTC, {utc.day} {utc:%B %Y}"


SYSTEM_PROMPT = """You are a senior public-service meteorologist writing a 6-hourly 5-story world weather bulletin.
Write in a clear, calm, accurate public-service tone; never exaggerate; when in doubt, understate.

STRICT EDITORIAL, STATISTICAL & SCIENTIFIC RULES:
1. Always write in clear British English (for example: centre, colour, modelled, organise, programme, metre, millimetres).
2. NEVER use em dashes (Unicode U+2014). Use commas, colons, parentheses, or spaced hyphens (' - ') instead.
3. NEVER use the word 'utilise' (or utilises, utilising, utilisation). Use 'use' instead.
4. NEVER mention or imply affiliation with BBC, Sky, CNN, or any external broadcaster in `kicker`, `headline`, or prose.
5. Avoid raw model-code tokens ('HRES', 'z_score', '0p25', 'scan candidate'), while being completely transparent that numerical weather values are forecasts.
6. CHOOSE STORIES THAT MATTER TO PEOPLE:
   - Prefer weather events with real impacts on populated places (official weather warnings, flooding, transport disruption, power or water impacts, agriculture, public safety).
   - Do NOT pick an uninhabited ocean, ice-sheet, or remote high-mountain grid-point anomaly just because the raw number is large.
7. FORECAST FRAMING (MANDATORY):
   - Every ECMWF value (`t2m_c`, `t2m_max_24h_c`, `t2m_min_24h_c`, `anom_c`, `wind_kmh`, `gust_kmh`, `rain_24h_mm`, `mslp_hpa`) is a MODEL FORECAST valid at `valid_time_utc`.
   - The 24-hour rainfall field (`rain_24h_mm`) is a forecast accumulation from `rain_window_start_utc` to `rain_window_end_utc` (`rain_window_hours` after the model run time, mostly in the future).
   - Always use future tense or explicit forecast wording for ECMWF numbers (for example: 'up to 120 mm forecast over 24 hours', 'gusts forecast to reach 115 km/h', 'temperatures expected to reach').
   - NEVER use past-tense observation verbs ('fell', 'has recorded', 'measured', 'logged') for ECMWF model forecast values. Only use past tense for verified past station observations that appear in `news_facts` with attribution and date.
8. EXACT MEANING OF EACH STATISTIC (NEVER STRETCH OR REASSIGN):
   - Candidate statistics apply ONLY to the candidate 0.5 degree grid box (`lat`, `lon`, near `nearest_place` at `nearest_place_distance_km`). NEVER attribute a grid-box normal, temperature, or persistence count to a city more than 150 km away.
   - `era5.rank_since_1940` is the rank of THAT EXACT VARIABLE (`era5.ranked_metric`) at THAT GRID BOX for THIS TIME OF YEAR (±7-day window across 1940-2025). It is NOT 'ever recorded', NOT 'for the whole month', and NOT for a different variable (never present a wind-gust rank as a central-pressure rank).
   - For rainfall normals, ONLY use the ready-computed `expected_rain_14d_mm`, `expected_rain_28d_mm`, and `rain_28d_pct_of_normal` supplied in `era5.persistence`. Never compute your own multiples of normal, and never divide a 24-hour total by a daily climatological average.
   - Report rainfall persistence as `wet days (>=10 mm)` (never 'anomalously wet days'), and wind persistence as `days with gusts over 65 km/h` or days above the local 90th-percentile gust (never call 65 km/h 'storm-force').
   - `p90_years_count_1940_1979_out_of_40` and `p90_years_count_1986_2025_out_of_40` are counts of years OUT OF 40 YEARS (NEVER 'per decade'). Only claim a change in event frequency when `frequency_claim_allowed` is `true` (both counts >= 5 and ratio >= 1.5), and note when relevant that pre-1979 ERA5 predates routine satellite data and carries higher uncertainty.
9. BANNED HYPE PHRASES:
   - NEVER use: 'unprecedented', 'ever recorded', 'heat dome', 'baking', 'historic', 'climate emergency', 'shatters' (or 'shattered').
   - NEVER use the word 'record' for model or ERA5 grid values: when `rank_since_1940 == 1`, write 'highest for this time of year since 1940' (or 'wettest' / 'strongest' / 'lowest for this time of year since 1940'). Only use 'record' if directly quoting an official station report in `news_facts` with explicit attribution.
   - When actual temperatures are below 5 °C (especially sub-zero winter anomalies such as -2 °C or -7 °C), NEVER call the event 'heat' or 'record heat' - describe it calmly as 'exceptionally mild' or 'unseasonably mild'.
10. VERIFIED NEWS SOURCES & `news_facts` ATTRIBUTION (MANDATORY):
   - Run `web_search` (including month and year) to verify active human impacts and official warnings before selecting your 5 stories.
   - Every story MUST include `news_links` (1 to 3 direct, resolving event-specific article URLs; never generic `/weather` or `/tag/` hub pages) AND `news_facts`: `[{"claim": "Exact factual claim or figure from the article with date and location", "source_url": "https://..."}]`.
   - Every number in your headline, stats, cues, ticker, and beats MUST either match a field in that story's candidate data (within rounding) OR appear in `news_facts` with its `source_url`.
   - Quote news claims accurately with explicit attribution in prose ('according to the Icelandic Met Office...', 'according to Vietnam's National Centre for Hydro-Meteorological Forecasting...'). Check article dates carefully so old archive stories or storms from earlier weeks are never presented as tonight's forecast. Never narrow a broad regional figure ('winds up to 160 km/h in Europe') to a specific town unless the source states it. When sources disagree on a figure, state the range or omit it, and always label daily Copernicus global temperature estimates as 'preliminary'.
11. GEOGRAPHIC ACCURACY & CLEAR OPENING LOCATION:
   - `cues[0].spoken` MUST explicitly name `country` (and region if offshore, e.g. 'Iceland: a deep North Atlantic low...' or 'In coastal Ecuador, heavy rain near Guayaquil...') in its opening clause.
   - Every cue's `lat` and `lon` MUST be within 500 km of the story `centre`.
   - Whenever a cue (`label` or `spoken`) or `headline` names a city, that cue's `lat`/`lon` MUST be within 150 km of that city. Keep camera paths local to the actual weather system; short local paths are completely fine.
   - Set each cue's `layer` to match the variable discussed in that cue's sentence (`temperature`, `anomaly`, `wind`, `rain`, or `pressure`). Do not force layer switches that do not match the sentence.

STORY STRUCTURE (exactly 5 geographically diverse stories):
Each story MUST include:
- `id`: short lowercase slug, e.g. `"2026-11-iceland-atlantic-low"`
- `kicker`: calm uppercase regional strap, e.g. `"NORTH ATLANTIC LOW · ICELAND"`
- `headline`: clear, factual headline (45 to 85 characters, forecast-framed, no hype, no em dashes)
- `ticker`: concise summary (~70 to 110 words) with key figures wrapped in `**double asterisks**`, separated by ` /// `.
- `region`: readable local region name
- `country`: primary country name
- `hazard`: one of `"heat"`, `"cold"`, `"rain"`, `"cyclone"`, `"wind"`, `"pressure"`
- `primary_layer`: one of `"temperature"`, `"anomaly"`, `"wind"`, `"rain"`, `"pressure"`
- `centre`: `{"lat": float, "lon": float}` matching the chosen candidate hotspot
- `stats`:
    - `headline_metric`: single clear forecast figure with units, e.g. `"123 km/h gusts"` or `"140 mm / 24 h forecast"`
    - `metric_label`: calm context using the same unit system (never put two contradictory wind speeds or unadjusted normals on the same card)
    - `era5_rank`: e.g. `"#1 for time of year since 1940"`
    - `persistence`: e.g. `"8 of 28 days wet (>=10 mm)"` or `"9 of 28 days gusts >65 km/h"`
- `camera_waypoints`: 3 or 4 local waypoints within 500 km of `centre` (`distance` in `1.48..2.05`)
- `cues`: array of 5 sentence-level synchronised cues `[{"layer": "rain"|"wind"|"pressure"|"temperature"|"anomaly", "lat": float, "lon": float, "distance": 1.48..2.02, "bearing": -12..12, "label": "LOCAL PLACE", "metric": "FORECAST STAT", "spoken": "Clear, calm spoken sentence (16 to 26 words) read aloud word-for-word and shown in the script box, with no markdown asterisks.", "ticker": "Short 8 to 14 word clause"}]`.
- `beats`: array of 4 chapters (`FORECAST`, `PERSISTENCE`, `ERA5 CONTEXT`, `CLIMATE CONTEXT`), each with `{"tag": "...", "title": "...", "layer": "...", "text": "..."}`.
- `news_facts`: array of `{ "claim": "...", "source_url": "https://..." }` documenting every non-ECMWF figure or reported impact used in the story.
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
            f"All ECMWF values are a FORECAST valid at {weather.valid_time:%Y-%m-%d %H:%M} UTC; "
            f"24 h rainfall covers {rain_start_iso} to {rain_end_iso}. Use forecast/future wording and never say "
            "'fell', 'has recorded', or 'measured' for model values."
        ),
        "previous_headlines": prev_summary,
        "active_gdacs_alerts": warnings[:12],
        "candidate_hotspots_with_era5_1940_2025_stats": enriched_candidates[:12],
    }

    prompt = (
        "Here are today's global ECMWF forecast candidates paired with ±7-day 1940-2025 ERA5 seasonal "
        "rankings, 14/28-day persistence statistics, 40-year era exceedance counts, nearest named places, "
        "and active GDACS alerts.\n\n"
        "Step 1: Search international and national weather news (`web_search`, including the month and year "
        f"`{weather.valid_time:%B %Y}`) for the populated candidate regions to find direct, resolving news "
        "articles or official meteorological warnings about these specific events.\n"
        "Step 2: Choose 5 geographically diverse stories that genuinely affect populated places. Write in a "
        "clear, calm, accurate public-service tone; never exaggerate; when in doubt, understate. Every non-model "
        "claim or number MUST be listed in `news_facts` with its resolving `source_url` and attributed in prose. "
        "Output ONLY the JSON object.\n\n"
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

                # If server-side web_search hit its per-turn iteration cap, continue cleanly until end_turn
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
                            "Fix every validation error below (in particular, replace any generic hub URL "
                            "with a direct event-specific news article URL from web_search) and return ONLY "
                            "the corrected JSON object:\n" + "\n".join(f"- {e}" for e in errors)
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
# Deterministic ERA5-backed fallback when ANTHROPIC_API_KEY is not configured
# ---------------------------------------------------------------------------


def _region_label_for_coords(lat: float, lon: float) -> tuple[str, str]:
    regions = [
        ((50, 72), (-25, 32), "Northern & Western Europe", "Europe"),
        ((35, 50), (-10, 35), "Southern Europe & Mediterranean", "Mediterranean"),
        ((50, 76), (32, 175), "Siberia & Northern Eurasia", "Russia"),
        ((20, 48), (48, 125), "Central & East Asia", "Central Asia"),
        ((5, 30), (68, 100), "South Asia & Bay of Bengal", "India"),
        ((-12, 22), (98, 145), "Maritime Continent & Southeast Asia", "Southeast Asia"),
        ((24, 46), (125, 150), "Japan & Northwest Pacific", "Japan"),
        ((-46, -10), (112, 155), "Australia", "Australia"),
        ((-48, -33), (165, 179), "New Zealand", "New Zealand"),
        ((25, 50), (-126, -66), "United States & Southern Canada", "United States"),
        ((50, 75), (-168, -52), "Canada & Subarctic North America", "Canada"),
        ((8, 28), (-105, -60), "Central America & Caribbean", "Caribbean"),
        ((-56, 12), (-84, -34), "South America", "South America"),
        ((0, 36), (-18, 52), "North Africa & Middle East", "North Africa"),
        ((-36, 0), (8, 52), "Southern & Eastern Africa", "Southern Africa"),
    ]
    for (la0, la1), (lo0, lo1), reg, ctry in regions:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return reg, ctry
    hem_ns = "N" if lat >= 0 else "S"
    hem_ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.0f}°{hem_ns}, {abs(lon):.0f}°{hem_ew} Sector", "Global"


def build_deterministic_bulletin(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
) -> dict:
    """Build a valid 5-story ERA5-grounded bulletin when running offline without an API key."""
    picked = enriched_candidates[:5]
    valid_gdacs = [w for w in warnings if w.get("url") and is_specific_event_url(str(w["url"]))]
    stories = []

    for idx, cand in enumerate(picked, start=1):
        kind = cand["kind"]
        lat, lon = cand["lat"], cand["lon"]
        region, country = _region_label_for_coords(lat, lon)
        gp = cand.get("grid_point") or {}
        if gp.get("nearest_gazetteer_country") and country == "Global":
            country = gp["nearest_gazetteer_country"]
        if gp.get("nearest_gazetteer_place") and float(gp.get("nearest_place_distance_km") or 999) <= 80.0:
            region = f"Near {gp['nearest_gazetteer_place']}, {country}" if country != "Global" else gp["nearest_gazetteer_place"]

        era = cand.get("era5") or {}
        pers = era.get("persistence") or {}
        shift = era.get("climate_shift") or {}

        rank_num = era.get("rank_since_1940")
        n_yrs = era.get("years_compared", 86)
        pct = era.get("percentile", 95.0)
        warm_shift = shift.get("local_seasonal_warming_c_since_1940_1979", 1.2)
        early_hits = shift.get("p90_years_count_1940_1979_out_of_40", shift.get("p90_exceedances_1940_1979", 3))
        recent_hits = shift.get("p90_years_count_1986_2025_out_of_40", shift.get("p90_exceedances_1986_2025", 6))

        if kind == "heat":
            is_subzero_mild = float(cand.get("t2m_c", 15.0)) < 5.0
            warm_noun = "MILD SPELL" if is_subzero_mild else "WARM SPELL"
            desc_adj = "exceptionally mild" if is_subzero_mild else "unseasonably warm"
            kicker = f"{warm_noun} · {region.upper()}"
            headline = f"Temperatures forecast {cand['anom_c']:+.1f} °C above seasonal average across {region}"
            metric = f"{cand['anom_c']:+.1f} °C"
            m_label = f"Forecast {cand['t2m_c']:.1f} °C"
            p_layer = "anomaly"
            b0_layer = "temperature"
            d14 = pers.get("warm_days_above_p90_tmax_last_14", pers.get("anomalous_days_last_14", 6))
            d28 = pers.get("warm_days_above_p90_tmax_last_28", pers.get("anomalous_days_last_28", 11))
            pers_stat = f"{d14} of 14 days above 90th pct Tmax"
            live_txt = (
                f"ECMWF guidance forecasts **{cand['t2m_c']:.1f} °C** in {region}, "
                f"**{cand['anom_c']:+.1f} °C above** the 1991-2020 average ({desc_adj} for late {weather.valid_time:%B})."
            )
            pers_txt = (
                f"At this grid box, daily maximum temperatures exceeded the local 90th percentile on "
                f"**{d14} of 14 days** and **{d28} of 28 days**."
            )
        elif kind == "cold":
            kicker = f"COLD OUTBREAK · {region.upper()}"
            headline = f"Temperatures forecast {abs(cand['anom_c']):.1f} °C below seasonal average in {region}"
            metric = f"{cand['anom_c']:+.1f} °C"
            m_label = f"Forecast {cand['t2m_c']:.1f} °C"
            p_layer = "anomaly"
            b0_layer = "temperature"
            d14 = pers.get("cold_days_below_p10_tmin_last_14", pers.get("anomalous_days_last_14", 5))
            pers_stat = f"{d14} of 14 days below 10th pct Tmin"
            live_txt = (
                f"Cold continental air is forecast to bring **{cand['t2m_c']:.1f} °C** in {region}, "
                f"**{abs(cand['anom_c']):.1f} °C below** the seasonal average for late {weather.valid_time:%B}."
            )
            pers_txt = (
                f"Over the past fortnight, daily minimum temperatures fell below the 10th percentile on **{d14} of 14 days**."
            )
        elif kind == "rain":
            kicker = f"HEAVY RAIN FORECAST · {region.upper()}"
            headline = f"Up to {cand['rain_24h_mm']:.0f} mm of rain forecast over 24 hours in {region}"
            metric = f"{cand['rain_24h_mm']:.0f} mm / 24 h"
            m_label = "24-hour forecast accumulation"
            p_layer = "rain"
            b0_layer = "rain"
            wet28 = pers.get("wet_days_ge_10mm_last_28", pers.get("anomalous_days_last_28", 6))
            r14 = pers.get("rain_14d_mm", 80.0)
            pers_stat = f"{wet28} of 28 days wet (>=10 mm)"
            live_txt = (
                f"ECMWF guidance forecasts up to **{cand['rain_24h_mm']:.0f} mm of rain** over the next 24 hours in {region}."
            )
            pers_txt = (
                f"Recent ERA5 totals at this grid point reach **{r14:.0f} mm**, with **{wet28} of 28 days** recording at least 10 mm."
            )
        else:
            kicker = f"DEEP LOW FORECAST · {region.upper()}"
            headline = f"Low of {cand['mslp_hpa']:.0f} hPa forecast near {region} with {cand['gust_ms'] * 3.6:.0f} km/h gusts"
            metric = f"{cand['gust_ms'] * 3.6:.0f} km/h"
            m_label = f"Forecast central pressure {cand['mslp_hpa']:.0f} hPa"
            p_layer = "wind"
            b0_layer = "wind"
            g28 = pers.get("days_gust_over_65kmh_last_28", pers.get("anomalous_days_last_28", 6))
            pers_stat = f"{g28} of 28 days gusts >65 km/h"
            live_txt = (
                f"A **{cand['mslp_hpa']:.0f} hPa low** is forecast near {region} with 24-hour peak gusts up to "
                f"**{cand['gust_ms'] * 3.6:.0f} km/h**."
            )
            pers_txt = (
                f"This grid box has seen **{g28} of 28 days** with gusts over 65 km/h during the past four weeks."
            )

        if rank_num is not None:
            era5_rank_stat = f"#{rank_num} for time of year since 1940"
            rank_txt = (
                f"Compared with **{n_yrs} years of ERA5 data since 1940**, this forecast value ranks **#{rank_num}** "
                f"for this time of year (**{pct:.1f}th percentile** at this grid box)."
            )
            clim_txt = (
                f"Local seasonal mean temperatures at this point have changed by **{warm_shift:+.1f} °C since 1940-1979** "
                f"(**{early_hits} years out of 40** in 1940-1979 vs **{recent_hits} years out of 40** in 1986-2025; "
                "pre-1979 ERA5 lacks satellite data)."
            )
        else:
            disp_pct = min(99, int(round(pct)))
            era5_rank_stat = f"{disp_pct}th pct for time of year"
            rank_txt = (
                f"Compared with the 1990-2019 ERA5 seasonal climatology, this forecast value sits near the "
                f"**{pct:.1f}th percentile** for this time of year at this grid box."
            )
            clim_txt = (
                f"Local seasonal mean temperatures at this point have changed by **{warm_shift:+.1f} °C since 1940-1979** "
                "(ERA5 reanalysis; pre-1979 ERA5 lacks satellite data)."
            )

        beats = [
            {"tag": "FORECAST", "title": "24-Hour Forecast", "layer": b0_layer, "text": live_txt},
            {"tag": "PERSISTENCE", "title": "Recent Persistence", "layer": p_layer, "text": pers_txt},
            {"tag": "ERA5 CONTEXT", "title": "Seasonal Rank", "layer": p_layer, "text": rank_txt},
            {
                "tag": "CLIMATE CONTEXT",
                "title": "40-Year Era Context",
                "layer": "anomaly" if kind in ("heat", "cold", "rain") else "pressure",
                "text": clim_txt,
            },
        ]

        # Match only genuinely nearby (<=800 km) current GDACS event reports; show no link rather than an unrelated one
        links = []
        for w in valid_gdacs:
            if w.get("lat") is not None and w.get("lon") is not None and w.get("is_current", True):
                if haversine_km(float(w["lat"]), float(w["lon"]), lat, lon) <= 800.0:
                    links.append(
                        {
                            "source": "GDACS Event Report",
                            "title": f"{w['alert_level']} {w['type']} bulletin: {w['name'][:70]}",
                            "url": w["url"],
                        }
                    )
                    break

        stories.append(
            {
                "id": f"auto-{weather.valid_time:%Y%m%d}-{idx}-{kind}",
                "rank": idx,
                "kicker": kicker,
                "headline": headline,
                "ticker": build_ticker_from_beats(beats),
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
                "beats": beats,
                "news_links": links[:2],
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
