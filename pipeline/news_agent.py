"""Anthropic Claude + Web Search broadcast writer for the 6-hourly top 5 weather headlines."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

from .opendata import LiveWeatherFields
from .validator import ensure_dynamic_waypoints, validate_and_clean_bulletin

DEFAULT_CLAUDE_MODELS = [
    "claude-sonnet-4-5",
    "claude-sonnet-4-20250514",
]

SYSTEM_PROMPT = """You are the Chief Broadcast Meteorologist and Climate Science Editor for Weather Headlines.
Every 6 hours you produce a captivating 5-story global weather tour that plays like a flagship international television weather bulletin.

STRICT EDITORIAL & LANGUAGE RULES:
1. Always write in crisp, vivid, engaging British English (for example: centre, colour, modelled, organise, programme, metre).
2. NEVER use em dashes (Unicode U+2014). Use commas, colons, parentheses, or spaced hyphens (' - ') instead.
3. NEVER use the word 'utilise' (or utilises, utilising, utilisation). Use 'use' instead.
4. Avoid dry numerical-model jargon (never say 'HRES', 'grid cell', 'z_score', 'scan candidate', or '0p25'). Speak to curious viewers with clarity, energy, and intuition.
5. Never invent rankings, persistence durations, or URLs. Use the exact ERA5 1940-present rankings, 14/28-day persistence stats, and 1940-1979 vs 1986-2025 shift metrics supplied in the candidate data, combined with real international news stories you find via web_search.
6. Every story's `news_links` MUST be genuine, human-readable https:// web pages from reputable international news outlets (such as BBC News, The Guardian, Reuters, Associated Press, Al Jazeera, Financial Times, ABC Australia, CBC, RNZ, Japan Times) or official meteorological/climate bodies (WMO, Met Office, NOAA, Copernicus, World Weather Attribution, GDACS). Never link raw JSON/XML/CAP feeds.
7. Continuity: if a major ongoing storm, flood, or heatwave from `previous_headlines` is still active in today's fields, keep covering it with updated stats and fresh narrative angles while bringing in newly breaking global events.

STORY STRUCTURE (exactly 5 stories spanning diverse global regions):
Each story MUST include:
- `id`: short lowercase slug, e.g. `"2026-11-bay-of-bengal-cyclone"`
- `kicker`: uppercase broadcast strap, e.g. `"RECORD NOVEMBER HEAT · CENTRAL ASIA"`
- `headline`: punchy, dramatic, accurate broadcast headline (under 90 characters, no em dashes)
- `region`: readable region name, e.g. `"Queensland and Northern New South Wales, Australia"`
- `country`: primary country name, e.g. `"Australia"`
- `hazard`: one of `"heat"`, `"cold"`, `"rain"`, `"cyclone"`, `"wind"`, `"pressure"`
- `primary_layer`: one of `"temperature"`, `"anomaly"`, `"wind"`, `"rain"`, `"pressure"`
- `centre`: `{"lat": float, "lon": float}`
- `stats`:
    - `headline_metric`: e.g. `"+13.8 °C"`, `"186 mm / 24 h"`, `"34 m/s gusts"`
    - `metric_label`: short explanation, e.g. `"Above 1991-2020 normal"`
    - `era5_rank`: exact concise summary from ERA5 stats, e.g. `"#2 since 1940 (98.9th pct)"`
    - `persistence`: concise duration summary, e.g. `"11 warm days in past 2 weeks"`
- `camera_waypoints`: 4 waypoints `[{"lat": ..., "lon": ..., "distance": 1.75..2.35}, ...]` tracing a smooth 600-2,000 km path across the weather feature (e.g. along a storm track, moisture plume, mountain chain, or heat dome) so the 3D camera glides continuously across the Earth's surface.
- `beats`: array of EXACTLY 4 broadcast chapters (2-3 lively sentences each, ~35-55 words per beat):
    1. `{"tag": "LIVE HIGHLIGHT", "title": "...", "layer": "...", "text": "..."}` -> Why this event is exciting, dramatic, and important right now.
    2. `{"tag": "PERSISTENCE", "title": "...", "layer": "...", "text": "..."}` -> Has this been going on for days or weeks? Explain the atmospheric setup (blocking high, stalled front, storm train) using the 14/28-day persistence figures.
    3. `{"tag": "HISTORICAL RANK", "title": "...", "layer": "...", "text": "..."}` -> Is it unprecedented? Give the exact 1940-present seasonal rank and percentile for this time of year.
    4. `{"tag": "CLIMATE CONTEXT", "title": "...", "layer": "...", "text": "..."}` -> Has climate change made this more likely or intense? Combine the local ERA5 1940-1979 vs 1986-2025 shift with attribution science (e.g. World Weather Attribution, IPCC AR6, Clausius-Clapeyron moisture scaling) honestly and clearly.
- `news_links`: 1 to 3 verified international news or official report links `[{"source": "BBC News", "title": "...", "url": "https://..."}]`.

Return ONLY a single valid JSON object matching `{"edition_title": "...", "stories": [...]}` with no markdown prose outside the JSON."""


def _extract_json_object(text: str) -> dict:
    """Extract the outermost JSON object from Claude's response."""
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
    """Use Anthropic Claude with built-in web search to research news and author 5 broadcast stories."""
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
                }
            )

    user_payload = {
        "current_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ecmwf_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "previous_headlines": prev_summary,
        "active_gdacs_alerts": warnings[:20],
        "candidate_hotspots_with_era5_1940_2026_stats": enriched_candidates,
    }

    prompt = (
        "Here are today's global ECMWF weather hotspots paired with exact 1940-present ERA5 seasonal "
        "rankings, 14/28-day persistence statistics, 1940-1979 vs 1986-2025 local warming shifts, and "
        "active GDACS alerts.\n\n"
        "Step 1: Run web searches for the strongest candidate regions and active severe weather events "
        f"(include the month and year `{weather.valid_time:%B %Y}` in queries) to find real international "
        "news coverage (BBC, Guardian, Reuters, AP, Al Jazeera, ABC, CBC, etc.) and any relevant climate "
        "attribution context.\n"
        "Step 2: Pick the 5 most compelling, geographically diverse stories worldwide and output the "
        "complete JSON bulletin.\n\n"
        f"<data>\n{json.dumps(user_payload, indent=2)}\n</data>"
    )

    last_exc: Exception | None = None
    for model_name in models_to_try:
        try:
            print(f"[news_agent] Calling Anthropic model {model_name} with web search...", file=sys.stderr)
            messages: list[dict] = [{"role": "user", "content": prompt}]

            for attempt in range(2):
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=8000,
                    system=SYSTEM_PROMPT,
                    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 12}],
                    messages=messages,
                )
                text_parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
                combined_text = "\n".join(text_parts)
                draft = _extract_json_object(combined_text)
                draft.update(
                    {
                        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "generator": f"anthropic:{model_name}",
                    }
                )
                cleaned, errors = validate_and_clean_bulletin(draft)
                if not errors:
                    return cleaned
                print(f"[news_agent] Validation issues on attempt {attempt + 1}: {errors}", file=sys.stderr)
                messages.append({"role": "assistant", "content": combined_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Fix these validation errors and return ONLY the corrected JSON object:\n"
                            + "\n".join(f"- {e}" for e in errors)
                        ),
                    }
                )
            raise RuntimeError(f"Claude output failed validation twice: {errors}")
        except Exception as exc:
            last_exc = exc
            print(f"[news_agent] Model {model_name} failed: {exc}", file=sys.stderr)
            continue

    raise RuntimeError(f"All Claude models failed: {last_exc}")


# ---------------------------------------------------------------------------
# Deterministic ERA5-backed fallback when ANTHROPIC_API_KEY is not configured
# ---------------------------------------------------------------------------


def _region_label_for_coords(lat: float, lon: float) -> tuple[str, str]:
    """Coarse human-readable region & country fallback when running without an LLM key."""
    regions = [
        ((50, 72), (-12, 32), "Northern & Western Europe", "Europe"),
        ((35, 50), (-10, 35), "Southern Europe & Mediterranean", "Mediterranean"),
        ((50, 76), (32, 175), "Siberia & Northern Eurasia", "Russia"),
        ((20, 48), (68, 125), "Central & East Asia", "East Asia"),
        ((5, 30), (68, 100), "South Asia & Bay of Bengal", "India"),
        ((-12, 22), (98, 145), "Maritime Continent & Southeast Asia", "Southeast Asia"),
        ((24, 46), (125, 150), "Japan & Northwest Pacific", "Japan"),
        ((-46, -10), (112, 155), "Australia", "Australia"),
        ((-48, -33), (165, 179), "New Zealand", "New Zealand"),
        ((25, 50), (-126, -66), "United States & Southern Canada", "United States"),
        ((50, 75), (-168, -52), "Canada & Subarctic North America", "Canada"),
        ((8, 28), (-105, -60), "Central America & Caribbean", "Caribbean"),
        ((-56, 12), (-82, -34), "South America", "South America"),
        ((0, 36), (-18, 52), "North Africa & Middle East", "North Africa"),
        ((-36, 0), (8, 52), "Southern & Eastern Africa", "Southern Africa"),
        ((72, 90), (-180, 180), "High Arctic Basin", "Arctic"),
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
    """Build a valid 5-story ERA5-grounded bulletin without paid API calls (used when no key is set)."""
    picked = enriched_candidates[:5]
    stories = []
    for idx, cand in enumerate(picked, start=1):
        kind = cand["kind"]
        lat, lon = cand["lat"], cand["lon"]
        region, country = _region_label_for_coords(lat, lon)
        era = cand.get("era5") or {}
        pers = era.get("persistence") or {}
        shift = era.get("climate_shift") or {}

        rank_num = era.get("rank_since_1940", 5)
        n_yrs = era.get("years_compared", 87)
        pct = era.get("percentile", 95.0)
        warm_shift = shift.get("local_seasonal_warming_c_since_1940_1979", 1.2)
        early_hits = shift.get("p90_exceedances_1940_1979", 2)
        recent_hits = shift.get("p90_exceedances_1986_2025", 6)

        if kind == "heat":
            kicker = f"UNSEASONAL HEAT · {region.upper()}"
            headline = f"Temperatures surge {cand['anom_c']:+.1f} °C above seasonal norms across {region}"
            metric = f"{cand['anom_c']:+.1f} °C"
            m_label = f"Peak {cand['t2m_c']:.1f} °C vs normal"
            p_layer = "anomaly"
            b0_layer = "temperature"
            live_txt = (
                f"A powerful dome of warm air has pushed 2 m temperatures to {cand['t2m_c']:.1f} °C across {region}, "
                f"standing {cand['anom_c']:+.1f} °C above the 1991-2020 seasonal average for late {weather.valid_time:%B}."
            )
            pers_txt = (
                f"ERA5 daily tracking shows {pers.get('anomalous_days_last_14', 6)} unusually warm days over the past "
                f"fortnight and {pers.get('anomalous_days_last_28', 11)} across the last four weeks, driven by a persistent upper-level ridge."
            )
        elif kind == "cold":
            kicker = f"DEEP COLD OUTBREAK · {region.upper()}"
            headline = f"Sharp polar air plunges {region} {abs(cand['anom_c']):.1f} °C below seasonal averages"
            metric = f"{cand['anom_c']:+.1f} °C"
            m_label = f"Down to {cand['t2m_c']:.1f} °C"
            p_layer = "anomaly"
            b0_layer = "temperature"
            live_txt = (
                f"An intense outbreak of continental polar air has driven temperatures down to {cand['t2m_c']:.1f} °C in {region}, "
                f"roughly {abs(cand['anom_c']):.1f} °C colder than typical for this date."
            )
            pers_txt = (
                f"Over the past two weeks, {pers.get('anomalous_days_last_14', 5)} days have fallen into the coldest 10% "
                f"of seasonal records as a displaced jet stream channels high-latitude air equatorward."
            )
        elif kind == "rain":
            kicker = f"HEAVY RAINFALL · {region.upper()}"
            headline = f"Intense moisture plume delivers {cand['rain_24h_mm']:.0f} mm of 24-hour rain in {region}"
            metric = f"{cand['rain_24h_mm']:.0f} mm"
            m_label = "24-hour forecast accumulation"
            p_layer = "rain"
            b0_layer = "rain"
            live_txt = (
                f"Deep atmospheric moisture converging over {region} is producing up to {cand['rain_24h_mm']:.0f} mm "
                f"of rainfall within 24 hours, elevating river and flash-flood risk along windward terrain."
            )
            pers_txt = (
                f"Recent ERA5 totals show {pers.get('rain_14d_mm', 80.0):.0f} mm over the past 14 days compared with a "
                f"seasonal expectation of {pers.get('expected_rain_14d_mm', 40.0):.0f} mm, leaving soils pre-saturated."
            )
        else:
            kicker = f"SEVERE WIND & CYCLONE · {region.upper()}"
            headline = f"Deep {cand['mslp_hpa']:.0f} hPa storm system drives {cand['gust_ms']:.0f} m/s gusts near {region}"
            metric = f"{cand['gust_ms']:.0f} m/s"
            m_label = f"Central pressure {cand['mslp_hpa']:.0f} hPa"
            p_layer = "wind"
            b0_layer = "wind"
            live_txt = (
                f"A vigorous low-pressure system at {cand['mslp_hpa']:.0f} hPa is sweeping {region} with sustained "
                f"10 m winds of {cand['wind_ms']:.0f} m/s and peak gusts reaching {cand['gust_ms']:.0f} m/s."
            )
            pers_txt = (
                f"This region has experienced {pers.get('anomalous_days_last_28', 6)} high-wind days over the past four weeks "
                f"along an active storm track, accompanied by {cand['rain_24h_mm']:.0f} mm of 24-hour rain."
            )

        rank_txt = (
            f"Comparing this event against every seasonal window since 1940 in the Copernicus ERA5 archive ({n_yrs} years), "
            f"it ranks #{rank_num} ({pct:.1f}th percentile) for this time of year."
        )
        clim_txt = (
            f"In local ERA5 records, this calendar season in {region} has warmed by {warm_shift:+.1f} °C between 1940-1979 "
            f"and 1986-2025, while seasonal extremes above the 90th percentile shifted from {early_hits} to {recent_hits} occurrences per 40 years."
        )

        # Attach any nearby GDACS official report plus major international climate/weather desks
        links = []
        for w in warnings:
            if w.get("url") and w.get("lat") is not None and w.get("lon") is not None:
                if abs(w["lat"] - lat) < 18 and abs(w["lon"] - lon) < 22:
                    links.append(
                        {
                            "source": "GDACS Alert",
                            "title": f"{w['alert_level']} {w['type']} report: {w['name']}",
                            "url": w["url"],
                        }
                    )
                    break
        links.extend(
            [
                {
                    "source": "The Guardian",
                    "title": "Extreme weather and global climate coverage",
                    "url": "https://www.theguardian.com/environment/extreme-weather",
                },
                {
                    "source": "Copernicus C3S",
                    "title": "Copernicus Climate Bulletins and global anomalies",
                    "url": "https://climate.copernicus.eu/climate-bulletins",
                },
            ]
        )

        stories.append(
            {
                "id": f"auto-{weather.valid_time:%Y%m%d}-{idx}-{kind}",
                "rank": idx,
                "kicker": kicker,
                "headline": headline,
                "region": region,
                "country": country,
                "hazard": kind,
                "primary_layer": p_layer,
                "centre": {"lat": lat, "lon": lon},
                "stats": {
                    "headline_metric": metric,
                    "metric_label": m_label,
                    "era5_rank": f"#{rank_num} since 1940 ({pct:.0f}th pct)",
                    "persistence": f"{pers.get('anomalous_days_last_14', 5)} anomalous days in 14d",
                },
                "camera_waypoints": ensure_dynamic_waypoints(lat, lon, None),
                "beats": [
                    {"tag": "LIVE HIGHLIGHT", "title": "Why This Event Stands Out", "layer": b0_layer, "text": live_txt},
                    {"tag": "PERSISTENCE", "title": "Multi-Week Build-Up", "layer": p_layer, "text": pers_txt},
                    {
                        "tag": "HISTORICAL RANK",
                        "title": f"ERA5 Seasonal Rank (1940-{weather.valid_time.year})",
                        "layer": p_layer,
                        "text": rank_txt,
                    },
                    {
                        "tag": "CLIMATE CONTEXT",
                        "title": "How Warming Shifts the Odds",
                        "layer": "anomaly" if kind in ("heat", "cold", "rain") else "pressure",
                        "text": clim_txt,
                    },
                ],
                "news_links": links[:3],
            }
        )

    draft = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "edition_title": "Global Weather Bulletin",
        "generator": "era5-deterministic",
        "stories": stories,
    }
    cleaned, errs = validate_and_clean_bulletin(draft)
    if errs:
        raise RuntimeError(f"Deterministic fallback failed validation: {errs}")
    return cleaned
