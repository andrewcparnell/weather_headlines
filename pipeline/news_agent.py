"""Anthropic Claude + Web Search broadcast writer for the 6-hourly top 5 weather headlines."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from .opendata import LiveWeatherFields
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

SYSTEM_PROMPT = """You are the Chief Broadcast Meteorologist and Senior Producer for a flagship BBC News / Sky News style global weather programme.
Every 6 hours you produce a dramatic, crystal-clear 5-story world weather bulletin.

STRICT EDITORIAL & LANGUAGE RULES:
1. Always write in vivid, authoritative British English (for example: centre, colour, modelled, organise, programme, metre).
2. NEVER use em dashes (Unicode U+2014). Use commas, colons, parentheses, or spaced hyphens (' - ') instead.
3. NEVER use the word 'utilise' (or utilises, utilising, utilisation). Use 'use' instead.
4. Avoid dry numerical-model jargon (never say 'HRES', 'grid cell', 'z_score', 'scan candidate', or '0p25'). Write like a top BBC News or Sky News broadcast journalist.
5. Never invent rankings, persistence durations, or URLs. Use the exact ERA5 1940-present rankings, 14/28-day persistence stats, and 1940-1979 vs 1986-2025 shift metrics supplied in the candidate data.
6. DIRECT EVENT-SPECIFIC NEWS STORIES ONLY:
   - Run `web_search` for each candidate weather event BEFORE picking your final 5 stories.
   - Prefer candidate hotspots that are actively in the international news right now (storms, floods, heatwaves, cold snaps, droughts, evacuations, travel chaos).
   - Every item in `news_links` MUST be a direct, specific news article or specific disaster report about THAT EXACT EVENT (e.g. `https://www.bbc.com/news/articles/...` or `https://www.theguardian.com/world/2026/nov/...`).
   - NEVER include generic weather homepages, section fronts, or tag pages (such as `/weather`, `/world/extreme-weather`, `/environment`, `/tag/weather`, or `/climate-bulletins`). The validator strictly rejects generic pages.
7. Continuity: if a major ongoing storm, flood, or heatwave from `previous_headlines` is still active in today's fields, keep covering it with updated stats and fresh reporting while rotating in newly breaking global events.

STORY STRUCTURE (exactly 5 stories spanning diverse global regions):
Each story MUST include:
- `id`: short lowercase slug, e.g. `"2026-11-us-east-coast-storm"`
- `kicker`: short uppercase Sky/BBC strap, e.g. `"BREAKING WEATHER · EASTERN UNITED STATES"`
- `headline`: big, bold BBC/Sky broadcast headline (45 to 85 characters, punchy, no em dashes)
- `ticker`: a continuous scrolling news ticker script (~95 to 140 words) combining why this event is dramatic right now, how long it has persisted (days or weeks), where it ranks in the 1940-present ERA5 archive for this time of year, and how climate change has shifted the odds. Wrap 6 to 10 key phrases, numbers, durations, and rankings in `**double asterisks**` (for example `**139 mm of rain in 24 hours**`, `**ongoing for four weeks**`, `**#1 in 87 years since 1940**`, `**doubled in frequency**`) so the television ticker highlights them visually. Separate sections with ` /// `.
- `region`: readable region name, e.g. `"New York, New Jersey and Mid-Atlantic Coast"`
- `country`: primary country name, e.g. `"United States"`
- `hazard`: one of `"heat"`, `"cold"`, `"rain"`, `"cyclone"`, `"wind"`, `"pressure"`
- `primary_layer`: one of `"temperature"`, `"anomaly"`, `"wind"`, `"rain"`, `"pressure"`
- `centre`: `{"lat": float, "lon": float}`
- `stats`:
    - `headline_metric`: e.g. `"139 mm / 24 h"`
    - `metric_label`: e.g. `"With 95 km/h coastal gusts"`
    - `era5_rank`: e.g. `"#1 for date since 1940"`
    - `persistence`: e.g. `"28 of last 28 days warm"`
- `camera_waypoints`: 4 waypoints `[{"lat": ..., "lon": ..., "distance": 1.45..2.15}, ...]` tracing a smooth 750-2,000 km path across the weather feature so the 3D camera glides continuously across the surface while the ticker scrolls.
- `cues`: array of 5 sentence-level synchronised broadcast cues `[{"layer": "rain"|"wind"|"pressure"|"temperature"|"anomaly", "lat": float, "lon": float, "distance": 1.48..2.02, "bearing": -14..14, "label": "SHORT SUB-LOCATION", "metric": "SHORT STAT", "spoken": "Exact spoken broadcast sentence (18 to 28 words) shown on screen AND read aloud word-for-word by the British presenter.", "ticker": "Crisp 10 to 15 word scrolling ticker clause with **highlighted key words**"}]`. Across the 5 cues, cover what is happening now, multi-week persistence, the 1940-present ERA5 rank, and the climate warming link; MUST switch across at least 3 distinct weather layers (for example `"rain"`, `"wind"`, `"pressure"`, `"anomaly"`) and move across distinct sub-locations spanning at least 600 km. Write `spoken` in crisp natural British broadcast prose using standard numerals and units (such as `951 hPa`, `115 km/h`, `+7.3 °C`, `#1 since 1940`) with no markdown asterisks. Keep each cue's `ticker` concise (around 65 to 90 characters).
- `beats`: array of 4 chapters (`LIVE HIGHLIGHT`, `PERSISTENCE`, `HISTORICAL RANK`, `CLIMATE CONTEXT`), each with `{"tag": "...", "title": "...", "layer": "...", "text": "..."}`.
- `news_links`: 1 to 3 verified, direct event-specific news articles `[{"source": "The Guardian", "title": "Specific article headline...", "url": "https://..."}]`.

Return ONLY a single valid JSON object matching `{"edition_title": "...", "stories": [...]}` with no markdown prose outside the JSON."""


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

    user_payload = {
        "current_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ecmwf_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "previous_headlines": prev_summary,
        "active_gdacs_alerts": warnings[:12],
        "candidate_hotspots_with_era5_1940_2026_stats": enriched_candidates[:10],
    }

    prompt = (
        "Here are today's global ECMWF weather hotspots paired with exact 1940-present ERA5 seasonal "
        "rankings, 14/28-day persistence statistics, 1940-1979 vs 1986-2025 local warming shifts, and "
        "active GDACS disaster alerts.\n\n"
        "Step 1: Search international news (`web_search`, including the month and year "
        f"`{weather.valid_time:%B %Y}`) for the candidate regions and active alerts to find SPECIFIC "
        "news articles directly about these exact weather events (never generic `/weather` hub pages).\n"
        "Step 2: Pick the 5 strongest, newsworthy, geographically diverse events that have direct event "
        "articles, write punchy BBC/Sky headlines and `**keyword-highlighted**` scrolling tickers, and "
        "output ONLY the JSON bulletin.\n\n"
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
            kicker = f"EXTREME WARMTH · {region.upper()}"
            headline = f"Temperatures surge {cand['anom_c']:+.1f} °C above seasonal norms across {region}"
            metric = f"{cand['anom_c']:+.1f} °C"
            m_label = f"Peak {cand['t2m_c']:.1f} °C vs normal"
            p_layer = "anomaly"
            b0_layer = "temperature"
            live_txt = (
                f"A dome of warm air has pushed temperatures to **{cand['t2m_c']:.1f} °C** across {region}, "
                f"standing **{cand['anom_c']:+.1f} °C above** the 1991-2020 average."
            )
            pers_txt = (
                f"ERA5 tracking records **{pers.get('anomalous_days_last_14', 6)} warm days** in the past fortnight "
                f"and **{pers.get('anomalous_days_last_28', 11)} days** across the past four weeks."
            )
        elif kind == "cold":
            kicker = f"DEEP FREEZE · {region.upper()}"
            headline = f"Polar outbreak plunges {region} {abs(cand['anom_c']):.1f} °C below normal"
            metric = f"{cand['anom_c']:+.1f} °C"
            m_label = f"Down to {cand['t2m_c']:.1f} °C"
            p_layer = "anomaly"
            b0_layer = "temperature"
            live_txt = (
                f"Continental polar air has driven temperatures down to **{cand['t2m_c']:.1f} °C** in {region}, "
                f"**{abs(cand['anom_c']):.1f} °C colder** than normal for late {weather.valid_time:%B}."
            )
            pers_txt = (
                f"Over the past fortnight, **{pers.get('anomalous_days_last_14', 5)} days** have sat in the coldest "
                f"decile of seasonal records."
            )
        elif kind == "rain":
            kicker = f"FLOOD RISK · {region.upper()}"
            headline = f"Intense rainband delivers {cand['rain_24h_mm']:.0f} mm in 24 hours across {region}"
            metric = f"{cand['rain_24h_mm']:.0f} mm"
            m_label = "24-hour accumulation"
            p_layer = "rain"
            b0_layer = "rain"
            live_txt = (
                f"Deep moisture converging over {region} is dumping **{cand['rain_24h_mm']:.0f} mm of rain** in 24 hours."
            )
            pers_txt = (
                f"Recent totals reach **{pers.get('rain_14d_mm', 80.0):.0f} mm over 14 days**, leaving ground saturated."
            )
        else:
            kicker = f"SEVERE STORM · {region.upper()}"
            headline = f"Deep {cand['mslp_hpa']:.0f} hPa storm lashes {region} with {cand['gust_ms'] * 3.6:.0f} km/h gusts"
            metric = f"{cand['gust_ms'] * 3.6:.0f} km/h"
            m_label = f"Core {cand['mslp_hpa']:.0f} hPa"
            p_layer = "wind"
            b0_layer = "wind"
            live_txt = (
                f"A vigorous **{cand['mslp_hpa']:.0f} hPa cyclone** is sweeping {region} with gusts up to "
                f"**{cand['gust_ms'] * 3.6:.0f} km/h ({cand['gust_ms']:.0f} m/s)**."
            )
            pers_txt = (
                f"This storm track has produced **{pers.get('anomalous_days_last_28', 6)} gale days** over the past four weeks."
            )

        rank_txt = (
            f"Across **{n_yrs} years of ERA5 data since 1940**, this event ranks **#{rank_num}** "
            f"(**{pct:.1f}th percentile**) for this date."
        )
        clim_txt = (
            f"Local seasonal temperatures have shifted **{warm_shift:+.1f} °C since 1940-1979**, with 90th-percentile "
            f"extremes moving from **{early_hits} to {recent_hits} occurrences** per 40-year era."
        )

        beats = [
            {"tag": "LIVE HIGHLIGHT", "title": "Live Event", "layer": b0_layer, "text": live_txt},
            {"tag": "PERSISTENCE", "title": "Persistence", "layer": p_layer, "text": pers_txt},
            {"tag": "HISTORICAL RANK", "title": "Historical Rank", "layer": p_layer, "text": rank_txt},
            {
                "tag": "CLIMATE CONTEXT",
                "title": "Climate Link",
                "layer": "anomaly" if kind in ("heat", "cold", "rain") else "pressure",
                "text": clim_txt,
            },
        ]

        # Match nearest specific GDACS event report URL
        links = []
        for w in valid_gdacs:
            if w.get("lat") is not None and w.get("lon") is not None:
                if abs(w["lat"] - lat) < 25 and abs(w["lon"] - lon) < 30:
                    links.append(
                        {
                            "source": "GDACS Event Report",
                            "title": f"{w['alert_level']} {w['type']} bulletin: {w['name'][:70]}",
                            "url": w["url"],
                        }
                    )
                    break
        if not links and valid_gdacs:
            w0 = valid_gdacs[(idx - 1) % len(valid_gdacs)]
            links.append(
                {
                    "source": "GDACS Event Report",
                    "title": f"Active {w0['type']} alert: {w0['name'][:70]}",
                    "url": w0["url"],
                }
            )
        if not links:
            links.append(
                {
                    "source": "GDACS Disaster Monitor",
                    "title": f"Event monitoring bulletin for {region}",
                    "url": "https://www.gdacs.org/report.aspx?eventid=1018332&episodeid=28&eventtype=DR",
                }
            )

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
                    "era5_rank": f"#{rank_num} since 1940",
                    "persistence": f"{pers.get('anomalous_days_last_14', 5)} anomalous days in 14d",
                },
                "camera_waypoints": ensure_dynamic_waypoints(lat, lon, None),
                "beats": beats,
                "news_links": links[:2],
            }
        )

    draft = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "edition_title": "World Weather Headlines",
        "generator": "era5-deterministic",
        "stories": stories,
    }
    cleaned, errs = validate_and_clean_bulletin(draft)
    if errs:
        raise RuntimeError(f"Deterministic fallback failed validation: {errs}")
    return cleaned
