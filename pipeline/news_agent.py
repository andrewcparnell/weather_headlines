"""Claude writer for the 6-hourly world weather headlines (plus news-lead lookup and template fallback)."""
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

from .config import USER_AGENT
from .opendata import LiveWeatherFields
from .fallback_writer import fmt_2sf  # noqa: F401  (re-exported for callers and tests)
from .validator import is_specific_event_url, validate_and_clean_bulletin

# Tried in order; override with the ANTHROPIC_MODEL environment variable (or repository variable in CI).
# Sonnet 5.5 at low effort without web search costs roughly $0.05-0.10 per run; Haiku 5.5 about a tenth of that.
DEFAULT_CLAUDE_MODELS = [
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
]

# Optional web searches per request (each costs $0.01 plus the tokens of the results it reads back in,
# typically $0.02-0.05 more per search). Off by default: Claude works from the free `news_leads` instead.
WEB_SEARCHES_PER_REQUEST = int(os.environ.get("WEATHER_WEB_SEARCHES", "0") or 0)

# Effort controls how much the model thinks before writing; "low" keeps output tokens (the main cost) down.
WRITER_EFFORT = os.environ.get("WEATHER_WRITER_EFFORT", "low").strip() or "low"

MAX_ATTEMPTS = 2  # one retry; stories that still fail are swapped for template stories

# USD per million tokens: (input, output, cache write, cache read). Used only for the cost estimate in the log.
MODEL_PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 3.75, 0.30),
}
WEB_SEARCH_PRICE = 0.01  # USD per search


def _web_search_tool(model_name: str, max_uses: int) -> dict:
    """Newest web search tool version the model supports (dynamic filtering needs a 4.6+ model)."""
    basic_only = model_name.startswith(("claude-haiku-4", "claude-sonnet-4-5", "claude-opus-4-5", "claude-3"))
    tool_type = "web_search_20250305" if basic_only else "web_search_20260209"
    return {"type": tool_type, "name": "web_search", "max_uses": max_uses}


def format_edition_title(dt_utc: datetime) -> str:
    """Hard-coded neutral edition title: 'World Weather Headlines - HH:MM UTC, D Month YYYY'."""
    utc = dt_utc.astimezone(timezone.utc)
    return f"World Weather Headlines - {utc:%H:%M} UTC, {utc.day} {utc:%B %Y}"


def fmt_pct(val: float) -> int:
    """Round a percentage or percentile to the nearest integer."""
    if not math.isfinite(val):
        return 0
    return int(round(val))


SYSTEM_PROMPT = """You are the writer and presenter of a short, 6-hourly world weather bulletin: five stories, each read aloud over a 3D globe in about 40 seconds.

Your job is to tell viewers what is happening to PEOPLE and PLACES because of the weather, in a way that is warm, vivid, specific and true. Think of a well-loved television weather presenter who has also read the morning's news: curious about the places they visit, quick to explain why something matters, never breathless, never dull.

WHAT GOOD LOOKS LIKE
- Lead with the human angle whenever the news gives you one: farmers racing to bring in a harvest, schools closed, ferries cancelled, a city clearing drains, a ski season that cannot start, a fishing fleet kept in port. Attribute it ("local media report...", "Vietnam's disaster agency says...").
- Paint the place: name the nearest town or city from `location`, and say what kind of place it is (a coffee-growing highland, a delta of rice paddies, an Arctic oil town) when you are confident it is true.
- Explain what the numbers mean in everyday terms: "more rain in a day than usually falls here in a month", "warm enough for T-shirts in a city that normally has snow on the ground by now", "gusts strong enough to bring down trees".
- Give each story its own shape. Pick the two or three most interesting angles for THAT story (impacts, how unusual it is, what happens next, how long it has gone on, the long-term trend) rather than marching through the same checklist every time.
- Vary your language across the bulletin. Never open two stories the same way, never reuse a stock phrase more than once per bulletin, and vary sentence length.
- Keep the tone calm and grounded. Vivid is good; hype is not.

ACCURACY RULES (a validator checks these; failing stories are replaced with plain template text)
1. Model numbers are FORECASTS valid at `valid_time_utc`; the 24 h rain total covers `rain_window_start_utc` to `rain_window_end_utc`. Use forecast wording ("up to 140 mm is forecast", "gusts could reach 120 km/h"). Never "fell", "has recorded" or "measured" for model values.
2. Every number with a unit (°C, mm, hPa, km/h, %, "N of M days", "#N", "N times the average") must either match the candidate's data (rounded to 2 significant figures; percentages to whole numbers) or appear in that story's `news_facts` with its `source_url`. Numbers from the news must be quoted as reported, with attribution, and never stretched (a figure about another place, another week or another country is not about this event). Call early estimates "preliminary", and when sources disagree give the range or leave the figure out.
3. Candidate statistics describe ONLY the point at `lat`/`lon`. `location` gives the country, state/province, nearest town and distance. Never transfer a local normal or rank to a city more than 150 km away.
4. `history.rank_since_1940` (when present) is the rank of that 24 h value for this time of year since 1940 at that point. Only when it is 1 AND `history.clear_record` is true say "the highest/wettest/strongest for this time of year since 1940" (you may add the previous high, `previous_record_value` in `previous_record_year`). When it is 1 but `clear_record` is false, say "among the highest for this time of year since 1940". With no rank, make no ranking claim. Never say "ever recorded" or "record" (unless quoting an official station record from the news).
5. Rain normals: use only the normals in `history`. Wet days mean days with at least 10 mm. 65 km/h gusts are not "storm force".
6. Climate trend: `warming_since_mid_century_c` may be null; then say nothing about the local trend. Only claim a change in how often such events happen when `frequency_claim_allowed` is true.
7. Never use: "unprecedented", "ever recorded", "heat dome", "baking", "historic", "climate emergency", "shatters". Below 5 °C a warm anomaly is "exceptionally mild", not "heat" or "heatwave".
8. Never mention data sources or jargon in the prose (ECMWF, ERA5, GDACS, reanalysis, grid point, model run) and never read out year ranges like "1991-2020". Never imply an affiliation with any broadcaster.
9. British English. No em dashes (use commas, colons or spaced hyphens). Never "utilise".

PLACE NAMES IN HEADLINES (important)
- `headline` and `cues[0].spoken` MUST start with the candidate's `location.headline_prefix` followed by a colon, for example "Oklahoma, USA: ...", "Krasnoyarsk Krai, Russia: ..." or "Vietnam: ...".
- `country` must be `location.country`.

CHOOSING STORIES
- Pick 5 geographically varied stories from `candidate_hotspots`, preferring places where people are affected and the news confirms impacts. Skip temperature anomalies over open sea or empty ice. Keep a big ongoing story from `previous_headlines` if it is still active, with fresh angles.

OUTPUT: a single compact JSON object `{"stories": [...]}` and nothing else. Each story has ONLY these keys:
- `id`: short slug, e.g. "2026-11-central-vietnam-floods"
- `kicker`: 2 to 4 uppercase words naming the hazard and place, e.g. "FLOODS · CENTRAL VIETNAM"
- `headline`: 45 to 90 characters, starting with `location.headline_prefix` and a colon, specific and engaging (not a bare number)
- `region`, `country`, `hazard` (heat|cold|rain|cyclone|wind|pressure), `primary_layer` (temperature|anomaly|wind|rain|pressure)
- `centre`: {"lat", "lon"} copied from the chosen candidate
- `stats`: {"headline_metric", "metric_label", "era5_rank", "persistence"}: very short card text; "" where you have no data
- `cues`: exactly 5 spoken sentences, each {"layer", "lat", "lon", "label" (short place or theme), "metric" (short stat or ""), "spoken" (16 to 30 words, read aloud word for word, no asterisks), "ticker" (8 to 14 words with key figures in **double asterisks**)}. Put each cue where it is about (within 500 km of `centre`, within 150 km of any city it names) and set its layer to the variable it discusses.
- `news_facts`: [{"claim", "source_url"}] for every fact taken from the news (the claim must contain any number you quote); [] if none
- `news_links`: up to 3 of the article URLs you drew on [{"source", "title", "url"}]; [] if none"""

NEWS_NOTE_LEADS_ONLY = """

NEWS SOURCES FOR THIS BULLETIN
`news_leads` lists recent headlines from reputable outlets about each candidate's area. They are your ONLY news source: you may describe what a headline says, attributed to its outlet, and cite its exact `url` in `news_facts` and `news_links`. Do not add news facts, casualty figures or impacts from memory. If a candidate has no leads, write from the weather data alone."""

NEWS_NOTE_WITH_SEARCH = """

NEWS SOURCES FOR THIS BULLETIN
`news_leads` lists recent headlines from reputable outlets about each candidate's area. You also have a small number of `web_search` calls: use them only for the most promising stories, to confirm impacts or warnings (include the month and year in queries). Cite only articles you actually saw, with direct article URLs (never section fronts or tag pages)."""


def _extract_json_object(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Claude output.")
    return json.loads(raw[start : end + 1])


def _iso(value) -> str:
    """ISO-8601 UTC string for a datetime (or pass a string through unchanged)."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value) if value is not None else ""


def _r(val, nd: int = 1):
    """Round floats for the brief (fewer tokens); pass everything else through."""
    if isinstance(val, float) and math.isfinite(val):
        return round(val, nd)
    return val


def slim_candidate(cand: dict) -> dict:
    """Only what the writer needs from an enriched candidate, so the brief costs as few tokens as possible.

    The validator still checks the story against the FULL candidate data.
    """
    loc = cand.get("location") or {}
    era = cand.get("era5") or {}
    pers = era.get("persistence") or {}
    shift = era.get("climate_shift") or {}
    kind = cand["kind"]
    out: dict = {
        "id": cand.get("id"),
        "kind": kind,
        "lat": _r(float(cand["lat"]), 2),
        "lon": _r(float(cand["lon"]), 2),
        "location": {
            k: loc.get(k)
            for k in (
                "country",
                "state",
                "headline_prefix",
                "place_phrase",
                "nearest_town",
                "nearest_town_km",
                "nearest_big_town",
                "nearest_big_town_km",
                "offshore",
            )
            if loc.get(k) not in (None, "")
        },
        "forecast": {
            "t2m_max_24h_c": _r(cand.get("t2m_max_24h_c")),
            "t2m_min_24h_c": _r(cand.get("t2m_min_24h_c")),
            "anomaly_c": _r(cand.get("anom_c")),
            "rain_24h_mm": _r(cand.get("rain_24h_mm")),
            "wind_kmh": _r(cand.get("wind_kmh", (cand.get("wind_ms") or 0) * 3.6), 0),
            "gust_kmh": _r(cand.get("gust_kmh", (cand.get("gust_ms") or 0) * 3.6), 0),
            "mslp_hpa": _r(cand.get("mslp_hpa"), 0),
        },
    }
    low = cand.get("low_centre")
    if low:
        out["forecast"]["nearby_low"] = {
            "mslp_hpa": _r(low.get("mslp_hpa"), 0),
            "distance_km": low.get("distance_km"),
            "direction": low.get("direction_from_hotspot"),
        }
    hist: dict = {}
    if era.get("rank_available", era.get("rank_since_1940") is not None) and era.get("rank_since_1940") is not None:
        hist.update(
            {
                "ranked": era.get("variable_ranked"),
                "rank_since_1940": era.get("rank_since_1940"),
                "clear_record": era.get("clear_record"),
                "previous_record_value": era.get("previous_record_value"),
                "previous_record_units": era.get("units"),
                "previous_record_year": era.get("previous_record_year"),
            }
        )
    if era.get("ranked_value_note"):
        hist["note"] = era["ranked_value_note"]
    for key_out, key_in in (("typical_high_c", "wmo_tmax_normal_c"), ("typical_low_c", "wmo_tmin_normal_c")):
        if era.get(key_in) is not None:
            hist[key_out] = era[key_in]
    if kind == "rain" or (cand.get("rain_24h_mm") or 0) >= 20:
        for k in ("daily_rain_normal_mm", "expected_rain_28d_mm", "rain_28d_mm", "rain_28d_pct_of_normal", "wet_days_ge_10mm_last_28"):
            if pers.get(k) is not None:
                hist[k] = pers[k]
    if kind == "heat":
        hist["warm_days_last_14_in_top_10pct"] = pers.get("warm_days_above_p90_tmax_last_14")
    elif kind == "cold":
        hist["cold_nights_last_14_in_bottom_10pct"] = pers.get("cold_days_below_p10_tmin_last_14")
    elif kind in ("cyclone", "wind"):
        hist["days_gusts_over_65kmh_last_28"] = pers.get("days_gust_over_65kmh_last_28")
        hist["local_top10pct_gust_kmh"] = pers.get("local_gust_p90_kmh")
    hist["warming_since_mid_century_c"] = shift.get("local_seasonal_warming_c_since_1940_1979")
    hist["frequency_claim_allowed"] = bool(shift.get("frequency_claim_allowed"))
    out["history"] = {k: v for k, v in hist.items() if v is not None}
    out["news_leads"] = [
        {"source": n.get("source"), "title": n.get("title"), "url": n.get("url")} for n in (cand.get("news_leads") or [])[:3]
    ]
    return out


def build_writer_payload(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    prev_summary: list[dict],
    fetch_news: bool = True,
    max_candidates: int = 8,
) -> dict:
    """JSON-serialisable brief for the writer: timing, slimmed candidates with place names, and news leads."""
    candidates = [dict(c) for c in enriched_candidates[:max_candidates]]
    if fetch_news:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=5) as pool:
            leads = list(pool.map(_safe_news, candidates))
        for cand, items in zip(candidates, leads):
            cand["news_leads"] = items
    return {
        "valid_time_utc": _iso(weather.valid_time),
        "rain_window_start_utc": _iso(weather.rain_window_start_utc or weather.run_time),
        "rain_window_end_utc": _iso(weather.rain_window_end_utc or weather.valid_time),
        "previous_headlines": [p.get("headline") for p in prev_summary if p.get("headline")],
        "active_disaster_alerts": [
            {k: w.get(k) for k in ("type", "name", "country", "alert_level", "url")} for w in warnings[:6]
        ],
        "candidate_hotspots": [slim_candidate(c) for c in candidates],
    }


def _safe_news(cand: dict) -> list[dict]:
    try:
        return news_for_candidate(cand, max_links=3)
    except Exception:
        return []


def complete_story_fields(draft: dict) -> None:
    """Fill the fields we no longer ask the model to write (to save output tokens): beats and ticker
    are built from the cues; camera waypoints and cue distances are generated by the validator."""
    for st in draft.get("stories") or []:
        if not isinstance(st, dict):
            continue
        cues = [c for c in (st.get("cues") or []) if isinstance(c, dict) and c.get("spoken")]
        if not st.get("beats") and len(cues) >= 4:
            layer = lambda i: cues[i].get("layer") or st.get("primary_layer") or "temperature"  # noqa: E731
            st["beats"] = [
                {"tag": "THE FORECAST", "title": "The forecast", "layer": layer(0), "text": cues[0]["spoken"]},
                {"tag": "WHAT IT MEANS", "title": "What it means", "layer": layer(1), "text": cues[1]["spoken"]},
                {"tag": "THE BACKGROUND", "title": "The background", "layer": layer(2), "text": " ".join(c["spoken"] for c in cues[2:-1])},
                {"tag": "IN CONTEXT", "title": "In context", "layer": layer(len(cues) - 1), "text": cues[-1]["spoken"]},
            ]
        if not st.get("ticker"):
            st["ticker"] = "  ///  ".join(str(c.get("ticker") or "") for c in cues if c.get("ticker"))


def _restrict_news_to_leads(draft: dict, lead_urls: set[str]) -> None:
    """Without web search, the writer may only cite the headlines we gave it: drop anything else."""
    for st in draft.get("stories") or []:
        if not isinstance(st, dict):
            continue
        st["news_links"] = [lk for lk in (st.get("news_links") or []) if isinstance(lk, dict) and lk.get("url") in lead_urls]
        st["news_facts"] = [nf for nf in (st.get("news_facts") or []) if isinstance(nf, dict) and nf.get("source_url") in lead_urls]


class UsageMeter:
    """Accumulate token usage across calls and estimate the cost of a run."""

    def __init__(self) -> None:
        self.by_model: dict[str, dict[str, float]] = {}

    def add(self, model: str, resp) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        m = self.by_model.setdefault(model, {"in": 0, "out": 0, "cw": 0, "cr": 0, "search": 0})
        m["in"] += getattr(u, "input_tokens", 0) or 0
        m["out"] += getattr(u, "output_tokens", 0) or 0
        m["cw"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        m["cr"] += getattr(u, "cache_read_input_tokens", 0) or 0
        stu = getattr(u, "server_tool_use", None)
        m["search"] += (getattr(stu, "web_search_requests", 0) or 0) if stu is not None else 0

    def cost_usd(self) -> float:
        total = 0.0
        for model, m in self.by_model.items():
            p_in, p_out, p_cw, p_cr = MODEL_PRICES.get(model, MODEL_PRICES["claude-sonnet-5-5"])
            total += (m["in"] * p_in + m["out"] * p_out + m["cw"] * p_cw + m["cr"] * p_cr) / 1e6
            total += m["search"] * WEB_SEARCH_PRICE
        return total

    def summary(self) -> str:
        parts = [
            f"{model}: {int(m['in'] + m['cw'] + m['cr'])} in ({int(m['cr'])} cached), {int(m['out'])} out, {int(m['search'])} searches"
            for model, m in self.by_model.items()
        ]
        return "; ".join(parts) + f" = about ${self.cost_usd():.3f}"


def generate_headlines_with_claude(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    previous_bulletin: dict | None = None,
) -> dict:
    """Ask Claude to write the 5-story bulletin from the candidates and news leads, validate it, and
    repair or replace stories that fail. Logs token use and an estimated cost."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    preferred = os.environ.get("ANTHROPIC_MODEL", "").strip()
    models_to_try = [preferred] if preferred else DEFAULT_CLAUDE_MODELS

    prev_summary = [
        {"headline": s.get("headline")}
        for s in ((previous_bulletin or {}).get("stories") or [])[:5]
        if isinstance(s, dict)
    ]
    user_payload = build_writer_payload(weather, enriched_candidates, warnings, prev_summary)
    lead_urls = {
        n["url"] for c in user_payload["candidate_hotspots"] for n in c.get("news_leads") or [] if n.get("url")
    } | {w["url"] for w in user_payload["active_disaster_alerts"] if w.get("url")}

    searches = max(0, WEB_SEARCHES_PER_REQUEST)
    system_text = SYSTEM_PROMPT + (NEWS_NOTE_WITH_SEARCH if searches else NEWS_NOTE_LEADS_ONLY)
    brief = (
        "Write today's bulletin from this data. Start every headline and first cue with the place's "
        "`headline_prefix`. Output ONLY the JSON object.\n\n"
        f"<data>{json.dumps(user_payload, separators=(',', ':'), default=str, ensure_ascii=False)}</data>"
    )
    meter = UsageMeter()

    try:
        last_exc: Exception | None = None
        for model_name in models_to_try:
            try:
                print(
                    f"[news_agent] Calling {model_name} (effort {WRITER_EFFORT}, {searches} web searches)...",
                    file=sys.stderr,
                )
                # Cache the instructions and the brief: a retry then reads them at a tenth of the price.
                system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
                messages: list[dict] = [
                    {"role": "user", "content": [{"type": "text", "text": brief, "cache_control": {"type": "ephemeral"}}]}
                ]
                # Keep `tools` identical for every request in this conversation: newer models bind their
                # thinking blocks to the exact tool list, and reject a replay if it changes.
                extra: dict = {"output_config": {"effort": WRITER_EFFORT}}
                if searches:
                    extra["tools"] = [_web_search_tool(model_name, searches)]

                def _call(msgs: list[dict]):
                    with client.messages.stream(
                        model=model_name, max_tokens=32000, system=system, messages=msgs, **extra
                    ) as stream:
                        resp = stream.get_final_message()
                    meter.add(model_name, resp)
                    return resp

                last_cleaned: dict | None = None
                last_errors: list[str] = []
                for attempt in range(MAX_ATTEMPTS):
                    resp = _call(messages)
                    pauses = 0
                    while getattr(resp, "stop_reason", None) == "pause_turn" and pauses < 2:
                        pauses += 1
                        messages = messages + [{"role": "assistant", "content": resp.content}]
                        resp = _call(messages)
                    if getattr(resp, "stop_reason", None) == "refusal":
                        raise RuntimeError(f"{model_name} declined the request (stop_reason=refusal)")

                    text = "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                    messages = messages + [{"role": "assistant", "content": resp.content}]
                    try:
                        draft = _extract_json_object(text)
                    except (ValueError, json.JSONDecodeError) as parse_exc:
                        print(
                            f"[news_agent] JSON parse error on attempt {attempt + 1} "
                            f"(stop_reason={getattr(resp, 'stop_reason', None)}): {parse_exc}",
                            file=sys.stderr,
                        )
                        messages.append(
                            {"role": "user", "content": f"That was not complete valid JSON ({parse_exc}). Return ONLY the complete JSON object."}
                        )
                        continue

                    complete_story_fields(draft)
                    if not searches:
                        _restrict_news_to_leads(draft, lead_urls)
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
                        draft, candidates=enriched_candidates, verify_urls=True, require_links=False
                    )
                    if not errors:
                        return cleaned
                    last_cleaned, last_errors = cleaned, errors
                    print(f"[news_agent] Validation issues on attempt {attempt + 1}: {errors}", file=sys.stderr)
                    if attempt + 1 < MAX_ATTEMPTS:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Fix every problem below and return ONLY the corrected JSON object "
                                    "(all five stories):\n" + "\n".join(f"- {e}" for e in errors)
                                ),
                            }
                        )
                if last_cleaned is not None:
                    salvaged = salvage_bulletin(last_cleaned, last_errors, weather, enriched_candidates)
                    if salvaged is not None:
                        return salvaged
                raise RuntimeError(f"Claude output failed validation after {MAX_ATTEMPTS} attempts on {model_name}")
            except anthropic.AuthenticationError as auth_exc:
                raise RuntimeError(f"Anthropic authentication failed (check ANTHROPIC_API_KEY secret): {auth_exc}") from auth_exc
            except anthropic.BadRequestError as bad_exc:
                msg_low = str(bad_exc).lower()
                if "credit balance" in msg_low or "billing" in msg_low:
                    raise RuntimeError(
                        f"Anthropic API credit balance is empty (add prepaid API credits at console.anthropic.com): {bad_exc}"
                    ) from bad_exc
                last_exc = bad_exc
                print(f"[news_agent] {model_name} BadRequestError: {bad_exc}", file=sys.stderr)
            except anthropic.RateLimitError as rate_exc:
                last_exc = rate_exc
                print(f"[news_agent] {model_name} hit a rate limit; trying the next model: {rate_exc}", file=sys.stderr)
                time.sleep(10)
            except Exception as exc:
                last_exc = exc
                print(f"[news_agent] {model_name} failed: {exc}", file=sys.stderr)
        raise RuntimeError(f"All Claude models failed: {last_exc}")
    finally:
        if meter.by_model:
            print(f"[news_agent] Claude usage: {meter.summary()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# News leads (Google News RSS from reputable outlets), template fallback & salvage
# ---------------------------------------------------------------------------


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
    "jiji.com",
    "efe.com",
    "aa.com.tr",
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


# Low-quality, aggregator or non-news sites that should never be cited even on a national domain
_UNRELIABLE_SOURCES = ("nature world news", "ftn news", "oilprice", "audacy", "aol", "msn", "britannica", "wikipedia")


def _is_reputable_source(source_url: str, source_name: str, country_domain: str = "") -> bool:
    """True for reputable international outlets and agencies, and for news sites on the story country's
    own domain (e.g. `.vn` papers for a Vietnam story), which usually have the best local detail."""
    s_name_low = (source_name or "").lower()
    if any(bad in s_name_low for bad in _UNRELIABLE_SOURCES):
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
    if country_domain and len(country_domain) == 2 and host.endswith("." + country_domain):
        return True
    return any(host == dom or host.endswith("." + dom) for dom in REPUTABLE_PUBLISHER_DOMAINS)


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
    state: str = "",
    country_domain: str = "",
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
            'heatwave OR "high temperatures" OR "warm weather" OR "mild weather" OR "record temperatures"',
            r"\b(heatwave|heat|hot|temperatures?|mild|warm\w*|wildfires?)\b",
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

    # Keep queries concise (well under Google News's 32-word operator limit) and filter publisher domains in Python
    # Only recent, place-specific stories: anything older than about a week is rarely about today's weather
    queries = []
    if state:
        queries.append((f'"{state}" ({hazard_query}) when:7d', (state.lower(),)))
    queries += [
        (f'"{country}" ({hazard_query}) when:7d', country_tokens),
        (f'"{country}" ({hazard_query}) when:14d', country_tokens),
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
            if not _is_reputable_source(source_url, source, country_domain):
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


def news_for_candidate(cand: dict, max_links: int = 3) -> list[dict]:
    """Recent reputable news headlines about the weather where `cand` is (used as leads and fallback links)."""
    loc = cand.get("location") or {}
    country = loc.get("country") or cand.get("country") or ""
    if not country:
        return []
    return fetch_google_news_rss_links(
        country,
        cand["kind"],
        max_links=max_links,
        lat=float(cand["lat"]),
        lon=float(cand["lon"]),
        state=loc.get("state") or "",
        country_domain=loc.get("country_domain") or "",
    )


def build_deterministic_bulletin(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    fetch_live_rss: bool = False,
) -> dict:
    """Template-based bulletin used when Claude is unavailable (see `pipeline.fallback_writer`)."""
    from .fallback_writer import build_fallback_bulletin
    from .places import describe_location

    for cand in enriched_candidates:
        if "location" not in cand:
            cand["location"] = describe_location(float(cand["lat"]), float(cand["lon"]))
    return build_fallback_bulletin(
        weather,
        enriched_candidates,
        warnings,
        news_lookup=news_for_candidate if fetch_live_rss else None,
    )


def salvage_bulletin(
    cleaned: dict,
    errors: list[str],
    weather: LiveWeatherFields,
    candidates: list[dict],
    max_replaced: int = 3,
) -> dict | None:
    """Keep the stories that passed validation and replace only the failing ones with template stories.

    Returns None when the draft is unusable (bulletin-level errors or too many failing stories).
    """
    from .fallback_writer import PhrasePicker, build_fallback_story

    stories = list(cleaned.get("stories") or [])
    bad: set[int] = set()
    for err in errors:
        m = re.match(r"stories\[(\d+)\]", err)
        if not m:
            return None
        bad.add(int(m.group(1)))
    if len(stories) < 5 or len(bad) > max_replaced:
        return None

    good = [st for i, st in enumerate(stories) if i not in bad]
    from .scanner import haversine_km

    unused = [
        c
        for c in candidates
        if all(haversine_km(float(c["lat"]), float(c["lon"]), st["centre"]["lat"], st["centre"]["lon"]) > 550 for st in good)
    ]
    picker = PhrasePicker(seed=f"{weather.valid_time:%Y%m%d%H}-salvage")
    for i in sorted(bad):
        if not unused:
            return None
        cand = unused.pop(0)
        stories[i] = build_fallback_story(cand, weather, picker, _safe_news(cand), rank=i + 1)
    draft = dict(cleaned, stories=stories)
    fixed, errs = validate_and_clean_bulletin(draft, candidates=candidates, verify_urls=False)
    if errs:
        return None
    print(f"[news_agent] Published Claude bulletin with {len(bad)} template story(ies) replacing failed ones.", file=sys.stderr)
    return fixed
