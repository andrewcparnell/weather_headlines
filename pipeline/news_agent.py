"""Claude writer for the 6-hourly world weather headlines (plus news-lead lookup and template fallback)."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

from .fallback_writer import fmt_2sf  # noqa: F401  (re-exported for callers and tests)
from .opendata import LiveWeatherFields
from .validator import validate_and_clean_bulletin

# Tried in order; override with the ANTHROPIC_MODEL environment variable (or repository variable in CI).
# Sonnet 5.5 at low effort without web search costs roughly $0.05-0.10 per run; Haiku 5.5 about a tenth of that.
DEFAULT_CLAUDE_MODELS = [
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
]

# Web searches are only used to top up stories that the free news lookup found fewer than 3 articles for.
# Each costs $0.01 plus the tokens of the results it reads (roughly $0.02-0.04 in all). Set 0 to disable.
# (An unset GitHub repository variable arrives as an empty string, which means "use the default".)
WEB_SEARCHES_PER_REQUEST = int(os.environ.get("WEATHER_WEB_SEARCHES", "").strip() or 4)
NEWS_PER_STORY = 3

# Effort controls how much the model thinks before writing; "low" keeps output tokens (the main cost) down.
WRITER_EFFORT = os.environ.get("WEATHER_WRITER_EFFORT", "low").strip() or "low"

MAX_ATTEMPTS = 3  # retries ask only for the failing stories; if they still fail, the run fails (no template text)

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
4. `history.rank_since_1940` (when present) is the rank of that 24 h value for this time of year since 1940 at that point. Only when it is 1 AND `history.clear_record` is true say "the highest/wettest/strongest for this time of year since 1940" (you may add the previous high, `previous_record_value` in `previous_record_year`). When it is 1 but `clear_record` is false, say "among the highest for this time of year since 1940", and never "beating" or "a new high". With no rank, make no ranking claim. Never say "ever recorded" or "record" (unless quoting an official station record from the news).
5. Rain normals: use only the normals in `history`. Wet days mean days with at least 10 mm. 65 km/h gusts are not "storm force".
6. Climate trend: `warming_since_mid_century_c` may be null; then say nothing about the local trend. Only claim a change in how often such events happen when `frequency_claim_allowed` is true.
7. Never use: "unprecedented", "ever recorded", "heat dome", "baking", "historic", "climate emergency", "shatters". Below 5 °C a warm anomaly is "exceptionally mild", not "heat" or "heatwave".
8. Never mention data sources or jargon in the prose (ECMWF, ERA5, GDACS, reanalysis, grid point, model run) and never read out year ranges like "1991-2020". Never imply an affiliation with any broadcaster.
9. British English. No em dashes (use commas, colons or spaced hyphens). Never "utilise".

PLACE NAMES IN HEADLINES (important)
- `headline` and `cues[0].spoken` MUST start with the candidate's `location.headline_prefix` followed by a colon, for example "Oklahoma, USA: ...", "Krasnoyarsk Krai, Russia: ..." or "Vietnam: ...".
- `country` must be `location.country`.

THE STORIES
- Write exactly one story for each entry in `stories_to_write`, in the same order (they have already been chosen). If a story continues one in `previous_headlines`, find a fresh angle.
- Temperatures: compare like with like. Use `history.high_vs_typical_c` (forecast high minus the typical high) or `history.low_vs_typical_c` (forecast low minus the typical low) for any "warmer/colder than usual" claim. Do not compute differences yourself.

OUTPUT: a single compact JSON object `{"stories": [...]}` and nothing else. Each story has ONLY these keys:
- `id`: short slug, e.g. "2026-11-central-vietnam-floods"
- `kicker`: 2 to 4 uppercase words naming the hazard and place, e.g. "FLOODS · CENTRAL VIETNAM"
- `headline`: 45 to 90 characters, starting with `location.headline_prefix` and a colon, specific and engaging (not a bare number)
- `region`, `country`, `hazard` (heat|cold|rain|cyclone|wind|pressure), `primary_layer` (temperature|anomaly|wind|rain|pressure)
- `centre`: {"lat", "lon"} copied from the story's entry
- `stats`: {"headline_metric", "metric_label", "era5_rank", "persistence"}: very short card text; "" where you have no data
- `cues`: exactly 5 spoken sentences, each {"layer", "lat", "lon", "label" (short place or theme), "metric" (short stat or ""), "spoken" (16 to 30 words, read aloud word for word, no asterisks), "ticker" (8 to 14 words with key figures in **double asterisks**)}. Put each cue where it is about (within 500 km of `centre`, within 150 km of any city it names) and set its layer to the variable it discusses.
- `news_facts`: [{"claim", "source_url"}] for every fact taken from the news (the claim must contain any number you quote)
- `news_links`: 3 articles about this story [{"source", "title", "url"}] (fewer only if fewer exist)"""

NEWS_NOTE_LEADS_ONLY = """

NEWS FOR EACH STORY
`news_leads` lists up to 3 recent articles about each story's area (headline, outlet, date and, where available, a summary). Use them: build at least one or two sentences of each story around what they report (who is affected, what is closed, what officials are saying), attributed to the outlet, and list every lead in `news_links`. They are your ONLY news source: do not add news facts, casualty figures or impacts from memory."""

NEWS_NOTE_WITH_SEARCH = """

NEWS FOR EACH STORY
`news_leads` lists recent articles about each story's area (headline, outlet, date and, where available, a summary). Use them: build at least one or two sentences of each story around what they report (who is affected, what is closed, what officials are saying), attributed to the outlet, and list every lead in `news_links`.
Some stories have `needs_more_news` > 0: for those only, use `web_search` (include the month and year) to find that many more recent articles about this weather in that place, from reputable outlets, and use and cite them too. Cite only articles you actually saw, with direct article URLs (never section fronts or tag pages). Do not search for stories that already have 3 leads."""


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
    if kind == "rain":
        # Rain is ranked on an ~80 km area average, so the old total is not comparable with the headline
        # peak: give the year only, never the number.
        hist.pop("previous_record_value", None)
        hist.pop("previous_record_units", None)
    if era.get("ranked_value_note"):
        hist["note"] = era["ranked_value_note"]
    for key_out, key_in in (
        ("typical_high_c", "wmo_tmax_normal_c"),
        ("typical_low_c", "wmo_tmin_normal_c"),
        ("high_vs_typical_c", "high_vs_typical_c"),
        ("low_vs_typical_c", "low_vs_typical_c"),
    ):
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
    leads = (cand.get("news_leads") or [])[:NEWS_PER_STORY]
    out["news_leads"] = [
        {k: n.get(k) for k in ("source", "date", "title", "summary", "url") if n.get(k)} for n in leads
    ]
    if len(leads) < NEWS_PER_STORY:
        out["needs_more_news"] = NEWS_PER_STORY - len(leads)
    return out


def build_writer_payload(
    weather: LiveWeatherFields,
    stories: list[dict],
    warnings: list[dict],
    prev_summary: list[dict],
    fetch_news: bool = True,
    max_candidates: int = 5,
) -> dict:
    """JSON-serialisable brief for the writer: timing, the chosen stories with place names, and their news."""
    candidates = [dict(c) for c in stories[:max_candidates]]
    missing = [c for c in candidates if "news_leads" not in c]
    if fetch_news and missing:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=5) as pool:
            for cand, items in zip(missing, pool.map(_safe_news, missing)):
                cand["news_leads"] = items
    return {
        "valid_time_utc": _iso(weather.valid_time),
        "rain_window_start_utc": _iso(weather.rain_window_start_utc or weather.run_time),
        "rain_window_end_utc": _iso(weather.rain_window_end_utc or weather.valid_time),
        "previous_headlines": [p.get("headline") for p in prev_summary if p.get("headline")],
        "stories_to_write": [slim_candidate(c) for c in candidates],
    }


def _safe_news(cand: dict) -> list[dict]:
    from .news_leads import gather_news

    try:
        return gather_news(cand)
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


class WriterValidationError(RuntimeError):
    """The writer's stories kept failing the fact check: use template text rather than paying for another model."""


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


def check_story_news(stories: list[dict], payload: dict) -> list[str]:
    """Each story must cite at least 3 working articles, or all of its leads if fewer than 3 were found.

    (Links that fail to load are removed by the validator, so a dead lead the writer rightly skipped
    never blocks the bulletin.)"""
    errs = []
    briefs = payload.get("stories_to_write") or []
    for i, (st, brief) in enumerate(zip(stories, briefs)):
        lead_urls = [n["url"] for n in brief.get("news_leads") or [] if n.get("url")]
        cited = [lk.get("url") for lk in st.get("news_links") or [] if isinstance(lk, dict) and lk.get("url")]
        needed = min(NEWS_PER_STORY, len(lead_urls))
        if len(cited) < needed:
            unused = [u for u in lead_urls if u not in cited]
            errs.append(
                f"stories[{i}]: cite at least {needed} news articles in news_links (only {len(cited)} working). "
                f"Unused leads: {unused}."
            )
    return errs


def _merge_fixed_stories(draft: dict, fix: dict) -> None:
    """Apply a retry answer: either {"stories": {"<index>": story, ...}} or a full {"stories": [...]} list."""
    fixed = fix.get("stories")
    if isinstance(fixed, list) and len(fixed) == len(draft["stories"]):
        draft["stories"] = fixed
    elif isinstance(fixed, dict):
        for key, story in fixed.items():
            try:
                idx = int(str(key).strip("[]"))
            except ValueError:
                continue
            if 0 <= idx < len(draft["stories"]) and isinstance(story, dict):
                draft["stories"][idx] = story


def generate_headlines_with_claude(
    weather: LiveWeatherFields,
    stories: list[dict],
    warnings: list[dict],
    previous_bulletin: dict | None = None,
) -> dict:
    """Ask Claude to write the chosen stories, validate them, and ask for fixes to any that fail.

    Raises if the bulletin still fails after MAX_ATTEMPTS: the pipeline then fails loudly instead of
    publishing template text. Logs token use and an estimated cost.
    """
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
    user_payload = build_writer_payload(weather, stories, warnings, prev_summary)
    lead_urls = {n["url"] for c in user_payload["stories_to_write"] for n in c.get("news_leads") or [] if n.get("url")}
    shortfall = sum(c.get("needs_more_news", 0) for c in user_payload["stories_to_write"])
    searches = min(max(0, WEB_SEARCHES_PER_REQUEST), shortfall)

    system_text = SYSTEM_PROMPT + (NEWS_NOTE_WITH_SEARCH if searches else NEWS_NOTE_LEADS_ONLY)
    brief = (
        "Write the bulletin from this data. Start every headline and first cue with the story's "
        "`headline_prefix`. Output ONLY the JSON object.\n\n"
        f"<data>{json.dumps(user_payload, separators=(',', ':'), default=str, ensure_ascii=False)}</data>"
    )
    meter = UsageMeter()

    try:
        last_exc: Exception | None = None
        for model_name in models_to_try:
            try:
                print(
                    f"[news_agent] Calling {model_name} (effort {WRITER_EFFORT}, up to {searches} web searches)...",
                    file=sys.stderr,
                )
                # Cache the instructions and the brief: retries then read them at a tenth of the price.
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

                draft: dict | None = None
                errors: list[str] = []
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
                        answer = _extract_json_object(text)
                    except (ValueError, json.JSONDecodeError) as parse_exc:
                        errors = [f"Reply was not complete valid JSON ({parse_exc})."]
                        print(f"[news_agent] Attempt {attempt + 1}: {errors[0]}", file=sys.stderr)
                        messages.append({"role": "user", "content": errors[0] + " Return ONLY the complete JSON object."})
                        continue

                    if draft is None or not isinstance(answer.get("stories"), dict):
                        draft = answer if isinstance(answer.get("stories"), list) else {"stories": []}
                    else:
                        _merge_fixed_stories(draft, answer)
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
                        draft, candidates=stories, verify_urls=True, require_links=False
                    )
                    if not any(e.startswith("Expected 5 stories") for e in errors):
                        errors += check_story_news(cleaned.get("stories") or [], user_payload)
                    if not errors:
                        return cleaned
                    print(f"[news_agent] Validation issues on attempt {attempt + 1}: {errors}", file=sys.stderr)
                    bad = sorted({int(m.group(1)) for e in errors if (m := re.match(r"stories\[(\d+)\]", e))})
                    if bad and all(re.match(r"stories\[\d+\]", e) for e in errors):
                        # Only the failing stories come back, which keeps retries cheap
                        request = (
                            "Fix every problem below. Return ONLY a JSON object of the corrected stories, keyed by "
                            'their index, for example {"stories": {"2": {...}}}. Include just these stories: '
                            + ", ".join(str(i) for i in bad)
                            + ".\n"
                        )
                    else:
                        request = "Fix every problem below and return ONLY the complete corrected JSON object (all stories).\n"
                    messages.append({"role": "user", "content": request + "\n".join(f"- {e}" for e in errors)})
                raise WriterValidationError(
                    f"{model_name} could not produce a bulletin that passes the fact check after "
                    f"{MAX_ATTEMPTS} attempts. Remaining problems: {errors}"
                )
            except WriterValidationError:
                raise
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


def build_deterministic_bulletin(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    fetch_live_rss: bool = False,
) -> dict:
    """Template-based bulletin, for local development and tests only (`--template` flag).

    Production never publishes template text: if Claude fails, the pipeline fails.
    """
    from .fallback_writer import build_fallback_bulletin
    from .places import describe_location

    for cand in enriched_candidates:
        if "location" not in cand:
            cand["location"] = describe_location(float(cand["lat"]), float(cand["lon"]))
    return build_fallback_bulletin(
        weather,
        enriched_candidates,
        warnings,
        news_lookup=_safe_news if fetch_live_rss else None,
    )
