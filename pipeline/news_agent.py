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

from .config import USER_AGENT
from .opendata import LiveWeatherFields
from .fallback_writer import fmt_2sf  # noqa: F401  (re-exported for callers and tests)
from .validator import is_specific_event_url, validate_and_clean_bulletin

# Tried in order; override with the ANTHROPIC_MODEL environment variable (or repository variable in CI)
DEFAULT_CLAUDE_MODELS = [
    "claude-opus-5-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]


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


SYSTEM_PROMPT = """You are the writer and presenter of a short, 6-hourly world weather bulletin: five stories, each read aloud over a 3D globe in about 45 seconds.

Your job is to tell viewers what is happening to PEOPLE and PLACES because of the weather, in a way that is warm, vivid, specific and true. Think of a well-loved television weather presenter who has also read the morning's news: curious about the places they visit, quick to explain why something matters, never breathless, never dull.

WHAT GOOD LOOKS LIKE
- Lead with the human angle whenever the news gives you one: farmers racing to bring in a harvest, schools closed, ferries cancelled, a city clearing drains, a ski season that cannot start, a fishing fleet kept in port. Use `news_leads` and your own `web_search` to find these, and attribute them ("local media report...", "Vietnam's disaster agency says...").
- Paint the place: name the nearest town or city from `location`, say what kind of place it is (a coffee-growing highland, a delta of rice paddies, an Arctic oil town) when you are confident it is true.
- Explain what the numbers mean in everyday terms: "more rain in a day than usually falls here in a month", "warm enough for T-shirts in a city that normally has snow on the ground by now", "gusts strong enough to bring down trees".
- Give each story its own shape. Pick the two or three most interesting angles for THAT story (impacts, how unusual it is, what happens next, how long it has gone on, the long-term trend) rather than marching through the same checklist every time. Not every story needs a persistence line, a historical rank or a climate line.
- Vary your language across the bulletin. Never open two stories the same way, never reuse a stock phrase (for example "tilting the odds", "staying power", "a surge of", "since the mid-twentieth century") more than once per bulletin, and vary sentence length.
- Keep the tone calm and grounded. Vivid is good; hype is not.

ACCURACY RULES (a validator checks these; failing stories are replaced with plain template text)
1. Model numbers are FORECASTS valid at `valid_time_utc`; the 24 h rain total covers `rain_window_start_utc` to `rain_window_end_utc`. Use forecast wording ("up to 140 mm is forecast", "gusts could reach 120 km/h"). Never "fell", "has recorded" or "measured" for model values.
2. Every number with a unit (°C, mm, hPa, km/h, %, "N of M days", "#N", "N times the average") must either match the candidate's data (rounded to 2 significant figures; percentages to whole numbers) or appear in that story's `news_facts` with its `source_url`. Numbers from the news must be quoted as reported, with attribution and date where relevant, and never stretched (a figure about another place, another week or another country is not about this event). Call early estimates "preliminary" (for example daily global temperature figures), and when sources disagree give the range or leave the figure out.
3. Candidate statistics describe ONLY the point at `lat`/`lon`. `location` tells you the country, state/province, nearest town and distance. Never transfer a local normal or rank to a city more than 150 km away.
4. `era5.rank_since_1940` (when `era5.rank_available` is true) is the rank of that 24 h value for this time of year (±7 days) since 1940 at that point. Only when it is 1 AND `era5.clear_record` is true say "the highest/wettest/strongest for this time of year since 1940" (you may add the previous high, `previous_record_value` in `previous_record_year`). When it is 1 but `clear_record` is false, the forecast only edges past the old high, so say "among the highest for this time of year since 1940". Never say "ever recorded" or "record" (unless quoting an official station record from the news). When `rank_available` is false there is NO rank: do not invent one, and do not use percentiles like "top 1%".
5. Rain normals: use only `expected_rain_14d_mm`, `expected_rain_28d_mm`, `rain_28d_pct_of_normal` and `daily_rain_normal_mm` from `era5.persistence`. Wet days mean days with at least 10 mm. 65 km/h gusts are not "storm force".
6. Climate trend: `local_seasonal_warming_c_since_1940_1979` may be null; then say nothing about the local trend. 40-year exceedance counts are "years out of 40", never "per decade"; only claim a change in frequency when `frequency_claim_allowed` is true.
7. Never use: "unprecedented", "ever recorded", "heat dome", "baking", "historic", "climate emergency", "shatters". Below 5 °C a warm anomaly is "exceptionally mild", not "heat" or "heatwave".
8. Never mention data sources or jargon in the prose (ECMWF, ERA5, GDACS, reanalysis, grid point, model run) and never read out year ranges like "1991-2020". Never imply an affiliation with any broadcaster.
9. British English. No em dashes (use commas, colons or spaced hyphens). Never "utilise".

PLACE NAMES IN HEADLINES (important)
- `headline` and `cues[0].spoken` MUST start with the candidate's `location.headline_prefix` followed by a colon. For large countries this already includes the state or province, for example "Oklahoma, USA: ..." or "Krasnoyarsk Krai, Russia: ...". For other countries it is just the country, for example "Vietnam: ...".
- `country` must be the sovereign country from `location.country`.

CHOOSING STORIES
- Pick 5 geographically varied stories from `candidate_hotspots`, preferring places where people are affected and the news confirms impacts. Skip temperature anomalies over open sea or empty ice. Keep a big ongoing story from `previous_headlines` if it is still active, with fresh angles.

OUTPUT: a single JSON object `{"stories": [...]}`. Each story:
- `id`: short slug, e.g. "2026-11-central-vietnam-floods"
- `kicker`: 2 to 4 uppercase words naming the hazard and place, e.g. "FLOODS · CENTRAL VIETNAM"
- `headline`: 45 to 90 characters, starting with `location.headline_prefix` and a colon, specific and engaging (not a bare number)
- `ticker`: 60 to 110 words of scrolling summary, key facts in `**double asterisks**`, clauses separated by ` /// `
- `region`, `country`, `hazard` (heat|cold|rain|cyclone|wind|pressure), `primary_layer` (temperature|anomaly|wind|rain|pressure)
- `centre`: {"lat", "lon"} of the chosen candidate
- `stats`: {"headline_metric", "metric_label", "era5_rank", "persistence"}, short card text; use "" for any item you have no data for
- `camera_waypoints`: 3 or 4 points within 500 km of `centre` (`distance` 1.48 to 2.05)
- `cues`: exactly 5 spoken sentences, each {"layer", "lat", "lon", "distance" (1.48 to 2.02), "bearing" (-12 to 12), "label" (short place or theme), "metric" (short stat or ""), "spoken" (16 to 30 words, read aloud word for word, no asterisks), "ticker" (8 to 14 words with **highlights**)}. Put each cue where it is about (within 500 km of `centre`, within 150 km of any city it names) and set its layer to the variable it discusses.
- `beats`: 4 short paragraphs {"tag", "title", "layer", "text"} summarising the story; choose tags that fit the story (for example "WHAT'S HAPPENING", "WHO'S AFFECTED", "HOW UNUSUAL", "WHAT NEXT")
- `news_facts`: [{"claim", "source_url"}] for every fact or figure taken from the news (the claim must contain any number you quote)
- `news_links`: 1 to 3 direct article URLs about this event [{"source", "title", "url"}] (never section fronts or tag pages)

Return ONLY the JSON object, with no prose before or after it."""


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


def build_writer_payload(
    weather: LiveWeatherFields,
    enriched_candidates: list[dict],
    warnings: list[dict],
    prev_summary: list[dict],
    fetch_news: bool = True,
) -> dict:
    """JSON-serialisable brief for the writer: timing, candidates with place names, and news leads."""
    candidates = [dict(c) for c in enriched_candidates[:10]]
    if fetch_news:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=5) as pool:
            leads = list(pool.map(lambda c: _safe_news(c), candidates))
        for cand, items in zip(candidates, leads):
            cand["news_leads"] = items
    return {
        "current_utc": _iso(datetime.now(timezone.utc)),
        "model_run_utc": _iso(weather.run_time),
        "valid_time_utc": _iso(weather.valid_time),
        "rain_window_start_utc": _iso(weather.rain_window_start_utc or weather.run_time),
        "rain_window_end_utc": _iso(weather.rain_window_end_utc or weather.valid_time),
        "previous_headlines": prev_summary,
        "active_disaster_alerts": warnings[:12],
        "candidate_hotspots": candidates,
    }


def _safe_news(cand: dict) -> list[dict]:
    try:
        return news_for_candidate(cand, max_links=4)
    except Exception:
        return []


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

    user_payload = build_writer_payload(weather, enriched_candidates, warnings, prev_summary)
    prompt = (
        "Here are today's forecast hotspots, with their place names, recent weather, historical context "
        "and a few recent news headlines from each area (`news_leads`).\n\n"
        "1. Use `web_search` (include the month and year, "
        f"`{weather.valid_time:%B %Y}`) to find out what is actually happening to people in the most promising "
        "places: warnings, disruption, damage, how communities are coping.\n"
        "2. Choose 5 varied stories and write them so a viewer learns something about each place and why the "
        "weather matters there. Start every headline and first cue with the place's `headline_prefix`.\n"
        "3. Output ONLY the JSON object.\n\n"
        f"<data>\n{json.dumps(user_payload, indent=2, default=str, ensure_ascii=False)}\n</data>"
    )

    last_exc: Exception | None = None
    for model_name in models_to_try:
        try:
            print(f"[news_agent] Calling Anthropic model {model_name} with web search...", file=sys.stderr)
            messages: list[dict] = [{"role": "user", "content": prompt}]
            last_cleaned: dict | None = None
            last_errors: list[str] = []

            # Keep `tools` identical for every request in this conversation: newer models bind their
            # thinking blocks to the exact tool list, and reject a replay if it changes.
            tools = [_web_search_tool(model_name, 8)]

            def _call(msgs: list[dict]):
                # Stream so a long answer (thinking + five stories of JSON) can't hit HTTP timeouts
                with client.messages.stream(
                    model=model_name,
                    max_tokens=48000,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=msgs,
                ) as stream:
                    return stream.get_final_message()

            for attempt in range(3):
                resp = _call(messages)

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
                                '{"stories": [...]} with no prose outside the JSON.'
                            ),
                        }
                    )
                    resp = _call(messages)
                    accumulated_text_parts.extend(
                        block.text for block in resp.content if getattr(block, "type", None) == "text"
                    )

                if getattr(resp, "stop_reason", None) == "refusal":
                    raise RuntimeError(f"{model_name} declined the request (stop_reason=refusal)")

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
                last_cleaned, last_errors = cleaned, errors
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
            if last_cleaned is not None:
                salvaged = salvage_bulletin(last_cleaned, last_errors, weather, enriched_candidates)
                if salvaged is not None:
                    return salvaged
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
