"""Validate and sanitise headlines.json before publishing to the static site."""
from __future__ import annotations

import json
import math
import re
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from .config import LAYERS, SITE_DIR, USER_AGENT
from .scanner import haversine_km

VALID_LAYERS = set(LAYERS.keys())
VALID_HAZARDS = {"heat", "cold", "rain", "cyclone", "wind", "pressure"}
FORBIDDEN_URL_EXTENSIONS = (".json", ".xml", ".cap", ".rss", ".atom", ".grib", ".grib2")

GENERIC_HUB_PATHS = {
    "",
    "/",
    "/weather",
    "/news",
    "/world",
    "/environment",
    "/world/extreme-weather",
    "/environment/extreme-weather",
    "/business/environment",
    "/climate-bulletins",
    "/media/news",
    "/warnings",
    "/warnings/home",
}
GENERIC_PATH_PREFIXES = (
    "/tag/",
    "/tags/",
    "/topics/",
    "/topic/",
    "/series/",
    "/category/",
)

PAYWALL_OR_WAF_DOMAINS = (
    "reuters.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "telegraph.co.uk",
    "thetimes.com",
    "economist.com",
    "apnews.com",
    "aljazeera.com",
    "axios.com",
    "forbes.com",
)

COUNTRY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "united states": ("united states", "u.s.", "america"),
    "united kingdom": (
        "united kingdom",
        "u.k.",
        "britain",
        "british",
        "england",
        "scotland",
        "wales",
        "northern ireland",
    ),
    "russia": ("russia", "russian"),
    "new zealand": ("new zealand",),
    "south africa": ("south africa",),
}

LOWERCASE_LEAD_WORDS = {
    "a",
    "an",
    "the",
    "at",
    "in",
    "on",
    "off",
    "over",
    "across",
    "along",
    "around",
    "near",
    "from",
    "throughout",
    "beneath",
    "amid",
    "heavy",
    "torrential",
    "fierce",
    "relentless",
    "extreme",
    "record",
    "intense",
    "severe",
    "powerful",
    "deep",
    "unprecedented",
    "continental",
    "polar",
    "tropical",
    "exceptionally",
}

_AMBIGUOUS_CITY_ALIASES = {"la", "hk", "rio", "nyc", "cdmx"}
_GAZETTEER_CITIES: list[dict] | None = None


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s or "") if unicodedata.category(ch) != "Mn")


def _load_gazetteer_cities() -> list[dict]:
    global _GAZETTEER_CITIES
    if _GAZETTEER_CITIES is not None:
        return _GAZETTEER_CITIES
    gaz_path = SITE_DIR / "assets" / "gazetteer.json"
    if not gaz_path.exists():
        _GAZETTEER_CITIES = []
        return _GAZETTEER_CITIES
    try:
        raw = json.loads(gaz_path.read_text())
        cities = []
        for entry in raw:
            if entry.get("kind") != "City":
                continue
            names = [entry["name"], *(entry.get("aliases") or [])]
            prose_pats = []
            caps_pats = []
            norm_keys = []
            for n in names:
                clean = _strip_accents(n).strip()
                low = clean.lower()
                if len(low) < 3 or low in _AMBIGUOUS_CITY_ALIASES:
                    continue
                norm_keys.append(low)
                # In mixed-case prose, proper city names start with a capital letter (avoids matching lowercase 'hue')
                prose_pats.append(re.compile(r"\b" + re.escape(clean) + r"\b"))
                caps_pats.append(re.compile(r"\b" + re.escape(clean.upper()) + r"\b"))
            if not norm_keys:
                continue
            cities.append(
                {
                    "name": entry["name"],
                    "norm_key": norm_keys[0],
                    "sub": entry.get("sub") or "",
                    "lat": float(entry["lat"]),
                    "lon": float(entry["lon"]),
                    "prose_patterns": prose_pats,
                    "caps_patterns": caps_pats,
                }
            )
        _GAZETTEER_CITIES = cities
    except Exception:
        _GAZETTEER_CITIES = []
    return _GAZETTEER_CITIES


def is_specific_event_url(url: str) -> bool:
    """Return True only if url points to a specific article or event report rather than a generic hub."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    path = parsed.path.rstrip("/")
    low_path = path.lower()
    if low_path in GENERIC_HUB_PATHS or not low_path:
        return False
    if any(low_path.startswith(pref) for pref in GENERIC_PATH_PREFIXES):
        return False
    if low_path.endswith(FORBIDDEN_URL_EXTENSIONS):
        return False
    if "eventid=" in parsed.query.lower():
        return True
    if "news.google.com" in parsed.netloc.lower() and "/articles/" in low_path:
        return True
    segments = [seg for seg in low_path.split("/") if seg]
    if not segments:
        return False
    last_seg = segments[-1]
    has_article_slug = ("-" in last_seg and len(last_seg) >= 12) or any(ch.isdigit() for ch in low_path)
    return has_article_slug and len(segments) >= 1


def check_url_resolves(url: str, timeout: int = 8) -> bool:
    """HEAD-check `url` (with lightweight GET fallback on 405/403). Rejects 404/410/5xx/unreachable links."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    host = parsed.netloc.lower().removeprefix("www.")
    is_paywall_host = any(host == d or host.endswith("." + d) for d in PAYWALL_OR_WAF_DOMAINS)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    }

    for method, extra_hdr in (("HEAD", {}), ("GET", {"Range": "bytes=0-2047"})):
        req = urllib.request.Request(url, method=method, headers={**headers, **extra_hdr})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = int(getattr(resp, "status", 200))
                if code < 400:
                    return True
        except urllib.error.HTTPError as hexc:
            if hexc.code < 400:
                return True
            if hexc.code in (401, 403) and is_paywall_host:
                return True
            if method == "HEAD" and hexc.code in (403, 405, 406):
                continue
            return False
        except Exception:
            if method == "HEAD":
                continue
            return False
    return False


def _friendly_span_phrase(y0: int, y1: int) -> str:
    span = abs(y1 - y0) + 1
    if y0 <= 1955 and y1 <= 1985:
        return "the mid-twentieth century"
    if y1 >= 2018 and 18 <= span <= 24:
        return "the previous 20 years"
    if 25 <= span <= 35:
        return "the past 30 years" if y1 >= 2015 else "a 30-year baseline"
    if 36 <= span <= 46:
        return "the past four decades"
    if span >= 70:
        return "more than 80 years of records"
    return f"the previous {int(round(span, -1))} years"


def sanitise_prose(text: str) -> str:
    """Enforce house style: no em dashes, no 'utilise' variants, listener-friendly eras, clean whitespace."""
    if not text:
        return ""
    cleaned = text.replace("\u2014", " - ").replace(" \u2013 ", " - ")
    cleaned = re.sub(r"\butilis(e|es|ed|ing|ation)\b", lambda m: _replace_utilise(m.group(0)), cleaned, flags=re.I)
    cleaned = re.sub(r"\butiliz(e|es|ed|ing|ation)\b", lambda m: _replace_utilise(m.group(0)), cleaned, flags=re.I)
    # Replace clunky baseline year ranges (e.g. 1980-2009, 1991-2020, 1940-1979) with natural broadcast phrasing
    cleaned = re.sub(
        r"\bthe\s+(?:19\d{2}|20\d{2})[-\u2013](?:19\d{2}|20\d{2})\s+(?:ERA5\s+)?(?:seasonal\s+)?(?:average|mean|climatology|normal)\b",
        "the 30-year seasonal average",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\bsince\s+1940[-\u2013]1979\b", "since the mid-twentieth century", cleaned, flags=re.I)
    cleaned = re.sub(r"\bin\s+1940[-\u2013]1979\b", "in the mid-twentieth century", cleaned, flags=re.I)
    cleaned = re.sub(r"\bin\s+1986[-\u2013]2025\b", "over the past four decades", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\b(19\d{2}|20\d{2})[-\u2013](19\d{2}|20\d{2})\b",
        lambda m: _friendly_span_phrase(int(m.group(1)), int(m.group(2)))
        if abs(int(m.group(2)) - int(m.group(1))) >= 9
        else m.group(0),
        cleaned,
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


def _replace_utilise(word: str) -> str:
    low = word.lower()
    if low.endswith("ing"):
        rep = "using"
    elif low.endswith("ed"):
        rep = "used"
    elif low.endswith("es"):
        rep = "uses"
    elif low.endswith("ation"):
        rep = "use"
    else:
        rep = "use"
    return rep.capitalize() if word[0].isupper() else rep


def auto_highlight_keywords(text: str) -> str:
    """Ensure key figures, rankings, and durations carry **highlight** markers if none were supplied."""
    if "**" in text:
        return text
    pattern = re.compile(
        r"([+-]?\d+(?:\.\d+)?\s*(?:°C|°F|mm|km/h|mph|m/s|hPa|%)"
        r"|#\d+(?:\s+in\s+\d+\s+years)?"
        r"|\d+(?:\.\d+)?(?:st|nd|rd|th)\s+percentile"
        r"|\b\d+\s+(?:days|weeks|years|rivers|counties)\b)",
        flags=re.I,
    )
    highlighted = pattern.sub(r"**\1**", text)
    if "**" in highlighted or not highlighted.strip():
        return highlighted
    words = highlighted.split()
    span = min(4, len(words))
    return "**" + " ".join(words[:span]) + "**" + (" " + " ".join(words[span:]) if len(words) > span else "")


def build_ticker_from_beats(beats: list[dict]) -> str:
    parts = [auto_highlight_keywords(b["text"]) for b in beats if b.get("text")]
    return "  ///  ".join(parts)


def infer_cue_layer(text: str, fallback: str) -> str:
    """Match the weather variable layer to the meteorological phenomenon discussed in a sentence."""
    low = (text or "").lower()
    if any(k in low for k in ("1940-1979", "warmed", "climate", "above normal", "below normal", "above seasonal", "anomaly")):
        return "anomaly"
    if any(k in low for k in ("hpa", "blocking high", "trough", "isobar", "depression", "low pressure")):
        return "pressure"
    if any(k in low for k in ("km/h", "mph", "m/s", "gust", "gale", "winds")):
        return "wind"
    if any(k in low for k in (" mm", "rain", "downpour", "deluge", "rivers", "flood", "wet days")):
        return "rain"
    if "°c" in low or "degrees" in low:
        return "temperature"
    return fallback if fallback in VALID_LAYERS else "temperature"


_CONTINENT_OR_GENERIC_LABELS = {
    "global",
    "world",
    "international",
    "south america",
    "north america",
    "central america",
    "europe",
    "north africa",
    "southern africa",
    "east africa",
    "west africa",
    "central asia",
    "southeast asia",
    "south asia",
    "east asia",
    "middle east",
    "mediterranean",
    "caribbean",
}


def opening_mentions_location(text: str, country: str, region: str) -> bool:
    """Return True if the opening clause (~first 65 chars) of `text` names the story's country."""
    opening_window = (text or "")[:68].lower()
    c_clean = (country or "").strip()
    if c_clean and c_clean.lower() not in _CONTINENT_OR_GENERIC_LABELS:
        c_low = c_clean.lower()
        accepted = COUNTRY_SYNONYMS.get(c_low, (c_low,))
        if any(tok in opening_window for tok in accepted):
            return True
        if len(c_low) >= 5 and c_low[:-1] in opening_window:
            return True
        return False

    reg_first = re.split(r"[,·/-]", region or "")[0].strip().lower()
    return bool(reg_first and reg_first not in _CONTINENT_OR_GENERIC_LABELS and reg_first in opening_window)


def ensure_opening_cue_names_location(cues: list[dict], country: str, region: str) -> None:
    """Guarantee `cues[0]['spoken']` explicitly names the country right in its opening words."""
    if not cues:
        return
    spoken0 = (cues[0].get("spoken") or "").strip()
    if not spoken0 or opening_mentions_location(spoken0, country, region):
        return

    anchor = (country or "").strip()
    if not anchor or anchor.lower() in _CONTINENT_OR_GENERIC_LABELS:
        anchor = re.split(r"[,·/-]", region or "")[0].strip()
    if not anchor:
        return

    first_word = spoken0.split()[0].rstrip(",:;")
    if first_word.lower() in LOWERCASE_LEAD_WORDS and not first_word.isupper():
        rest = spoken0[0].lower() + spoken0[1:]
    else:
        rest = spoken0
    cues[0]["spoken"] = sanitise_prose(f"{anchor}: {rest}")


def _interpolate_along_waypoints(waypoints: list[dict], frac: float) -> tuple[float, float, float]:
    """Interpolate (lat, lon, distance) cleanly along camera waypoints for frac in [0, 1]."""
    if not waypoints:
        return 0.0, 0.0, 1.75
    if len(waypoints) == 1 or frac <= 0.0:
        w = waypoints[0]
        return w["lat"], w["lon"], w["distance"]
    if frac >= 1.0:
        w = waypoints[-1]
        return w["lat"], w["lon"], w["distance"]

    pos = frac * (len(waypoints) - 1)
    idx = min(int(pos), len(waypoints) - 2)
    t = pos - idx
    a, b = waypoints[idx], waypoints[idx + 1]
    dlon = ((b["lon"] - a["lon"] + 180.0) % 360.0) - 180.0
    lat = round(max(-82.0, min(82.0, a["lat"] + t * (b["lat"] - a["lat"]))), 2)
    lon = round(((a["lon"] + t * dlon + 180.0) % 360.0) - 180.0, 2)
    dist = round(max(1.42, min(2.4, a["distance"] + t * (b["distance"] - a["distance"]))), 2)
    return lat, lon, dist


def _assign_default_cue_timings(cues: list[dict]) -> None:
    """Ensure every cue carries monotonic [start_sec, end_sec] even before neural TTS runs."""
    if not cues:
        return
    valid_existing = True
    prev_end = 0.0
    for c in cues:
        s = c.get("start_sec")
        e = c.get("end_sec")
        if s is None or e is None or not (float(e) > float(s) >= prev_end - 0.05):
            valid_existing = False
            break
        prev_end = float(e)
    if valid_existing:
        return

    weights = [max(14, len((c.get("spoken") or c.get("ticker") or "").strip())) for c in cues]
    total_chars = sum(weights)
    est_duration = max(24.0, round(total_chars / 14.5, 2))
    acc = 0.0
    n = len(cues)
    for i, w in enumerate(weights):
        t0 = (acc / total_chars) * est_duration
        acc += w
        t1 = est_duration if i == n - 1 else (acc / total_chars) * est_duration
        cues[i]["start_sec"] = round(t0, 2)
        cues[i]["end_sec"] = round(max(t0 + 0.5, t1), 2)


def clean_or_build_cues(
    st: dict,
    cleaned_beats: list[dict],
    waypoints: list[dict],
    primary_layer: str,
    kicker: str,
) -> list[dict]:
    """Validate explicit `cues` WITHOUT silently moving explicit lat/lon or overwriting explicit layers."""
    raw_cues = st.get("cues")
    default_bearings = [-10.0, -5.0, 0.0, 5.0, 10.0, -6.0, 6.0, 0.0]
    cues_out: list[dict] = []

    if isinstance(raw_cues, list) and len(raw_cues) >= 3:
        for i, c in enumerate(raw_cues[:8]):
            if not isinstance(c, dict):
                continue
            fb_lat, fb_lon, fb_dist = _interpolate_along_waypoints(
                waypoints, i / max(1, len(raw_cues[:8]) - 1)
            )
            has_explicit_coords = ("lat" in c and c["lat"] is not None) and ("lon" in c and c["lon"] is not None)
            try:
                lat = round(max(-85.0, min(85.0, float(c["lat"] if has_explicit_coords else fb_lat))), 2)
                lon = round(((float(c["lon"] if has_explicit_coords else fb_lon) + 180.0) % 360.0) - 180.0, 2)
                dist = round(max(1.36, min(2.6, float(c.get("distance", fb_dist)))), 2)
                bearing = round(
                    max(-45.0, min(45.0, float(c.get("bearing", default_bearings[i % len(default_bearings)])))),
                    1,
                )
            except (TypeError, ValueError):
                lat, lon, dist = fb_lat, fb_lon, fb_dist
                bearing = default_bearings[i % len(default_bearings)]

            spoken = sanitise_prose(str(c.get("spoken") or c.get("text") or ""))
            if not spoken:
                continue
            ticker_part = auto_highlight_keywords(sanitise_prose(str(c.get("ticker") or spoken)))
            raw_layer = str(c.get("layer") or "").strip().lower()
            # Section 2.2: keep explicit valid layer as-is; otherwise infer from sentence
            layer = raw_layer if raw_layer in VALID_LAYERS else infer_cue_layer(spoken, primary_layer)

            entry: dict = {
                "layer": layer,
                "lat": lat,
                "lon": lon,
                "distance": dist,
                "bearing": bearing,
                "label": sanitise_prose(str(c.get("label") or kicker))[:38].upper(),
                "metric": sanitise_prose(str(c.get("metric") or "")),
                "spoken": spoken,
                "ticker": ticker_part,
            }
            if c.get("start_sec") is not None and c.get("end_sec") is not None:
                try:
                    entry["start_sec"] = round(max(0.0, float(c["start_sec"])), 2)
                    entry["end_sec"] = round(max(entry["start_sec"] + 0.5, float(c["end_sec"])), 2)
                except (TypeError, ValueError):
                    pass
            cues_out.append(entry)

    # Only synthesise cues from beats when explicit cues were omitted (<3 cues provided)
    if len(cues_out) < 3:
        headline_txt = sanitise_prose(str(st.get("headline") or "")).rstrip(".")
        stats_in = st.get("stats") or {}
        metric_str = sanitise_prose(str(stats_in.get("headline_metric") or ""))
        region_str = sanitise_prose(str(st.get("region") or kicker)).upper()[:36]

        fallback_templates: list[dict] = []
        if headline_txt:
            la0, lo0, d0 = _interpolate_along_waypoints(waypoints, 0.0)
            fallback_templates.append(
                {
                    "layer": primary_layer,
                    "lat": la0,
                    "lon": lo0,
                    "distance": d0,
                    "bearing": default_bearings[0],
                    "label": region_str or kicker[:36],
                    "metric": metric_str,
                    "spoken": f"{headline_txt}.",
                    "ticker": auto_highlight_keywords(headline_txt),
                }
            )

        for b_idx, beat in enumerate(cleaned_beats):
            frac = (b_idx + 1) / max(1, len(cleaned_beats))
            la, lo, dist = _interpolate_along_waypoints(waypoints, frac)
            b_txt = beat["text"]
            b_layer = beat["layer"] if beat.get("layer") in VALID_LAYERS else infer_cue_layer(b_txt, primary_layer)
            fallback_templates.append(
                {
                    "layer": b_layer,
                    "lat": la,
                    "lon": lo,
                    "distance": round(max(1.44, min(2.1, dist)), 2),
                    "bearing": default_bearings[(b_idx + 1) % len(default_bearings)],
                    "label": (beat.get("title") or beat.get("tag") or kicker)[:36].upper(),
                    "metric": beat.get("tag") or metric_str,
                    "spoken": b_txt.replace("**", ""),
                    "ticker": auto_highlight_keywords(b_txt),
                }
            )

        if not cues_out:
            cues_out = fallback_templates[:5]
        else:
            for existing in cues_out:
                existing.pop("start_sec", None)
                existing.pop("end_sec", None)
            for cand_cue in fallback_templates:
                if len(cues_out) >= 5:
                    break
                if not any(cand_cue["spoken"][:28].lower() in x["spoken"].lower() for x in cues_out):
                    cues_out.append(cand_cue)

    _assign_default_cue_timings(cues_out)
    return cues_out


def ensure_dynamic_waypoints(centre_lat: float, centre_lon: float, waypoints: list[dict] | None) -> list[dict]:
    """Keep valid local camera waypoints (within 500 km of centre), or create a compact ~90 km local drift."""
    cleaned: list[dict] = []
    for wp in waypoints or []:
        try:
            la = max(-85.0, min(85.0, float(wp["lat"])))
            lo = ((float(wp["lon"]) + 180.0) % 360.0) - 180.0
            if haversine_km(la, lo, centre_lat, centre_lon) > 500.0:
                continue
            dist = max(1.42, min(2.6, float(wp.get("distance", 1.75))))
            cleaned.append({"lat": round(la, 2), "lon": round(lo, 2), "distance": round(dist, 2)})
        except (KeyError, TypeError, ValueError):
            continue

    if len(cleaned) >= 2:
        return cleaned[:5]

    # Compact ~85 km local camera drift around centre (always well inside 150 km and 500 km)
    cos_lat = max(0.35, math.cos(math.radians(centre_lat)))
    dlon = min(1.6, 0.45 / cos_lat)
    raw = [
        (centre_lat - 0.35, centre_lon - dlon, 1.82),
        (centre_lat - 0.12, centre_lon - dlon * 0.3, 1.68),
        (centre_lat + 0.15, centre_lon + dlon * 0.35, 1.64),
        (centre_lat + 0.35, centre_lon + dlon, 1.76),
    ]
    return [
        {
            "lat": round(max(-85.0, min(85.0, la)), 2),
            "lon": round(((lo + 180.0) % 360.0) - 180.0, 2),
            "distance": round(d, 2),
        }
        for la, lo, d in raw
    ]


# ---------------------------------------------------------------------------
# Section 2.3: Geographic sanity check (<=500 km from centre; <=150 km from named city)
# ---------------------------------------------------------------------------


def _cities_mentioned_in_text(text: str, ref_lat: float, ref_lon: float, is_all_caps: bool = False) -> list[dict]:
    clean = _strip_accents(text or "")
    if not clean:
        return []
    by_name: dict[str, list[dict]] = {}
    for city in _load_gazetteer_cities():
        pats = city["caps_patterns"] if is_all_caps else city["prose_patterns"]
        if any(pat.search(clean) for pat in pats):
            by_name.setdefault(city["norm_key"], []).append(city)

    # Disambiguate homonym cities (e.g. San Jose CR vs US) by choosing the one closest to ref_lat/ref_lon
    resolved: list[dict] = []
    for group in by_name.values():
        best = min(group, key=lambda c: haversine_km(ref_lat, ref_lon, c["lat"], c["lon"]))
        resolved.append(best)
    return resolved


def validate_story_geography(
    prefix: str,
    c_lat: float,
    c_lon: float,
    cues: list[dict],
    headline: str = "",
) -> list[str]:
    """Verify every cue lies <=500 km from story centre and <=150 km from every gazetteer city it names."""
    errs: list[str] = []

    for h_city in _cities_mentioned_in_text(headline, c_lat, c_lon, is_all_caps=False):
        d_hc = haversine_km(c_lat, c_lon, h_city["lat"], h_city["lon"])
        min_cue_d = min(
            (haversine_km(float(c["lat"]), float(c["lon"]), h_city["lat"], h_city["lon"]) for c in cues),
            default=d_hc,
        )
        if d_hc > 500.0 or min_cue_d > 150.0:
            errs.append(
                f"{prefix}.headline: names {h_city['name']} ({h_city['lat']:.2f}, {h_city['lon']:.2f}), "
                f"which is {d_hc:.0f} km from story centre ({c_lat:.2f}, {c_lon:.2f}) and {min_cue_d:.0f} km "
                "from the nearest cue (must be <=500 km from centre and <=150 km from its cue)."
            )

    for i, c in enumerate(cues):
        lat, lon = float(c["lat"]), float(c["lon"])
        d_centre = haversine_km(lat, lon, c_lat, c_lon)
        if d_centre > 500.0:
            errs.append(
                f"{prefix}.cues[{i}]: coordinates ({lat}, {lon}) are {d_centre:.0f} km from story centre "
                f"({c_lat:.2f}, {c_lon:.2f}); every cue must be within 500 km of the story centre."
            )

        seen_city_names: set[str] = set()
        for city in (
            _cities_mentioned_in_text(c.get("label", ""), c_lat, c_lon, is_all_caps=True)
            + _cities_mentioned_in_text(c.get("spoken", ""), c_lat, c_lon, is_all_caps=False)
        ):
            if city["name"] in seen_city_names:
                continue
            seen_city_names.add(city["name"])
            d_city = haversine_km(lat, lon, city["lat"], city["lon"])
            if d_city > 150.0:
                errs.append(
                    f"{prefix}.cues[{i}]: cue names {city['name']} ({city['lat']:.2f}, {city['lon']:.2f}) "
                    f"but cue coordinates ({lat}, {lon}) are {d_city:.0f} km away (must be within 150 km)."
                )
    return errs


# ---------------------------------------------------------------------------
# Section 2.4: Numeric fact check & internal contradiction check
# ---------------------------------------------------------------------------

_RE_TEMP_C = re.compile(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)\s*°\s*C\b", flags=re.I)
_RE_RAIN_MM = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(?:mm|millimetres|millimeters)\b", flags=re.I)
_RE_PRES_HPA = re.compile(r"(?<![A-Za-z0-9])(\d{3,4}(?:\.\d+)?)\s*(?:hPa|hectopascals)\b", flags=re.I)
_RE_SPEED = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(km/h|kilometres per hour|kilometres an hour|m/s|metres per second|mph)\b",
    flags=re.I,
)
_RE_RANK = re.compile(r"#(\d+)\b")
_RE_DAYS_RATIO = re.compile(r"\b(\d{1,2})\s+of\s+(?:the\s+)?(?:last\s+|past\s+)?(\d{1,2})\s+days\b", flags=re.I)
_RE_MULT_NORM = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:times|x|×)\s+(?:the\s+)?(?:seasonal\s+|climatological\s+|daily\s+|expected\s+|28-day\s+|14-day\s+)?(?:norm|normal|average|expectation)\b",
    flags=re.I,
)


_RE_PCT_NORM = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*%\s*of\s+(?:the\s+)?(?:seasonal\s+|climatological\s+|28-day\s+|14-day\s+|expected\s+)?(?:norm|normal|average)\b",
    flags=re.I,
)
_RE_DAYS_IN_WINDOW = re.compile(
    r"\b(\d{1,2})\s+(?:[A-Za-z-]+\s+){0,3}?days\s+(?:in|over|across)\s+(?:the\s+)?(?:past\s+|last\s+)?(14\s*d(?:ays)?|fortnight|28\s*d(?:ays)?|four\s+weeks)\b",
    flags=re.I,
)


def _speed_to_ms(val: float, unit_raw: str) -> float:
    u = unit_raw.lower()
    if "km" in u:
        return val / 3.6
    if "mph" in u:
        return val * 0.44704
    return val


def extract_numbers_by_unit(text: str) -> dict[str, list]:
    """Extract all quantitative meteorological figures by unit family from `text`."""
    s = (text or "").replace("**", "")
    s_norm = re.sub(r"\bminus\s+(\d+(?:\.\d+)?)\s+degrees\s+Celsius\b", r"-\1 °C", s, flags=re.I)
    s_norm = re.sub(r"\bplus\s+(\d+(?:\.\d+)?)\s+degrees\s+Celsius\b", r"+\1 °C", s_norm, flags=re.I)
    s_norm = re.sub(r"\b(\d+(?:\.\d+)?)\s+degrees\s+Celsius\b", r"\1 °C", s_norm, flags=re.I)

    # Strip standard wet-day persistence threshold qualifiers (e.g. '>=10 mm', 'at least 10 mm') so threshold labels are not mistaken for event totals
    s_rain = re.sub(
        r"(?:\(\s*(?:>=|≥)\s*10\s*mm\s*\)|\bat\s+least\s+10\s*(?:mm|millimetres|millimeters)\b)",
        "",
        s_norm,
        flags=re.I,
    )
    temps = [(float(m.group(1)), m.group(1)[0] in "+-") for m in _RE_TEMP_C.finditer(s_norm)]
    rains = [float(m.group(1)) for m in _RE_RAIN_MM.finditer(s_rain)]
    press = [float(m.group(1)) for m in _RE_PRES_HPA.finditer(s_norm)]
    speeds_ms = [
        (_speed_to_ms(float(m.group(1)), m.group(2)), m.group(0))
        for m in _RE_SPEED.finditer(s_norm)
    ]
    ranks = [int(m.group(1)) for m in _RE_RANK.finditer(s_norm)]
    day_ratios = [(int(m.group(1)), int(m.group(2))) for m in _RE_DAYS_RATIO.finditer(s_norm)]
    for m in _RE_DAYS_IN_WINDOW.finditer(s_norm):
        win_raw = m.group(2).lower()
        denom = 14 if ("14" in win_raw or "fortnight" in win_raw) else 28
        day_ratios.append((int(m.group(1)), denom))
    multiples = [float(m.group(1)) for m in _RE_MULT_NORM.finditer(s_norm)]
    pct_norms = [float(m.group(1)) for m in _RE_PCT_NORM.finditer(s_norm)]

    return {
        "temp_c": temps,
        "rain_mm": rains,
        "pres_hpa": press,
        "speed_ms": speeds_ms,
        "ranks": ranks,
        "day_ratios": day_ratios,
        "multiples": multiples,
        "pct_norms": pct_norms,
    }


_BANNED_HYPE_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bunprecedented\b", flags=re.I), "unprecedented"),
    (re.compile(r"\bever\s+recorded\b", flags=re.I), "ever recorded"),
    (re.compile(r"\bheat\s+dome\b", flags=re.I), "heat dome"),
    (re.compile(r"\bbaking\b", flags=re.I), "baking"),
    (re.compile(r"\bhistoric\b", flags=re.I), "historic"),
    (re.compile(r"\bclimate\s+emergency\b", flags=re.I), "climate emergency"),
    (re.compile(r"\bshatters?\b|\bshattered\b|\bshattering\b", flags=re.I), "shatters"),
    (re.compile(r"\b(?:bbc|sky\s+news)\b", flags=re.I), "BBC/Sky affiliation"),
)


def validate_editorial_tone_and_framing(
    prefix: str,
    st: dict,
    matched_cand: dict | None = None,
) -> list[str]:
    """Enforce Section 3 calm public-service tone, banned hype phrases, and sub-zero mildness framing."""
    errs: list[str] = []
    stats = st.get("stats") or {}
    links = st.get("news_links") or []
    has_bbc_or_sky_source = any(
        "bbc." in str(lk.get("url") or "").lower() or "sky.com" in str(lk.get("url") or "").lower()
        for lk in links
        if isinstance(lk, dict)
    )

    kicker_and_head = f"{st.get('kicker', '')} | {st.get('headline', '')}"
    if re.search(r"\b(?:bbc|sky)\b", kicker_and_head, flags=re.I):
        errs.append(f"{prefix}: remove BBC/Sky branding from kicker and headline.")

    prose_parts = [
        str(st.get("kicker") or ""),
        str(st.get("headline") or ""),
        str(st.get("ticker") or ""),
        str(stats.get("headline_metric") or ""),
        str(stats.get("metric_label") or ""),
        str(stats.get("era5_rank") or ""),
        str(stats.get("persistence") or ""),
        *(f"{c.get('label', '')} {c.get('metric', '')} {c.get('spoken', '')} {c.get('ticker', '')}" for c in (st.get("cues") or [])),
        *(f"{b.get('title', '')} {b.get('text', '')}" for b in (st.get("beats") or [])),
    ]
    blob = " \n ".join(prose_parts)

    for pat, label in _BANNED_HYPE_PATTERNS:
        if label == "BBC/Sky affiliation" and has_bbc_or_sky_source:
            continue
        if pat.search(blob):
            errs.append(
                f"{prefix}: banned phrase {label!r} found; use a calm, understated public-service tone."
            )

    # Sub-zero positive anomalies must not be called 'record heat' or 'extreme heat'
    is_subzero_actual = False
    if matched_cand is not None and matched_cand.get("t2m_c") is not None and float(matched_cand["t2m_c"]) < 2.0:
        is_subzero_actual = True
    elif re.search(r"actual\s+temp(?:erature)?[^.\n]{0,20}?-\d", blob, flags=re.I):
        is_subzero_actual = True
    if is_subzero_actual and re.search(r"\b(?:record\s+heat|extreme\s+heat|heatwave)\b", blob, flags=re.I):
        errs.append(
            f"{prefix}: sub-zero or near-freezing warm anomaly must be called 'exceptionally mild', never 'record heat' or 'heatwave'."
        )

    # When candidate data is present and news_facts is empty, forbid past-tense model verbs ('fell', 'has recorded', 'measured')
    if matched_cand is not None and not (st.get("news_facts") or []):
        if re.search(r"\b(?:mm\s+of\s+rain\s+fell|millimetres\s+of\s+rain\s+fell|has\s+recorded\s+\d|ever\s+measured)\b", blob, flags=re.I):
            errs.append(
                f"{prefix}: ECMWF values are a forecast; use forecast/future wording rather than 'fell', 'has recorded', or 'measured'."
            )

    return errs


def validate_internal_consistency(prefix: str, st: dict) -> list[str]:
    """Detect internal numerical contradictions within a story even without candidate lookup."""
    errs: list[str] = []
    stats = st.get("stats") or {}
    stats_blob = f"{stats.get('headline_metric', '')} | {stats.get('metric_label', '')}"

    # 1. Contradictory wind/gust speeds in the same stats card (e.g. 34.3 m/s vs 44 m/s / 160 km/h)
    stat_speeds = extract_numbers_by_unit(stats_blob)["speed_ms"]
    if len(stat_speeds) >= 2:
        vals = [v for v, _ in stat_speeds]
        hi, lo = max(vals), min(vals)
        if hi > 0 and (hi - lo) > max(2.5, 0.18 * hi):
            raw_labels = ", ".join(lbl for _, lbl in stat_speeds)
            errs.append(
                f"{prefix}.stats: contradictory wind/gust figures in stats card ({raw_labels}); "
                "report a single consistent forecast wind/gust value."
            )

    # 2. Temperature vs Normal vs Anomaly triangle contradiction (e.g. -7.7 C vs normal -7.6 C with +14.4 C anomaly)
    anom_claim = None
    full_blob = " \n ".join(
        [
            str(st.get("headline") or ""),
            stats_blob,
            *(f"{c.get('metric', '')} {c.get('spoken', '')}" for c in (st.get("cues") or [])),
            *(str(b.get("text") or "") for b in (st.get("beats") or [])),
        ]
    )
    for m in re.finditer(
        r"([+-]\d+(?:\.\d+)?)\s*°\s*C\s*(?:anomaly|above|below)",
        json.dumps(stats) + " " + str(st.get("headline", "")),
        flags=re.I,
    ):
        anom_claim = float(m.group(1))
        break

    vs_norm_pat = re.compile(
        r"([+-]?\d+(?:\.\d+)?)[ \t]*°[ \t]*C[ \t]+(?:vs\.?|versus|where[ \t]+normal[ \t]+is|against[ \t]+(?:a[ \t]+)?(?:seasonal[ \t]+)?norm(?:al)?(?:[ \t]+of)?)[ \t]*(?:normal[ \t]+)?([+-]?\d+(?:\.\d+)?)[ \t]*°[ \t]*C",
        flags=re.I,
    )
    found_contradiction = False
    for m in vs_norm_pat.finditer(full_blob):
        actual_t = float(m.group(1))
        norm_t = float(m.group(2))
        implied_anom = actual_t - norm_t
        if anom_claim is not None and abs(implied_anom - anom_claim) > 4.0:
            errs.append(
                f"{prefix}: internal temperature contradiction - actual {actual_t:+.1f} °C vs normal "
                f"{norm_t:+.1f} °C implies a {implied_anom:+.1f} °C departure, contradicting the "
                f"claimed {anom_claim:+.1f} °C anomaly."
            )
            found_contradiction = True
            break

    if not found_contradiction and anom_claim is not None:
        norm_m = re.search(
            r"(?:seasonal\s+(?:normal|average|norm)|normal\s+for\s+this\s+(?:date|time\s+of\s+year))[^.\n]{0,35}?([+-]?\d+(?:\.\d+)?)\s*°\s*C",
            full_blob,
            flags=re.I,
        )
        act_m = re.search(
            r"actual\s+temperature[^.\n]{0,35}?([+-]?\d+(?:\.\d+)?)\s*°\s*C",
            full_blob,
            flags=re.I,
        )
        if norm_m and act_m:
            actual_t = float(act_m.group(1))
            norm_t = float(norm_m.group(1))
            implied_anom = actual_t - norm_t
            if abs(implied_anom - anom_claim) > 4.0:
                errs.append(
                    f"{prefix}: internal temperature contradiction - actual {actual_t:+.1f} °C vs seasonal normal "
                    f"{norm_t:+.1f} °C implies {implied_anom:+.1f} °C, contradicting {anom_claim:+.1f} °C anomaly."
                )
    return errs


def _round_2sf(val: float) -> float:
    if val == 0 or not math.isfinite(val):
        return 0.0
    mag = math.floor(math.log10(abs(val)))
    step = 10.0 ** (mag - 1)
    return round(val / step) * step


def _with_2sf_variants(vals: list[float]) -> list[float]:
    out = list(vals)
    for v in vals:
        out.append(_round_2sf(v))
        out.append(float(round(v)))
    return out
def _allowed_candidate_numbers(cand: dict) -> dict[str, list]:
    era = cand.get("era5") or {}
    pers = era.get("persistence") or {}
    shift = era.get("climate_shift") or {}

    temps_signed: list[float] = []
    temps_diffs_abs: list[float] = []
    for k in ("t2m_c", "t2m_max_24h_c", "t2m_min_24h_c", "anom_c"):
        if cand.get(k) is not None:
            v = float(cand[k])
            temps_signed.append(v)
            if k == "anom_c":
                temps_diffs_abs.append(abs(v))
    for k in ("seasonal_normal_1991_2020", "wmo_tmax_normal_c", "wmo_tmin_normal_c"):
        if (era.get(k) is not None and era.get("units") == "°C") or (k.startswith("wmo_") and era.get(k) is not None):
            temps_signed.append(float(era[k]))
    for k in ("tmax_p90_threshold_c", "tmin_p10_threshold_c"):
        if pers.get(k) is not None:
            temps_signed.append(float(pers[k]))
    if shift.get("local_seasonal_warming_c_since_1940_1979") is not None:
        w = float(shift["local_seasonal_warming_c_since_1940_1979"])
        temps_signed.append(w)
        temps_diffs_abs.append(abs(w))

    rains: list[float] = []
    if cand.get("rain_24h_mm") is not None:
        rains.append(float(cand["rain_24h_mm"]))
    for k in ("daily_rain_normal_mm", "rain_14d_mm", "expected_rain_14d_mm", "rain_28d_mm", "expected_rain_28d_mm"):
        if pers.get(k) is not None:
            rains.append(float(pers[k]))
    if era.get("units") == "mm" and era.get("seasonal_normal_1991_2020") is not None:
        rains.append(float(era["seasonal_normal_1991_2020"]))

    press = [float(cand["mslp_hpa"])] if cand.get("mslp_hpa") is not None else []

    speeds_ms = [65.0 / 3.6]  # standard 65 km/h persistence threshold
    for k in ("wind_ms", "gust_ms"):
        if cand.get(k) is not None:
            speeds_ms.append(float(cand[k]))
    if pers.get("local_gust_p90_kmh") is not None:
        speeds_ms.append(float(pers["local_gust_p90_kmh"]) / 3.6)
    if era.get("units") == "m/s" and era.get("seasonal_normal_1991_2020") is not None:
        speeds_ms.append(float(era["seasonal_normal_1991_2020"]))

    ranks = [int(era["rank_since_1940"])] if era.get("rank_since_1940") is not None else []

    day_ratios: list[tuple[int, int]] = []
    for k14 in (
        "warm_days_above_p90_tmax_last_14",
        "cold_days_below_p10_tmin_last_14",
        "days_gust_over_65kmh_last_14",
        "days_gust_above_local_p90_last_14",
        "wet_days_ge_10mm_last_14",
        "anomalous_days_last_14",
    ):
        if pers.get(k14) is not None:
            day_ratios.append((int(pers[k14]), 14))
    for k28 in (
        "warm_days_above_p90_tmax_last_28",
        "cold_days_below_p10_tmin_last_28",
        "days_gust_over_65kmh_last_28",
        "days_gust_above_local_p90_last_28",
        "wet_days_ge_10mm_last_28",
        "anomalous_days_last_28",
    ):
        if pers.get(k28) is not None:
            day_ratios.append((int(pers[k28]), 28))

    multiples: list[float] = []
    pct_norms: list[float] = []
    if pers.get("rain_28d_multiple_of_normal") is not None:
        multiples.append(float(pers["rain_28d_multiple_of_normal"]))
    if pers.get("rain_28d_pct_of_normal") is not None:
        pct_norms.append(float(pers["rain_28d_pct_of_normal"]))
    if (pers.get("expected_rain_14d_mm") or 0) >= 5.0 and pers.get("rain_14d_mm") is not None:
        m14 = float(pers["rain_14d_mm"]) / float(pers["expected_rain_14d_mm"])
        multiples.append(m14)
        pct_norms.append(m14 * 100.0)
    if shift.get("frequency_claim_allowed") and shift.get("frequency_ratio") is not None:
        multiples.append(float(shift["frequency_ratio"]))

    return {
        "temp_signed": _with_2sf_variants(temps_signed),
        "temp_diff_abs": _with_2sf_variants(temps_diffs_abs),
        "rain_mm": _with_2sf_variants(rains),
        "pres_hpa": [float(round(p)) for p in press] + press,
        "speed_ms": [s / 3.6 for s in _with_2sf_variants([x * 3.6 for x in speeds_ms])] + speeds_ms,
        "ranks": ranks,
        "day_ratios": day_ratios,
        "multiples": _with_2sf_variants(multiples),
        "pct_norms": [float(round(p)) for p in pct_norms] + _with_2sf_variants(pct_norms),
    }


def validate_story_numbers_against_candidates(
    prefix: str,
    st: dict,
    candidates: list[dict],
) -> list[str]:
    """Verify every unit-bearing figure in `st` matches its candidate data or an attributed `news_facts` item."""
    errs: list[str] = []
    c_lat = float(st["centre"]["lat"])
    c_lon = float(st["centre"]["lon"])

    matched = min(
        candidates,
        key=lambda c: haversine_km(c_lat, c_lon, float(c["lat"]), float(c["lon"])),
        default=None,
    )
    if matched is None or haversine_km(c_lat, c_lon, float(matched["lat"]), float(matched["lon"])) > 550.0:
        errs.append(
            f"{prefix}: story centre ({c_lat:.2f}, {c_lon:.2f}) does not match any scanned candidate hotspot within 550 km."
        )
        return errs

    allowed = _allowed_candidate_numbers(matched)

    # Also allow numbers explicitly documented in `news_facts` with a valid source URL
    news_facts = st.get("news_facts") or []
    fact_text = " ".join(
        str(nf.get("claim") or "") for nf in news_facts if isinstance(nf, dict) and nf.get("source_url")
    )
    fact_nums = extract_numbers_by_unit(fact_text)

    allowed_temps_signed = allowed["temp_signed"] + [v for v, _ in fact_nums["temp_c"]]
    allowed_temps_unsigned = allowed_temps_signed + allowed["temp_diff_abs"] + [abs(v) for v, _ in fact_nums["temp_c"]]
    allowed_rains = allowed["rain_mm"] + fact_nums["rain_mm"]
    allowed_press = allowed["pres_hpa"] + fact_nums["pres_hpa"]
    allowed_speeds = allowed["speed_ms"] + [v for v, _ in fact_nums["speed_ms"]]
    allowed_ranks = set(allowed["ranks"] + fact_nums["ranks"])
    allowed_ratios = set(allowed["day_ratios"] + fact_nums["day_ratios"])
    allowed_mults = allowed["multiples"] + fact_nums["multiples"]
    allowed_pcts = allowed["pct_norms"] + fact_nums["pct_norms"]

    stats = st.get("stats") or {}
    story_segments = [
        str(st.get("headline") or ""),
        str(st.get("ticker") or ""),
        str(stats.get("headline_metric") or ""),
        str(stats.get("metric_label") or ""),
        str(stats.get("era5_rank") or ""),
        str(stats.get("persistence") or ""),
        *(f"{c.get('metric', '')} {c.get('spoken', '')} {c.get('ticker', '')}" for c in (st.get("cues") or [])),
        *(str(b.get("text") or "") for b in (st.get("beats") or [])),
    ]
    found = extract_numbers_by_unit(" \n ".join(story_segments))

    for t_val, has_sign in found["temp_c"]:
        pool = allowed_temps_signed if has_sign else allowed_temps_unsigned
        if not any(abs(t_val - a) <= max(1.0, 0.08 * abs(a)) for a in pool):
            errs.append(
                f"{prefix}: temperature figure '{t_val:g} °C' does not match candidate data "
                f"({sorted({round(x, 1) for x in allowed['temp_signed']})}) or news_facts."
            )
            break

    for r_val in found["rain_mm"]:
        if not any(abs(r_val - a) <= max(2.0, 0.08 * max(1.0, a)) for a in allowed_rains):
            errs.append(
                f"{prefix}: rainfall figure '{r_val:g} mm' does not match candidate data "
                f"({sorted({round(x, 1) for x in allowed['rain_mm']})}) or news_facts."
            )
            break

    for p_val in found["pres_hpa"]:
        if not any(abs(p_val - a) <= 2.0 for a in allowed_press):
            errs.append(
                f"{prefix}: pressure figure '{p_val:g} hPa' does not match candidate MSLP "
                f"({allowed['pres_hpa']}) or news_facts."
            )
            break

    for s_ms, raw_lbl in found["speed_ms"]:
        if not any(abs(s_ms - a) <= max(1.4, 0.08 * a) for a in allowed_speeds):
            errs.append(
                f"{prefix}: wind/gust figure '{raw_lbl}' ({s_ms:.1f} m/s) does not match candidate wind/gust "
                f"({[round(x, 1) for x in allowed['speed_ms']]}) or news_facts."
            )
            break

    for rk in found["ranks"]:
        if rk not in allowed_ranks:
            errs.append(
                f"{prefix}: rank '#{rk}' does not match candidate rank_since_1940 ({allowed['ranks']}) or news_facts."
            )
            break

    for pair in found["day_ratios"]:
        if pair not in allowed_ratios:
            errs.append(
                f"{prefix}: persistence count '{pair[0]} of {pair[1]} days' does not match candidate persistence "
                f"({sorted(allowed_ratios)}) or news_facts."
            )
            break

    for mult in found["multiples"]:
        if not any(abs(mult - a) <= max(0.35, 0.22 * a) for a in allowed_mults):
            errs.append(
                f"{prefix}: multiple-of-normal claim '{mult:g}x' is not supported by candidate rainfall/frequency ratios "
                f"({[round(x, 1) for x in allowed['multiples']]})."
            )
            break

    for pct in found["pct_norms"]:
        if not any(abs(pct - a) <= max(15.0, 0.15 * a) for a in allowed_pcts):
            errs.append(
                f"{prefix}: percentage-of-normal claim '{pct:g}%' is not supported by candidate data "
                f"({[round(x, 0) for x in allowed['pct_norms']]})."
            )
            break

    # Section 1.5 enforcement: forbid frequency-increase claims if frequency_claim_allowed is False
    shift = (matched.get("era5") or {}).get("climate_shift") or {}
    if shift and shift.get("frequency_claim_allowed") is False:
        full_low = " ".join(story_segments).lower()
        if re.search(r"\b(per\s+decade|doubled|tripled|\d+\s*(?:times|x)\s+more\s+frequent)\b", full_low):
            errs.append(
                f"{prefix}: climate frequency claim ('per decade' / 'doubled' / 'more frequent') is forbidden "
                "when 40-year exceedance counts are below 5 or ratio < 1.5."
            )

    return errs


# ---------------------------------------------------------------------------
# Main Bulletin Validator & Cleaner
# ---------------------------------------------------------------------------


def validate_and_clean_bulletin(
    payload: dict,
    candidates: list[dict] | None = None,
    verify_urls: bool = False,
) -> tuple[dict, list[str]]:
    """Validate and sanitise a bulletin dict. Returns (cleaned_payload, errors)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {}, ["Bulletin root must be a JSON object."]

    stories = payload.get("stories")
    if not isinstance(stories, list) or len(stories) < 5:
        errors.append(
            f"Expected 5 stories, got {len(stories) if isinstance(stories, list) else 'non-list'}."
        )
        return payload, errors

    generator_tag = str(payload.get("generator") or "claude")
    allow_empty_links = generator_tag == "era5-deterministic"

    # Pre-verify candidate news URLs in parallel when live URL checking is enabled
    url_ok_map: dict[str, bool] = {}
    if verify_urls:
        all_urls: list[str] = []
        for st in stories[:5]:
            if isinstance(st, dict):
                for lk in st.get("news_links") or []:
                    if isinstance(lk, dict) and lk.get("url"):
                        u = str(lk["url"]).strip()
                        if is_specific_event_url(u) and u not in all_urls:
                            all_urls.append(u)
                for nf in st.get("news_facts") or []:
                    if isinstance(nf, dict) and nf.get("source_url"):
                        u = str(nf["source_url"]).strip()
                        if is_specific_event_url(u) and u not in all_urls:
                            all_urls.append(u)
        if all_urls:
            with ThreadPoolExecutor(max_workers=6) as pool:
                statuses = list(pool.map(check_url_resolves, all_urls))
            url_ok_map = dict(zip(all_urls, statuses))

    cleaned_stories: list[dict] = []
    for idx, st in enumerate(stories[:5], start=1):
        prefix = f"stories[{idx - 1}]"
        if not isinstance(st, dict):
            errors.append(f"{prefix}: must be an object.")
            continue

        headline = sanitise_prose(str(st.get("headline") or ""))
        if len(headline) < 12:
            errors.append(f"{prefix}: headline is missing or too short.")

        primary_layer = str(st.get("primary_layer") or "temperature").strip().lower()
        if primary_layer not in VALID_LAYERS:
            errors.append(f"{prefix}: invalid primary_layer {primary_layer!r}.")

        hazard = str(st.get("hazard") or "wind").strip().lower()
        if hazard not in VALID_HAZARDS:
            hazard = "wind"

        centre = st.get("centre") or {}
        try:
            c_lat = float(centre["lat"])
            c_lon = ((float(centre["lon"]) + 180.0) % 360.0) - 180.0
            if not (-90.0 <= c_lat <= 90.0):
                raise ValueError("lat out of range")
        except Exception:
            errors.append(f"{prefix}: invalid centre coordinates {centre!r}.")
            continue

        beats_in = st.get("beats")
        if not isinstance(beats_in, list) or len(beats_in) < 4:
            errors.append(f"{prefix}: must include 4 story beats (live, persistence, rank, climate).")
            continue

        cleaned_beats: list[dict] = []
        for b_idx, beat in enumerate(beats_in[:4]):
            b_text = sanitise_prose(str(beat.get("text") or ""))
            if len(b_text) < 25:
                errors.append(f"{prefix}.beats[{b_idx}]: narrative text too short.")
            b_layer = str(beat.get("layer") or primary_layer).strip().lower()
            if b_layer not in VALID_LAYERS:
                b_layer = primary_layer
            cleaned_beats.append(
                {
                    "tag": sanitise_prose(str(beat.get("tag") or f"PART {b_idx + 1}")).upper(),
                    "title": sanitise_prose(str(beat.get("title") or "")),
                    "layer": b_layer,
                    "text": b_text,
                }
            )

        region_clean = sanitise_prose(str(st.get("region") or ""))
        country_clean = sanitise_prose(str(st.get("country") or ""))

        dyn_waypoints = ensure_dynamic_waypoints(c_lat, c_lon, st.get("camera_waypoints"))
        kicker_clean = sanitise_prose(str(st.get("kicker") or region_clean or "WORLD WEATHER")).upper()
        cleaned_cues = clean_or_build_cues(st, cleaned_beats, dyn_waypoints, primary_layer, kicker_clean)
        ensure_opening_cue_names_location(cleaned_cues, country_clean, region_clean)

        # Section 2.3: Geographic sanity checks (never silently move cues!)
        errors.extend(validate_story_geography(prefix, c_lat, c_lon, cleaned_cues, headline=headline))

        raw_ticker = sanitise_prose(str(st.get("ticker") or ""))
        if len(raw_ticker) >= 60:
            ticker = auto_highlight_keywords(raw_ticker)
        else:
            ticker = "  ///  ".join(c["ticker"] for c in cleaned_cues if c.get("ticker"))

        voice_script = " ".join(c["spoken"] for c in cleaned_cues if c.get("spoken")).strip()
        if not voice_script:
            voice_script = sanitise_prose(str(st.get("voice_script") or headline))
        audio_url = str(st.get("audio_url") or "").strip()
        duration_sec = None
        if st.get("duration_sec") is not None:
            try:
                duration_sec = round(max(5.0, float(st["duration_sec"])), 2)
            except (TypeError, ValueError):
                duration_sec = None

        cleaned_links: list[dict] = []
        rejected_urls: list[str] = []
        dead_urls: list[str] = []
        for link in st.get("news_links") or []:
            if not isinstance(link, dict):
                continue
            url = str(link.get("url") or "").strip()
            if not is_specific_event_url(url):
                if url:
                    rejected_urls.append(url)
                continue
            if verify_urls and not url_ok_map.get(url, False):
                dead_urls.append(url)
                continue
            parsed = urlparse(url)
            cleaned_links.append(
                {
                    "source": sanitise_prose(str(link.get("source") or parsed.netloc.removeprefix("www."))),
                    "title": sanitise_prose(str(link.get("title") or "Read event report")),
                    "url": url,
                }
            )
        if not cleaned_links and not allow_empty_links:
            if dead_urls:
                errors.append(
                    f"{prefix}: news_links URLs {dead_urls!r} failed HTTP HEAD/GET check (404/unreachable); supply live resolving article URLs."
                )
            elif rejected_urls:
                errors.append(
                    f"{prefix}: URLs {rejected_urls!r} look like generic hub/tag pages; supply at least one direct article URL about this event."
                )
            else:
                errors.append(f"{prefix}: at least one direct, event-specific news story URL is required.")

        cleaned_facts: list[dict] = []
        valid_link_urls = {lk["url"] for lk in cleaned_links}
        for nf in st.get("news_facts") or []:
            if not isinstance(nf, dict):
                continue
            claim = sanitise_prose(str(nf.get("claim") or ""))
            s_url = str(nf.get("source_url") or "").strip()
            if not claim or not s_url:
                continue
            if verify_urls and not url_ok_map.get(s_url, False):
                errors.append(f"{prefix}.news_facts: source_url {s_url!r} did not resolve.")
                continue
            if not is_specific_event_url(s_url):
                errors.append(f"{prefix}.news_facts: source_url {s_url!r} must be a specific article URL.")
                continue
            cleaned_facts.append({"claim": claim, "source_url": s_url})
            # Ensure cited news_facts URL is also included in news_links if room remains
            if s_url not in valid_link_urls and len(cleaned_links) < 3:
                parsed = urlparse(s_url)
                cleaned_links.append(
                    {
                        "source": parsed.netloc.removeprefix("www."),
                        "title": claim[:90],
                        "url": s_url,
                    }
                )
                valid_link_urls.add(s_url)

        stats_in = st.get("stats") or {}
        cleaned_stats = {
            "headline_metric": sanitise_prose(str(stats_in.get("headline_metric") or "")),
            "metric_label": sanitise_prose(str(stats_in.get("metric_label") or "")),
            "era5_rank": sanitise_prose(str(stats_in.get("era5_rank") or "")),
            "persistence": sanitise_prose(str(stats_in.get("persistence") or "")),
        }

        story_obj = {
            "id": re.sub(r"[^a-z0-9-]+", "-", str(st.get("id") or f"story-{idx}").lower()).strip("-"),
            "rank": idx,
            "kicker": kicker_clean,
            "headline": headline,
            "ticker": ticker,
            "voice_script": voice_script,
            **({"audio_url": audio_url} if audio_url else {}),
            **({"duration_sec": duration_sec} if duration_sec else {}),
            "region": region_clean,
            "country": country_clean,
            "hazard": hazard,
            "primary_layer": primary_layer,
            "centre": {"lat": round(c_lat, 2), "lon": round(c_lon, 2)},
            "stats": cleaned_stats,
            "camera_waypoints": dyn_waypoints,
            "cues": cleaned_cues,
            "beats": cleaned_beats,
            "news_links": cleaned_links[:3],
            "news_facts": cleaned_facts,
        }

        # Section 2.4 & Section 3: Internal consistency, calm editorial tone, & candidate numeric fact checks
        matched_cand = None
        if candidates:
            matched_cand = min(
                candidates,
                key=lambda c: haversine_km(c_lat, c_lon, float(c["lat"]), float(c["lon"])),
                default=None,
            )
        errors.extend(validate_editorial_tone_and_framing(prefix, story_obj, matched_cand=matched_cand))
        errors.extend(validate_internal_consistency(prefix, story_obj))
        if candidates is not None:
            errors.extend(validate_story_numbers_against_candidates(prefix, story_obj, candidates))

        cleaned_stories.append(story_obj)

    raw_ed_title = sanitise_prose(str(payload.get("edition_title") or "World Weather Headlines"))
    if re.search(r"\b(?:bbc|sky)\b", raw_ed_title, flags=re.I):
        errors.append(f"edition_title {raw_ed_title!r} must not reference BBC or Sky.")
        raw_ed_title = "World Weather Headlines"

    out = {
        "version": 1,
        "generated_utc": str(payload.get("generated_utc") or ""),
        "model_run_utc": str(payload.get("model_run_utc") or ""),
        "valid_time_utc": str(payload.get("valid_time_utc") or ""),
        "edition_title": raw_ed_title,
        "generator": generator_tag,
        "stories": cleaned_stories,
    }
    return out, errors
