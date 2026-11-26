"""Validate and sanitise headlines.json before publishing to the static site."""
from __future__ import annotations

import math
import re
from urllib.parse import urlparse

from .config import LAYERS
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
    # Allow specific GDACS/ReliefWeb event reports with query IDs or deep article slugs
    if "eventid=" in parsed.query.lower():
        return True
    segments = [seg for seg in low_path.split("/") if seg]
    if not segments:
        return False
    last_seg = segments[-1]
    # A real news article slug almost always has multiple words or digits, or sits >= 2 levels deep
    has_article_slug = ("-" in last_seg and len(last_seg) >= 12) or any(ch.isdigit() for ch in low_path)
    return has_article_slug and len(segments) >= 1


def sanitise_prose(text: str) -> str:
    """Enforce house style: no em dashes, no 'utilise' variants, clean whitespace."""
    if not text:
        return ""
    cleaned = text.replace("\u2014", " - ").replace(" \u2013 ", " - ")
    cleaned = re.sub(r"\butilis(e|es|ed|ing|ation)\b", lambda m: _replace_utilise(m.group(0)), cleaned, flags=re.I)
    cleaned = re.sub(r"\butiliz(e|es|ed|ing|ation)\b", lambda m: _replace_utilise(m.group(0)), cleaned, flags=re.I)
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
    # Highlight measurements (+16.4 °C, 139 mm, 147 km/h, 975 hPa, #1, 99th percentile, N days/weeks/years)
    pattern = re.compile(
        r"([+-]?\d+(?:\.\d+)?\s*(?:°C|°F|mm|km/h|mph|m/s|hPa|%)"
        r"|#\d+(?:\s+in\s+\d+\s+years)?"
        r"|\d+(?:\.\d+)?(?:st|nd|rd|th)\s+percentile"
        r"|\b\d+\s+(?:days|weeks|years|rivers|counties)\b)",
        flags=re.I,
    )
    return pattern.sub(r"**\1**", text)


def build_ticker_from_beats(beats: list[dict]) -> str:
    parts = [auto_highlight_keywords(b["text"]) for b in beats if b.get("text")]
    return "  ///  ".join(parts)


def infer_cue_layer(text: str, fallback: str) -> str:
    """Match the weather variable layer to the meteorological phenomenon discussed in a sentence."""
    low = (text or "").lower()
    if any(k in low for k in ("1940-1979", "warmed", "climate", "above normal", "below normal", "above seasonal")):
        return "anomaly"
    if any(k in low for k in ("hpa", "blocking high", "trough", "isobar", "depression")):
        return "pressure"
    if any(k in low for k in ("km/h", "mph", "m/s", "gust", "gale", "scirocco", "winds")):
        return "wind"
    if any(k in low for k in (" mm", "rain", "downpour", "deluge", "rivers", "flood")):
        return "rain"
    if "°c" in low or "degrees" in low:
        return "temperature"
    return fallback if fallback in VALID_LAYERS else "temperature"


def clean_or_build_cues(
    st: dict,
    cleaned_beats: list[dict],
    waypoints: list[dict],
    primary_layer: str,
    kicker: str,
) -> list[dict]:
    """Validate explicit sentence-level `cues` or synthesise them from beats + waypoints."""
    raw_cues = st.get("cues")
    default_bearings = [-24, 22, -18, 24, -20, 18]
    cues_out: list[dict] = []

    if isinstance(raw_cues, list) and len(raw_cues) >= 3:
        for i, c in enumerate(raw_cues[:8]):
            if not isinstance(c, dict):
                continue
            wp_fb = waypoints[min(i, len(waypoints) - 1)]
            try:
                lat = round(max(-82.0, min(82.0, float(c.get("lat", wp_fb["lat"])))), 2)
                lon = round(((float(c.get("lon", wp_fb["lon"])) + 180.0) % 360.0) - 180.0, 2)
                dist = round(max(1.36, min(2.6, float(c.get("distance", wp_fb["distance"])))), 2)
                bearing = round(max(-45.0, min(45.0, float(c.get("bearing", default_bearings[i % len(default_bearings)])))), 1)
            except (TypeError, ValueError):
                lat, lon, dist = wp_fb["lat"], wp_fb["lon"], wp_fb["distance"]
                bearing = default_bearings[i % len(default_bearings)]

            spoken = sanitise_prose(str(c.get("spoken") or c.get("text") or ""))
            ticker_part = auto_highlight_keywords(sanitise_prose(str(c.get("ticker") or spoken)))
            raw_layer = str(c.get("layer") or "").strip().lower()
            layer = raw_layer if raw_layer in VALID_LAYERS else infer_cue_layer(ticker_part, primary_layer)

            entry: dict = {
                "layer": layer,
                "lat": lat,
                "lon": lon,
                "distance": dist,
                "bearing": bearing,
                "label": sanitise_prose(str(c.get("label") or kicker)).upper(),
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

    if len(cues_out) >= 3:
        return cues_out

    # Fallback: construct sentence-synced cues from the 4 beats + dynamic waypoints
    for i, beat in enumerate(cleaned_beats):
        wp = waypoints[min(i, len(waypoints) - 1)]
        b_txt = beat["text"]
        b_layer = beat["layer"]
        # Ensure consecutive cues vary layers when the prose discusses a secondary variable
        inferred = infer_cue_layer(b_txt, b_layer)
        if i > 0 and cues_out[-1]["layer"] == b_layer and inferred != b_layer:
            b_layer = inferred
        cues_out.append(
            {
                "layer": b_layer,
                "lat": wp["lat"],
                "lon": wp["lon"],
                "distance": round(max(1.42, min(2.2, wp["distance"] - (0.14 if i in (1, 2) else 0.0))), 2),
                "bearing": default_bearings[i % len(default_bearings)],
                "label": (beat.get("title") or kicker)[:34].upper(),
                "metric": beat.get("tag") or "",
                "spoken": b_txt.replace("**", ""),
                "ticker": auto_highlight_keywords(b_txt),
            }
        )
    return cues_out


def ensure_dynamic_waypoints(centre_lat: float, centre_lon: float, waypoints: list[dict] | None) -> list[dict]:
    """Ensure 4 smooth, moving camera waypoints spanning at least 450 km across the surface."""
    cleaned: list[dict] = []
    for wp in waypoints or []:
        try:
            la = max(-82.0, min(82.0, float(wp["lat"])))
            lo = ((float(wp["lon"]) + 180.0) % 360.0) - 180.0
            dist = max(1.42, min(2.9, float(wp.get("distance", 1.95))))
            cleaned.append({"lat": round(la, 2), "lon": round(lo, 2), "distance": round(dist, 2)})
        except (KeyError, TypeError, ValueError):
            continue

    total_km = 0.0
    for a, b in zip(cleaned, cleaned[1:]):
        total_km += haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])

    if len(cleaned) >= 3 and total_km >= 450.0:
        return cleaned[:5]

    cos_lat = max(0.35, math.cos(math.radians(centre_lat)))
    dlon = 9.0 / cos_lat
    raw = [
        (centre_lat - 4.5, centre_lon - dlon, 2.22),
        (centre_lat - 1.0, centre_lon - dlon * 0.25, 1.84),
        (centre_lat + 2.5, centre_lon + dlon * 0.45, 1.78),
        (centre_lat + 5.0, centre_lon + dlon, 2.08),
    ]
    return [
        {
            "lat": round(max(-82.0, min(82.0, la)), 2),
            "lon": round(((lo + 180.0) % 360.0) - 180.0, 2),
            "distance": round(d, 2),
        }
        for la, lo, d in raw
    ]


def validate_and_clean_bulletin(payload: dict) -> tuple[dict, list[str]]:
    """Validate and sanitise a bulletin dict. Returns (cleaned_payload, errors)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {}, ["Bulletin root must be a JSON object."]

    stories = payload.get("stories")
    if not isinstance(stories, list) or not (3 <= len(stories) <= 7):
        errors.append(
            f"Expected 5 stories (between 3 and 7 allowed), got {len(stories) if isinstance(stories, list) else 'non-list'}."
        )
        return payload, errors

    cleaned_stories: list[dict] = []
    for idx, st in enumerate(stories, start=1):
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

        dyn_waypoints = ensure_dynamic_waypoints(c_lat, c_lon, st.get("camera_waypoints"))
        kicker_clean = sanitise_prose(str(st.get("kicker") or st.get("region") or "WORLD WEATHER")).upper()
        cleaned_cues = clean_or_build_cues(st, cleaned_beats, dyn_waypoints, primary_layer, kicker_clean)

        raw_ticker = sanitise_prose(str(st.get("ticker") or ""))
        if len(raw_ticker) >= 60:
            ticker = auto_highlight_keywords(raw_ticker)
        else:
            ticker = "  ///  ".join(c["ticker"] for c in cleaned_cues if c.get("ticker"))

        voice_script = sanitise_prose(str(st.get("voice_script") or ""))
        if len(voice_script) < 40:
            voice_script = " ".join(c["spoken"] for c in cleaned_cues if c.get("spoken"))
        audio_url = str(st.get("audio_url") or "").strip()
        duration_sec = None
        if st.get("duration_sec") is not None:
            try:
                duration_sec = round(max(5.0, float(st["duration_sec"])), 2)
            except (TypeError, ValueError):
                duration_sec = None

        cleaned_links: list[dict] = []
        for link in st.get("news_links") or []:
            if not isinstance(link, dict):
                continue
            url = str(link.get("url") or "").strip()
            if not is_specific_event_url(url):
                errors.append(
                    f"{prefix}: URL {url!r} looks like a generic hub/tag page rather than a direct article about this event."
                )
                continue
            parsed = urlparse(url)
            cleaned_links.append(
                {
                    "source": sanitise_prose(str(link.get("source") or parsed.netloc.removeprefix("www."))),
                    "title": sanitise_prose(str(link.get("title") or "Read event report")),
                    "url": url,
                }
            )
        if not cleaned_links:
            errors.append(f"{prefix}: at least one direct, event-specific news story URL is required.")

        stats_in = st.get("stats") or {}
        cleaned_stats = {
            "headline_metric": sanitise_prose(str(stats_in.get("headline_metric") or "")),
            "metric_label": sanitise_prose(str(stats_in.get("metric_label") or "")),
            "era5_rank": sanitise_prose(str(stats_in.get("era5_rank") or "")),
            "persistence": sanitise_prose(str(stats_in.get("persistence") or "")),
        }

        cleaned_stories.append(
            {
                "id": re.sub(r"[^a-z0-9-]+", "-", str(st.get("id") or f"story-{idx}").lower()).strip("-"),
                "rank": idx,
                "kicker": kicker_clean,
                "headline": headline,
                "ticker": ticker,
                "voice_script": voice_script,
                **({"audio_url": audio_url} if audio_url else {}),
                **({"duration_sec": duration_sec} if duration_sec else {}),
                "region": sanitise_prose(str(st.get("region") or "")),
                "country": sanitise_prose(str(st.get("country") or "")),
                "hazard": hazard,
                "primary_layer": primary_layer,
                "centre": {"lat": round(c_lat, 2), "lon": round(c_lon, 2)},
                "stats": cleaned_stats,
                "camera_waypoints": dyn_waypoints,
                "cues": cleaned_cues,
                "beats": cleaned_beats,
                "news_links": cleaned_links[:3],
            }
        )

    out = {
        "version": 1,
        "generated_utc": str(payload.get("generated_utc") or ""),
        "model_run_utc": str(payload.get("model_run_utc") or ""),
        "valid_time_utc": str(payload.get("valid_time_utc") or ""),
        "edition_title": sanitise_prose(str(payload.get("edition_title") or "World Weather Headlines")),
        "generator": str(payload.get("generator") or "claude"),
        "stories": cleaned_stories,
    }
    return out, errors
