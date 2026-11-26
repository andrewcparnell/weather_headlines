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


def sanitise_prose(text: str) -> str:
    """Enforce house style: no em dashes, no 'utilise' variants, clean whitespace."""
    if not text:
        return ""
    # Replace em dashes and spaced en dashes with standard spaced hyphens
    cleaned = text.replace("\u2014", " - ").replace(" \u2013 ", " - ")
    # Remove any accidental 'utilise' / 'utilize' variants
    cleaned = re.sub(r"\butilis(e|es|ed|ing|ation)\b", lambda m: _replace_utilise(m.group(0)), cleaned, flags=re.I)
    cleaned = re.sub(r"\butiliz(e|es|ed|ing|ation)\b", lambda m: _replace_utilise(m.group(0)), cleaned, flags=re.I)
    # Collapse repeated spaces
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


def ensure_dynamic_waypoints(centre_lat: float, centre_lon: float, waypoints: list[dict] | None) -> list[dict]:
    """Ensure 4 smooth, moving camera waypoints spanning at least 500 km across the surface."""
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

    # Synthesise a sweeping 4-waypoint broadcast glide across the event region
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
        errors.append(f"Expected 5 stories (between 3 and 7 allowed), got {len(stories) if isinstance(stories, list) else 'non-list'}.")
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

        cleaned_links: list[dict] = []
        for link in st.get("news_links") or []:
            if not isinstance(link, dict):
                continue
            url = str(link.get("url") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                continue
            if parsed.path.lower().endswith(FORBIDDEN_URL_EXTENSIONS):
                continue
            cleaned_links.append(
                {
                    "source": sanitise_prose(str(link.get("source") or parsed.netloc.removeprefix("www."))),
                    "title": sanitise_prose(str(link.get("title") or "Read coverage")),
                    "url": url,
                }
            )
        if not cleaned_links:
            errors.append(f"{prefix}: at least one valid http/https news or agency link is required.")

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
                "kicker": sanitise_prose(str(st.get("kicker") or st.get("region") or "WORLD WEATHER")).upper(),
                "headline": headline,
                "region": sanitise_prose(str(st.get("region") or "")),
                "country": sanitise_prose(str(st.get("country") or "")),
                "hazard": hazard,
                "primary_layer": primary_layer,
                "centre": {"lat": round(c_lat, 2), "lon": round(c_lon, 2)},
                "stats": cleaned_stats,
                "camera_waypoints": ensure_dynamic_waypoints(c_lat, c_lon, st.get("camera_waypoints")),
                "beats": cleaned_beats,
                "news_links": cleaned_links[:4],
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
