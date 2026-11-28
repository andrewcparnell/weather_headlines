"""Offline reverse geocoding: country, state/province (for large countries) and nearest town for a lat/lon.

Uses the compact Natural Earth rasters and town list built by `pipeline.make_admin_lookup`.
"""
from __future__ import annotations

import gzip
import json
import math
from functools import lru_cache

import numpy as np

from .config import ROOT_DIR
from .scanner import haversine_km

DATA_DIR = ROOT_DIR / "pipeline" / "data"

# Short country names used in headline prefixes such as "Oklahoma, USA:"
COUNTRY_SHORT_NAMES = {
    "United States": "USA",
}

# Countries large enough that the state / province is worth naming in the headline
LARGE_COUNTRIES = {
    "United States",
    "Canada",
    "Russia",
    "China",
    "India",
    "Indonesia",
    "Brazil",
    "Australia",
    "South Africa",
    "Mexico",
    "Argentina",
}

MAX_OFFSHORE_SEARCH_KM = 450.0


@lru_cache(maxsize=1)
def _admin() -> dict | None:
    path = DATA_DIR / "admin_lookup.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    return {
        "res": float(npz["res"]),
        "admin0": npz["admin0"],
        "admin1": npz["admin1"],
        "country_names": [str(x) for x in npz["country_names"]],
        "country_iso2": [str(x) for x in npz["country_iso2"]] if "country_iso2" in npz.files else [],
        "state_names": [str(x) for x in npz["state_names"]],
        "state_country": [str(x) for x in npz["state_country"]],
    }


@lru_cache(maxsize=1)
def _towns() -> list[list]:
    path = DATA_DIR / "places.json.gz"
    if not path.exists():
        return []
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def _cell(lat: float, lon: float, res: float, shape: tuple[int, int]) -> tuple[int, int]:
    iy = int(min(shape[0] - 1, max(0, math.floor((90.0 - lat) / res))))
    ix = int(math.floor(((lon + 180.0) % 360.0) / res)) % shape[1]
    return iy, ix


def _nearest_land_cell(lat: float, lon: float, adm: dict) -> tuple[int, int, float] | None:
    """Return (iy, ix, distance_km) of the nearest country cell within MAX_OFFSHORE_SEARCH_KM."""
    a0 = adm["admin0"]
    res = adm["res"]
    iy, ix = _cell(lat, lon, res, a0.shape)
    if a0[iy, ix]:
        return iy, ix, 0.0
    max_cells = int(MAX_OFFSHORE_SEARCH_KM / (111.0 * res)) + 1
    cos_lat = max(0.15, math.cos(math.radians(lat)))
    best = None
    for r in range(1, max_cells + 1):
        rx = min(a0.shape[1] // 2, int(math.ceil(r / cos_lat)))
        y0, y1 = max(0, iy - r), min(a0.shape[0], iy + r + 1)
        xs = np.arange(ix - rx, ix + rx + 1) % a0.shape[1]
        win = a0[y0:y1][:, xs]
        hits = np.argwhere(win > 0)
        if hits.size:
            for hy, hx in hits[:: max(1, len(hits) // 200)]:
                cy, cx = y0 + int(hy), int(xs[hx])
                clat = 90.0 - (cy + 0.5) * res
                clon = (cx + 0.5) * res - 180.0
                d = haversine_km(lat, lon, clat, clon)
                if best is None or d < best[2]:
                    best = (cy, cx, d)
            if best is not None and best[2] <= r * 111.0 * res:
                break
    if best is None or best[2] > MAX_OFFSHORE_SEARCH_KM:
        return None
    return best


def nearest_town(lat: float, lon: float, country: str | None = None, min_pop: int = 0) -> dict | None:
    """Nearest populated place (optionally restricted to `country`), with distance and bearing words."""
    best = None
    for name, tlat, tlon, pop, adm1, tcountry in _towns():
        if pop < min_pop:
            continue
        if country and tcountry and tcountry != country:
            continue
        if abs(tlat - lat) > 8.0:
            continue
        d = haversine_km(lat, lon, tlat, tlon)
        if best is None or d < best["distance_km"]:
            best = {
                "name": name,
                "lat": tlat,
                "lon": tlon,
                "population": pop,
                "admin1": adm1,
                "country": tcountry,
                "distance_km": round(d),
            }
    return best


def _compass(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> str:
    """Compass direction of `to` as seen from `from` (e.g. 'north-east')."""
    dlon = math.radians(((to_lon - from_lon + 180.0) % 360.0) - 180.0)
    la1, la2 = math.radians(from_lat), math.radians(to_lat)
    x = math.sin(dlon) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    brg = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return names[int((brg + 22.5) // 45) % 8]


def describe_location(lat: float, lon: float) -> dict:
    """Describe where (lat, lon) is in plain words for headlines and narration.

    Returns keys: country, state, offshore, offshore_km, headline_prefix, place_phrase,
    nearest_town, nearest_town_km, nearest_town_direction, nearest_big_town, nearest_big_town_km.
    """
    adm = _admin()
    out: dict = {
        "country": "",
        "state": "",
        "offshore": False,
        "offshore_km": 0,
        "headline_prefix": "",
        "place_phrase": "",
        "nearest_town": "",
        "nearest_town_km": None,
        "nearest_town_direction": "",
        "nearest_big_town": "",
        "nearest_big_town_km": None,
        "country_domain": "",
    }
    if adm is None:
        return out

    land = _nearest_land_cell(lat, lon, adm)
    if land is None:
        return out
    cy, cx, dist_km = land
    c_idx = int(adm["admin0"][cy, cx])
    country = adm["country_names"][c_idx]
    iso2 = adm["country_iso2"][c_idx] if c_idx < len(adm["country_iso2"]) else ""
    out["country_domain"] = {"GB": "uk"}.get(iso2, iso2.lower()) if iso2 else ""
    state = ""
    if country in LARGE_COUNTRIES:
        s_idx = int(adm["admin1"][cy, cx])
        if s_idx and adm["state_country"][s_idx] == country:
            state = adm["state_names"][s_idx]

    offshore = dist_km > 25.0
    short_country = COUNTRY_SHORT_NAMES.get(country, country)
    where = f"{state}, {short_country}" if state else country

    out.update(
        {
            "country": country,
            "state": state,
            "offshore": offshore,
            "offshore_km": round(dist_km),
            "headline_prefix": where,
        }
    )

    town = nearest_town(lat, lon, country=country)
    big = nearest_town(lat, lon, country=country, min_pop=250_000)
    if town is not None:
        out["nearest_town"] = town["name"]
        out["nearest_town_km"] = town["distance_km"]
        out["nearest_town_direction"] = _compass(town["lat"], town["lon"], lat, lon)
    if big is not None:
        out["nearest_big_town"] = big["name"]
        out["nearest_big_town_km"] = big["distance_km"]

    if offshore:
        out["place_phrase"] = f"off the coast of {where}"
    elif town is not None and town["distance_km"] <= 200:
        # Avoid "Novosibirsk, Novosibirsk, Russia" when a town shares its state's name
        town_where = short_country if state and town["name"] == state else where
        if town["distance_km"] <= 40:
            out["place_phrase"] = f"around {town['name']}, {town_where}"
        else:
            out["place_phrase"] = (
                f"{town['distance_km']} km {out['nearest_town_direction']} of {town['name']}, {town_where}"
            )
    else:
        out["place_phrase"] = f"in {'remote parts of ' if town is None or town['distance_km'] > 200 else ''}{where}"
    return out


def _nearby_low_centre(lat: float, lon: float, mslp_hpa: np.ndarray, radius_km: float = 900.0) -> dict | None:
    """Deepest sea-level pressure within `radius_km` of (lat, lon) on the 0.25 degree grid."""
    ny, nx = mslp_hpa.shape
    res = 180.0 / (ny - 1)
    iy = int(round((90.0 - lat) / res))
    ix = int(round(((lon + 180.0) % 360.0) / res)) % nx
    r_lat = int(radius_km / (111.0 * res)) + 1
    r_lon = min(nx // 2, int(radius_km / (111.0 * res * max(0.15, math.cos(math.radians(lat))))) + 1)
    y0, y1 = max(0, iy - r_lat), min(ny, iy + r_lat + 1)
    xs = np.arange(ix - r_lon, ix + r_lon + 1) % nx
    win = np.array(mslp_hpa[y0:y1][:, xs], dtype=np.float32)
    # Mask grid cells outside the search radius (great-circle distance) before finding the minimum
    lats_w = np.radians(90.0 - np.arange(y0, y1) * res)[:, None]
    lons_w = np.radians(xs * res - 180.0)[None, :]
    la0, lo0 = math.radians(lat), math.radians(lon)
    cosd = np.sin(la0) * np.sin(lats_w) + np.cos(la0) * np.cos(lats_w) * np.cos(lons_w - lo0)
    dist_km = 6371.0 * np.arccos(np.clip(cosd, -1.0, 1.0))
    win[dist_km > radius_km] = np.nan
    if not np.isfinite(win).any():
        return None
    flat = int(np.nanargmin(win))
    wy, wx = divmod(flat, win.shape[1])
    clat = 90.0 - (y0 + wy) * res
    clon = float(xs[wx]) * res - 180.0
    dist = haversine_km(lat, lon, clat, clon)
    if dist > 0.85 * radius_km:
        return None  # the minimum sits on the search edge: part of a more distant system, not a local low
    return {
        "mslp_hpa": round(float(win[wy, wx]), 1),
        "lat": round(clat, 2),
        "lon": round(clon, 2),
        "distance_km": round(dist),
        "direction_from_hotspot": _compass(lat, lon, clat, clon),
    }


def _nearest_mapped_low(lat: float, lon: float, centres: list[dict], radius_km: float = 800.0) -> dict | None:
    """Nearest 'L' centre from the map's own pressure-centre list (so narration matches the L label shown)."""
    best = None
    for c in centres or []:
        if c.get("kind") != "L":
            continue
        d = haversine_km(lat, lon, float(c["lat"]), float(c["lon"]))
        if d <= radius_km and (best is None or d < best[0]):
            best = (d, c)
    if best is None:
        return None
    d, c = best
    return {
        "mslp_hpa": float(c["hpa"]),
        "lat": float(c["lat"]),
        "lon": float(c["lon"]),
        "distance_km": round(d),
        "direction_from_hotspot": _compass(lat, lon, float(c["lat"]), float(c["lon"])),
    }


def attach_location_context(candidates: list[dict], weather, pressure_centres: list[dict] | None = None) -> list[dict]:
    """Add plain-language `location` (country, state, nearest towns) and, for storms, the nearby low centre."""
    for cand in candidates:
        lat, lon = float(cand["lat"]), float(cand["lon"])
        loc = describe_location(lat, lon)
        cand["location"] = loc
        if loc.get("country"):
            cand["country"] = loc["country"]
        if cand.get("kind") in ("cyclone", "wind", "pressure", "rain"):
            if pressure_centres is not None:
                low = _nearest_mapped_low(lat, lon, pressure_centres)
            else:
                low = _nearby_low_centre(lat, lon, weather.mslp_hpa)
            if low is not None and low["mslp_hpa"] < 1000.0:
                low_loc = describe_location(low["lat"], low["lon"])
                low["near"] = low_loc.get("place_phrase") or ""
                cand["low_centre"] = low
    return candidates
