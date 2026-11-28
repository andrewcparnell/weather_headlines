"""Build `pipeline/data/admin_lookup.npz`: a 0.1 degree raster of countries and (for large countries) states.

Run once (or whenever borders change); the output is committed so the 6-hourly pipeline never needs the
network for place names:

    uv run python -m pipeline.make_admin_lookup

Also writes `pipeline/data/places.json.gz`, a compact list of ~7,000 populated places used to name the
nearest town to each weather hotspot.

Source: Natural Earth 1:50m admin-0 countries and admin-1 states/provinces (public domain), 1:10m admin-1
for Mexico and Argentina (not covered at 1:50m), and 1:10m populated places.
"""
from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import ROOT_DIR, USER_AGENT

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
OUT_PATH = ROOT_DIR / "pipeline" / "data" / "admin_lookup.npz"
PLACES_PATH = ROOT_DIR / "pipeline" / "data" / "places.json.gz"
RES = 0.1  # degrees per raster cell
NX, NY = int(360 / RES), int(180 / RES)

# Friendlier English names for broadcast prose
COUNTRY_NAME_OVERRIDES = {
    "United States of America": "United States",
    "People's Republic of China": "China",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of the Congo": "Republic of the Congo",
    "Turkish Republic of Northern Cyprus": "Northern Cyprus",
    "East Timor": "Timor-Leste",
    "Czech Republic": "Czechia",
    "The Gambia": "Gambia",
    "The Bahamas": "Bahamas",
    "Federated States of Micronesia": "Micronesia",
    "Republic of China": "Taiwan",
    "Macau": "Macao",
}

# Large countries where a state / province name makes a headline far more specific
ADMIN1_COUNTRIES_50M = {
    "United States of America",
    "Canada",
    "Russia",
    "China",
    "India",
    "Indonesia",
    "Brazil",
    "Australia",
    "South Africa",
}
ADMIN1_COUNTRIES_10M = {"Mexico", "Argentina"}


def _download(name: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.geojson"
    if not path.exists():
        print(f"[admin] Downloading {name}...", file=sys.stderr)
        req = urllib.request.Request(f"{NE_BASE}/{name}.geojson", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text())


def _polygons(geom: dict) -> list[list[list[tuple[float, float]]]]:
    if geom is None:
        return []
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return list(geom["coordinates"])
    return []


def _xy(lon: float, lat: float) -> tuple[float, float]:
    return (lon + 180.0) / RES, (90.0 - lat) / RES


def _burn(draw: ImageDraw.ImageDraw, geom: dict, value: int) -> None:
    for poly in _polygons(geom):
        outer, holes = poly[0], poly[1:]
        draw.polygon([_xy(x, y) for x, y in outer], fill=value)
        for hole in holes:
            draw.polygon([_xy(x, y) for x, y in hole], fill=0)


def build(cache_dir: Path) -> None:
    adm0 = _download("ne_50m_admin_0_countries", cache_dir)
    adm1_50 = _download("ne_50m_admin_1_states_provinces", cache_dir)
    adm1_10 = _download("ne_10m_admin_1_states_provinces", cache_dir)
    towns = _download("ne_10m_populated_places_simple", cache_dir)

    country_names = [""]
    country_iso2 = [""]
    img0 = Image.new("I", (NX, NY), 0)
    d0 = ImageDraw.Draw(img0)
    # Draw large features first so small enclaves/islands drawn later stay visible
    feats0 = sorted(adm0["features"], key=lambda f: -float(f["properties"].get("POP_EST") or 0))
    for f in feats0:
        p = f["properties"]
        raw = str(p.get("NAME_EN") or p.get("NAME") or "").strip()
        name = COUNTRY_NAME_OVERRIDES.get(raw, raw)
        country_names.append(name)
        iso2 = str(p.get("ISO_A2_EH") or p.get("ISO_A2") or "").strip()
        country_iso2.append(iso2 if len(iso2) == 2 and iso2 != "-9" else "")
        _burn(d0, f["geometry"], len(country_names) - 1)

    state_names = [""]
    state_country = [""]
    img1 = Image.new("I", (NX, NY), 0)
    d1 = ImageDraw.Draw(img1)
    feats1 = [f for f in adm1_50["features"] if f["properties"].get("admin") in ADMIN1_COUNTRIES_50M]
    feats1 += [f for f in adm1_10["features"] if f["properties"].get("admin") in ADMIN1_COUNTRIES_10M]
    for f in feats1:
        p = f["properties"]
        name = str(p.get("name_en") or p.get("name") or "").strip()
        if not name:
            continue
        admin = str(p.get("admin") or "")
        state_names.append(name)
        state_country.append(COUNTRY_NAME_OVERRIDES.get(admin, admin))
        _burn(d1, f["geometry"], len(state_names) - 1)

    a0 = np.asarray(img0, dtype=np.int32).astype(np.uint16)
    a1 = np.asarray(img1, dtype=np.int32).astype(np.uint16)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        res=np.float32(RES),
        admin0=a0,
        admin1=a1,
        country_names=np.array(country_names),
        country_iso2=np.array(country_iso2),
        state_names=np.array(state_names),
        state_country=np.array(state_country),
    )
    # Compact populated-places list: [name, lat, lon, population, admin-1 name, country]
    rows = []
    for f in towns["features"]:
        p = f["properties"]
        name = str(p.get("name") or p.get("nameascii") or "").strip()
        if not name:
            continue
        country = COUNTRY_NAME_OVERRIDES.get(str(p.get("adm0name") or ""), str(p.get("adm0name") or ""))
        rows.append(
            [
                name,
                round(float(p["latitude"]), 3),
                round(float(p["longitude"]), 3),
                int(p.get("pop_max") or 0),
                str(p.get("adm1name") or ""),
                country,
            ]
        )
    PLACES_PATH.write_bytes(gzip.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    print(f"[admin] Wrote {PLACES_PATH} ({PLACES_PATH.stat().st_size / 1024:.0f} KB, {len(rows)} places).", file=sys.stderr)

    kb = OUT_PATH.stat().st_size / 1024
    print(
        f"[admin] Wrote {OUT_PATH} ({kb:.0f} KB, {len(country_names) - 1} countries, {len(state_names) - 1} states).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    build(ROOT_DIR / "pipeline_cache" / "naturalearth")
