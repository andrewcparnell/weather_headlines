"""Objective global weather hotspot scanner, pressure centre detector, and isobar tracer."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .config import SITE_DIR
from .opendata import LiveWeatherFields

_POPULATED_COORDS: tuple[np.ndarray, np.ndarray] | None = None


def _populated_proximity_weight_05(lats_05: np.ndarray, lons_05: np.ndarray) -> np.ndarray:
    """Weight [361, 720] favouring populated regions near gazetteer cities over uninhabited tundra/ocean."""
    global _POPULATED_COORDS
    if _POPULATED_COORDS is None:
        gaz_path = SITE_DIR / "assets" / "gazetteer.json"
        if gaz_path.exists():
            raw = json.loads(gaz_path.read_text())
            pts = [(float(e["lat"]), float(e["lon"])) for e in raw if e.get("kind") == "City"]
            _POPULATED_COORDS = (
                np.array([p[0] for p in pts], dtype=np.float32),
                np.array([p[1] for p in pts], dtype=np.float32),
            )
        else:
            _POPULATED_COORDS = (np.array([], dtype=np.float32), np.array([], dtype=np.float32))

    c_lats, c_lons = _POPULATED_COORDS
    if len(c_lats) == 0:
        return np.ones((len(lats_05), len(lons_05)), dtype=np.float32)

    lat_rad = np.radians(lats_05)[:, None]
    lon_rad = np.radians(lons_05)[None, :]
    min_d_km = np.full((len(lats_05), len(lons_05)), 9999.0, dtype=np.float32)

    # Vectorised great-circle distance to each gazetteer city (~120 cities -> ~12 ms total)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    for cla, clo in zip(c_lats, c_lons):
        p_c = math.radians(float(cla))
        l_c = math.radians(float(clo))
        cos_ang = np.clip(
            sin_lat * math.sin(p_c) + cos_lat * math.cos(p_c) * np.cos(lon_rad - l_c),
            -1.0,
            1.0,
        )
        d_km = 6371.0 * np.arccos(cos_ang)
        min_d_km = np.minimum(min_d_km, d_km)

    return np.where(min_d_km <= 450.0, 1.0, np.where(min_d_km <= 850.0, 0.62, 0.28)).astype(np.float32)


@dataclass
class WeatherCandidate:
    kind: str  # "heat", "cold", "rain", "cyclone", "wind"
    primary_layer: str  # "temperature", "anomaly", "rain", "wind", "pressure"
    lat: float
    lon: float
    score: float
    peak_value: float
    units: str
    t2m_c: float
    anom_c: float
    z_score: float
    rain_24h_mm: float
    wind_ms: float
    gust_ms: float
    mslp_hpa: float
    on_land: bool
    summary_hint: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _smooth_periodic(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    """Light Gaussian-like 3x3 smoothing periodic in longitude."""
    out = arr.astype(np.float32, copy=True)
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        padded = np.concatenate([padded[:, -1:], padded, padded[:, :1]], axis=1)
        out = (
            4.0 * padded[1:-1, 1:-1]
            + 2.0 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
            + (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:])
        ) / 16.0
    return out


def _coastal_land_mask(lsm: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask True over land and within ~125 km of coastlines (excludes uninhabited open ocean)."""
    if lsm is None or lsm.shape != shape:
        return np.ones(shape, dtype=bool)
    land = lsm > 0.30
    expanded = land.copy()
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            expanded |= np.roll(np.roll(land, dy, axis=0), dx, axis=1)
    return expanded


def _terrain_mask_05(
    ifs_elev_025_m: np.ndarray | None,
    era5_elev_025_m: np.ndarray | None,
    ifs_sdor_025_m: np.ndarray | None,
    shape_05: tuple[int, int],
) -> np.ndarray:
    """Boolean mask True where terrain is trustworthy (excludes steep relief & >300 m IFS/ERA5 mismatch)."""
    valid = np.ones(shape_05, dtype=bool)
    if ifs_elev_025_m is not None:
        elev_05 = ifs_elev_025_m[::2, ::2]
        if era5_elev_025_m is not None and era5_elev_025_m.shape == ifs_elev_025_m.shape:
            diff_05 = np.abs(elev_05 - era5_elev_025_m[::2, ::2].astype(np.float32))
            valid &= diff_05 <= 300.0
        # Exclude steep local elevation jumps (>600 m across 3x3 0.5 deg neighbourhood)
        pad = np.pad(elev_05, ((1, 1), (0, 0)), mode="edge")
        pad = np.concatenate([pad[:, -1:], pad, pad[:, :1]], axis=1)
        loc_max = np.full_like(elev_05, -np.inf)
        loc_min = np.full_like(elev_05, np.inf)
        for dy in range(3):
            for dx in range(3):
                sl = pad[dy : dy + shape_05[0], dx : dx + shape_05[1]]
                loc_max = np.maximum(loc_max, sl)
                loc_min = np.minimum(loc_min, sl)
        valid &= (loc_max - loc_min) <= 600.0
    if ifs_sdor_025_m is not None:
        valid &= ifs_sdor_025_m[::2, ::2] <= 250.0
    return valid


def detect_global_candidates(
    weather: LiveWeatherFields,
    climo_mean_c: np.ndarray,
    climo_std_c: np.ndarray,
    max_candidates: int = 14,
    era5_elev_025_m: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Scan global 0.25 deg fields and return spatially diverse candidate hotspots plus (anom_c, z_t2m)."""
    anom_c = (weather.t2m_c - climo_mean_c).astype(np.float32)
    z_t2m = (anom_c / np.maximum(climo_std_c, 1.0)).astype(np.float32)

    # Subsample to 0.5 deg for fast candidate ranking while keeping exact local maxima
    lats = weather.lats[::2]
    lons = weather.lons[::2]
    t2m = weather.t2m_c[::2, ::2]
    tmax_24 = (weather.t2m_max_24h_c if weather.t2m_max_24h_c is not None else weather.t2m_c)[::2, ::2]
    tmin_24 = (weather.t2m_min_24h_c if weather.t2m_min_24h_c is not None else weather.t2m_c)[::2, ::2]
    anom = _smooth_periodic(anom_c[::2, ::2], passes=1)
    z_sm = _smooth_periodic(z_t2m[::2, ::2], passes=1)
    rain = weather.rain_24h_mm[::2, ::2]
    wind = weather.wind_speed[::2, ::2]
    gust = weather.gust_max[::2, ::2]
    mslp = _smooth_periodic(weather.mslp_hpa[::2, ::2], passes=1)
    near_land = _coastal_land_mask(weather.lsm[::2, ::2] if weather.lsm is not None else None, t2m.shape)
    clean_terrain = _terrain_mask_05(weather.ifs_elev_m, era5_elev_025_m, weather.ifs_sdor_m, t2m.shape)

    lat_2d = lats[:, None]
    habitable = (lat_2d > -58.0) & (lat_2d < 71.5)  # exclude Antarctica and high-polar sea ice
    pop_weight = _populated_proximity_weight_05(lats, lons)

    raw_pool: list[WeatherCandidate] = []

    def _add_top_points(
        score_grid: np.ndarray,
        kind: str,
        layer: str,
        val_grid: np.ndarray,
        units: str,
        count: int,
        min_sep_km: float = 1600.0,
    ) -> None:
        flat_order = np.argsort(score_grid.ravel())[::-1]
        picked_here: list[tuple[float, float]] = []
        nx = score_grid.shape[1]
        for idx in flat_order:
            sc = float(score_grid.ravel()[idx])
            if not math.isfinite(sc) or sc <= 0.0:
                break
            iy, ix = divmod(int(idx), nx)
            la, lo = float(lats[iy]), float(lons[ix])
            if any(haversine_km(la, lo, p[0], p[1]) < min_sep_km for p in picked_here):
                continue
            picked_here.append((la, lo))
            on_l = bool(near_land[iy, ix])
            pv = float(val_grid[iy, ix])
            hint = (
                f"{kind.upper()} signal at ({la:+.1f}, {lo:+.1f}): "
                f"peak={pv:.1f} {units}, T2m={float(t2m[iy, ix]):.1f} C (anom {float(anom[iy, ix]):+.1f} C, z={float(z_sm[iy, ix]):+.1f}), "
                f"24h rain={float(rain[iy, ix]):.1f} mm, wind={float(wind[iy, ix]):.1f} m/s (gust {float(gust[iy, ix]):.1f} m/s), "
                f"MSLP={float(mslp[iy, ix]):.1f} hPa"
            )
            raw_pool.append(
                WeatherCandidate(
                    kind=kind,
                    primary_layer=layer,
                    lat=round(la, 2),
                    lon=round(lo, 2),
                    score=round(sc, 2),
                    peak_value=round(pv, 1),
                    units=units,
                    t2m_c=round(float(t2m[iy, ix]), 1),
                    anom_c=round(float(anom[iy, ix]), 1),
                    z_score=round(float(z_sm[iy, ix]), 2),
                    rain_24h_mm=round(float(rain[iy, ix]), 1),
                    wind_ms=round(float(wind[iy, ix]), 1),
                    gust_ms=round(float(gust[iy, ix]), 1),
                    mslp_hpa=round(float(mslp[iy, ix]), 1),
                    on_land=on_l,
                    summary_hint=hint,
                )
            )
            if len(picked_here) >= count:
                break

    # 1. Extreme heat / mildness anomalies (exclude high-relief / >300 m orography-mismatch cells; prefer populated places)
    heat_weight = np.where(near_land & habitable & clean_terrain, 1.35 * pop_weight, 0.0)
    abs_heat_bonus = np.clip((tmax_24 - 32.0) * 0.18, 0.0, 3.0)
    heat_score = np.where(anom > 5.0, (anom * 0.55 + np.maximum(0.0, z_sm) * 1.8 + abs_heat_bonus) * heat_weight, 0.0)
    _add_top_points(heat_score, "heat", "anomaly", anom, "°C", count=5)

    # 2. Severe cold outbreaks (strong negative anomalies over trustworthy populated land terrain)
    cold_weight = np.where(near_land & habitable & clean_terrain, 1.35 * pop_weight, 0.0)
    cold_score = np.where(anom < -6.0, ((-anom) * 0.55 + np.maximum(0.0, -z_sm) * 1.8) * cold_weight, 0.0)
    _add_top_points(cold_score, "cold", "anomaly", anom, "°C", count=4)

    # 3. Heavy rainfall (24h rain >= 45 mm over inhabited land/coastal zones; exclude open-ocean ITCZ)
    rain_weight = np.where(near_land & habitable, 1.5 * pop_weight, 0.0)
    rain_score = np.where(rain >= 45.0, (rain / 12.0) * rain_weight, 0.0)
    _add_top_points(rain_score, "rain", "rain", rain, "mm", count=5)

    # 4. Cyclones, deep lows, and severe windstorms affecting populated land or coastal regions
    tropical = np.abs(lat_2d) <= 32.0
    low_depth = np.where(tropical, np.maximum(0.0, 1006.0 - mslp) * 0.45, np.maximum(0.0, 988.0 - mslp) * 0.28)
    storm_land_weight = np.where(near_land & habitable, 1.35 * np.maximum(0.55, pop_weight), 0.0)
    storm_score = np.where(
        (gust >= 24.0) | (low_depth > 4.0),
        (np.maximum(0.0, gust - 16.0) * 0.55 + low_depth) * storm_land_weight,
        0.0,
    )
    _add_top_points(storm_score, "cyclone", "wind", gust, "m/s", count=5)

    # Deduplicate co-located multi-hazard signals (e.g. a typhoon's rain + wind -> keep highest score, note both)
    raw_pool.sort(key=lambda c: c.score, reverse=True)
    deduped: list[WeatherCandidate] = []
    for cand in raw_pool:
        existing = next((d for d in deduped if haversine_km(cand.lat, cand.lon, d.lat, d.lon) < 1100.0), None)
        if existing is not None:
            continue
        deduped.append(cand)
        if len(deduped) >= max_candidates:
            break

    serialised = []
    for i, c in enumerate(deduped):
        iy05 = int(round((90.0 - c.lat) / 0.5))
        ix05 = int(round(((c.lon + 180.0) % 360.0) / 0.5)) % 720
        serialised.append(
            {
                "id": f"cand-{i + 1:02d}-{c.kind}",
                "kind": c.kind,
                "primary_layer": c.primary_layer,
                "lat": c.lat,
                "lon": c.lon,
                "score": c.score,
                "peak_value": c.peak_value,
                "units": c.units,
                "t2m_c": c.t2m_c,
                "t2m_max_24h_c": round(float(tmax_24[iy05, ix05]), 1),
                "t2m_min_24h_c": round(float(tmin_24[iy05, ix05]), 1),
                "anom_c": c.anom_c,
                "z_score": c.z_score,
                "rain_24h_mm": c.rain_24h_mm,
                "wind_ms": c.wind_ms,
                "wind_kmh": round(c.wind_ms * 3.6, 0),
                "gust_ms": c.gust_ms,
                "gust_kmh": round(c.gust_ms * 3.6, 0),
                "mslp_hpa": c.mslp_hpa,
                "on_land": c.on_land,
                "summary_hint": c.summary_hint,
            }
        )
    return serialised, anom_c, z_t2m


# ---------------------------------------------------------------------------
# Global H / L pressure centres and smooth 4 hPa isobars for the 3D globe
# ---------------------------------------------------------------------------


def extract_pressure_features(mslp_05: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Extract major H/L centres and smoothed 4 hPa isobar polylines from 0.5 deg MSLP [361, 720]."""
    # Downsample to 1.0 deg [181, 360] and smooth so contours look clean and meteorological
    mslp_1d = _smooth_periodic(mslp_05[::2, ::2], passes=3)
    lats = np.linspace(90.0, -90.0, mslp_1d.shape[0])
    lons = (np.arange(mslp_1d.shape[1]) * 1.0) - 180.0

    # 1. Local High and Low centres (radius = 8 deg neighbourhood, excluding poles |lat| > 80)
    rad = 8
    pad = np.pad(mslp_1d, ((rad, rad), (0, 0)), mode="edge")
    pad = np.concatenate([pad[:, -rad:], pad, pad[:, :rad]], axis=1)
    ny, nx = mslp_1d.shape
    local_min = np.full_like(mslp_1d, np.inf)
    local_max = np.full_like(mslp_1d, -np.inf)
    for dy in range(2 * rad + 1):
        for dx in range(2 * rad + 1):
            sl = pad[dy : dy + ny, dx : dx + nx]
            local_min = np.minimum(local_min, sl)
            local_max = np.maximum(local_max, sl)

    centres: list[dict] = []
    for iy in range(10, ny - 10):
        la = float(lats[iy])
        for ix in range(nx):
            lo = float(lons[ix])
            val = float(mslp_1d[iy, ix])
            if val <= local_min[iy, ix] and val <= 998.0:
                if not any(c["kind"] == "L" and haversine_km(la, lo, c["lat"], c["lon"]) < 1200.0 for c in centres):
                    centres.append({"kind": "L", "lat": round(la, 1), "lon": round(lo, 1), "hpa": round(val)})
            elif val >= local_max[iy, ix] and val >= 1026.0:
                if not any(c["kind"] == "H" and haversine_km(la, lo, c["lat"], c["lon"]) < 1400.0 for c in centres):
                    centres.append({"kind": "H", "lat": round(la, 1), "lon": round(lo, 1), "hpa": round(val)})

    # Keep strongest ~22 lows and ~18 highs so the globe stays crisp and legible
    lows = sorted([c for c in centres if c["kind"] == "L"], key=lambda x: x["hpa"])[:22]
    highs = sorted([c for c in centres if c["kind"] == "H"], key=lambda x: -x["hpa"])[:18]

    # 2. Trace smooth 8 hPa global isobars on a 2 deg grid [91, 180] to keep manifest compact (~35 KB)
    coarse = _smooth_periodic(mslp_1d[::2, ::2], passes=1)
    c_lats = np.linspace(90.0, -90.0, coarse.shape[0])
    c_lons = (np.arange(coarse.shape[1]) * 2.0) - 180.0
    # Wrap first column at +180 deg so contours close seamlessly across the antimeridian
    coarse_w = np.concatenate([coarse, coarse[:, :1]], axis=1)
    c_lons_w = np.append(c_lons, 180.0)

    isobars: list[dict] = []
    for level in range(960, 1049, 8):
        polylines = _marching_squares_polylines(coarse_w, c_lats, c_lons_w, float(level))
        for poly in polylines:
            if len(poly) >= 5:
                # Quantise to 0.1 deg flat [lon, lat, lon, lat, ...]
                flat = [round(coord, 1) for pt in poly for coord in pt]
                isobars.append({"hpa": level, "path": flat})

    return lows + highs, isobars


def _marching_squares_polylines(
    grid: np.ndarray, lats: np.ndarray, lons: np.ndarray, level: float
) -> list[list[tuple[float, float]]]:
    """Simple marching-squares contour segment stitcher returning [(lon, lat), ...] polylines."""
    ny, nx = grid.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def _interp(v1: float, v2: float, p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
        denom = v2 - v1
        t = 0.5 if abs(denom) < 1e-6 else max(0.0, min(1.0, (level - v1) / denom))
        return (round(p1[0] + t * (p2[0] - p1[0]), 2), round(p1[1] + t * (p2[1] - p1[1]), 2))

    # Avoid polar singularities beyond 84 deg
    for j in range(3, ny - 4):
        lat0, lat1 = float(lats[j]), float(lats[j + 1])
        for i in range(nx - 1):
            lon0, lon1 = float(lons[i]), float(lons[i + 1])
            v00 = float(grid[j, i])
            v10 = float(grid[j, i + 1])
            v11 = float(grid[j + 1, i + 1])
            v01 = float(grid[j + 1, i])
            idx = (
                (1 if v00 >= level else 0)
                | (2 if v10 >= level else 0)
                | (4 if v11 >= level else 0)
                | (8 if v01 >= level else 0)
            )
            if idx in (0, 15):
                continue
            top = _interp(v00, v10, (lon0, lat0), (lon1, lat0))
            right = _interp(v10, v11, (lon1, lat0), (lon1, lat1))
            bottom = _interp(v01, v11, (lon0, lat1), (lon1, lat1))
            left = _interp(v00, v01, (lon0, lat0), (lon0, lat1))

            table = {
                1: [(left, top)],
                2: [(top, right)],
                3: [(left, right)],
                4: [(right, bottom)],
                5: [(left, bottom), (top, right)],
                6: [(top, bottom)],
                7: [(left, bottom)],
                8: [(bottom, left)],
                9: [(bottom, top)],
                10: [(top, left), (bottom, right)],
                11: [(bottom, right)],
                12: [(right, left)],
                13: [(right, top)],
                14: [(top, left)],
            }
            segments.extend(table.get(idx, []))

    if not segments:
        return []

    # Stitch segments sharing endpoints
    adj: dict[tuple[float, float], list[int]] = {}
    for s_idx, (a, b) in enumerate(segments):
        adj.setdefault(a, []).append(s_idx)
        adj.setdefault(b, []).append(s_idx)

    used = [False] * len(segments)
    lines: list[list[tuple[float, float]]] = []
    for s_idx, (a, b) in enumerate(segments):
        if used[s_idx]:
            continue
        used[s_idx] = True
        chain = [a, b]
        # Extend forward
        while True:
            cur = chain[-1]
            nxt_id = next((k for k in adj.get(cur, []) if not used[k]), None)
            if nxt_id is None:
                break
            used[nxt_id] = True
            sa, sb = segments[nxt_id]
            chain.append(sb if sa == cur else sa)
        # Extend backward
        while True:
            cur = chain[0]
            nxt_id = next((k for k in adj.get(cur, []) if not used[k]), None)
            if nxt_id is None:
                break
            used[nxt_id] = True
            sa, sb = segments[nxt_id]
            chain.insert(0, sb if sa == cur else sa)
        # Light decimation every 2nd vertex while keeping endpoints
        if len(chain) > 8:
            chain = chain[::2] + ([chain[-1]] if chain[-1] != chain[::2][-1] else [])
        lines.append(chain)
    return lines
