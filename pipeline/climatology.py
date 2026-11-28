"""ERA5 seasonal climatology (1991-2020), ±7-day 1940-present like-for-like 24 h rankings, and persistence.

Key statistical design:
1. Precomputed offline 0.5 deg global ERA5 climatology + 0.25 deg ERA5 surface orography
   (`pipeline/data/era5_climo_0p5.npz`), loaded at runtime in ~40 ms with zero network downloads.
2. Standard-atmosphere 6.5 °C/km lapse-rate adjustment (`climo - 0.0065 * (ifs_elev_m - era5_elev_m)`)
   eliminating high-mountain IFS (9 km) vs ERA5 (31 km) resolution artefacts.
3. Like-for-like 24-hour historical rankings across a ±7-day seasonal window (15 days x 86 years = 1,290
   daily snapshots, 1940-2025) from ERA5 daily `temperature_2m_max`, `temperature_2m_min`,
   `precipitation_sum` (24 h total), and `wind_gusts_10m_max` (24 h max).
4. Un-inflated persistence counts comparing recent daily Tmax/Tmin/gusts against local seasonal
   percentiles, honest rainfall normals (`daily_rain_normal * 14` and `* 28`), and strict 40-year era
   exceedance rules (`1940-1979` vs `1986-2025`, noting higher pre-1979 uncertainty).
"""
from __future__ import annotations

import json
import math
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, ROOT_DIR, SITE_DIR
from .opendata import LiveWeatherFields, http_get
from .scanner import haversine_km

PRECOMPUTED_CLIMO_PATH = ROOT_DIR / "pipeline" / "data" / "era5_climo_0p5.npz"
LAPSE_RATE_C_PER_M = 0.0065  # 6.5 deg C / km standard tropospheric lapse rate

_GAZETTEER_CACHE: list[dict] | None = None


@dataclass
class PrecomputedClimatology:
    t2m_harmonics: np.ndarray  # [4, 7, 361, 720] float32
    t2m_std_harmonics: np.ndarray  # [3, 361, 720] float32
    rain_harmonics: np.ndarray  # [7, 361, 720] float32 (mm/day)
    wind_harmonics: np.ndarray  # [5, 361, 720] float32 (m/s)
    gust_p90_harmonics: np.ndarray  # [3, 361, 720] float32 (m/s)
    era5_elev_025_m: np.ndarray  # [721, 1440] float32 (m)


def _eval_harmonics(coeffs: np.ndarray, doy: float) -> np.ndarray:
    """Evaluate Fourier harmonic array [1 + 2*K, ...] at 1-based day-of-year `doy`."""
    n_basis = coeffs.shape[0]
    n_harm = (n_basis - 1) // 2
    theta = 2.0 * math.pi * (doy - 1.0) / 365.25
    weights = [1.0]
    for k in range(1, n_harm + 1):
        weights.append(math.cos(k * theta))
        weights.append(math.sin(k * theta))
    w_arr = np.array(weights, dtype=np.float32).reshape((n_basis,) + (1,) * (coeffs.ndim - 1))
    return np.sum(coeffs * w_arr, axis=0, dtype=np.float32)


def _upsample_05_to_025(grid_05: np.ndarray) -> np.ndarray:
    """Smoothly bilinearly upsample [361, 720] (0.5 deg) to [721, 1440] (0.25 deg), periodic in lon."""
    out = np.empty((721, 1440), dtype=np.float32)
    # Even rows & cols
    out[0::2, 0::2] = grid_05
    # Odd cols (average horizontally with wrap-around at antimeridian)
    next_col = np.roll(grid_05, -1, axis=1)
    out[0::2, 1::2] = 0.5 * (grid_05 + next_col)
    # Odd rows (average vertically between adjacent even rows)
    out[1::2, :] = 0.5 * (out[0:-1:2, :] + out[2::2, :])
    return out


def _latitudinal_fallback_climatology(valid_time: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Analytical seasonal fallback [721, 1440] only if precomputed asset is missing."""
    lats = np.linspace(90.0, -90.0, 721, dtype=np.float32)[:, None]
    lons = ((np.arange(1440, dtype=np.float32) * 0.25) - 180.0)[None, :]
    doy = valid_time.timetuple().tm_yday
    season_phase = math.cos(2.0 * math.pi * (doy - 205.0) / 365.25)
    lat_rad = np.radians(lats)
    base = 27.5 * (np.cos(lat_rad) ** 1.25) - 22.0 * (np.sin(lat_rad) ** 2)
    seasonal = 14.0 * np.sin(lat_rad) * season_phase
    diurnal = -2.5 * np.cos(np.radians(lons + (valid_time.hour - 14.0) * 15.0)) * np.cos(lat_rad)
    mean_c = np.broadcast_to(base + seasonal + diurnal, (721, 1440)).astype(np.float32)
    std_c = np.broadcast_to(2.0 + 5.5 * (np.abs(np.sin(lat_rad)) ** 1.3), (721, 1440)).astype(np.float32)
    return mean_c, std_c


def load_precomputed_climatology(path: Path = PRECOMPUTED_CLIMO_PATH) -> PrecomputedClimatology | None:
    if not path.exists():
        return None
    npz = np.load(path)
    if "t2m_harmonics_i16" in npz.files:
        inv = 1.0 / float(npz["scale"][0])
        return PrecomputedClimatology(
            t2m_harmonics=npz["t2m_harmonics_i16"].astype(np.float32) * inv,
            t2m_std_harmonics=npz["t2m_std_harmonics_i16"].astype(np.float32) * inv,
            rain_harmonics=npz["rain_harmonics_i16"].astype(np.float32) * inv,
            wind_harmonics=npz["wind_harmonics_i16"].astype(np.float32) * inv,
            gust_p90_harmonics=npz["gust_p90_harmonics_i16"].astype(np.float32) * inv,
            era5_elev_025_m=npz["era5_elev_025_m"].astype(np.float32),
        )
    return PrecomputedClimatology(
        t2m_harmonics=npz["t2m_harmonics_f16"].astype(np.float32),
        t2m_std_harmonics=npz["t2m_std_harmonics_f16"].astype(np.float32),
        rain_harmonics=npz["rain_harmonics_f16"].astype(np.float32),
        wind_harmonics=npz["wind_harmonics_f16"].astype(np.float32),
        gust_p90_harmonics=npz["gust_p90_harmonics_f16"].astype(np.float32),
        era5_elev_025_m=npz["era5_elev_025_m"].astype(np.float32),
    )


def orography_is_plausible(ifs_elev_m: np.ndarray, era5_elev_m: np.ndarray) -> bool:
    """The two model terrains differ by a few hundred metres at most in typical places; a field that is
    kilometres off everywhere (e.g. a pressure-level height read by mistake) must never be used."""
    diff = np.asarray(ifs_elev_m, dtype=np.float32) - np.asarray(era5_elev_m, dtype=np.float32)
    return bool(np.isfinite(diff).any() and abs(float(np.nanmedian(diff))) < 50.0 and float(np.nanpercentile(np.abs(diff), 99)) < 1500.0)


def load_era5_historical_context(
    valid_time: datetime,
    ifs_elev_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str, PrecomputedClimatology | None]:
    """Load precomputed ±7-day ERA5 1990-2020 climatology and apply 6.5 °C/km lapse-rate adjustment."""
    vt = valid_time.astimezone(timezone.utc)
    doy = float(vt.timetuple().tm_yday)
    frac_hour = vt.hour + vt.minute / 60.0

    climo = load_precomputed_climatology()
    if climo is None:
        mean_c, std_c = _latitudinal_fallback_climatology(vt)
        return mean_c, std_c, "Analytical seasonal baseline", None

    # Interpolate smoothly between adjacent 6-hourly synoptic climatology grids (00, 06, 12, 18Z)
    h_pos = (frac_hour % 24.0) / 6.0
    h0 = int(math.floor(h_pos)) % 4
    h1 = (h0 + 1) % 4
    w1 = h_pos - math.floor(h_pos)

    grid_h0 = _eval_harmonics(climo.t2m_harmonics[h0], doy)
    grid_h1 = _eval_harmonics(climo.t2m_harmonics[h1], doy)
    mean_05_c = (1.0 - w1) * grid_h0 + w1 * grid_h1
    std_05_c = np.maximum(0.85, _eval_harmonics(climo.t2m_std_harmonics, doy))

    mean_025_c = _upsample_05_to_025(mean_05_c)
    std_025_c = _upsample_05_to_025(std_05_c)

    label = "ERA5 ±7-day seasonal baseline (precomputed)"
    if ifs_elev_m is not None and ifs_elev_m.shape == climo.era5_elev_025_m.shape:
        orog_diff_m = ifs_elev_m - climo.era5_elev_025_m
        if not orography_is_plausible(ifs_elev_m, climo.era5_elev_025_m):
            print(
                "[climatology] ERROR: forecast surface elevation looks wrong "
                f"(median difference from ERA5 {float(np.nanmedian(orog_diff_m)):.0f} m); skipping the lapse-rate adjustment.",
                file=sys.stderr,
            )
        else:
            mean_025_c = (mean_025_c - LAPSE_RATE_C_PER_M * orog_diff_m).astype(np.float32)
            label = "ERA5 ±7-day seasonal baseline (6.5 °C/km lapse-rate adjusted)"

    return mean_025_c, std_025_c, label, climo


def get_global_t2m_climatology(valid_time: datetime) -> tuple[np.ndarray, np.ndarray, str]:
    mean_c, std_c, label, _ = load_era5_historical_context(valid_time)
    return mean_c, std_c, label


# ---------------------------------------------------------------------------
# Gazetteer attachment (Section 1.6)
# ---------------------------------------------------------------------------


def _load_gazetteer_places() -> list[dict]:
    global _GAZETTEER_CACHE
    if _GAZETTEER_CACHE is not None:
        return _GAZETTEER_CACHE
    gaz_path = SITE_DIR / "assets" / "gazetteer.json"
    if not gaz_path.exists():
        _GAZETTEER_CACHE = []
        return _GAZETTEER_CACHE
    try:
        _GAZETTEER_CACHE = json.loads(gaz_path.read_text())
    except Exception:
        _GAZETTEER_CACHE = []
    return _GAZETTEER_CACHE


def nearest_gazetteer_place(lat: float, lon: float) -> dict:
    """Return nearest named settlement/place (preferring non-country places) and country from gazetteer.json."""
    entries = _load_gazetteer_places()
    places = [e for e in entries if e.get("kind") != "Country"]
    countries = [e for e in entries if e.get("kind") == "Country"]

    best_place = None
    best_dist = float("inf")
    for p in places or entries:
        d = haversine_km(lat, lon, float(p["lat"]), float(p["lon"]))
        if d < best_dist:
            best_dist = d
            best_place = p

    best_country = None
    best_c_dist = float("inf")
    for c in countries:
        d = haversine_km(lat, lon, float(c["lat"]), float(c["lon"]))
        if d < best_c_dist:
            best_c_dist = d
            best_country = c

    place_name = best_place["name"] if best_place else f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"
    place_sub = (best_place.get("sub") or "") if best_place else ""
    country_name = place_sub.split(",")[-1].strip() if place_sub else (best_country["name"] if best_country else "")
    dist_km = round(best_dist, 0) if math.isfinite(best_dist) else 0.0
    return {
        "nearest_place": place_name,
        "nearest_country": country_name,
        "distance_km": dist_km,
    }


# ---------------------------------------------------------------------------
# Like-for-like ±7-day 1940-2025 daily ERA5 rankings & 28-day persistence
# ---------------------------------------------------------------------------


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def fetch_batched_recent_persistence(candidates: list[dict]) -> list[dict]:
    """Fetch 28-day past daily series for all candidates in 1 Open-Meteo request."""
    if not candidates:
        return []
    lats_str = ",".join(f"{c['lat']:.2f}" for c in candidates)
    lons_str = ",".join(f"{c['lon']:.2f}" for c in candidates)
    qs = urllib.parse.urlencode(
        {
            "latitude": lats_str,
            "longitude": lons_str,
            "past_days": 28,
            "forecast_days": 1,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_gusts_10m_max",
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{qs}"
    try:
        payload = json.loads(http_get(url, timeout=25, attempts=3))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "daily" in payload:
            return [payload]
    except Exception as exc:
        print(f"[climatology] Warning: batched 28-day persistence query failed: {exc}", file=sys.stderr)
    return [{} for _ in candidates]


ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_HISTORY_YEARS = range(1940, 2026)  # complete years; "since 1940" in the prose
ERA5_DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_gusts_10m_max"
OPEN_METEO_BUDGET_PER_MIN = 540.0  # free tier allows 600 "calls" per minute; one call = one location x <=14 days


class _MinuteBudget:
    """Thread-safe sliding one-minute budget for Open-Meteo's per-location call weights."""

    def __init__(self, per_minute: float) -> None:
        import threading

        self.per_minute = per_minute
        self.events: list[tuple[float, float]] = []
        self.lock = threading.Lock()

    def take(self, cost: float) -> None:
        import time

        while True:
            with self.lock:
                now = time.monotonic()
                self.events = [(t, c) for t, c in self.events if now - t < 60.0]
                used = sum(c for _, c in self.events)
                if used + cost <= self.per_minute:
                    self.events.append((now, cost))
                    return
                wait = 60.0 - (now - self.events[0][0]) + 0.05
            time.sleep(max(0.05, wait))


def _season_window(center, year: int):
    """Dates [centre - 7 days, centre + 6 days] in `year` (14 days, so each location costs one API call)."""
    from datetime import timedelta

    try:
        c = center.replace(year=year)
    except ValueError:  # 29 February
        c = center.replace(year=year, day=28)
    return c - timedelta(days=7), c + timedelta(days=6)


def _get_json_with_rate_limit(url: str, timeout: int = 45) -> list | dict:
    """GET JSON from Open-Meteo, waiting out a 429 (rate limit) once before giving up."""
    import time
    import urllib.error
    import urllib.request

    from .config import USER_AGENT

    for attempt in range(4):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 1:
                time.sleep(65.0)
                continue
            if exc.code >= 500 and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise IOError(f"Unreachable: {url}")


def fetch_era5_seasonal_windows(candidates: list[dict], valid_time: datetime, max_points: int = 12) -> list[dict | None]:
    """Daily ERA5 history (1940-2025) for the 14 days around this date in every year, for each candidate.

    Requesting only the seasonal window costs 86 API calls per location instead of ~2,200 for the full
    daily record, so every candidate can be ranked within Open-Meteo's free limits. One request per year
    covers all uncached locations at once, paced to stay under the per-minute budget.
    Returns per-candidate dicts {years, doys, tmax, tmin, rain, gust} (or None if unavailable).
    """
    from datetime import date, timedelta

    center = valid_time.astimezone(timezone.utc).date()
    latest_ok = datetime.now(timezone.utc).date() - timedelta(days=6)  # ERA5 lags real time by ~5 days
    years = [y for y in ERA5_HISTORY_YEARS if _season_window(center, y)[1] <= latest_ok]

    results: list[dict | None] = [None] * len(candidates)
    todo: list[tuple[int, float, float, Path]] = []
    for idx, cand in enumerate(candidates[:max_points]):
        qlat = round(float(cand["lat"]) * 4.0) / 4.0
        qlon = round(float(cand["lon"]) * 4.0) / 4.0
        cache_file = CACHE_DIR / "era5_points" / f"era5_window_{qlat:+07.2f}_{qlon:+08.2f}_{center:%m%d}.npz"
        if cache_file.exists():
            try:
                npz = np.load(cache_file)
                results[idx] = {k: npz[k] for k in npz.files}
                continue
            except Exception:
                pass
        todo.append((idx, qlat, qlon, cache_file))
    if not todo or not years:
        return results

    budget = _MinuteBudget(OPEN_METEO_BUDGET_PER_MIN)
    lat_str = ",".join(f"{la:.2f}" for _, la, _, _ in todo)
    lon_str = ",".join(f"{lo:.2f}" for _, _, lo, _ in todo)

    def _fetch_year(year: int):
        start, end = _season_window(center, year)
        qs = urllib.parse.urlencode(
            {
                "latitude": lat_str,
                "longitude": lon_str,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "models": "era5",
                "daily": ERA5_DAILY_VARS,
                "timezone": "UTC",
                "wind_speed_unit": "ms",
            }
        )
        budget.take(float(len(todo)))
        try:
            payload = _get_json_with_rate_limit(f"{ERA5_ARCHIVE_URL}?{qs}")
        except Exception as exc:
            return year, None, str(exc)
        if isinstance(payload, dict):
            payload = [payload]
        return year, payload, ""

    with ThreadPoolExecutor(max_workers=6) as pool:
        year_rows = list(pool.map(_fetch_year, years))
    # Second pass for transient failures (timeouts / 5xx), one at a time
    for i, (year, payload, _) in enumerate(year_rows):
        if payload is None or len(payload) != len(todo):
            year_rows[i] = _fetch_year(year)

    failed = [(y, err) for y, p, err in year_rows if p is None or len(p) != len(todo)]
    if len(failed) > 2:
        print(
            f"[climatology] ERA5 seasonal history incomplete ({len(failed)} of {len(years)} years failed, "
            f"e.g. {failed[0][1][:80]}); no historical ranks this run.",
            file=sys.stderr,
        )
        return results

    for k, (idx, _, _, cache_file) in enumerate(todo):
        cols: dict[str, list] = {"years": [], "doys": [], "tmax": [], "tmin": [], "rain": [], "gust": []}
        for year, payload, _ in year_rows:
            if payload is None or len(payload) != len(todo):
                continue
            daily = (payload[k] or {}).get("daily") or {}
            for i, t in enumerate(daily.get("time") or []):
                d = date.fromisoformat(t)
                cols["years"].append(year)
                cols["doys"].append(d.timetuple().tm_yday)
                for key, var in (
                    ("tmax", "temperature_2m_max"),
                    ("tmin", "temperature_2m_min"),
                    ("rain", "precipitation_sum"),
                    ("gust", "wind_gusts_10m_max"),
                ):
                    v = (daily.get(var) or [None] * (i + 1))[i]
                    cols[key].append(np.nan if v is None else float(v))
        hist = {
            "years": np.array(cols["years"], dtype=np.int16),
            "doys": np.array(cols["doys"], dtype=np.int16),
            **{key: np.array(cols[key], dtype=np.float32) for key in ("tmax", "tmin", "rain", "gust")},
        }
        if len(np.unique(hist["years"])) < 80:
            continue
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, **hist)
        results[idx] = hist
    n_ok = sum(r is not None for r in results)
    print(f"[climatology] ERA5 seasonal history available for {n_ok} of {len(candidates)} hotspots.", file=sys.stderr)
    return results


def _circular_doy_mask(doys: np.ndarray, target_doy: int, half_window: int = 7) -> np.ndarray:
    diff = np.abs(doys.astype(np.int32) - int(target_doy))
    circ = np.minimum(diff, 365 - diff)
    return circ <= half_window


def compute_rain_persistence_metrics(rain_daily_mm: list[float], climo_daily_rain_mm: float) -> dict:
    """Compute 14-day and 28-day rainfall totals, honest normals, and wet-day counts (Section 1.1 & 1.4)."""
    obs = [max(0.0, float(x)) for x in rain_daily_mm]
    last_14 = obs[-14:] if len(obs) >= 14 else obs
    last_28 = obs[-28:] if len(obs) >= 28 else obs

    rain_14d = round(sum(last_14), 1)
    rain_28d = round(sum(last_28), 1)
    daily_norm = max(0.0, float(climo_daily_rain_mm))
    expected_14d = round(daily_norm * 14.0, 1)
    expected_28d = round(daily_norm * 28.0, 1)

    # Only compute % / multiple of normal when 28-day normal is >= 5 mm (avoids desert division artefacts)
    pct_28d = round(100.0 * rain_28d / expected_28d, 0) if expected_28d >= 5.0 else None
    mult_28d = round(rain_28d / expected_28d, 1) if expected_28d >= 5.0 else None

    wet_14 = sum(1 for v in last_14 if v >= 10.0)
    wet_28 = sum(1 for v in last_28 if v >= 10.0)

    return {
        "daily_rain_normal_mm": round(daily_norm, 2),
        "rain_14d_mm": rain_14d,
        "expected_rain_14d_mm": expected_14d,
        "rain_28d_mm": rain_28d,
        "expected_rain_28d_mm": expected_28d,
        "rain_28d_pct_of_normal": pct_28d,
        "rain_28d_multiple_of_normal": mult_28d,
        "wet_days_ge_10mm_last_14": wet_14,
        "wet_days_ge_10mm_last_28": wet_28,
        "wet_days_label": "wet days (>=10 mm)",
    }


def compute_era_frequency_metrics(
    years: np.ndarray,
    doys: np.ndarray,
    series: np.ndarray,
    target_doy: int,
    tail: str,
) -> dict:
    """Compute independent 40-year era exceedance counts (`1940-1979` vs `1986-2025`, Section 1.5)."""
    win_mask = _circular_doy_mask(doys, target_doy, half_window=7) & np.isfinite(series)
    if not np.any(win_mask):
        return {
            "p90_years_count_1940_1979_out_of_40": 0,
            "p90_years_count_1986_2025_out_of_40": 0,
            "count_unit": "years out of 40 years (NOT per decade)",
            "frequency_claim_allowed": False,
            "frequency_ratio": None,
            "era5_pre_1979_note": "ERA5 before 1979 assimilates no satellite observations and is less reliable.",
        }

    # Summarize each year by its ±7-day window peak (high tail) or minimum (low tail)
    yr_vals: dict[int, float] = {}
    for yr in range(1940, 2026):
        ym = win_mask & (years == yr)
        if np.any(ym):
            yr_vals[yr] = float(np.min(series[ym]) if tail == "low" else np.max(series[ym]))

    all_yr_arr = np.array(list(yr_vals.values()), dtype=np.float32)
    thresh = float(np.percentile(all_yr_arr, 10 if tail == "low" else 90))

    def _hit(v: float) -> bool:
        if tail == "low":
            return v <= thresh and v < float(np.max(all_yr_arr))
        if thresh <= 0.0 and np.min(all_yr_arr) >= 0.0:
            return v > 0.0
        return v >= thresh

    early_hits = sum(1 for yr in range(1940, 1980) if yr in yr_vals and _hit(yr_vals[yr]))
    recent_hits = sum(1 for yr in range(1986, 2026) if yr in yr_vals and _hit(yr_vals[yr]))
    ratio = round(recent_hits / early_hits, 2) if early_hits >= 5 else None
    allowed = bool(early_hits >= 5 and recent_hits >= 5 and (recent_hits / early_hits) >= 1.5)

    return {
        "p90_years_count_1940_1979_out_of_40": early_hits,
        "p90_years_count_1986_2025_out_of_40": recent_hits,
        "count_unit": "years out of 40 years (NOT per decade)",
        "frequency_claim_allowed": allowed,
        "frequency_ratio": ratio,
        "era5_pre_1979_note": "ERA5 before 1979 assimilates no satellite observations and is less reliable.",
    }


def enrich_all_candidates(
    candidates: list[dict],
    weather: LiveWeatherFields,
    climo: PrecomputedClimatology | None,
) -> list[dict]:
    """Enrich candidates with like-for-like ±7-day 1940-2025 ERA5 rankings, persistence & place context."""
    if not candidates:
        return []

    recent_list = fetch_batched_recent_persistence(candidates)
    hist_list = fetch_era5_seasonal_windows(candidates, weather.valid_time)

    vt = weather.valid_time.astimezone(timezone.utc)
    target_doy = vt.timetuple().tm_yday

    rain_05_climo = _eval_harmonics(climo.rain_harmonics, float(target_doy)) if climo is not None else None
    gust_p90_05_climo = _eval_harmonics(climo.gust_p90_harmonics, float(target_doy)) if climo is not None else None
    t2m_std_05_climo = (
        np.maximum(0.85, _eval_harmonics(climo.t2m_std_harmonics, float(target_doy))) if climo is not None else None
    )
    t2m_synoptic_05 = (
        np.stack([_eval_harmonics(climo.t2m_harmonics[h], float(target_doy)) for h in range(4)], axis=0)
        if climo is not None
        else None
    )

    enriched: list[dict] = []
    for cand, rec, hist in zip(candidates, recent_list, hist_list):
        out = dict(cand)
        kind = cand["kind"]
        lat, lon = float(cand["lat"]), float(cand["lon"])
        iy025 = int(round((90.0 - lat) / 0.25))
        ix025 = int(round(((lon + 180.0) % 360.0) / 0.25)) % 1440
        iy05 = int(round((90.0 - lat) / 0.5))
        ix05 = int(round(((lon + 180.0) % 360.0) / 0.5)) % 720

        ifs_elev = float(weather.ifs_elev_m[iy025, ix025]) if weather.ifs_elev_m is not None else 0.0
        era5_elev = float(climo.era5_elev_025_m[iy025, ix025]) if climo is not None else ifs_elev
        elev_diff = round(ifs_elev - era5_elev, 0)
        if weather.ifs_elev_m is not None and climo is not None and not orography_is_plausible(
            weather.ifs_elev_m, climo.era5_elev_025_m
        ):
            ifs_elev = era5_elev  # bad elevation field: no adjustment rather than a wrong one
        lapse_adj_c = -LAPSE_RATE_C_PER_M * (ifs_elev - era5_elev)

        place_ctx = nearest_gazetteer_place(lat, lon)
        grid_label = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'} {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
        out["grid_point"] = {
            "lat": lat,
            "lon": lon,
            "label": grid_label,
            "ifs_elevation_m": round(ifs_elev, 0),
            "era5_elevation_m": round(era5_elev, 0),
            "elevation_diff_m": elev_diff,
            "nearest_gazetteer_place": place_ctx["nearest_place"],
            "nearest_gazetteer_country": place_ctx["nearest_country"],
            "nearest_place_distance_km": place_ctx["distance_km"],
            "attribution_rule": (
                f"All grid-box statistics apply ONLY to ({lat:+.2f}, {lon:+.2f}) "
                f"({place_ctx['distance_km']:.0f} km from {place_ctx['nearest_place']}). "
                f"Never attribute these normals or rankings to a distant city."
            ),
        }

        # Parse past observed days from Open-Meteo (drop trailing forecast day index -1)
        daily_rec = (rec or {}).get("daily") or {}
        obs_rain = [float(x) for x in (daily_rec.get("precipitation_sum") or [])[:-1] if x is not None]
        obs_tmax = [float(x) for x in (daily_rec.get("temperature_2m_max") or [])[:-1] if x is not None]
        obs_tmin = [float(x) for x in (daily_rec.get("temperature_2m_min") or [])[:-1] if x is not None]
        obs_gust = [float(x) for x in (daily_rec.get("wind_gusts_10m_max") or [])[:-1] if x is not None]

        # Default seasonal rain normal from precomputed 0.5 deg grid (overridden below if exact daily ERA5 loaded)
        climo_daily_rain_mm = float(max(0.0, rain_05_climo[iy05, ix05])) if rain_05_climo is not None else 2.5

        era_info: dict = {}
        if hist is not None:
            years = hist["years"]
            doys = hist["doys"]
            win_mask = _circular_doy_mask(doys, target_doy, half_window=7)
            wmo_win = win_mask & (years >= 1991) & (years <= 2020)

            tmax_s = hist["tmax"] + lapse_adj_c
            tmin_s = hist["tmin"] + lapse_adj_c
            tmean_s = 0.5 * (tmax_s + tmin_s)
            rain_s = np.maximum(0.0, hist["rain"])
            gust_s = np.maximum(0.0, hist["gust"])

            if np.any(wmo_win & np.isfinite(rain_s)):
                climo_daily_rain_mm = float(np.nanmean(rain_s[wmo_win]))

            wmo_tmax_mean = float(np.nanmean(tmax_s[wmo_win]))
            wmo_tmax_p90 = float(np.nanpercentile(tmax_s[wmo_win], 90))
            wmo_tmin_mean = float(np.nanmean(tmin_s[wmo_win]))
            wmo_tmin_p10 = float(np.nanpercentile(tmin_s[wmo_win], 10))
            wmo_gust_mean = float(np.nanmean(gust_s[wmo_win]))
            wmo_gust_p90 = float(np.nanpercentile(gust_s[wmo_win], 90))

            # Select strictly like-for-like 24-hour variable for ranking
            if kind == "heat":
                pool_mask = win_mask & np.isfinite(tmax_s)
                pool = tmax_s[pool_mask]
                pool_years = years[pool_mask]
                live_val = float(cand.get("t2m_max_24h_c", cand["t2m_c"]))
                climo_normal = round(wmo_tmax_mean, 1)
                var_ranked = "24 h maximum 2 m temperature (Tmax)"
                tail = "high"
                units = "°C"
                full_series = tmax_s
            elif kind == "cold":
                pool_mask = win_mask & np.isfinite(tmin_s)
                pool = tmin_s[pool_mask]
                pool_years = years[pool_mask]
                live_val = float(cand.get("t2m_min_24h_c", cand["t2m_c"]))
                climo_normal = round(wmo_tmin_mean, 1)
                var_ranked = "24 h minimum 2 m temperature (Tmin)"
                tail = "low"
                units = "°C"
                full_series = tmin_s
            elif kind == "rain":
                pool_mask = win_mask & np.isfinite(rain_s)
                pool = rain_s[pool_mask]
                pool_years = years[pool_mask]
                # Rank the forecast rain averaged over 3x3 grid cells (~80 km): the forecast model resolves
                # sharper downpours than ERA5, so a single-cell value would make "wettest since 1940" too easy.
                rain_box = weather.rain_24h_mm[max(0, iy025 - 1) : iy025 + 2, [(ix025 + d) % 1440 for d in (-1, 0, 1)]]
                live_val = float(np.nanmean(rain_box)) if np.isfinite(rain_box).any() else float(cand["rain_24h_mm"])
                climo_normal = round(climo_daily_rain_mm, 1)
                var_ranked = "24 h accumulated precipitation"
                tail = "high"
                units = "mm"
                full_series = rain_s
            else:
                pool_mask = win_mask & np.isfinite(gust_s)
                pool = gust_s[pool_mask]
                pool_years = years[pool_mask]
                live_val = float(cand["gust_ms"])
                climo_normal = round(wmo_gust_mean, 1)
                var_ranked = "24 h maximum 10 m wind gust"
                tail = "high"
                units = "m/s"
                full_series = gust_s

            n_samples = int(len(pool)) + 1
            if tail == "low":
                more_ext_mask = pool < live_val
            else:
                more_ext_mask = pool > live_val

            more_ext_days = int(np.sum(more_ext_mask))
            years_exceeded = int(len(np.unique(pool_years[more_ext_mask])))
            n_years = int(len(np.unique(pool_years)))
            rank = more_ext_days + 1

            # Previous most extreme day in the record (value and year), and whether the forecast beats it
            # by a clear margin. Forecast models run hotter/wetter/gustier than ERA5, and we scan the whole
            # planet for extremes, so a bare "#1" is often a near-tie rather than a genuine new high.
            if len(pool):
                rec_i = int(np.argmin(pool) if tail == "low" else np.argmax(pool))
                prev_rec_val = float(pool[rec_i])
                prev_rec_year = int(pool_years[rec_i])
                margin = (prev_rec_val - live_val) if tail == "low" else (live_val - prev_rec_val)
                if kind in ("heat", "cold"):
                    needed = 0.5
                elif kind == "rain":
                    needed = max(10.0, 0.10 * prev_rec_val)
                else:
                    needed = max(2.0, 0.07 * prev_rec_val)
                clear_record = bool(rank == 1 and margin >= needed)
            else:
                prev_rec_val, prev_rec_year, margin, clear_record = None, None, None, False
            pct = round(100.0 * (1.0 - (rank - 0.5) / n_samples), 1)

            early_t = tmean_s[win_mask & (years >= 1940) & (years <= 1979)]
            recent_t = tmean_s[win_mask & (years >= 1986) & (years <= 2025)]
            warming_c = (
                round(float(np.nanmean(recent_t) - np.nanmean(early_t)), 2)
                if len(early_t) > 30 and len(recent_t) > 30
                else 0.0
            )
            freq_metrics = compute_era_frequency_metrics(years, doys, full_series, target_doy, tail)

            rain_pers = compute_rain_persistence_metrics(obs_rain, climo_daily_rain_mm)
            hot_thr = max(wmo_tmax_p90, wmo_tmax_mean + 3.0)
            cold_thr = min(wmo_tmin_p10, wmo_tmin_mean - 3.0)

            hot_14 = sum(1 for v in obs_tmax[-14:] if v >= hot_thr)
            hot_28 = sum(1 for v in obs_tmax[-28:] if v >= hot_thr)
            cold_14 = sum(1 for v in obs_tmin[-14:] if v <= cold_thr)
            cold_28 = sum(1 for v in obs_tmin[-28:] if v <= cold_thr)
            gust65_14 = sum(1 for v in obs_gust[-14:] if v >= (65.0 / 3.6))
            gust65_28 = sum(1 for v in obs_gust[-28:] if v >= (65.0 / 3.6))
            gust_p90_14 = sum(1 for v in obs_gust[-14:] if v >= wmo_gust_p90)
            gust_p90_28 = sum(1 for v in obs_gust[-28:] if v >= wmo_gust_p90)

            if kind == "heat":
                primary_pers_summary = (
                    f"{hot_28} of last 28 days with Tmax above local 90th percentile ({hot_thr:.1f} °C)"
                )
            elif kind == "cold":
                primary_pers_summary = (
                    f"{cold_28} of last 28 days with Tmin below local 10th percentile ({cold_thr:.1f} °C)"
                )
            elif kind == "rain":
                primary_pers_summary = (
                    f"{rain_pers['wet_days_ge_10mm_last_28']} wet days (>=10 mm) in last 28 days "
                    f"({rain_pers['rain_28d_mm']} mm vs {rain_pers['expected_rain_28d_mm']} mm 28-day normal)"
                )
            else:
                primary_pers_summary = (
                    f"{gust65_28} days with gusts over 65 km/h in last 28 days "
                    f"({gust_p90_28} days above local 90th-percentile gust of {wmo_gust_p90 * 3.6:.0f} km/h)"
                )

            if rank == 1:
                rank_word = "Lowest" if tail == "low" else "Highest"
                rank_summary = f"{rank_word} for this time of year since 1940 (#1 of {n_samples} seasonal days across {n_years} years)"
            else:
                rank_summary = (
                    f"#{rank} of {n_samples} seasonal days for this time of year since 1940 "
                    f"({pct}th percentile; exceeded in {years_exceeded} of {n_years} years)"
                )

            era_info = {
                "variable_ranked": var_ranked,
                "live_24h_value": round(live_val, 1),
                **(
                    {"ranked_value_note": "Rain is ranked using the forecast averaged over ~80 km, to match the historical record's resolution."}
                    if kind == "rain"
                    else {}
                ),
                "units": units,
                "window_description": "14-day seasonal window around this date in each year 1940-2025",
                "rank_since_1940": rank,
                "rank_available": True,
                "samples_compared": n_samples,
                "years_compared": n_years,
                "years_with_more_extreme_day": years_exceeded,
                "percentile": pct,
                "is_highest_for_time_of_year_since_1940": rank == 1,
                "clear_record": clear_record,
                "previous_record_value": round(prev_rec_val, 1) if prev_rec_val is not None else None,
                "previous_record_year": prev_rec_year,
                "beats_previous_record_by": round(margin, 1) if margin is not None and margin > 0 else None,
                "seasonal_normal_1991_2020": climo_normal,
                "wmo_tmax_normal_c": round(wmo_tmax_mean, 1),
                "wmo_tmin_normal_c": round(wmo_tmin_mean, 1),
                "rank_summary": rank_summary,
                "persistence": {
                    "summary": primary_pers_summary,
                    "warm_days_above_p90_tmax_last_14": hot_14,
                    "warm_days_above_p90_tmax_last_28": hot_28,
                    "tmax_p90_threshold_c": round(hot_thr, 1),
                    "cold_days_below_p10_tmin_last_14": cold_14,
                    "cold_days_below_p10_tmin_last_28": cold_28,
                    "tmin_p10_threshold_c": round(cold_thr, 1),
                    "days_gust_over_65kmh_last_14": gust65_14,
                    "days_gust_over_65kmh_last_28": gust65_28,
                    "days_gust_above_local_p90_last_14": gust_p90_14,
                    "days_gust_above_local_p90_last_28": gust_p90_28,
                    "local_gust_p90_kmh": round(wmo_gust_p90 * 3.6, 0),
                    **rain_pers,
                },
                "climate_shift": {
                    "local_seasonal_warming_c_since_1940_1979": warming_c,
                    **freq_metrics,
                },
            }
        else:
            # Zero-network precomputed 0.5 deg ERA5 diurnal Tmax/Tmin/rain/gust harmonics + 28-day observations
            rain_pers = compute_rain_persistence_metrics(obs_rain, climo_daily_rain_mm)
            gp90 = float(gust_p90_05_climo[iy05, ix05]) if gust_p90_05_climo is not None else 20.0
            local_std = float(t2m_std_05_climo[iy05, ix05]) if t2m_std_05_climo is not None else 3.0
            if t2m_synoptic_05 is not None:
                syn_col = t2m_synoptic_05[:, iy05, ix05] + lapse_adj_c
                wmo_tmax_fb = float(np.max(syn_col)) + 0.8
                wmo_tmin_fb = float(np.min(syn_col)) - 0.8
            else:
                approx_mean_c = float(cand["t2m_c"] - cand["anom_c"])
                wmo_tmax_fb = approx_mean_c + 3.0
                wmo_tmin_fb = approx_mean_c - 3.0

            hot_thr_fb = wmo_tmax_fb + max(1.2816 * local_std, 2.5)
            cold_thr_fb = wmo_tmin_fb - max(1.2816 * local_std, 2.5)
            hot_14 = sum(1 for v in obs_tmax[-14:] if v >= hot_thr_fb)
            hot_28 = sum(1 for v in obs_tmax[-28:] if v >= hot_thr_fb)
            cold_14 = sum(1 for v in obs_tmin[-14:] if v <= cold_thr_fb)
            cold_28 = sum(1 for v in obs_tmin[-28:] if v <= cold_thr_fb)
            gust65_14 = sum(1 for v in obs_gust[-14:] if v >= (65.0 / 3.6))
            gust65_28 = sum(1 for v in obs_gust[-28:] if v >= (65.0 / 3.6))
            gust_p90_14 = sum(1 for v in obs_gust[-14:] if v >= gp90)
            gust_p90_28 = sum(1 for v in obs_gust[-28:] if v >= gp90)

            if kind == "heat":
                fb_pers_summary = (
                    f"{hot_28} of last 28 days with Tmax above local 90th percentile ({hot_thr_fb:.1f} °C)"
                )
                fb_norm = round(wmo_tmax_fb, 1)
                z_est = (float(cand.get("t2m_max_24h_c", cand["t2m_c"])) - wmo_tmax_fb) / max(1.0, local_std)
            elif kind == "cold":
                fb_pers_summary = (
                    f"{cold_28} of last 28 days with Tmin below local 10th percentile ({cold_thr_fb:.1f} °C)"
                )
                fb_norm = round(wmo_tmin_fb, 1)
                z_est = (wmo_tmin_fb - float(cand.get("t2m_min_24h_c", cand["t2m_c"]))) / max(1.0, local_std)
            elif kind == "rain":
                fb_pers_summary = (
                    f"{rain_pers['wet_days_ge_10mm_last_28']} wet days (>=10 mm) in last 28 days "
                    f"({rain_pers['rain_28d_mm']} mm vs {rain_pers['expected_rain_28d_mm']} mm 28-day normal)"
                )
                fb_norm = round(climo_daily_rain_mm, 1)
                z_est = 2.2
            else:
                fb_pers_summary = (
                    f"{gust65_28} days with gusts over 65 km/h in last 28 days "
                    f"({gust_p90_28} days above local 90th-percentile gust of {gp90 * 3.6:.0f} km/h)"
                )
                fb_norm = round(gp90 / 1.45, 1)
                z_est = 1.28 + max(0.0, (float(cand["gust_ms"]) - gp90) / 4.5)

            # Without the exact 1940-2025 daily series we cannot rank honestly, so publish NO percentile or
            # rank (never a guessed "top 1%"). Instead give simple, defensible comparisons with the normal.
            live_fb = float(cand["peak_value"])
            if kind == "heat":
                live_fb = float(cand.get("t2m_max_24h_c", cand["t2m_c"]))
                vs_normal = f"{live_fb - wmo_tmax_fb:+.1f} °C compared with a typical daytime high of {wmo_tmax_fb:.0f} °C"
            elif kind == "cold":
                live_fb = float(cand.get("t2m_min_24h_c", cand["t2m_c"]))
                vs_normal = f"{live_fb - wmo_tmin_fb:+.1f} °C compared with a typical overnight low of {wmo_tmin_fb:.0f} °C"
            elif kind == "rain":
                mult = live_fb / climo_daily_rain_mm if climo_daily_rain_mm >= 0.5 else None
                vs_normal = (
                    f"about {mult:.0f} times the average daily rainfall ({climo_daily_rain_mm:.1f} mm) for the time of year"
                    if mult is not None and mult >= 2
                    else "far above the usual daily rainfall for the time of year"
                )
            else:
                vs_normal = (
                    f"above the local 90th-percentile gust of {gp90 * 3.6:.0f} km/h"
                    if float(cand["gust_ms"]) > gp90
                    else f"close to the local 90th-percentile gust of {gp90 * 3.6:.0f} km/h"
                )
            era_info = {
                "variable_ranked": f"24 h {kind}",
                "live_24h_value": round(live_fb, 1),
                "units": cand["units"],
                "window_description": "±7-day seasonal climatology (1990-2019 precomputed at 0.5 deg); exact daily history unavailable this run",
                "rank_since_1940": None,
                "rank_available": False,
                "samples_compared": None,
                "years_compared": None,
                "years_with_more_extreme_day": None,
                "percentile": None,
                "estimated_departure_sd": round(z_est, 1) if kind in ("heat", "cold") else None,
                "comparison_with_normal": vs_normal,
                "is_highest_for_time_of_year_since_1940": False,
                "seasonal_normal_1991_2020": fb_norm,
                "wmo_tmax_normal_c": round(wmo_tmax_fb, 1),
                "wmo_tmin_normal_c": round(wmo_tmin_fb, 1),
                "rank_summary": f"No historical rank available this run; {vs_normal}.",
                "persistence": {
                    "summary": fb_pers_summary,
                    "warm_days_above_p90_tmax_last_14": hot_14,
                    "warm_days_above_p90_tmax_last_28": hot_28,
                    "tmax_p90_threshold_c": round(hot_thr_fb, 1),
                    "cold_days_below_p10_tmin_last_14": cold_14,
                    "cold_days_below_p10_tmin_last_28": cold_28,
                    "tmin_p10_threshold_c": round(cold_thr_fb, 1),
                    "days_gust_over_65kmh_last_14": gust65_14,
                    "days_gust_over_65kmh_last_28": gust65_28,
                    "days_gust_above_local_p90_last_14": gust_p90_14,
                    "days_gust_above_local_p90_last_28": gust_p90_28,
                    "local_gust_p90_kmh": round(gp90 * 3.6, 0),
                    **rain_pers,
                },
                "climate_shift": {
                    "local_seasonal_warming_c_since_1940_1979": None,
                    "p90_years_count_1940_1979_out_of_40": None,
                    "p90_years_count_1986_2025_out_of_40": None,
                    "count_unit": "years out of 40 years (NOT per decade)",
                    "frequency_claim_allowed": False,
                    "frequency_ratio": None,
                    "era5_pre_1979_note": "ERA5 before 1979 assimilates no satellite observations and is less reliable.",
                },
            }

        # Ready-made daily comparisons for the writer: forecast high vs typical high, forecast low vs
        # typical low. (The map's anomaly is for one hour of the day and must not be mixed with these.)
        if era_info.get("wmo_tmax_normal_c") is not None and cand.get("t2m_max_24h_c") is not None:
            era_info["high_vs_typical_c"] = round(float(cand["t2m_max_24h_c"]) - float(era_info["wmo_tmax_normal_c"]), 1)
        if era_info.get("wmo_tmin_normal_c") is not None and cand.get("t2m_min_24h_c") is not None:
            era_info["low_vs_typical_c"] = round(float(cand["t2m_min_24h_c"]) - float(era_info["wmo_tmin_normal_c"]), 1)
        out["era5"] = era_info
        enriched.append(out)
    return enriched
