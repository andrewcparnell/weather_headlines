"""One-off offline builder for the compressed 0.5 degree ERA5 seasonal climatology & orography asset.

Downloads 24 half-monthly (15-day window) global slices across all 4 synoptic hours (00, 06, 12, 18 UTC)
from WeatherBench 2's 1990-2019 ERA5 climatology (`1990-2019_6h_1440x721.zarr`), plus static ERA5 surface
geopotential (`geopotential_at_surface`) and historical spread/quantile samples from ARCO-ERA5, then fits
compact harmonic seasonal tables saved to `pipeline/data/era5_climo_0p5.npz` (~3.5 MB).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numcodecs import Blosc

from .config import ROOT_DIR
from .opendata import http_get, to_half_degree

WB2_CLIMO_ROOT = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
)
ARCO_ERA5_STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)
OUT_ASSET_PATH = ROOT_DIR / "pipeline" / "data" / "era5_climo_0p5.npz"

# 24 evenly spaced day-of-year anchors (1-indexed DOY: 8, 23, 38, ..., 353)
SAMPLE_DOY_0IDX = [7 + 15 * k for k in range(24)]  # 0-based indices into 0..365


def _decode_zarr_chunk(url: str, shape: tuple[int, ...], dtype: str = "<f4") -> np.ndarray:
    compressed = http_get(url, timeout=60, attempts=4)
    raw = Blosc().decode(compressed)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _orient_to_standard_025(arr_2d: np.ndarray, lat_ascending: bool, lon_0_360: bool) -> np.ndarray:
    """Orient [721, 1440] array to standard [lat 90..-90, lon -180..179.75]."""
    out = arr_2d
    if lat_ascending:
        out = out[::-1, :]
    if lon_0_360:
        out = np.concatenate([out[:, 720:], out[:, :720]], axis=1)
    return out.astype(np.float32)


def _fetch_wb2_var_at_doy(var_name: str, doy_0idx: int, lat_asc: bool, lon_0360: bool) -> np.ndarray:
    """Return [4 hours (0,6,12,18), 361, 720] at 0.5 deg for a single 0-based day-of-year."""
    d_chunk = doy_0idx // 3
    d_off = doy_0idx % 3
    c0 = _decode_zarr_chunk(f"{WB2_CLIMO_ROOT}/{var_name}/0.{d_chunk}.0.0", (3, 3, 721, 1440))
    c1 = _decode_zarr_chunk(f"{WB2_CLIMO_ROOT}/{var_name}/1.{d_chunk}.0.0", (3, 3, 721, 1440))
    slices_05 = []
    for hr_arr in (c0[0, d_off], c0[1, d_off], c0[2, d_off], c1[0, d_off]):
        oriented = _orient_to_standard_025(hr_arr, lat_asc, lon_0360)
        slices_05.append(to_half_degree(oriented))
    return np.stack(slices_05, axis=0)  # [4, 361, 720]


def _fit_harmonics(y_samples: np.ndarray, doys_1idx: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    """Fit Fourier seasonal basis [1 + 2*K, ...] along axis 0 via least squares."""
    theta = 2.0 * np.pi * (doys_1idx - 1.0) / 365.25
    cols = [np.ones_like(theta)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    design = np.stack(cols, axis=1)  # [M, 1 + 2*K]
    flat = y_samples.reshape(y_samples.shape[0], -1)
    coeffs, _, _, _ = np.linalg.lstsq(design, flat, rcond=None)
    return coeffs.reshape((design.shape[1],) + y_samples.shape[1:]).astype(np.float32)


def build_precomputed_climatology(out_path: Path = OUT_ASSET_PATH) -> Path:
    print("[precompute] Inspecting WeatherBench2 coordinate orientation...", file=sys.stderr)
    lats_wb = _decode_zarr_chunk(f"{WB2_CLIMO_ROOT}/latitude/0", (721,), "<f4")
    lons_wb = _decode_zarr_chunk(f"{WB2_CLIMO_ROOT}/longitude/0", (1440,), "<f4")
    lat_asc = bool(lats_wb[0] < lats_wb[-1])
    lon_0360 = bool(lons_wb[0] >= -1.0 and lons_wb[-1] > 180.0)

    print("[precompute] Fetching 24 half-monthly 6-hourly ERA5 1990-2019 global slices...", file=sys.stderr)

    def _worker_doy(doy0: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        t_k = _fetch_wb2_var_at_doy("2m_temperature", doy0, lat_asc, lon_0360) - 273.15
        tp_m = _fetch_wb2_var_at_doy("total_precipitation_24hr", doy0, lat_asc, lon_0360)
        ws = _fetch_wb2_var_at_doy("10m_wind_speed", doy0, lat_asc, lon_0360)
        # Daily mean 24h rain in mm/day (each synoptic 24h window average)
        rain_mm_day = np.maximum(0.0, np.mean(tp_m, axis=0) * 1000.0)
        wind_mean = np.maximum(0.0, np.mean(ws, axis=0))
        return doy0, t_k, rain_mm_day, wind_mean

    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(_worker_doy, SAMPLE_DOY_0IDX))

    rows.sort(key=lambda r: r[0])
    doys_1idx = np.array([r[0] + 1 for r in rows], dtype=np.float64)
    t2m_6h = np.stack([r[1] for r in rows], axis=0)  # [24, 4, 361, 720]
    rain_24 = np.stack([r[2] for r in rows], axis=0)  # [24, 361, 720]
    wind_24 = np.stack([r[3] for r in rows], axis=0)  # [24, 361, 720]

    # Fit 3-harmonic seasonal curves for each of the 4 synoptic UTC hours [0, 6, 12, 18]
    # Shape: [4, 7, 361, 720]
    t2m_harmonics = np.stack(
        [_fit_harmonics(t2m_6h[:, h_idx, :, :], doys_1idx, n_harmonics=3) for h_idx in range(4)],
        axis=0,
    )
    rain_harmonics = _fit_harmonics(rain_24, doys_1idx, n_harmonics=3)  # [7, 361, 720]
    wind_harmonics = _fit_harmonics(wind_24, doys_1idx, n_harmonics=2)  # [5, 361, 720]

    # Fetch ERA5 static surface geopotential at 0.25 deg for exact 6.5 C/km orography lapse-rate adjustment
    print("[precompute] Fetching ARCO-ERA5 0.25 deg surface geopotential (orography)...", file=sys.stderr)
    ref_idx = int((datetime(2020, 1, 1, 0, tzinfo=timezone.utc) - ERA5_EPOCH).total_seconds() // 3600)
    z_raw = _decode_zarr_chunk(f"{ARCO_ERA5_STORE}/geopotential_at_surface/{ref_idx}.0.0", (721, 1440), "<f4")
    z_std = _orient_to_standard_025(z_raw, lat_ascending=False, lon_0_360=True)
    era5_elev_025_m = np.rint(z_std / 9.80665).astype(np.int16)

    # Estimate seasonal synoptic day-to-day temperature std [4 seasons, 361, 720] from 12 sample WMO slices per season
    print("[precompute] Sampling WMO 1991-2020 interannual temperature spread & gust/rain 90th percentiles...", file=sys.stderr)
    season_months = [1, 4, 7, 10]
    sample_years = [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]

    def _fetch_arco_slice(args: tuple[str, int, int]) -> tuple[str, int, np.ndarray]:
        vname, m_idx, yr = args
        month = season_months[m_idx]
        dt = datetime(yr, month, 15, 12, tzinfo=timezone.utc)
        h_idx = int((dt - ERA5_EPOCH).total_seconds() // 3600)
        arr = _decode_zarr_chunk(f"{ARCO_ERA5_STORE}/{vname}/{h_idx}.0.0", (721, 1440), "<f4")
        arr_std = _orient_to_standard_025(arr, lat_ascending=False, lon_0_360=True)
        return vname, m_idx, arr_std[::2, ::2]

    tasks = []
    for m_idx in range(4):
        for yr in sample_years:
            tasks.append(("2m_temperature", m_idx, yr))
            tasks.append(("10m_wind_gust_since_previous_post_processing", m_idx, yr))

    with ThreadPoolExecutor(max_workers=16) as pool:
        arco_res = list(pool.map(_fetch_arco_slice, tasks))

    t2m_std_seasons = np.zeros((4, 361, 720), dtype=np.float32)
    gust_p90_seasons = np.zeros((4, 361, 720), dtype=np.float32)
    for m_idx in range(4):
        t_stack = np.stack([a - 273.15 for v, mi, a in arco_res if v == "2m_temperature" and mi == m_idx], axis=0)
        g_stack = np.stack([a for v, mi, a in arco_res if v != "2m_temperature" and mi == m_idx], axis=0)
        t2m_std_seasons[m_idx] = np.maximum(0.85, np.std(t_stack, axis=0))
        # Scale 1-hour gust p90 by 1.18 to match 24-hour daily max gust distribution
        gust_p90_seasons[m_idx] = np.percentile(g_stack, 90, axis=0) * 1.18

    season_doys = np.array([15.0, 105.0, 196.0, 288.0], dtype=np.float64)
    t2m_std_harmonics = _fit_harmonics(t2m_std_seasons, season_doys, n_harmonics=1)  # [3, 361, 720]
    gust_p90_harmonics = _fit_harmonics(gust_p90_seasons, season_doys, n_harmonics=1)  # [3, 361, 720]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Quantise harmonics to 0.02-unit int16 (50x) so deflate achieves ~4x better compression than noisy float16
    np.savez_compressed(
        out_path,
        version=np.array([3], dtype=np.int16),
        scale=np.array([50.0], dtype=np.float32),
        t2m_harmonics_i16=np.rint(np.clip(t2m_harmonics * 50.0, -32000, 32000)).astype(np.int16),
        t2m_std_harmonics_i16=np.rint(np.clip(t2m_std_harmonics * 50.0, -32000, 32000)).astype(np.int16),
        rain_harmonics_i16=np.rint(np.clip(rain_harmonics * 50.0, -32000, 32000)).astype(np.int16),
        wind_harmonics_i16=np.rint(np.clip(wind_harmonics * 50.0, -32000, 32000)).astype(np.int16),
        gust_p90_harmonics_i16=np.rint(np.clip(gust_p90_harmonics * 50.0, -32000, 32000)).astype(np.int16),
        era5_elev_025_m=era5_elev_025_m,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[precompute] Saved {out_path} ({size_mb:.2f} MB).", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    build_precomputed_climatology()
