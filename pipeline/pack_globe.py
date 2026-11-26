"""Encode 0.5 degree global weather layers into compact lossless WebP textures and manifest.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .config import LAYERS, layer_metadata
from .opendata import LiveWeatherFields, to_half_degree
from .scanner import extract_pressure_features


def quantise_u8(arr: np.ndarray, lo: float, hi: float, scale: str = "linear") -> np.ndarray:
    """Quantise float array to uint8 [0..254], reserving 255 for NaN/missing."""
    out = np.full(arr.shape, 255, dtype=np.uint8)
    valid = np.isfinite(arr)
    norm = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    if scale == "sqrt":
        norm = np.sqrt(norm)
    out[valid] = np.rint(norm * 254.0).astype(np.uint8)
    return out


def dequantise_u8(encoded: np.ndarray, lo: float, hi: float, scale: str = "linear") -> np.ndarray:
    """Inverse of quantise_u8 (used in unit tests and mirrored in browser JS)."""
    vals = np.full(encoded.shape, np.nan, dtype=np.float32)
    valid = encoded < 255
    t = encoded[valid].astype(np.float32) / 254.0
    if scale == "sqrt":
        t = t * t
    vals[valid] = lo + t * (hi - lo)
    return vals


def save_lossless_webp(img_arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGB" if img_arr.ndim == 3 else "L"
    img = Image.fromarray(img_arr, mode=mode)
    tmp = path.with_suffix(".tmp.webp")
    img.save(tmp, format="WEBP", lossless=True, method=4)
    tmp.replace(path)


def package_globe_layers(
    weather: LiveWeatherFields,
    anom_025_c: np.ndarray,
    climo_source: str,
    out_dir: Path,
) -> dict:
    """Write all 5 global 0.5 deg WebP textures and manifest.json under out_dir."""
    layers_dir = out_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    t2m_05 = to_half_degree(weather.t2m_c)
    anom_05 = to_half_degree(anom_025_c)
    u10_05 = to_half_degree(weather.u10)
    v10_05 = to_half_degree(weather.v10)
    wspd_05 = np.hypot(u10_05, v10_05).astype(np.float32)
    rain_05 = to_half_degree(weather.rain_24h_mm)
    mslp_05 = to_half_degree(weather.mslp_hpa)

    # 1. Scalar 8-bit greyscale layers
    for layer_id, grid in (
        ("temperature", t2m_05),
        ("anomaly", anom_05),
        ("rain", rain_05),
        ("pressure", mslp_05),
    ):
        spec = LAYERS[layer_id]
        u8 = quantise_u8(grid, spec.lo, spec.hi, spec.scale)
        save_lossless_webp(u8, layers_dir / f"{layer_id}.webp")

    # 2. Wind RGB WebP: R = u10 [-45..45], G = v10 [-45..45], B = speed [0..45]
    wind_spec = LAYERS["wind"]
    u_u8 = quantise_u8(u10_05, -wind_spec.hi, wind_spec.hi, "linear")
    v_u8 = quantise_u8(v10_05, -wind_spec.hi, wind_spec.hi, "linear")
    s_u8 = quantise_u8(wspd_05, wind_spec.lo, wind_spec.hi, "linear")
    wind_rgb = np.stack([u_u8, v_u8, s_u8], axis=-1)
    save_lossless_webp(wind_rgb, layers_dir / "wind.webp")

    # 3. Pressure centres & smooth global isobars
    centres, isobars = extract_pressure_features(mslp_05)

    now_utc = datetime.now(timezone.utc)
    manifest = {
        "version": 1,
        "generated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step_hours": weather.step_hours,
        "rain_window_hours": list(weather.rain_window_hours),
        "climatology_baseline": climo_source,
        "grid": {
            "nx": 720,
            "ny": 361,
            "step_deg": 0.5,
            "lon_first": -180.0,
            "lat_first": 90.0,
            "wind_uv_range": [-wind_spec.hi, wind_spec.hi],
        },
        "attribution": {
            "forecast": f"Data: © {now_utc.year} ECMWF, CC BY 4.0 (IFS Open Data, regridded).",
            "climatology": "Climatology & rankings: Copernicus Climate Change Service ERA5 (1940-present).",
            "maps": "Outlines: Natural Earth (public domain). Places: GeoNames (CC BY 4.0).",
        },
        "layers": layer_metadata(),
        "pressure_centres": centres,
        "isobars": isobars,
    }

    manifest_path = out_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(manifest, separators=(",", ":")))
    tmp.replace(manifest_path)
    return manifest
