"""Encode 0.5 degree global weather layers into compact lossless WebP textures and manifest.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import LAYERS, SITE_DIR, layer_metadata
from .opendata import LiveWeatherFields, to_half_degree
from .scanner import extract_pressure_features


def _decode_delta_ring(delta_arr: list[int], scale: float = 100.0) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    x = 0
    y = 0
    for i in range(0, len(delta_arr) - 1, 2):
        x += delta_arr[i]
        y += delta_arr[i + 1]
        pts.append((x / scale, y / scale))
    return pts


def render_basemap_webp(
    src_json: Path = SITE_DIR / "assets" / "basemap.json",
    out_webp: Path = SITE_DIR / "assets" / "basemap.webp",
    width: int = 2048,
    height: int = 1024,
) -> Path:
    """Pre-render `basemap.json` ocean, land, coastlines, and borders into a compact antialiased WebP texture."""
    topo = json.loads(src_json.read_text())
    scale = float(topo.get("scale") or 100.0)

    # Render at 2x supersampling (4096x2048) and downsample for clean antialiased coastlines & borders
    ss = 2
    W = width * ss
    H = height * ss

    y_norm = np.linspace(0.0, 1.0, H, dtype=np.float32)
    top_rgb = np.array([9.0, 27.0, 50.0], dtype=np.float32)
    mid_rgb = np.array([7.0, 20.0, 38.0], dtype=np.float32)
    bot_rgb = np.array([9.0, 26.0, 48.0], dtype=np.float32)
    bg = np.empty((H, 3), dtype=np.uint8)
    upper = y_norm <= 0.5
    t_up = (y_norm[upper] / 0.5)[:, None]
    t_lo = ((y_norm[~upper] - 0.5) / 0.5)[:, None]
    bg[upper] = np.rint(top_rgb + t_up * (mid_rgb - top_rgb)).astype(np.uint8)
    bg[~upper] = np.rint(mid_rgb + t_lo * (bot_rgb - mid_rgb)).astype(np.uint8)
    canvas_arr = np.broadcast_to(bg[:, None, :], (H, W, 3)).copy()

    img = Image.fromarray(canvas_arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    land_fill = (24, 43, 66)
    coast_col = (145, 178, 210)
    border_col = (68, 96, 126)

    # First pass: land polygon fills
    coast_polylines: list[list[tuple[float, float]]] = []
    for encoded_ring in topo.get("land") or []:
        pts = _decode_delta_ring(encoded_ring, scale)
        if len(pts) < 3:
            continue
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        min_lon, max_lon = min(lons), max(lons)
        avg_lat = sum(lats) / len(lats)
        is_antarctica = avg_lat < -62.0 and (max_lon - min_lon) > 300.0

        for shift in (-360.0, 0.0, 360.0):
            if shift != 0.0 and not is_antarctica and min_lon >= -180.0 and max_lon <= 180.0:
                continue
            poly = [(((lo + shift + 180.0) / 360.0) * W, ((90.0 - la) / 180.0) * H) for lo, la in pts]
            if is_antarctica and shift == 0.0:
                poly.append((((pts[-1][0] + 180.0) / 360.0) * W, float(H)))
                poly.append((((pts[0][0] + 180.0) / 360.0) * W, float(H)))
            draw.polygon(poly, fill=land_fill)

        # Split dateline jumps for coastline strokes
        seg: list[tuple[float, float]] = []
        for idx, (lo, la) in enumerate(pts):
            if idx > 0 and abs(lo - pts[idx - 1][0]) > 40.0:
                if len(seg) >= 2:
                    coast_polylines.append(seg)
                seg = []
            seg.append((((lo + 180.0) / 360.0) * W, ((90.0 - la) / 180.0) * H))
        if len(seg) >= 2:
            coast_polylines.append(seg)

    # Second pass: country borders
    for encoded_line in topo.get("borders") or []:
        pts = _decode_delta_ring(encoded_line, scale)
        if len(pts) < 2:
            continue
        seg = []
        for idx, (lo, la) in enumerate(pts):
            if idx > 0 and abs(lo - pts[idx - 1][0]) > 40.0:
                if len(seg) >= 2:
                    draw.line(seg, fill=border_col, width=2)
                seg = []
            seg.append((((lo + 180.0) / 360.0) * W, ((90.0 - la) / 180.0) * H))
        if len(seg) >= 2:
            draw.line(seg, fill=border_col, width=2)

    # Third pass: crisp coastlines on top
    for seg in coast_polylines:
        draw.line(seg, fill=coast_col, width=3)

    final_img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
    out_webp.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_webp.with_suffix(".tmp.webp")
    final_img.save(tmp, format="WEBP", quality=88, method=6)
    tmp.replace(out_webp)
    return out_webp


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

    # Ensure pre-rendered basemap.webp exists for instant client startup
    basemap_webp = SITE_DIR / "assets" / "basemap.webp"
    basemap_json = SITE_DIR / "assets" / "basemap.json"
    if not basemap_webp.exists() and basemap_json.exists():
        render_basemap_webp(basemap_json, basemap_webp)

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
