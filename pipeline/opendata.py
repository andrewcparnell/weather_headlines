"""Fetch and decode global 0.25 degree fields from ECMWF Open Data via HTTP Range requests.

All data comes from ECMWF Open Data (CC BY 4.0), reading the AWS S3 mirror first with
data.ecmwf.int as automatic fallback. Supports both 00z/12z (`oper`) and 06z/18z (`scda`)
6-hourly forecast cycles so refreshes every 6 hours always pick up the newest cycle.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, USER_AGENT

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOTS = [
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "https://data.ecmwf.int/forecasts",
]
HOUR = timedelta(hours=1)
HEADERS = {"User-Agent": USER_AGENT}

_index_cache: dict[str, tuple[str, list[dict]]] = {}
_index_lock = threading.Lock()


@dataclass
class LiveWeatherFields:
    run_time: datetime
    valid_time: datetime
    step_hours: int
    rain_window_hours: tuple[int, int]
    lats: np.ndarray  # (721,) 90.0 down to -90.0
    lons: np.ndarray  # (1440,) -180.0 up to 179.75
    t2m_c: np.ndarray  # 2 m temperature in deg C at valid_time
    u10: np.ndarray  # 10 m zonal wind in m/s
    v10: np.ndarray  # 10 m meridional wind in m/s
    wind_speed: np.ndarray  # 10 m wind speed in m/s
    gust_max: np.ndarray  # max 10 m wind gust in m/s over 24 h window
    rain_24h_mm: np.ndarray  # 24 h accumulated precipitation in mm
    mslp_hpa: np.ndarray  # mean sea level pressure in hPa
    lsm: np.ndarray | None = None  # land-sea mask (0..1)
    t2m_max_24h_c: np.ndarray | None = None  # max 2 m temperature in deg C over 24 h window
    t2m_min_24h_c: np.ndarray | None = None  # min 2 m temperature in deg C over 24 h window
    ifs_elev_m: np.ndarray | None = None  # IFS surface elevation in metres (z / 9.80665)
    ifs_sdor_m: np.ndarray | None = None  # IFS sub-gridscale orography std in metres
    rain_window_start_utc: datetime | None = None
    rain_window_end_utc: datetime | None = None


def floor6(dt: datetime) -> datetime:
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.replace(hour=dt_utc.hour - (dt_utc.hour % 6), minute=0, second=0, microsecond=0)


def streams_for_hour(hour: int) -> tuple[str, ...]:
    return ("oper",) if hour in (0, 12) else ("scda", "oper")


def rel_path(run: datetime, stream: str, step: int, res: str = "0p25") -> str:
    return f"{run:%Y%m%d}/{run:%H}z/ifs/{res}/{stream}/{run:%Y%m%d%H}0000-{step}h-{stream}-fc"


def http_get(url: str, headers: dict | None = None, timeout: int = 60, attempts: int = 4) -> bytes:
    merged = {**HEADERS, **(headers or {})}
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=merged)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if headers and "Range" in headers and resp.status != 206:
                    raise IOError(f"Expected HTTP 206 Partial Content from {url}, got {resp.status}")
                return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, IOError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (2**attempt))
    raise IOError(f"Unreachable URL: {url}")


def index_exists(rel: str) -> bool:
    for root in ROOTS:
        try:
            req = urllib.request.Request(f"{root}/{rel}.index", method="HEAD", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def find_latest_run(now: datetime | None = None, min_horizon_h: int = 36) -> tuple[datetime, str]:
    """Return (run_datetime, stream) for the newest 6-hourly ECMWF Open Data run available."""
    ref = floor6(now or datetime.now(timezone.utc))
    for back in range(0, 10):
        candidate = ref - back * 6 * HOUR
        for stream in streams_for_hour(candidate.hour):
            if index_exists(rel_path(candidate, stream, min_horizon_h)) and index_exists(
                rel_path(candidate, stream, 0)
            ):
                return candidate, stream
    raise RuntimeError("Could not locate any recent ECMWF Open Data 0.25 deg run.")


def get_index_rows(rel: str) -> tuple[str, list[dict]]:
    with _index_lock:
        if rel in _index_cache:
            return _index_cache[rel]
    for root in ROOTS:
        try:
            raw = http_get(f"{root}/{rel}.index", timeout=30).decode("utf-8")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            with _index_lock:
                _index_cache[rel] = (root, rows)
            return root, rows
        except Exception:
            continue
    raise RuntimeError(f"Unable to download index for {rel}")


def select_index_row(rows: list[dict], param: str, levelist: int | None = None) -> dict | None:
    """Pick the GRIB message for `param`: a surface field unless a pressure level is given.

    Several parameters (notably `z`, geopotential) exist both at the surface and on pressure levels, and
    the order of rows in the index varies between runs. Taking "the first `z`" sometimes returned the
    300 hPa height (~9 km) as the surface elevation, which shifted the whole temperature anomaly map
    by about +60 °C.
    """
    want_levtype = "pl" if levelist is not None else "sfc"
    for r in rows:
        if r.get("param") != param:
            continue
        if levelist is not None and str(r.get("levelist")) != str(levelist):
            continue
        if (r.get("levtype") or "sfc") != want_levtype:
            continue
        return r
    return None


def fetch_grib_message(run: datetime, stream: str, step: int, param: str, levelist: int | None = None) -> Path:
    """Download a single GRIB2 message using its byte offset in the .index file (cached on disk)."""
    lev_tag = f"_pl{levelist}" if levelist is not None else "_sfc"
    out_path = CACHE_DIR / "opendata" / f"{run:%Y%m%d%H}_{stream}" / f"{param}{lev_tag}_step{step:03d}.grib2"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    rel = rel_path(run, stream, step)
    root, rows = get_index_rows(rel)
    match = select_index_row(rows, param, levelist)
    if match is None:
        raise KeyError(f"Parameter {param!r} not found in {rel}.index")

    offset = int(match["_offset"])
    length = int(match["_length"])
    byte_range = {"Range": f"bytes={offset}-{offset + length - 1}"}

    mirrors = [root] + [r for r in ROOTS if r != root]
    last_err: Exception | None = None
    for mirror in mirrors:
        try:
            payload = http_get(f"{mirror}/{rel}.grib2", headers=byte_range, timeout=60)
            if len(payload) != length:
                raise IOError(f"Incomplete GRIB slice ({len(payload)} != {length} bytes)")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(out_path)
            return out_path
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed downloading {param} step {step} from all mirrors: {last_err}")


def decode_grib_025(path: Path) -> np.ndarray:
    """Decode a 0.25 degree global GRIB2 message to float32 [721 lat (90..-90), 1440 lon (-180..179.75)]."""
    import eccodes

    raw = path.read_bytes()
    gid = eccodes.codes_new_from_message(raw)
    try:
        ni = int(eccodes.codes_get(gid, "Ni"))
        nj = int(eccodes.codes_get(gid, "Nj"))
        lat_first = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
        lon_first = float(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
        eccodes.codes_set(gid, "missingValue", -9999.0)
        vals = eccodes.codes_get_values(gid).reshape(nj, ni).astype(np.float32)
        vals[vals == -9999.0] = np.nan
    finally:
        eccodes.codes_release(gid)

    if lat_first < 0:
        vals = vals[::-1, :]
    step_deg = 360.0 / ni
    shift = int(round(((-180.0 - lon_first) % 360.0) / step_deg))
    if shift:
        vals = np.roll(vals, -shift, axis=1)
    return vals


def fetch_live_weather(now: datetime | None = None) -> LiveWeatherFields:
    """Fetch the latest global 0.25 degree ECMWF forecast snapshot and 24 h rain/gust windows."""
    target_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run, stream = find_latest_run(target_now)

    raw_hours = max(3.0, (target_now - run).total_seconds() / 3600.0)
    step = int(round(raw_hours / 3.0) * 3)
    step = max(3, min(24, step))
    valid_time = run + step * HOUR

    rain_start = max(0, step - 12)
    rain_end = rain_start + 24
    rain_start_utc = run + rain_start * HOUR
    rain_end_utc = run + rain_end * HOUR

    gust_steps = list(range(max(3, rain_start + 3), rain_end + 1, 3))

    requests_to_fetch: list[tuple[str, int, str]] = [
        ("t2m", step, "2t"),
        ("u10", step, "10u"),
        ("v10", step, "10v"),
        ("msl", step, "msl"),
        ("lsm", 0, "lsm"),
        ("z_sfc", 0, "z"),
        ("sdor", 0, "sdor"),
        ("tp_end", rain_end, "tp"),
    ]
    if rain_start > 0:
        requests_to_fetch.append(("tp_start", rain_start, "tp"))
    for gs in gust_steps:
        requests_to_fetch.append((f"gust_{gs}", gs, "10fg"))
        requests_to_fetch.append((f"mx2t_{gs}", gs, "mx2t3"))
        requests_to_fetch.append((f"mn2t_{gs}", gs, "mn2t3"))

    print(
        f"[opendata] Fetching ECMWF {run:%Y-%m-%d %H}Z ({stream}), valid {valid_time:%Y-%m-%d %H:%M}Z (step +{step}h)...",
        file=sys.stderr,
    )

    optional_prefixes = ("lsm", "z_sfc", "sdor", "gust_", "mx2t_", "mn2t_")

    def _worker(item: tuple[str, int, str]) -> tuple[str, np.ndarray | None]:
        key, st, param = item
        try:
            p = fetch_grib_message(run, stream, st, param)
            return key, decode_grib_025(p)
        except Exception as exc:
            if key.startswith(optional_prefixes):
                print(f"[opendata] Optional field {param} step {st} unavailable: {exc}", file=sys.stderr)
                return key, None
            raise

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(_worker, requests_to_fetch))

    t2m_c = results["t2m"] - 273.15
    u10 = results["u10"]
    v10 = results["v10"]
    wind_speed = np.hypot(u10, v10).astype(np.float32)
    mslp_hpa = results["msl"] / 100.0

    tp_end = results["tp_end"]
    tp_start = results.get("tp_start")
    tp_diff_m = tp_end if tp_start is None else (tp_end - tp_start)
    rain_24h_mm = np.maximum(0.0, tp_diff_m * 1000.0).astype(np.float32)

    gust_arrays = [results[f"gust_{gs}"] for gs in gust_steps if results.get(f"gust_{gs}") is not None]
    if gust_arrays:
        gust_max = np.nanmax(np.stack(gust_arrays, axis=0), axis=0).astype(np.float32)
    else:
        gust_max = (wind_speed * 1.35).astype(np.float32)

    mx_arrays = [results[f"mx2t_{gs}"] - 273.15 for gs in gust_steps if results.get(f"mx2t_{gs}") is not None]
    t2m_max_24h_c = (
        np.nanmax(np.stack(mx_arrays, axis=0), axis=0).astype(np.float32)
        if mx_arrays
        else t2m_c.astype(np.float32)
    )
    mn_arrays = [results[f"mn2t_{gs}"] - 273.15 for gs in gust_steps if results.get(f"mn2t_{gs}") is not None]
    t2m_min_24h_c = (
        np.nanmin(np.stack(mn_arrays, axis=0), axis=0).astype(np.float32)
        if mn_arrays
        else t2m_c.astype(np.float32)
    )

    z_sfc = results.get("z_sfc")
    ifs_elev_m = (z_sfc / 9.80665).astype(np.float32) if z_sfc is not None else None
    ifs_sdor_m = results.get("sdor").astype(np.float32) if results.get("sdor") is not None else None

    lats = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lons = (np.arange(1440, dtype=np.float32) * 0.25) - 180.0

    return LiveWeatherFields(
        run_time=run,
        valid_time=valid_time,
        step_hours=step,
        rain_window_hours=(rain_start, rain_end),
        lats=lats,
        lons=lons,
        t2m_c=t2m_c.astype(np.float32),
        u10=u10.astype(np.float32),
        v10=v10.astype(np.float32),
        wind_speed=wind_speed,
        gust_max=gust_max,
        rain_24h_mm=rain_24h_mm,
        mslp_hpa=mslp_hpa.astype(np.float32),
        lsm=results.get("lsm"),
        t2m_max_24h_c=t2m_max_24h_c,
        t2m_min_24h_c=t2m_min_24h_c,
        ifs_elev_m=ifs_elev_m,
        ifs_sdor_m=ifs_sdor_m,
        rain_window_start_utc=rain_start_utc,
        rain_window_end_utc=rain_end_utc,
    )


def to_half_degree(field_025: np.ndarray) -> np.ndarray:
    """Downsample [721, 1440] (0.25 deg) to [361, 720] (0.5 deg) cleanly."""
    if field_025.shape != (721, 1440):
        raise ValueError(f"Expected (721, 1440) grid, got {field_025.shape}")
    rows = field_025[::2, :]
    blended = 0.5 * (rows + np.roll(rows, -1, axis=1))
    return blended[:, ::2].astype(np.float32)
