"""Fetch worldwide official severe weather alerts (GDACS and WMO CAP summary) for candidate context."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from .config import USER_AGENT

GDACS_TYPES = {"TC": "Tropical cyclone", "FL": "Flood", "DR": "Drought", "WF": "Wildfire"}


def _fetch_json(url: str, timeout: int = 12) -> dict | list | None:
    try:
        proc = subprocess.run(
            ["curl", "-sSfL", "--connect-timeout", "5", "-m", str(timeout), "-A", USER_AGENT, "-H", "Accept: application/json", url],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception:
        pass
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[warnings] Feed unavailable ({url[:60]}...): {exc}", file=sys.stderr)
        return None


def is_gdacs_event_current(props: dict) -> bool:
    """Return True only when GDACS `iscurrent` is genuinely true (handles string 'false' as False)."""
    raw = props.get("iscurrent", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"false", "0", "no", "n", "inactive"}
    return True


def fetch_global_warnings(days_back: int = 7, raw_payload: dict | None = None) -> list[dict]:
    """Return active current Orange/Red GDACS disaster alerts worldwide with coordinates and report URLs."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    url = (
        "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?"
        f"eventlist={';'.join(GDACS_TYPES)}&alertlevel=Orange;Red&fromdate={start}&todate={today}"
    )
    payload = raw_payload if raw_payload is not None else _fetch_json(url, timeout=12)
    if not isinstance(payload, dict):
        return []

    alerts: list[dict] = []
    for feat in payload.get("features") or []:
        props = feat.get("properties") or {}
        if not is_gdacs_event_current(props):
            continue
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") if geom.get("type") == "Point" else None
        lon = float(coords[0]) if coords and len(coords) >= 2 else None
        lat = float(coords[1]) if coords and len(coords) >= 2 else None
        report_url = (props.get("url") or {}).get("report")
        alerts.append(
            {
                "source": "GDACS",
                "type": GDACS_TYPES.get(props.get("eventtype", ""), props.get("eventtype", "Weather")),
                "name": props.get("name") or props.get("eventname") or "Active alert",
                "country": props.get("country") or "",
                "alert_level": props.get("alertlevel") or "Orange",
                "is_current": True,
                "lat": lat,
                "lon": lon,
                "from_date": props.get("fromdate"),
                "to_date": props.get("todate"),
                "url": report_url,
            }
        )
    return alerts
