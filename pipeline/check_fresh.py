"""Is a new bulletin due? Used by the hourly scheduled workflow to skip runs cheaply.

A bulletin is due when ECMWF has published a newer forecast run than the one the current stories were
written from (a new run appears every 6 hours), or when the stories are more than 7 hours old.
Writes `due=true|false` to $GITHUB_OUTPUT (and prints the reason). Costs nothing: no Claude call.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from .config import DATA_DIR
from .opendata import find_latest_run

MAX_AGE_HOURS = 7.0


def _parse(iso: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")) if iso else None
    except ValueError:
        return None


def bulletin_due(bulletin: dict | None, latest_run: datetime, now: datetime | None = None) -> tuple[bool, str]:
    now = now or datetime.now(timezone.utc)
    if not isinstance(bulletin, dict) or not bulletin.get("stories"):
        return True, "no current bulletin"
    if str(bulletin.get("generator") or "").startswith("era5-deterministic"):
        return True, "current bulletin is template text"
    generated = _parse(bulletin.get("generated_utc"))
    if generated is None or (now - generated).total_seconds() / 3600.0 > MAX_AGE_HOURS:
        return True, f"stories are more than {MAX_AGE_HOURS:.0f} hours old"
    written_from = _parse(bulletin.get("stories_model_run_utc"))
    if written_from is None:
        # Older bulletins lack the field; fall back to their age
        return (now - generated).total_seconds() / 3600.0 > 5.5, "age check (no stories_model_run_utc)"
    if latest_run > written_from:
        return True, f"new ECMWF run {latest_run:%Y-%m-%d %HZ} (stories use {written_from:%Y-%m-%d %HZ})"
    return False, f"stories already use the latest ECMWF run ({written_from:%Y-%m-%d %HZ})"


def main() -> int:
    path = DATA_DIR / "headlines.json"
    try:
        bulletin = json.loads(path.read_text()) if path.exists() else None
    except Exception:
        bulletin = None
    try:
        latest_run, _ = find_latest_run()
    except Exception as exc:  # ECMWF unreachable: run anyway and let the build report the problem
        due, reason = True, f"could not check ECMWF ({exc})"
    else:
        due, reason = bulletin_due(bulletin, latest_run)
    print(f"[check_fresh] due={str(due).lower()}: {reason}", file=sys.stderr)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"due={str(due).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
