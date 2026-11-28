"""End-to-end 6-hourly builder: ECMWF Open Data -> hotspots -> news -> ERA5 history -> Claude bulletin -> site.

If the bulletin cannot be written properly, the build FAILS (non-zero exit) instead of publishing template
text: GitHub then emails the failure, nothing is deployed, and the site keeps the last good bulletin.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .climatology import enrich_all_candidates, load_era5_historical_context
from .config import DATA_DIR
from .news_agent import build_deterministic_bulletin, generate_headlines_with_claude
from .news_leads import attach_news, choose_stories
from .opendata import fetch_live_weather
from .pack_globe import package_globe_layers
from .places import attach_location_context
from .scanner import detect_global_candidates
from .stamp_assets import stamp_frontend_assets
from .tts import build_bulletin_audio
from .validator import validate_and_clean_bulletin
from .warnings_feed import fetch_global_warnings

HOTSPOTS_TO_SCAN = 14  # more than we need, so the story choice can favour places that are in the news
STORIES_PER_BULLETIN = 5


class BulletinError(RuntimeError):
    """The bulletin could not be produced; the build must fail rather than publish template text."""


def _redact_secrets(msg: str) -> str:
    return re.sub(r"sk-ant-[A-Za-z0-9_-]+", "sk-ant-***", msg or "")


def _is_recent_valid_bulletin(bulletin: dict | None, max_age_hours: float = 18.0) -> bool:
    if not isinstance(bulletin, dict):
        return False
    _, errs = validate_and_clean_bulletin(bulletin, require_links=False)
    if errs:
        return False
    if str(bulletin.get("generator") or "").startswith("era5-deterministic"):
        return False  # never keep template text
    gen = str(bulletin.get("generated_utc") or "").strip()
    if not gen:
        return False
    try:
        dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        return age_h <= max_age_hours
    except ValueError:
        return False


def _annotate_error(title: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error title={title}::{message}", flush=True)


def run_pipeline(
    out_dir: Path = DATA_DIR,
    skip_headlines_if_no_key: bool = False,
    dump_candidates_path: Path | None = None,
    allow_template: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    headlines_path = out_dir / "headlines.json"
    previous_bulletin = None
    if headlines_path.exists():
        try:
            previous_bulletin = json.loads(headlines_path.read_text())
        except Exception:
            previous_bulletin = None

    # 1. Live global 0.25 deg ECMWF forecast
    weather = fetch_live_weather()

    # 2. Precomputed ERA5 climatology (lapse-rate adjusted) and global hotspot scan
    climo_mean_c, climo_std_c, climo_source, era5_climo = load_era5_historical_context(
        weather.valid_time, ifs_elev_m=weather.ifs_elev_m
    )
    era5_elev = era5_climo.era5_elev_025_m if era5_climo is not None else None
    candidates, anom_025_c, _ = detect_global_candidates(
        weather, climo_mean_c, climo_std_c, max_candidates=HOTSPOTS_TO_SCAN, era5_elev_025_m=era5_elev
    )

    # 3. Globe layers, isobars and H/L centres
    manifest = package_globe_layers(weather, anom_025_c, climo_source, out_dir)
    print(
        f"[build] Wrote 5 global WebP weather layers and manifest ({len(manifest['isobars'])} isobars, "
        f"{len(manifest['pressure_centres'])} H/L centres).",
        file=sys.stderr,
    )

    # 4. Place names, free news leads, and the choice of stories (land, in the news, varied)
    candidates = attach_location_context(candidates, weather, pressure_centres=manifest.get("pressure_centres"))
    candidates = attach_news(candidates)
    stories = choose_stories(candidates, n=STORIES_PER_BULLETIN)
    print(
        "[build] Stories: " + "; ".join(f"{(c.get('location') or {}).get('headline_prefix')} ({c['kind']})" for c in stories),
        file=sys.stderr,
    )

    # 5. ERA5 1940-2025 history (ranks, normals, persistence) for the chosen stories only
    enriched = enrich_all_candidates(stories, weather, era5_climo)
    warnings = fetch_global_warnings(days_back=7)

    if dump_candidates_path is not None:
        dump_candidates_path.parent.mkdir(parents=True, exist_ok=True)
        dump_candidates_path.write_text(
            json.dumps(
                {
                    "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "warnings": warnings,
                    "candidates": enriched,
                },
                indent=2,
                default=str,
            )
        )

    # 6. The bulletin
    run_iso = weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    valid_iso = weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    need_audio_synth = True
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    if has_key:
        try:
            bulletin = generate_headlines_with_claude(weather, enriched, warnings, previous_bulletin)
            bulletin["stories_model_run_utc"] = run_iso  # lets the hourly check see what the stories were written from
        except Exception as exc:
            msg = _redact_secrets(str(exc)).replace("\n", " ")
            _annotate_error("Headline writing failed", msg)
            raise BulletinError(f"Claude headline generation failed: {msg}") from exc
    elif allow_template:
        print("[build] --template: writing template text (local testing only; never deployed).", file=sys.stderr)
        bulletin = build_deterministic_bulletin(weather, enriched, warnings, fetch_live_rss=False)
    elif skip_headlines_if_no_key and _is_recent_valid_bulletin(previous_bulletin, max_age_hours=18.0):
        # Code pushes deploy without calling Claude: keep the latest Claude bulletin, refresh only the maps
        print("[build] No ANTHROPIC_API_KEY: keeping the recent bulletin and refreshing the maps.", file=sys.stderr)
        bulletin = dict(previous_bulletin)
        bulletin["model_run_utc"] = run_iso
        bulletin["valid_time_utc"] = valid_iso
        need_audio_synth = not all(
            st.get("audio_url") and (out_dir.parent / str(st["audio_url"])).exists()
            for st in (bulletin.get("stories") or [])
        )
    else:
        msg = (
            "ANTHROPIC_API_KEY is not set and there is no recent Claude-written bulletin to keep. "
            "Set the key (or use --template for local testing)."
        )
        _annotate_error("No bulletin", msg)
        raise BulletinError(msg)

    # 7. Neural British newsreader MP3s for newly written stories
    if need_audio_synth:
        try:
            bulletin = build_bulletin_audio(bulletin, out_dir)
        except Exception as exc:
            print(f"[build] Warning: neural TTS audio synthesis failed ({exc}); keeping text bulletin.", file=sys.stderr)

    tmp = headlines_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(bulletin, indent=2, ensure_ascii=False))
    tmp.replace(headlines_path)
    asset_ver = stamp_frontend_assets()
    print(
        f"[build] Published {len(bulletin['stories'])} weather headlines to {headlines_path} (asset version ?v={asset_ver}).",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Weather Headlines global layers and 5-story bulletin.")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="Output directory (default: site/data)")
    parser.add_argument(
        "--keep-existing-headlines-if-no-key",
        action="store_true",
        help="Without ANTHROPIC_API_KEY, keep a recent Claude bulletin and refresh only the maps (used on code pushes).",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Without ANTHROPIC_API_KEY, write template text (local testing only; never use in production).",
    )
    parser.add_argument("--dump-candidates", type=Path, default=None, help="Optional path to save the chosen stories' data as JSON.")
    args = parser.parse_args()
    try:
        return run_pipeline(
            out_dir=args.out,
            skip_headlines_if_no_key=args.keep_existing_headlines_if_no_key,
            dump_candidates_path=args.dump_candidates,
            allow_template=args.template,
        )
    except BulletinError as exc:
        print(f"[build] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
