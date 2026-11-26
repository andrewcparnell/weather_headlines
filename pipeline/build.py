"""End-to-end 6-hourly builder: ECMWF Open Data -> ERA5 Climatology -> WebP Globe + Claude Bulletin."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .climatology import enrich_all_candidates, load_era5_historical_context
from .config import DATA_DIR
from .news_agent import build_deterministic_bulletin, generate_headlines_with_claude
from .opendata import fetch_live_weather
from .pack_globe import package_globe_layers
from .scanner import detect_global_candidates
from .tts import build_bulletin_audio
from .warnings_feed import fetch_global_warnings


def run_pipeline(
    out_dir: Path = DATA_DIR,
    skip_headlines_if_no_key: bool = False,
    dump_candidates_path: Path | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    headlines_path = out_dir / "headlines.json"
    previous_bulletin = None
    if headlines_path.exists():
        try:
            previous_bulletin = json.loads(headlines_path.read_text())
        except Exception:
            previous_bulletin = None

    # 1. Fetch live global 0.25 deg ECMWF Open Data
    weather = fetch_live_weather()

    # 2. Load 1940-2025 ARCO-ERA5 archive + WMO 1991-2020 global 2t climatology & scan global hotspots
    climo_mean_c, climo_std_c, climo_source, era5_cube = load_era5_historical_context(weather.valid_time)
    candidates, anom_025_c, _ = detect_global_candidates(weather, climo_mean_c, climo_std_c, max_candidates=12)

    # 3. Pack all 5 global 0.5 deg WebP layers + isobars + H/L centres
    manifest = package_globe_layers(weather, anom_025_c, climo_source, out_dir)
    print(
        f"[build] Wrote 5 global WebP weather layers and manifest ({len(manifest['isobars'])} isobars, "
        f"{len(manifest['pressure_centres'])} H/L centres).",
        file=sys.stderr,
    )

    # 4. Enrich candidate hotspots with 1940-present ERA5 rankings, 28-day persistence & fetch GDACS alerts
    print(f"[build] Computing 1940-present ERA5 rankings & 28-day persistence for {len(candidates)} hotspots...", file=sys.stderr)
    enriched = enrich_all_candidates(candidates, weather, era5_cube)
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
            )
        )

    # 5. Generate 5-story broadcast tour with Claude + Web Search (or fallback cleanly if no key)
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if has_key:
        bulletin = generate_headlines_with_claude(weather, enriched, warnings, previous_bulletin)
    elif skip_headlines_if_no_key and previous_bulletin:
        print(
            "[build] ANTHROPIC_API_KEY not set; keeping existing headlines.json while updating live globe layers.",
            file=sys.stderr,
        )
        bulletin = previous_bulletin
    else:
        print(
            "[build] ANTHROPIC_API_KEY not set; building ERA5-grounded fallback bulletin.",
            file=sys.stderr,
        )
        bulletin = build_deterministic_bulletin(weather, enriched, warnings)

    # 6. Synthesise neural British female newsreader MP3s for all stories
    try:
        bulletin = build_bulletin_audio(bulletin, out_dir)
    except Exception as exc:
        print(f"[build] Warning: neural TTS audio synthesis failed ({exc}); keeping text bulletin.", file=sys.stderr)

    tmp = headlines_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(bulletin, indent=2, ensure_ascii=False))
    tmp.replace(headlines_path)
    print(f"[build] Published {len(bulletin['stories'])} weather headlines to {headlines_path}.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Weather Headlines global layers and 5-story bulletin.")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="Output directory (default: site/data)")
    parser.add_argument(
        "--keep-existing-headlines-if-no-key",
        action="store_true",
        help="Keep existing headlines.json when ANTHROPIC_API_KEY is unset (useful before GitHub Secret is added).",
    )
    parser.add_argument("--dump-candidates", type=Path, default=None, help="Optional path to save scanned ERA5 candidates JSON.")
    args = parser.parse_args()
    return run_pipeline(
        out_dir=args.out,
        skip_headlines_if_no_key=args.keep_existing_headlines_if_no_key,
        dump_candidates_path=args.dump_candidates,
    )


if __name__ == "__main__":
    raise SystemExit(main())
