"""Audit a generated headlines.json bulletin against candidate data, city distances, and rain normals."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import DATA_DIR
from .scanner import haversine_km
from .validator import (
    _cities_mentioned_in_text,
    extract_numbers_by_unit,
    validate_and_clean_bulletin,
)


def audit_bulletin(headlines_path: Path = DATA_DIR / "headlines.json", candidates_path: Path | None = None) -> int:
    hl = json.loads(headlines_path.read_text())
    cands = None
    if candidates_path and candidates_path.exists():
        raw_c = json.loads(candidates_path.read_text())
        cands = raw_c["candidates"] if isinstance(raw_c, dict) and "candidates" in raw_c else raw_c

    _, errs = validate_and_clean_bulletin(hl, candidates=cands, verify_urls=False)
    if errs:
        print("VALIDATION ERRORS:", errs, file=sys.stderr)
        return 1

    print(f"Edition: {hl.get('edition_title')} (generator: {hl.get('generator')})")
    print(f"Model run: {hl.get('model_run_utc')} | Valid time: {hl.get('valid_time_utc')}")

    for idx, st in enumerate(hl["stories"], start=1):
        c_lat = float(st["centre"]["lat"])
        c_lon = float(st["centre"]["lon"])
        matched = (
            min(cands, key=lambda c: haversine_km(c_lat, c_lon, float(c["lat"]), float(c["lon"])))
            if cands
            else None
        )
        print(f"\n--- Story {idx}: {st['headline']} ---")
        print(f"Region: {st.get('region')} ({st.get('country')}) | Centre: ({c_lat:.2f}, {c_lon:.2f})")
        if matched:
            gp = matched.get("grid_point") or {}
            era = matched.get("era5") or {}
            pers = era.get("persistence") or {}
            shift = era.get("climate_shift") or {}
            print(
                f"Matched candidate: {matched['id']} at ({matched['lat']}, {matched['lon']}) - "
                f"{gp.get('nearest_place_distance_km')} km from {gp.get('nearest_gazetteer_place')}, {gp.get('nearest_gazetteer_country')}"
            )
            print(
                f"Candidate fields: t2m_c={matched.get('t2m_c')} °C, t2m_max_24h_c={matched.get('t2m_max_24h_c')} °C, "
                f"t2m_min_24h_c={matched.get('t2m_min_24h_c')} °C, anom_c={matched.get('anom_c')} °C, "
                f"rain_24h_mm={matched.get('rain_24h_mm')} mm, gust_kmh={matched.get('gust_kmh')} km/h "
                f"({matched.get('gust_ms')} m/s), mslp_hpa={matched.get('mslp_hpa')} hPa"
            )
            print(
                f"ERA5 & persistence: rank={era.get('rank_since_1940')}, pct={era.get('percentile')}th, "
                f"normal={era.get('seasonal_normal_1991_2020')} {era.get('units')}, "
                f"rain_14d={pers.get('rain_14d_mm')} mm (exp {pers.get('expected_rain_14d_mm')} mm), "
                f"rain_28d={pers.get('rain_28d_mm')} mm (exp {pers.get('expected_rain_28d_mm')} mm, "
                f"pct_norm={pers.get('rain_28d_pct_of_normal')}%), "
                f"pers_summary={pers.get('summary')!r}, "
                f"warming={shift.get('local_seasonal_warming_c_since_1940_1979')} °C, "
                f"era_counts={shift.get('p90_years_count_1940_1979_out_of_40')}/40 vs {shift.get('p90_years_count_1986_2025_out_of_40')}/40"
            )

        if st.get("news_facts"):
            print("Attributed news_facts:")
            for nf in st["news_facts"]:
                print(f"  * {nf['claim']} [{nf['source_url']}]")

        stats = st.get("stats") or {}
        screen_segments = [
            f"headline: {st['headline']}",
            f"stats: {stats.get('headline_metric')} | {stats.get('metric_label')} | {stats.get('era5_rank')} | {stats.get('persistence')}",
            *(f"cue[{i}] ({c['lat']}, {c['lon']}, {c['layer']}, {c['label']}): {c.get('metric', '')} - {c['spoken']}" for i, c in enumerate(st["cues"])),
        ]
        print("On-screen text & cues:")
        for seg in screen_segments:
            print(" ", seg)

        extracted = extract_numbers_by_unit(" \n ".join(screen_segments))
        print("Extracted unit-bearing figures on screen:", extracted)

        max_centre_d = 0.0
        for ci, cue in enumerate(st["cues"]):
            dc = haversine_km(float(cue["lat"]), float(cue["lon"]), c_lat, c_lon)
            max_centre_d = max(max_centre_d, dc)
            cities = _cities_mentioned_in_text(cue.get("label", ""), c_lat, c_lon, True) + _cities_mentioned_in_text(
                cue.get("spoken", ""), c_lat, c_lon, False
            )
            seen = set()
            for city in cities:
                if city["name"] in seen:
                    continue
                seen.add(city["name"])
                dcity = haversine_km(float(cue["lat"]), float(cue["lon"]), city["lat"], city["lon"])
                print(f"  City check cue[{ci}] -> {city['name']}: {dcity:.1f} km (limit <= 150 km)")
        print(f"  Max cue distance from story centre: {max_centre_d:.1f} km (limit <= 500 km)")

    return 0


if __name__ == "__main__":
    cand_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(audit_bulletin(candidates_path=cand_arg))
