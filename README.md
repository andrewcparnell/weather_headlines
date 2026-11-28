# Weather Headlines (`headlines.aimsir.org`)

Live site: **<https://headlines.aimsir.org>**

An interactive 3D global weather explorer paired with a 6-hourly **World Weather Headlines** broadcast tour that presents the planet's top five weather stories like an international television news bulletin.

Inspired by Neil Gordon's [Weather Explorer](https://github.com/tehoro/weather-explorer-web), redesigned around a clean, single-screen 3D globe and statistically grounded ERA5 climate context.

## Features

- **Immediate Live 3D Globe:** Opens straight onto a smooth WebGL / Three.js Earth showing live global **ECMWF Open Data** fields at 0.5° resolution with crisp Natural Earth coastlines, country borders, and top-left report timestamps showing when the current bulletin was created and when the next 6-hourly report is due.
- **Five Instant Weather Layers:** Switch seamlessly down the right-hand side between:
  1. **Temperature** (2 m air temperature)
  2. **Temp Anomaly** (departure from the 1991-2020 Copernicus ERA5 seasonal mean for the exact date and UTC hour)
  3. **Winds** (10 m wind speed plus animated 3D flowing streamlines)
  4. **Rainfall** (24-hour accumulated precipitation with radar-wave animation)
  5. **Pressure** (mean sea-level pressure with smooth isobars and labelled `H` / `L` centres)
- **Country & City Search:** Top-left search bar covers every country on Earth plus major cities and regions (with live Open-Meteo geocoding fallback for smaller towns), flying smoothly across the globe and displaying a live multi-variable weather readout.
- **Broadcast World Tour (`Play Headlines`):** Clicking **Play Headlines** at the bottom centre launches a guided tour of **5 global weather highlights**:
  - **Sentence-Synchronised Globe Choreography:** Each story glides continuously along a smooth 3D Catmull-Rom spline across key locations while automatically switching weather variables (`Rainfall`, `Winds`, `Pressure`, `Temperature`, `Temp Anomaly`) and surface callouts in lockstep with each sentence.
  - **Independent `Text On/Off` & `Voice On/Off` Controls:** Defaults to **`📝 Text On`** (on-map presenter script cascading cleanly in the upper-left) and **`🔇 Voice Off`**, paired with pre-generated British female newsreader MP3 narration (`en-GB-SoniaNeural` via `edge-tts`) and a soft transition beat when **`🔊 Voice On`** is enabled.
  - **Dual-Speed Broadcast Banner:** Full-width frosted-glass lower-third banner featuring the main headline, a calm highlighted `STORY` ticker covering duration, **1940-present ERA5 historical rank**, and **climate change warming shift**, plus a slower CNN-style `IN THE NEWS` ticker linking directly to international news articles about that specific event.

## How the 6-Hourly Pipeline Works

Every 6 hours (`01:25`, `07:25`, `13:25`, `19:25` UTC via `.github/workflows/refresh.yml`), Python scripts in `pipeline/` run on GitHub Actions and redeploy `site/` to **<https://headlines.aimsir.org>**:

1. **`pipeline/opendata.py`** fetches only the required byte ranges of the latest 0.25° ECMWF IFS Open Data GRIB2 files (`00z`, `06z`, `12z`, `18z`).
2. **`pipeline/climatology.py`** uses the precomputed ERA5 climatology in `pipeline/data/` for the anomaly layer. For each hotspot it fetches daily ERA5 history from the Open-Meteo archive, but only the 14 days around today's date in each year 1940-2025. That costs one API call per location per year, about 86 instead of about 2,200 for the full record, so every hotspot can be ranked within the free limits; results are cached in `pipeline_cache/era5_points/`. From these it computes the seasonal rank, the previous high and its year, recent persistence, and the long-term warming shift. A forecast only counts as "the highest since 1940" when it beats the previous ERA5 high by a clear margin (0.5 °C, 10% of rainfall, or 7% of gust speed), because forecasts run more extreme than the reanalysis. Rain is ranked on an ~80 km area average to match ERA5's resolution.
3. **`pipeline/scanner.py` & `pipeline/pack_globe.py`** detect global hotspots, trace smooth global isobars and `H`/`L` centres, and pack all 5 global layers into ~200 KB total of lossless WebP textures (`site/data/layers/`).
4. **`pipeline/places.py`** names every hotspot offline from Natural Earth boundaries and towns (`pipeline/data/admin_lookup.npz`, `pipeline/data/places.json.gz`, rebuilt with `python -m pipeline.make_admin_lookup`). Large countries get their state or province, so headlines read "Oklahoma, USA: ..." or "Krasnoyarsk Krai, Russia: ...", while other countries just get "Vietnam: ...".
5. **`pipeline/news_agent.py`, `pipeline/validator.py` & `pipeline/tts.py`** pass Claude (with web search) the hotspots, place names and recent news leads, then validate every figure against the data and synthesise `en-GB-SoniaNeural` MP3 audio (`site/data/audio/`). Stories that fail validation are replaced one by one with template stories from **`pipeline/fallback_writer.py`**. The same writer produces the whole bulletin when Claude is unavailable, using varied, people-focused wording that only makes claims the data supports. The default writer model is `claude-opus-5-5`; set the `ANTHROPIC_MODEL` environment variable to change it.

## Deployment (`headlines.aimsir.org`)

1. **DNS (`aimsir.org` on GoDaddy):**
   Add a `CNAME` record with Host/Name `headlines` pointing to `andrewcparnell.github.io` (TTL: 1 Hour). `site/CNAME` is already included in the repository.
2. **GitHub Pages:**
   In **Settings -> Pages**, set **Build and deployment -> Source** to **GitHub Actions**, enter `headlines.aimsir.org` under **Custom domain**, and tick **Enforce HTTPS** once the DNS check turns green.
3. **Anthropic Secret for 6-Hourly Headlines:**
   In **Settings -> Secrets and variables -> Actions**, add `ANTHROPIC_API_KEY`.

## Running Locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync

# Refresh live ECMWF weather layers + ERA5 rankings + headlines + MP3 audio:
ANTHROPIC_API_KEY=sk-ant-... uv run python -m pipeline.build

# Serve the static 3D globe locally:
python3 -m http.server 8080 --directory site
```

Then open <http://localhost:8080> in your browser.

Run the test suite with:

```bash
uv run pytest -v
```

## Data Sources & Acknowledgements

- **Forecast Data:** © ECMWF, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) from [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data) (regridded and unit-converted).
- **Climatology & Historical Rankings:** Contains modified Copernicus Climate Change Service information (ERA5, 1940-present, Hersbach et al. 2020) via Google ARCO-ERA5 and [Open-Meteo](https://open-meteo.com/).
- **Official Alerts:** Global Disaster Alert and Coordination System ([GDACS](https://www.gdacs.org/)).
- **Map Outlines & Places:** [Natural Earth](https://www.naturalearthdata.com/) 1:50m (public domain) and [GeoNames](https://www.geonames.org/) (CC BY 4.0).
- **3D Rendering:** [Three.js](https://threejs.org/) (MIT Licence).

> **Disclaimer:** Weather Headlines presents automated model guidance and AI-curated summaries for educational and exploratory use. Always consult your National Meteorological Service for official forecasts and weather warnings.

## Licence

MIT Licence, © 2026 Andrew Parnell.
