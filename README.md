# Weather Headlines

An interactive 3D global weather explorer paired with a 6-hourly **World Weather Tour** that presents the planet's top five weather stories like an international television weather bulletin.

Inspired by Neil Gordon's [Weather Explorer](https://github.com/tehoro/weather-explorer-web), redesigned around a clean, single-screen globe and statistically grounded climate context.

## Features

- **Immediate Live 3D Globe:** Opens straight onto a smooth WebGL / Three.js Earth showing live global **ECMWF Open Data** fields at 0.5° resolution with crisp Natural Earth coastlines and country borders.
- **Five Instant Weather Layers:** Switch seamlessly down the right-hand side between:
  1. **Temperature** (2 m air temperature)
  2. **Temp Anomaly** (departure from the 1991-2020 Copernicus ERA5 seasonal mean for the exact date and UTC hour)
  3. **Winds** (10 m wind speed plus 4,000 animated 3D flowing wind streamlines)
  4. **Rainfall** (24-hour accumulated precipitation)
  5. **Pressure** (mean sea level pressure with smooth isobars and labelled `H` / `L` centres)
- **Country & City Search:** Top-left search bar covers every country on Earth plus major cities and regions (with live Open-Meteo geocoding fallback for smaller towns), flying smoothly across the globe and displaying a live multi-variable weather readout.
- **Broadcast World Tour (Play Button):** Clicking the Play button at the bottom centre launches a guided tour of **5 global weather highlights**:
  - **Continuous Camera Choreography:** Sweeps between continents along great-circle arcs and glides continuously across each weather system through 4 surface waypoints while automatically switching weather layers.
  - **Four Broadcast Chapters per Event:**
    1. *Live Event:* What is happening right now and why it matters.
    2. *Persistence:* Whether the pattern has been building or stalling for days or weeks (14-day and 28-day ERA5 tracking).
    3. *Historical Rank:* Exact rank and percentile among annual seasonal windows back to **1940** in ERA5 (~87 years).
    4. *Climate Link:* How local background warming (comparing 1940-1979 with 1986-2025 in ERA5, alongside attribution science) has shifted the odds or intensity of the event.
  - **International News Links & Optional Voiceover:** Direct links to international news coverage (BBC, The Guardian, Reuters, AP, Al Jazeera, and official disaster reports) plus an optional Web Speech presenter voice toggle.

## How the 6-Hourly Pipeline Works

Every 6 hours (via `.github/workflows/refresh.yml`), Python scripts in `pipeline/` run on GitHub Actions and redeploy `site/` to GitHub Pages:

1. **`pipeline/opendata.py`** fetches only the required byte ranges of the latest 0.25° ECMWF IFS Open Data GRIB2 files (`00z`, `06z`, `12z`, `18z`) from AWS S3 / `data.ecmwf.int`.
2. **`pipeline/climatology.py`** computes the WMO 1991-2020 seasonal 2 m temperature normal from Google's public ARCO-ERA5 Zarr store, and queries 1940-present daily ERA5 series (via Open-Meteo Historical Archive) at candidate hotspots to calculate exact seasonal ranks, multi-week persistence, and 85-year warming shifts.
3. **`pipeline/scanner.py` & `pipeline/pack_globe.py`** detect global hotspots, trace smooth global isobars and `H`/`L` centres, and pack all 5 global layers into ~200 KB total of lossless WebP textures (`site/data/layers/`).
4. **`pipeline/news_agent.py` & `pipeline/validator.py`** call **Anthropic Claude** with built-in web search to verify active international news coverage, write the 5 four-beat broadcast scripts and moving camera paths, validate every field, and publish `site/data/headlines.json`.

## Setup for Automated 6-Hourly Updates

1. **Enable GitHub Pages:**
   In your GitHub repository, go to **Settings -> Pages** and set **Build and deployment -> Source** to **GitHub Actions**.
2. **Add Your Anthropic API Key:**
   Go to **Settings -> Secrets and variables -> Actions -> New repository secret**, create a secret named `ANTHROPIC_API_KEY`, and paste your Anthropic key.
   *(Optional: set `ANTHROPIC_MODEL` if you wish to override the default Claude Sonnet 4.5 model.)*
3. **Trigger a Run Anytime:**
   Open the **Actions** tab, select **Refresh Weather & Deploy Pages**, and click **Run workflow** (it also runs automatically every 6 hours).

## Running Locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync

# Refresh live ECMWF weather layers + ERA5 rankings + headlines:
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
- **Climatology & Historical Rankings:** Contains modified Copernicus Climate Change Service information (ERA5, 1940-present, Hersbach et al. 2020) via Google ARCO-ERA5 and [Open-Meteo Historical Weather API](https://open-meteo.com/).
- **Official Alerts:** Global Disaster Alert and Coordination System ([GDACS](https://www.gdacs.org/)).
- **Map Outlines & Places:** [Natural Earth](https://www.naturalearthdata.com/) 1:50m (public domain) and [GeoNames](https://www.geonames.org/) (CC BY 4.0).
- **3D Rendering:** [Three.js](https://threejs.org/) (MIT Licence).

> **Disclaimer:** Weather Headlines presents automated model guidance and AI-curated summaries for educational and exploratory use. Always consult your National Meteorological Service for official forecasts and weather warnings.

## Licence

MIT Licence, © 2026 Andrew Parnell.
