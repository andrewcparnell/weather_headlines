"""Central configuration, layer definitions, and colour ramps for Weather Headlines."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT_DIR / "site"
DATA_DIR = SITE_DIR / "data"
CACHE_DIR = Path(os.environ.get("WEATHER_HEADLINES_CACHE") or (ROOT_DIR / "pipeline_cache"))

USER_AGENT = "weather-headlines/0.1 (+https://github.com/andrewcparnell/weather_headlines)"


@dataclass(frozen=True)
class LayerSpec:
    id: str
    label: str
    short_label: str
    units: str
    lo: float
    hi: float
    scale: str  # "linear" or "sqrt"
    description: str
    stops: list[tuple[float, str]]  # (value in units, #RRGGBBAA)


LAYERS: dict[str, LayerSpec] = {
    spec.id: spec
    for spec in [
        LayerSpec(
            id="temperature",
            label="2 m Temperature",
            short_label="Temperature",
            units="°C",
            lo=-45.0,
            hi=50.0,
            scale="linear",
            description="Near-surface air temperature 2 metres above ground or sea.",
            stops=[
                (-45.0, "#1e0836ee"),
                (-30.0, "#3b1c78ee"),
                (-15.0, "#1d65b8ee"),
                (0.0, "#43b8d9dd"),
                (10.0, "#6fd18cdd"),
                (20.0, "#f5d762dd"),
                (30.0, "#f47c3cee"),
                (40.0, "#c81d25f5"),
                (50.0, "#590016ff"),
            ],
        ),
        LayerSpec(
            id="anomaly",
            label="Temperature Anomaly",
            short_label="Temp Anomaly",
            units="°C",
            lo=-28.0,
            hi=28.0,
            scale="linear",
            description="Departure of 2 m temperature from the 1990-2019 ERA5 ±7-day seasonal average.",
            stops=[
                (-28.0, "#05143ef7"),
                (-18.0, "#0c2c84f2"),
                (-10.0, "#1d5aa8ee"),
                (-4.0, "#4da3d9cc"),
                (0.0, "#ffffff08"),
                (4.0, "#f78844cc"),
                (10.0, "#e34a33ee"),
                (18.0, "#b30000f4"),
                (28.0, "#4d000bf8"),
            ],
        ),
        LayerSpec(
            id="wind",
            label="10 m Wind Speed & Flow",
            short_label="Winds",
            units="m/s",
            lo=0.0,
            hi=45.0,
            scale="linear",
            description="Near-surface wind speed at 10 metres with animated flowing wind streamlines.",
            stops=[
                (0.0, "#00000000"),
                (5.0, "#386cb044"),
                (10.0, "#2ca25f99"),
                (16.0, "#7fcdbbcc"),
                (22.0, "#fed976dd"),
                (28.0, "#fd8d3cee"),
                (35.0, "#e31a1cf5"),
                (45.0, "#980043ff"),
            ],
        ),
        LayerSpec(
            id="rain",
            label="24 h Rainfall",
            short_label="Rainfall",
            units="mm",
            lo=0.0,
            hi=200.0,
            scale="sqrt",
            description="24-hour accumulated precipitation from ECMWF high-resolution guidance.",
            stops=[
                (0.0, "#00000000"),
                (0.8, "#00000000"),
                (2.5, "#7fcdbb88"),
                (8.0, "#41b6c4cc"),
                (20.0, "#1d91c0dd"),
                (45.0, "#225ea8ee"),
                (80.0, "#fed976f2"),
                (130.0, "#f03b20f8"),
                (200.0, "#bd0026ff"),
            ],
        ),
        LayerSpec(
            id="pressure",
            label="Mean Sea Level Pressure",
            short_label="Pressure",
            units="hPa",
            lo=955.0,
            hi=1048.0,
            scale="linear",
            description="Mean sea level pressure with 4 hPa isobars and high/low pressure centres.",
            stops=[
                (955.0, "#542788dd"),
                (978.0, "#2c7fb8bb"),
                (998.0, "#7fcdbb66"),
                (1013.0, "#ffffff05"),
                (1026.0, "#fdae6b66"),
                (1038.0, "#e6550dbb"),
                (1048.0, "#99000ddd"),
            ],
        ),
    ]
}

LAYER_ORDER = ["temperature", "anomaly", "wind", "rain", "pressure"]


def layer_metadata() -> list[dict]:
    """Serialisable layer catalogue for site/data/manifest.json."""
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "short_label": spec.short_label,
            "units": spec.units,
            "lo": spec.lo,
            "hi": spec.hi,
            "scale": spec.scale,
            "description": spec.description,
            "stops": [[val, col] for val, col in spec.stops],
        }
        for spec in (LAYERS[k] for k in LAYER_ORDER)
    ]
