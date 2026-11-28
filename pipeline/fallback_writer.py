"""Template-based bulletin writer used when Claude is unavailable.

Goals, in order: never state anything the data does not support; say what the weather means for people
living there; and avoid the "same sentence five times" feel by drawing from several phrasings and never
reusing one within a bulletin.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from .opendata import LiveWeatherFields
from .places import describe_location
from .scanner import haversine_km
from .validator import (
    _BANNED_HYPE_PATTERNS,
    _cities_mentioned_in_text,
    ensure_dynamic_waypoints,
    is_specific_event_url,
    validate_and_clean_bulletin,
)

GENERATOR_TAG = "era5-deterministic"


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def fmt_2sf(val: float, signed: bool = False) -> str:
    """Format a measurement to 2 significant figures without a trailing '.0'."""
    import math

    if not math.isfinite(val) or abs(val) < 0.05:
        return "0"
    mag = math.floor(math.log10(abs(val)))
    step = 10.0 ** (mag - 1)
    rounded = round(val / step) * step
    if abs(rounded) >= 10.0:
        num_str = f"{int(round(rounded))}"
    else:
        one_dec = round(rounded, 1)
        num_str = f"{int(round(one_dec))}" if abs(one_dec - round(one_dec)) < 1e-6 else f"{one_dec:.1f}"
    if signed and rounded > 0 and not num_str.startswith("+"):
        return f"+{num_str}"
    return num_str


def _ordinal_word(n: int) -> str:
    words = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    return words.get(n, f"{n}th")


def _beaufort(kmh: float) -> str:
    for limit, name in (
        (39, "a fresh breeze"),
        (50, "a strong breeze"),
        (62, "a near gale"),
        (75, "a gale"),
        (89, "a severe gale"),
        (103, "storm force"),
        (118, "violent storm force"),
    ):
        if kmh < limit:
            return name
    return "hurricane force"


# ---------------------------------------------------------------------------
# Phrase variety
# ---------------------------------------------------------------------------


class PhrasePicker:
    """Pick one phrasing per slot, never repeating a phrasing within a bulletin."""

    def __init__(self, seed: str) -> None:
        self.rng = random.Random(seed)
        self.used: set[str] = set()

    def pick(self, slot: str, options: list[str]) -> str:
        order = list(range(len(options)))
        self.rng.shuffle(order)
        for i in order:
            key = f"{slot}:{i}"
            if key not in self.used:
                self.used.add(key)
                return options[i]
        return options[order[0]]


# ---------------------------------------------------------------------------
# Location wording
# ---------------------------------------------------------------------------


def _location_words(cand: dict) -> dict:
    loc = cand.get("location") or describe_location(float(cand["lat"]), float(cand["lon"]))
    country = loc.get("country") or cand.get("country") or "this region"
    prefix = loc.get("headline_prefix") or country
    town = loc.get("nearest_town") or ""
    town_km = loc.get("nearest_town_km")
    direction = loc.get("nearest_town_direction") or ""
    offshore = bool(loc.get("offshore"))

    if offshore:
        local = "just off the coast" if (loc.get("offshore_km") or 0) <= 80 else "out at sea"
        near = local
    elif town and town_km is not None and town_km <= 30:
        local = f"in and around {town}"
        near = local
    elif town and town_km is not None and town_km <= 80:
        local = f"near {town}"
        near = local
    elif town and town_km is not None and town_km <= 150:
        local = f"about {town_km} km {direction} of {town}"
        near = local
    else:
        local = "across remote parts of the country"
        near = local
    has_town = bool(town) and town_km is not None and town_km <= 150 and not offshore
    return {
        "country": country,
        "prefix": prefix,
        "local": local,
        "near": near,
        "town": town if has_town else "",
        "town_km": town_km,
        "offshore": offshore,
        "offshore_km": loc.get("offshore_km") or 0,
        "label": (town if has_town else (f"OFF {prefix}" if offshore else prefix)).upper()[:36],
    }


def _cue_points(lat: float, lon: float) -> list[tuple[float, float, float, float]]:
    """Five gentle camera positions within ~60 km of the hotspot."""
    import math

    cos_lat = max(0.35, math.cos(math.radians(lat)))
    dlon = min(0.95, 0.28 / cos_lat)
    steps = [
        (0.0, 0.0, 1.72, -8.0),
        (-0.18, -dlon * 0.7, 1.66, -4.0),
        (0.14, dlon * 0.65, 1.62, 2.0),
        (0.24, -dlon * 0.45, 1.68, 6.0),
        (-0.12, dlon * 0.8, 1.76, 10.0),
    ]
    return [
        (round(lat + dla, 2), round(((lon + dlo + 180.0) % 360.0) - 180.0, 2), dist, brg)
        for dla, dlo, dist, brg in steps
    ]


def _news_line(
    news: list[dict], picker: PhrasePicker, lat: float, lon: float, must_mention: str = ""
) -> tuple[str, str, dict] | None:
    """Turn the first suitable news headline into an attributed spoken line and a news_fact.

    Headlines that use hype words, or that name a city far from this story, are skipped rather than
    risk contradicting the calm, local framing of the bulletin.
    """
    for item in news:
        title = str(item.get("title") or "").strip().rstrip(".")
        source = str(item.get("source") or "").strip()
        if not title or not source or len(title) > 110 or source == "GDACS":
            continue
        if any(pat.search(title) for pat, _ in _BANNED_HYPE_PATTERNS):
            continue
        # For a state-level story, only read out news that is about that state
        if must_mention and must_mention.lower() not in title.lower():
            continue
        if any(haversine_km(lat, lon, c["lat"], c["lon"]) > 120 for c in _cities_mentioned_in_text(title, lat, lon)):
            continue
        spoken = picker.pick(
            "news",
            [
                f"On the ground, {source} reports: {title}.",
                f"The impacts are already in the news, with {source} reporting: {title}.",
                f"{source} has been following events there, reporting: {title}.",
            ],
        )
        ticker = f"**{source}**: {title}"
        return spoken, ticker, {"claim": f"{title} ({source})", "source_url": item["url"]}
    return None


# ---------------------------------------------------------------------------
# Story builders per hazard
# ---------------------------------------------------------------------------


def _story_heat(cand: dict, L: dict, month: str, p: PhrasePicker) -> dict:
    era = cand.get("era5") or {}
    pers = era.get("persistence") or {}
    tmax = float(cand.get("t2m_max_24h_c", cand.get("t2m_c", 0.0)))
    anom = float(cand.get("anom_c", 0.0))
    t_s, a_s = fmt_2sf(tmax), fmt_2sf(abs(anom))
    normal = era.get("wmo_tmax_normal_c")
    mild = tmax < 8.0
    P = L["prefix"]

    if mild:
        kicker = "EXCEPTIONALLY MILD"
        headline = p.pick(
            "heat_head_mild",
            [
                f"{P}: winter on hold, with temperatures {a_s} °C above normal",
                f"{P}: exceptionally mild for late {month}, around {a_s} °C above normal",
            ],
        )
        lead = p.pick(
            "heat_lead_mild",
            [
                f"{P}: it is exceptionally mild {L['local']} for late {month}, with temperatures forecast to reach {t_s} °C.",
                f"{P}: winter is being held at bay {L['local']}, where temperatures are forecast to climb to {t_s} °C.",
            ],
        )
    else:
        kicker = "UNSEASONABLE HEAT"
        headline = p.pick(
            "heat_head",
            [
                f"{P}: highs near {t_s} °C, about {a_s} °C above normal for late {month}",
                f"{P}: an out-of-season hot spell pushes temperatures towards {t_s} °C",
                f"{P}: unseasonable heat builds, with temperatures up to {t_s} °C",
            ],
        )
        lead = p.pick(
            "heat_lead",
            [
                f"{P}: temperatures {L['local']} are forecast to climb to around {t_s} °C today.",
                f"{P}: it is going to be a hot one {L['local']}, with highs of around {t_s} °C expected.",
                f"{P}: unseasonable heat is building {L['local']}, with temperatures set to reach {t_s} °C.",
            ],
        )

    if normal is not None and abs(tmax - float(normal)) >= 2.0:
        compare = p.pick(
            "heat_compare",
            [
                f"A typical day here at this time of year would peak at about {fmt_2sf(float(normal))} °C.",
                f"For comparison, the usual afternoon high for late {month} is nearer {fmt_2sf(float(normal))} °C.",
            ],
        )
    else:
        compare = f"That is roughly {a_s} °C warmer than is normal for the time of year."

    if mild:
        impact = p.pick(
            "heat_imp_mild",
            [
                "In a place where frost and snow would normally be settling in by now, that is a striking change.",
                "Lakes and rivers that would usually be starting to freeze may stay open for longer.",
            ],
        )
    elif tmax >= 40:
        impact = (
            "Heat like this is dangerous, particularly for older people and outdoor workers, so shade, "
            "water and rest in the hottest hours really matter."
        )
    elif tmax >= 34:
        impact = p.pick(
            "heat_imp_hot",
            [
                "That is hot enough to make outdoor work and travel exhausting through the afternoon.",
                "Farmers, labourers and anyone outdoors will feel the strain in the middle of the day.",
            ],
        )
    else:
        impact = p.pick(
            "heat_imp_warm",
            [
                "For people living here it will feel far warmer than they would expect at this point in the year.",
                "It is the kind of warmth that feels out of place on the calendar.",
            ],
        )

    hot14_raw = pers.get("warm_days_above_p90_tmax_last_14")
    hot14 = int(hot14_raw) if hot14_raw is not None else None
    if hot14 is not None and hot14 >= 4:
        background = p.pick(
            "heat_pers",
            [
                f"And this is no one-off: {hot14} of the past 14 days have been among the warmest 10% for the season.",
                f"The warmth has been building for a while, with {hot14} of the past 14 days in the warmest 10% for the time of year.",
            ],
        )
        bg_ticker = f"**{hot14} of the past 14 days** in the warmest 10% for the season"
    elif hot14 is not None:
        background = "It is a sharp change after a fairly ordinary couple of weeks."
        bg_ticker = "A **sharp change** after an ordinary fortnight"
    else:
        background = "How long the warmth lasts will depend on when cooler air can push back in."
        bg_ticker = "How long will the **warmth** last?"

    context, ctx_ticker = _context_line(cand, "warmest", "hottest", p)
    metric = f"{t_s} °C"
    return {
        "kicker": kicker,
        "headline": headline,
        "hazard": "heat",
        "primary_layer": "anomaly",
        "metric": metric,
        "metric_label": f"{fmt_2sf(anom, signed=True)} °C vs normal",
        "persistence": f"{hot14} of 14 days in warmest 10%" if hot14 is not None else "",
        "lines": [
            ("temperature", L["label"], f"Up to {t_s} °C", lead, f"**{P}**: highs near **{t_s} °C**"),
            ("anomaly", "COMPARED WITH NORMAL", f"{fmt_2sf(anom, signed=True)} °C", compare, f"About **{a_s} °C above** normal for late {month}"),
            ("temperature", "WHAT IT MEANS", metric, impact, impact.split(",")[0].rstrip(".")),
            ("anomaly", "RECENT WEEKS", f"{hot14} of 14 days" if hot14 is not None else "", background, bg_ticker),
            ("anomaly", "IN CONTEXT", "Long-term view", context, ctx_ticker),
        ],
    }


def _story_cold(cand: dict, L: dict, month: str, p: PhrasePicker) -> dict:
    era = cand.get("era5") or {}
    pers = era.get("persistence") or {}
    tmin = float(cand.get("t2m_min_24h_c", cand.get("t2m_c", 0.0)))
    anom = float(cand.get("anom_c", 0.0))
    t_s, a_s = fmt_2sf(tmin), fmt_2sf(abs(anom))
    normal = era.get("wmo_tmin_normal_c")
    P = L["prefix"]

    headline = p.pick(
        "cold_head",
        [
            f"{P}: a cold snap brings temperatures down to {t_s} °C",
            f"{P}: unseasonably cold, around {a_s} °C below normal for late {month}",
            f"{P}: a sharp chill sets in, with lows near {t_s} °C",
        ],
    )
    lead = p.pick(
        "cold_lead",
        [
            f"{P}: a blast of cold air is set to drive temperatures down to around {t_s} °C {L['local']}.",
            f"{P}: it is turning bitterly cold {L['local']}, with temperatures forecast to fall to {t_s} °C.",
            f"{P}: people {L['local']} are in for a cold spell, with lows near {t_s} °C on the way.",
        ],
    )
    if normal is not None and abs(tmin - float(normal)) >= 2.0:
        compare = f"Nights here would normally bottom out nearer {fmt_2sf(float(normal))} °C at this time of year."
    else:
        compare = f"That is roughly {a_s} °C colder than is normal for late {month}."

    if tmin <= -25:
        impact = "At temperatures like these, exposed skin can be frostbitten within minutes."
    elif tmin <= -10:
        impact = "That is cold enough to freeze pipes and turn roads and pavements treacherous."
    elif tmin <= 0:
        impact = p.pick(
            "cold_imp_frost",
            [
                "Expect frost, with icy patches on untreated roads and pavements first thing.",
                "Gardeners and growers will want to protect anything tender from the frost.",
            ],
        )
    else:
        impact = "It should stay above freezing, but it will feel raw, particularly in any wind."

    cold14_raw = pers.get("cold_days_below_p10_tmin_last_14")
    cold14 = int(cold14_raw) if cold14_raw is not None else None
    if cold14 is not None and cold14 >= 3:
        background = f"The cold has been setting in for a while, with {cold14} of the past 14 nights among the coldest 10% for the season."
        bg_ticker = f"**{cold14} of the past 14 nights** in the coldest 10%"
    elif cold14 is not None:
        background = "It is a sudden change after a milder spell."
        bg_ticker = "A **sudden change** after a milder spell"
    else:
        background = "Wrap up well, and keep an eye on anyone vulnerable to the cold."
        bg_ticker = "**Wrap up** and check on vulnerable neighbours"

    context, ctx_ticker = _context_line(cand, "coldest", "coldest", p)
    return {
        "kicker": "COLD SNAP",
        "headline": headline,
        "hazard": "cold",
        "primary_layer": "anomaly",
        "metric": f"{t_s} °C",
        "metric_label": f"{fmt_2sf(anom, signed=True)} °C vs normal",
        "persistence": f"{cold14} of 14 nights in coldest 10%" if cold14 is not None else "",
        "lines": [
            ("temperature", L["label"], f"Down to {t_s} °C", lead, f"**{P}**: lows near **{t_s} °C**"),
            ("anomaly", "COMPARED WITH NORMAL", f"{fmt_2sf(anom, signed=True)} °C", compare, f"About **{a_s} °C below** normal for late {month}"),
            ("temperature", "WHAT IT MEANS", f"{t_s} °C", impact, impact.split(",")[0].rstrip(".")),
            ("anomaly", "RECENT WEEKS", f"{cold14} of 14 nights" if cold14 is not None else "", background, bg_ticker),
            ("anomaly", "IN CONTEXT", "Long-term view", context, ctx_ticker),
        ],
    }


def _story_rain(cand: dict, L: dict, month: str, p: PhrasePicker) -> dict:
    era = cand.get("era5") or {}
    pers = era.get("persistence") or {}
    rain = float(cand.get("rain_24h_mm", 0.0))
    r_s = fmt_2sf(rain)
    P = L["prefix"]

    headline = p.pick(
        "rain_head",
        [
            f"{P}: up to {r_s} mm of rain forecast in the next 24 hours",
            f"{P}: flood risk rises with up to {r_s} mm of rain on the way",
            f"{P}: {'torrential' if rain >= 100 else 'heavy'} downpours could bring {r_s} mm in a day",
        ],
    )
    if L["offshore"]:
        lead = f"{P}: a band of heavy rain {L['near']} could bring up to {r_s} mm over the next day, and coastal communities are in its path."
    else:
        lead = p.pick(
            "rain_lead",
            [
                f"{P}: people {L['local']} are bracing for heavy rain, with up to {r_s} mm forecast over the next 24 hours.",
                f"{P}: up to {r_s} mm of rain is forecast {L['local']} by this time tomorrow.",
                f"{P}: a very wet day lies ahead {L['local']}, with as much as {r_s} mm of rain forecast.",
            ],
        )

    exp28 = float(pers.get("expected_rain_28d_mm") or 0.0)
    daily = float(pers.get("daily_rain_normal_mm") or 0.0)
    if exp28 >= 10 and rain >= exp28:
        compare = "That is more rain in a single day than would normally fall here in four weeks at this time of year."
        cmp_ticker = "**More than a month's rain** in a day"
    elif daily >= 0.5 and rain / daily >= 3:
        k = rain / daily
        compare = f"That is around {fmt_2sf(k)} times the average daily rainfall for the time of year."
        cmp_ticker = f"Around **{fmt_2sf(k)} times** the usual daily rainfall"
    else:
        compare = "That is a lot of water to fall in a short time, even by local standards."
        cmp_ticker = "A **lot of water** in a short time"

    if rain >= 150:
        impact = p.pick(
            "rain_imp_big",
            [
                "Rain on this scale can trigger flash floods and landslides on steep ground, and rivers may burst their banks.",
                "Totals like this can overwhelm drains, cut off roads and send hillsides sliding, so local warnings matter.",
            ],
        )
    elif rain >= 80:
        impact = p.pick(
            "rain_imp_mid",
            [
                "Low-lying roads could flood quickly, and smaller rivers may rise sharply.",
                "Expect standing water, difficult driving and the risk of flooding in low-lying neighbourhoods.",
            ],
        )
    else:
        impact = "Drains may struggle to cope during the heaviest bursts."

    pct = pers.get("rain_28d_pct_of_normal")
    wet28_raw = pers.get("wet_days_ge_10mm_last_28")
    wet28 = int(wet28_raw) if wet28_raw is not None else None
    if wet28 is None:
        background = "Where the heaviest bursts set up over the next few hours will decide who sees the worst of it."
        bg_ticker = "Watch for the **heaviest bursts**"
    elif pct is not None and float(pct) >= 150 and wet28 >= 4:
        background = f"And the ground is already saturated: the past four weeks have brought {int(round(float(pct)))}% of the normal rainfall."
        bg_ticker = f"Past four weeks: **{int(round(float(pct)))}% of normal** rainfall"
    elif wet28 <= 1:
        background = "It follows a largely dry few weeks, and hard, dry ground can shed heavy rain quickly."
        bg_ticker = "After a **largely dry** few weeks"
    else:
        background = f"It has been an unsettled spell, with {wet28} of the past 28 days bringing at least 10 mm of rain."
        bg_ticker = f"**{wet28} of the past 28 days** brought at least 10 mm"

    low = cand.get("low_centre") or {}
    if low.get("mslp_hpa") is not None and low.get("distance_km", 9999) <= 600:
        context = (
            f"The rain is being fed by an area of low pressure about {low['distance_km']} km to the "
            f"{low['direction_from_hotspot']}, with a central pressure near {int(round(low['mslp_hpa']))} hPa."
        )
        ctx_ticker = f"Fed by a **{int(round(low['mslp_hpa']))} hPa** low nearby"
    else:
        context, ctx_ticker = _context_line(cand, "wettest", "wettest", p)

    return {
        "kicker": "HEAVY RAIN",
        "headline": headline,
        "hazard": "rain",
        "primary_layer": "rain",
        "metric": f"{r_s} mm / 24 h",
        "metric_label": "Forecast rainfall, next 24 hours",
        "persistence": f"{wet28} of 28 days wet (at least 10 mm)" if wet28 is not None else "",
        "lines": [
            ("rain", L["label"], f"Up to {r_s} mm", lead, f"**{P}**: up to **{r_s} mm** forecast"),
            ("rain", "COMPARED WITH NORMAL", f"{r_s} mm / 24 h", compare, cmp_ticker),
            ("rain", "WHAT IT MEANS", "Flood risk", impact, impact.split(",")[0].rstrip(".")),
            ("rain", "RECENT WEEKS", f"{wet28} wet days" if wet28 is not None else "", background, bg_ticker),
            ("pressure", "IN CONTEXT", "Weather pattern", context, ctx_ticker),
        ],
    }


def _story_wind(cand: dict, L: dict, month: str, p: PhrasePicker) -> dict:
    era = cand.get("era5") or {}
    pers = era.get("persistence") or {}
    gust = float(cand.get("gust_ms", 0.0)) * 3.6
    wind = float(cand.get("wind_ms", 0.0)) * 3.6
    rain = float(cand.get("rain_24h_mm", 0.0))
    g_s, w_s = fmt_2sf(gust), fmt_2sf(wind)
    P = L["prefix"]
    low = cand.get("low_centre") or {}

    headline = p.pick(
        "wind_head",
        [
            f"{P}: gusts of up to {g_s} km/h forecast as a storm passes",
            f"{P}: a stormy day ahead, with gusts near {g_s} km/h",
            f"{P}: wild weather on the way, with gusts up to {g_s} km/h",
        ],
    )
    if low.get("mslp_hpa") is not None:
        article = "a deep" if low["mslp_hpa"] < 980 else "an"
        lead = (
            f"{P}: {article} area of low pressure, with a central pressure near {int(round(low['mslp_hpa']))} hPa, "
            f"is set to bring gusts of up to {g_s} km/h {L['near']}."
        )
    else:
        lead = f"{P}: gusts of up to {g_s} km/h are forecast {L['near']} as a storm moves through."

    p90 = pers.get("local_gust_p90_kmh")
    if wind >= 62:
        compare = (
            f"Sustained winds of around {w_s} km/h would be {_beaufort(wind)} on the Beaufort scale, "
            "with stronger bursts in the gusts."
        )
        cmp_ticker = f"Sustained winds near **{w_s} km/h**"
    elif p90 is not None and gust > float(p90) * 1.2:
        compare = (
            f"For comparison, only one day in ten at this time of year usually brings gusts above "
            f"{fmt_2sf(float(p90))} km/h here."
        )
        cmp_ticker = f"Usual top-10% gust here: **{fmt_2sf(float(p90))} km/h**"
    else:
        compare = "The strongest gusts are likely in short, sharp bursts, especially on exposed coasts and high ground."
        cmp_ticker = "Strongest in **short, sharp bursts**"
    if L["offshore"] and L["offshore_km"] > 60:
        impact = (
            "The strongest winds are over open water, where shipping and fishing boats face very rough seas, "
            "and large waves are likely to reach exposed coasts."
        )
    elif gust >= 120:
        impact = "Gusts like that can bring down trees and power lines and make driving dangerous, especially for high-sided vehicles."
    elif gust >= 90:
        impact = p.pick(
            "wind_imp_mid",
            [
                "Expect disruption to ferries and flights, and difficult driving on exposed roads and bridges.",
                "Loose objects can be blown around and power cuts are possible where trees come down.",
            ],
        )
    else:
        impact = "It will be a blustery, unpleasant day, with rough seas along exposed coasts."

    g28_raw = pers.get("days_gust_over_65kmh_last_28")
    g28 = int(g28_raw) if g28_raw is not None else None
    if rain >= 20:
        background = f"The same system is also forecast to bring up to {fmt_2sf(rain)} mm of rain."
        bg_ticker = f"Plus up to **{fmt_2sf(rain)} mm** of rain"
        bg_layer = "rain"
    elif g28 is not None and g28 >= 5:
        background = f"It has been a stormy few weeks, with gusts topping 65 km/h on {g28} of the past 28 days."
        bg_ticker = f"Gusts over 65 km/h on **{g28} of the past 28 days**"
        bg_layer = "wind"
    else:
        background = "It is a marked change after a relatively calm few weeks."
        bg_ticker = "After a **relatively calm** few weeks"
        bg_layer = "wind"

    context, ctx_ticker = _context_line(cand, "windiest", "strongest", p)
    return {
        "kicker": "STORM",
        "headline": headline,
        "hazard": "cyclone",
        "primary_layer": "wind",
        "metric": f"{g_s} km/h gusts",
        "metric_label": (
            f"Low pressure near {int(round(low['mslp_hpa']))} hPa" if low.get("mslp_hpa") is not None else "Peak gusts, next 24 hours"
        ),
        "persistence": f"{g28} of 28 days with gusts over 65 km/h" if g28 is not None else "",
        "lines": [
            ("pressure", L["label"], f"Gusts {g_s} km/h", lead, f"**{P}**: gusts up to **{g_s} km/h**"),
            ("wind", "HOW STRONG", f"Gusts {g_s} km/h", compare, cmp_ticker),
            ("wind", "WHAT IT MEANS", "Disruption likely", impact, impact.split(",")[0].rstrip(".")),
            (bg_layer, "RECENT WEEKS", f"{g28} windy days" if g28 is not None else "", background, bg_ticker),
            ("wind", "IN CONTEXT", "Long-term view", context, ctx_ticker),
        ],
    }


def _context_line(cand: dict, adj_window: str, adj_rank: str, p: PhrasePicker) -> tuple[str, str]:
    """Historical rank if we genuinely have one, else long-term warming, else a calm outlook line."""
    era = cand.get("era5") or {}
    shift = era.get("climate_shift") or {}
    rank = era.get("rank_since_1940") if era.get("rank_available", era.get("rank_since_1940") is not None) else None
    prev_val, prev_year = era.get("previous_record_value"), era.get("previous_record_year")
    units = era.get("units") or ""
    if prev_val is not None and units == "m/s":
        prev_txt = f"{fmt_2sf(float(prev_val) * 3.6)} km/h"
    elif prev_val is not None:
        prev_txt = f"{fmt_2sf(float(prev_val))} {units}".strip()
    else:
        prev_txt = ""
    if rank == 1 and units == "mm":
        # Rain is ranked on an area average (see climatology), so compare years, not the headline number
        if era.get("clear_record", True):
            when = f", beating a very wet day in {prev_year}" if prev_year else ""
            return (
                f"Averaged over the surrounding area, it would be the wettest day for this time of year here since at least 1940{when}.",
                "Would be the **wettest for the time of year** since 1940",
            )
        when = f", alongside a deluge in {prev_year}" if prev_year else ""
        return (
            f"Averaged over the surrounding area, it would rank among the wettest days for this time of year here since 1940{when}.",
            "Among the **wettest days** for the time of year since 1940",
        )
    if rank == 1 and era.get("clear_record", True):
        if prev_txt and prev_year:
            return (
                p.pick(
                    "record",
                    [
                        f"If the forecast is right, it would be the {adj_rank} day for this time of year here since at least 1940, beating {prev_txt} in {prev_year}.",
                        f"That would set a new high for the time of year here: the previous {adj_rank} day since 1940 brought {prev_txt}, in {prev_year}.",
                    ],
                ),
                f"Would beat the **{prev_year}** high for the time of year",
            )
        return (
            f"If the forecast is right, this would be the {adj_rank} day for this time of year here in records going back to 1940.",
            f"Would be the **{adj_rank} for the time of year** since 1940",
        )
    if rank == 1:
        # Ahead of the old high, but not by enough to be sure given the difference between forecast and history
        tail = f", close to the {prev_txt} of {prev_year}" if prev_txt and prev_year else ""
        return (
            f"That would put it among the {adj_window} days for this time of year here since 1940{tail}.",
            f"Among the **{adj_window} days** for the time of year since 1940",
        )
    if rank is not None and 2 <= int(rank) <= 5:
        return (
            f"That would make it the {_ordinal_word(int(rank))} {adj_rank} day for this time of year here since 1940.",
            f"Would rank **#{int(rank)}** for the time of year since 1940",
        )
    warming = shift.get("local_seasonal_warming_c_since_1940_1979")
    if warming is not None and 0.4 <= float(warming) <= 2.5:  # larger values are more likely early-record artefacts
        w = fmt_2sf(float(warming))
        return (
            p.pick(
                "warming",
                [
                    f"Over the long run, temperatures here at this time of year have risen by about {w} °C since the mid-twentieth century.",
                    f"The backdrop is a warming climate: this time of year is now about {w} °C warmer here than it was half a century ago.",
                ],
            ),
            f"This time of year is now about **{w} °C** warmer here",
        )
    kind = cand.get("kind")
    specific = {
        "rain": [
            "Rivers can keep rising for a day or more after the rain eases, so the flood risk may linger.",
            "Exactly where the heaviest bursts set up will decide who sees the worst of it.",
        ],
        "heat": [
            "With warmth like this, the evenings are often the most pleasant part of the day.",
            "The key question now is how long it lasts before more seasonal air returns.",
        ],
        "cold": [
            "Clear skies overnight would let temperatures drop furthest, especially in sheltered valleys.",
            "The key question now is how quickly milder air can return.",
        ],
    }.get(kind, [
        "Winds are usually strongest around headlands, on bridges and over high ground.",
        "Conditions should ease as the system moves away, but not before a rough spell.",
    ])
    general = [
        "Forecasters will be keeping a close eye on how this develops over the next day or so.",
        "It is worth checking local forecasts and warnings, as details can change at short notice.",
    ]
    line = p.pick(f"outlook_{kind}", specific) if any(f"outlook_{kind}:{i}" not in p.used for i in range(len(specific))) else p.pick("outlook", general)
    return line, "Check **local forecasts** for the latest"


_BUILDERS = {"heat": _story_heat, "cold": _story_cold, "rain": _story_rain}


# ---------------------------------------------------------------------------
# Bulletin assembly
# ---------------------------------------------------------------------------


def _pick_candidates(candidates: list[dict], news_by_id: dict[str, list[dict]]) -> list[dict]:
    scored = []
    for idx, c in enumerate(candidates):
        loc = c.get("location") or {}
        offshore_km = float(loc.get("offshore_km") or 0)
        if c["kind"] in ("heat", "cold") and offshore_km > 25:
            penalty = 6.0  # temperature anomalies over open sea rarely affect anyone
        elif offshore_km > 250:
            penalty = 4.0
        else:
            penalty = 0.0
        news_bonus = 1.5 * min(2, len(news_by_id.get(c["id"], [])))
        scored.append((news_bonus - penalty - 0.4 * idx, idx, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    # First pass: at most two stories of any one hazard type, for variety; then top up in score order
    picked: list[dict] = []
    kinds: dict[str, int] = {}
    for _, _, c in scored:
        if len(picked) < 5 and kinds.get(c["kind"], 0) < 2:
            picked.append(c)
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    for _, _, c in scored:
        if len(picked) >= 5:
            break
        if c not in picked:
            picked.append(c)
    return picked


def build_fallback_story(cand: dict, weather: LiveWeatherFields, picker: PhrasePicker, news: list[dict], rank: int) -> dict:
    month = weather.valid_time.strftime("%B")
    L = _location_words(cand)
    builder = _BUILDERS.get(cand["kind"], _story_wind)
    spec = builder(cand, L, month, picker)

    lines = list(spec["lines"])
    news_facts: list[dict] = []
    state = (cand.get("location") or {}).get("state") or ""
    nl = _news_line(news, picker, float(cand["lat"]), float(cand["lon"]), must_mention=state)
    if nl is not None:
        spoken, ticker, fact = nl
        # Replace the weakest line (recent weeks) with real, attributed news from the ground
        lines[3] = (lines[3][0], "IN THE NEWS", "Local reports", spoken, ticker)
        news_facts.append(fact)

    lat, lon = float(cand["lat"]), float(cand["lon"])
    pts = _cue_points(lat, lon)
    cues = []
    for i, (layer, label, metric, spoken, ticker) in enumerate(lines):
        cues.append(
            {
                "layer": layer,
                "lat": pts[i][0],
                "lon": pts[i][1],
                "distance": pts[i][2],
                "bearing": pts[i][3],
                "label": label[:36],
                "metric": metric,
                "spoken": spoken,
                "ticker": ticker,
            }
        )

    beats = [
        {"tag": "THE FORECAST", "title": "The forecast", "layer": cues[0]["layer"], "text": f"{cues[0]['spoken']} {cues[1]['spoken']}"},
        {"tag": "WHAT IT MEANS", "title": "What it means", "layer": cues[2]["layer"], "text": cues[2]["spoken"]},
        {"tag": "THE BACKGROUND", "title": "The background", "layer": cues[3]["layer"], "text": cues[3]["spoken"]},
        {"tag": "IN CONTEXT", "title": "In context", "layer": cues[4]["layer"], "text": cues[4]["spoken"]},
    ]

    links = [{"source": n["source"], "title": n["title"], "url": n["url"]} for n in news[:3]]
    return {
        "id": f"auto-{weather.valid_time:%Y%m%d}-{rank}-{cand['kind']}",
        "rank": rank,
        "kicker": f"{spec['kicker']} · {L['prefix'].upper()}",
        "headline": spec["headline"],
        "ticker": "  ///  ".join(c["ticker"] for c in cues),
        "region": L["prefix"],
        "country": L["country"],
        "hazard": spec["hazard"],
        "primary_layer": spec["primary_layer"],
        "centre": {"lat": lat, "lon": lon},
        "stats": {
            "headline_metric": spec["metric"],
            "metric_label": spec["metric_label"],
            "era5_rank": _rank_stat(cand),
            "persistence": spec["persistence"],
        },
        "camera_waypoints": ensure_dynamic_waypoints(lat, lon, None),
        "cues": cues,
        "beats": beats,
        "news_links": links,
        "news_facts": news_facts,
    }


def _rank_stat(cand: dict) -> str:
    era = cand.get("era5") or {}
    rank = era.get("rank_since_1940")
    if rank is not None and era.get("rank_available", True):
        if int(rank) == 1 and not era.get("clear_record", True):
            return "Near the top for time of year since 1940"
        return f"#{int(rank)} for time of year since 1940"
    return "No historical rank this run"


def build_fallback_bulletin(
    weather: LiveWeatherFields,
    candidates: list[dict],
    warnings: list[dict],
    news_lookup=None,
) -> dict:
    """Build a validated 5-story bulletin from candidates without an LLM.

    `news_lookup(cand) -> list[{source,title,url}]` supplies reputable, event-relevant headlines (optional).
    """
    news_by_id: dict[str, list[dict]] = {}
    for cand in candidates[:9]:
        items: list[dict] = []
        for w in warnings:
            if (
                w.get("url")
                and is_specific_event_url(str(w["url"]))
                and w.get("lat") is not None
                and w.get("is_current", True)
                and haversine_km(float(w["lat"]), float(w["lon"]), float(cand["lat"]), float(cand["lon"])) <= 800
            ):
                items.append(
                    {
                        "source": "GDACS",
                        "title": f"{w['alert_level']} {w['type']} alert: {str(w['name'])[:70]}",
                        "url": w["url"],
                    }
                )
                break
        if news_lookup is not None:
            try:
                items.extend(news_lookup(cand))
            except Exception:
                pass
        news_by_id[cand["id"]] = items[:3]

    picked = _pick_candidates(candidates, news_by_id)
    now_utc = datetime.now(timezone.utc)

    errs: list[str] = []
    for use_news in (True, False):
        picker = PhrasePicker(seed=f"{weather.valid_time:%Y%m%d%H}")
        stories = [
            build_fallback_story(c, weather, picker, news_by_id.get(c["id"], []) if use_news else [], rank=i)
            for i, c in enumerate(picked, start=1)
        ]
        draft = {
            "generated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_run_utc": weather.run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_time_utc": weather.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "edition_title": f"World Weather Headlines - {now_utc:%H:%M} UTC, {now_utc.day} {now_utc:%B %Y}",
            "generator": GENERATOR_TAG,
            "stories": stories,
        }
        cleaned, errs = validate_and_clean_bulletin(draft, candidates=picked, verify_urls=False)
        if not errs:
            return cleaned
    raise RuntimeError(f"Fallback bulletin failed validation: {errs}")
