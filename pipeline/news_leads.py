"""Free news leads for each weather hotspot, and the choice of the five stories to write.

Sources (no API keys, no cost):
- Bing News RSS: direct article URLs plus a one-paragraph summary, which gives the writer real human detail.
- Google News RSS: headlines only (links go through a Google redirect); used to top up.

Only recent (last ~10 days), weather-related articles that mention the place and come from reputable
international outlets or news sites on the country's own domain are kept.
"""
from __future__ import annotations

import html
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .config import USER_AGENT
from .scanner import haversine_km
from .validator import is_specific_event_url

MAX_AGE_DAYS = 10
LEADS_PER_STORY = 3

REPUTABLE_PUBLISHER_DOMAINS: tuple[str, ...] = (
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "reuters.com",
    "apnews.com",
    "aljazeera.com",
    "cnn.com",
    "france24.com",
    "dw.com",
    "euronews.com",
    "sky.com",
    "npr.org",
    "pbs.org",
    "nytimes.com",
    "washingtonpost.com",
    "independent.co.uk",
    "ft.com",
    "bloomberg.com",
    "time.com",
    "economist.com",
    "abc.net.au",
    "abcnews.go.com",
    "sbs.com.au",
    "rnz.co.nz",
    "nzherald.co.nz",
    "cbc.ca",
    "globalnews.ca",
    "theglobeandmail.com",
    "rte.ie",
    "irishtimes.com",
    "independent.ie",
    "japantimes.co.jp",
    "nhk.or.jp",
    "jiji.com",
    "efe.com",
    "aa.com.tr",
    "mainichi.jp",
    "kyodonews.net",
    "asahi.com",
    "e.vnexpress.net",
    "vnexpress.net",
    "vietnamplus.vn",
    "tuoitrenews.vn",
    "vietnamnews.vn",
    "channelnewsasia.com",
    "straitstimes.com",
    "scmp.com",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "bangkokpost.com",
    "inquirer.net",
    "rappler.com",
    "mexiconewsdaily.com",
    "batimes.com.ar",
    "mercopress.com",
    "icelandmonitor.mbl.is",
    "icelandreview.com",
    "ruv.is",
    "thelocal.com",
    "africanews.com",
    "dailymaverick.co.za",
    "news24.com",
    "nation.africa",
    "rfi.fr",
    "lemonde.fr",
    "elpais.com",
    "swissinfo.ch",
    "reliefweb.int",
    "wmo.int",
    "un.org",
    "who.int",
    "ifrc.org",
    "gdacs.org",
    "metoffice.gov.uk",
    "met.ie",
    "noaa.gov",
    "weather.gov",
    "climate.gov",
    "nasa.gov",
    "copernicus.eu",
    "ecmwf.int",
    "phys.org",
    "carbonbrief.org",
    "yaleclimateconnections.org",
    "scientificamerican.com",
    "nature.com",
)

# Low-quality, aggregator or non-news sites that should never be cited, even on a national domain
UNRELIABLE_SOURCES = (
    "natureworldnews",
    "nature world news",
    "ftnnews",
    "ftn news",
    "oilprice",
    "audacy",
    "aol.",
    "msn.",
    "britannica",
    "wikipedia",
    "yahoo.",
    "guidetoiceland",
)

# (search words, words that must appear in the title or summary, words that rule an article out)
HAZARD_TERMS = {
    "rain": (
        ["floods", "heavy rain"],
        r"\b(rain\w*|flood\w*|downpours?|deluge|monsoon|landslides?|typhoon|cyclone|hurricane|tropical storm|mudslides?|storm\w*)\b",
        r"\b(heatwave|drought|stormwater)\b",
    ),
    "heat": (
        ["heatwave", "high temperatures"],
        r"\b(heatwave|heat|hot|temperatures?|mild|warm\w*|wildfires?|drought)\b",
        r"\b(snow\w*|blizzard\w*|freez\w*|frost)\b",
    ),
    "cold": (
        ["cold snap", "snow"],
        r"\b(cold|chill\w*|freez\w*|frost\w*|snow\w*|blizzard\w*|ice|icy|sub-zero|temperatures?|winter)\b",
        r"\b(heatwave|scorching)\b",
    ),
    "cyclone": (
        ["storm", "high winds"],
        r"\b(storm\w*|gales?|gusts?|winds?|cyclone|typhoon|hurricane|blizzard\w*|weather warning|waves?|flood\w*|rain\w*)\b",
        r"\b(heatwave)\b",
    ),
}
HAZARD_TERMS["wind"] = HAZARD_TERMS["cyclone"]
HAZARD_TERMS["pressure"] = HAZARD_TERMS["cyclone"]

IDIOMS = (
    "storm in a",
    "by storm",
    "hot air",
    "hot seat",
    "hot topic",
    "cold war",
    "cold shoulder",
    "cold case",
    "political storm",
    "firestorm",
    "brainstorm",
    "perfect storm",
)


def _host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def is_reputable(url: str, source_name: str = "", country_domain: str = "") -> bool:
    """Reputable international outlet/agency, or a news site on the story country's own domain."""
    host = _host(url)
    low = f"{host} {source_name}".lower()
    if not host or any(bad in low for bad in UNRELIABLE_SOURCES):
        return False
    if host.endswith((".gov", ".gov.uk", ".gob.mx", ".go.jp", ".gov.au", ".int", ".edu", ".ac.uk")):
        return True
    if country_domain and len(country_domain) == 2 and host.endswith("." + country_domain):
        return True
    return any(host == d or host.endswith("." + d) for d in REPUTABLE_PUBLISHER_DOMAINS)


def _get(url: str, timeout: int = 8) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml,*/*"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _published(raw: str) -> datetime | None:
    try:
        dt = parsedate_to_datetime(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def search_bing_news(query: str) -> list[dict]:
    """Bing News RSS: direct article URL, title, summary, source and date."""
    # qft interval "8" = published in the past week (Bing otherwise ranks years-old articles first)
    raw = _get(
        "https://www.bing.com/news/search?"
        + urllib.parse.urlencode({"q": query, "format": "rss", "mkt": "en-GB", "qft": 'interval="8"'})
    )
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for item in root.findall("./channel/item"):
        link = item.findtext("link") or ""
        target = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get("url", [""])[0] or link
        source = ""
        for child in item:
            if child.tag.endswith("Source") and child.text:
                source = child.text.strip()
        out.append(
            {
                "title": _clean(item.findtext("title") or ""),
                "url": target,
                "summary": _clean(item.findtext("description") or "")[:400],
                "source": source or _host(target),
                "published": _published(item.findtext("pubDate") or ""),
            }
        )
    return out


def search_google_news(query: str) -> list[dict]:
    """Google News RSS: headline, outlet and a redirect link (no summary)."""
    raw = _get(
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode({"q": f"{query} when:{MAX_AGE_DAYS}d", "hl": "en-GB", "gl": "GB", "ceid": "GB:en"})
    )
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for item in root.findall("./channel/item"):
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        source_url = (src_el.get("url") if src_el is not None else "") or ""
        title = _clean(item.findtext("title") or "")
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")].strip()
        out.append(
            {
                "title": title,
                "url": (item.findtext("link") or "").strip(),
                "summary": "",
                "source": source,
                "source_url": source_url,
                "published": _published(item.findtext("pubDate") or ""),
            }
        )
    return out


def _place_terms(cand: dict) -> list[str]:
    loc = cand.get("location") or {}
    terms = []
    for key in ("state", "nearest_big_town", "country"):
        val = str(loc.get(key) or "").strip()
        if key == "nearest_big_town" and (loc.get("nearest_big_town_km") or 9999) > 300:
            continue
        if val and val not in terms:
            terms.append(val)
    return terms


def _strip_accents(s: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def gather_news(cand: dict, max_items: int = 4) -> list[dict]:
    """Up to `max_items` recent, relevant, reputable articles about the weather where `cand` is."""
    loc = cand.get("location") or {}
    country_domain = loc.get("country_domain") or ""
    kind = cand.get("kind", "rain")
    words, must_pat, forbid_pat = HAZARD_TERMS.get(kind, HAZARD_TERMS["rain"])
    must_re, forbid_re = re.compile(must_pat, re.I), re.compile(forbid_pat, re.I)
    places = _place_terms(cand)
    if not places:
        return []
    place_res = [re.compile(r"\b" + re.escape(_strip_accents(p).lower())) for p in places]
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

    queries = [f"{p} {w}" for p in places[:2] for w in words[:1]] + [f"{places[-1]} {words[1]}"]
    picked: list[dict] = []
    seen: set[str] = set()

    def _keep(results: list[dict]) -> None:
        for art in results:
            title, summary, url = art["title"], art.get("summary", ""), art["url"]
            if not title or not url or not is_specific_event_url(url):
                continue
            if art.get("published") and art["published"] < cutoff:
                continue
            if not is_reputable(art.get("source_url") or url, art.get("source", ""), country_domain):
                continue
            text = _strip_accents(f"{title} {summary}").lower()
            if any(idiom in text for idiom in IDIOMS) or forbid_re.search(title) or not must_re.search(text):
                continue
            if not any(r.search(text) for r in place_res):
                continue
            key = re.sub(r"[^a-z0-9]+", "", title.lower())[:50]
            if key in seen:
                continue
            seen.add(key)
            picked.append(art)

    for q in queries:
        _keep(search_bing_news(q))
    if len(picked) < LEADS_PER_STORY:
        for q in queries:
            _keep(search_google_news(q))
            if len(picked) >= max_items:
                break

    # Only offer articles whose links actually load (the writer is required to cite what it is given)
    if picked:
        from .validator import check_url_resolves

        with ThreadPoolExecutor(max_workers=4) as pool:
            ok = list(pool.map(lambda a: check_url_resolves(a["url"]), picked))
        picked = [a for a, good in zip(picked, ok) if good]
    # Prefer articles with a summary (more detail for the writer), then the most recent
    picked.sort(key=lambda a: (not a.get("summary"), -(a["published"].timestamp() if a.get("published") else 0)))
    return [
        {
            "source": a["source"],
            "title": a["title"][:160],
            "summary": a.get("summary", ""),
            "date": a["published"].strftime("%d %b %Y") if a.get("published") else "",
            "url": a["url"],
        }
        for a in picked[:max_items]
    ]


def attach_news(candidates: list[dict]) -> list[dict]:
    """Add `news_leads` to every candidate (in parallel; failures just mean no leads)."""

    def _safe(c: dict) -> list[dict]:
        try:
            return gather_news(c)
        except Exception as exc:
            print(f"[news] Lookup failed for {c.get('id')}: {exc}", file=sys.stderr)
            return []

    with ThreadPoolExecutor(max_workers=6) as pool:
        leads = list(pool.map(_safe, candidates))
    for c, items in zip(candidates, leads):
        c["news_leads"] = items
    print(
        "[news] Leads per hotspot: "
        + ", ".join(f"{(c.get('location') or {}).get('headline_prefix') or c['id']}={len(c['news_leads'])}" for c in candidates),
        file=sys.stderr,
    )
    return candidates


def choose_stories(candidates: list[dict], n: int = 5) -> list[dict]:
    """Pick the bulletin's stories: significant weather, on land where people live, in the news, and varied."""

    def base_score(idx: int, c: dict) -> float:
        loc = c.get("location") or {}
        # Scanner order reflects how unusual the weather is; news coverage shows it matters to people.
        # News weighs more: each story should be able to cite three articles.
        n_news = min(LEADS_PER_STORY, len(c.get("news_leads") or []))
        score = -0.4 * idx + 3.0 * n_news - (4.0 if n_news == 0 else 0.0)
        if not loc.get("country"):
            score -= 12.0  # open ocean, nobody to tell
        offshore_km = float(loc.get("offshore_km") or 0)
        if c["kind"] in ("heat", "cold") and offshore_km > 25:
            score -= 6.0
        elif offshore_km > 120:
            score -= 4.0
        return score

    pool = [(base_score(i, c), c) for i, c in enumerate(candidates)]
    chosen: list[dict] = []
    while pool and len(chosen) < n:
        def adjusted(item):
            s, c = item
            for other in chosen:
                if other["kind"] == c["kind"]:
                    s -= 1.0
                    if haversine_km(float(c["lat"]), float(c["lon"]), float(other["lat"]), float(other["lon"])) < 3000:
                        s -= 4.0  # two near-identical stories side by side
                if (c.get("location") or {}).get("country") and (c.get("location") or {}).get("country") == (
                    other.get("location") or {}
                ).get("country"):
                    s -= 5.0
            return s

        best = max(pool, key=adjusted)
        chosen.append(best[1])
        pool.remove(best)
    return chosen
