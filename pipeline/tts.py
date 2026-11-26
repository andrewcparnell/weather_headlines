"""Generate studio-style British female newsreader MP3 narration for each weather headline.

Default synthesis uses Microsoft's neural `en-GB-SoniaNeural` British female voice via `edge-tts`
(zero API key required, runs identically locally and in GitHub Actions every 6 hours), paired with
phonetic broadcast normalisation so units, year ranges, rankings, and names such as Met Eireann
and El Nino are spoken with natural BBC / Sky News intonation.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

DEFAULT_VOICE = "en-GB-SoniaNeural"
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "-1Hz"


def normalise_for_newsreader(text: str) -> str:
    """Convert visual symbols, abbreviations, and names into natural spoken British broadcast prose."""
    s = text or ""
    # Strip markdown highlight markers and ticker separators
    s = s.replace("**", "")
    s = re.sub(r"\s*///\s*", ". ", s)
    s = s.replace("\u2014", ", ").replace(" - ", ", ")

    # Year ranges: 1940-1979 -> 1940 to 1979
    s = re.sub(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", r"\1 to \2", s)

    # Signed temperatures: +16.4 °C -> 16.4 degrees Celsius, -25 °C -> minus 25 degrees Celsius
    s = re.sub(r"\+(\d+(?:\.\d+)?)\s*°C\b", r"\1 degrees Celsius", s)
    s = re.sub(r"-(\d+(?:\.\d+)?)\s*°C\b", r"minus \1 degrees Celsius", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°C\b", r"\1 degrees Celsius", s)

    # Units spoken naturally by a TV presenter
    s = re.sub(r"\b(\d+(?:,\d+)?(?:\.\d+)?)\s*mm\s*/\s*24\s*h\b", r"\1 millimetres in 24 hours", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:,\d+)?(?:\.\d+)?)\s*mm\b", r"\1 millimetres", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*km/h\b", r"\1 kilometres an hour", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*mph\b", r"\1 miles an hour", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*m/s\b", r"\1 metres per second", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*hPa\b", r"\1 hectopascals", s)
    s = re.sub(r"\b2\s*m\s+temperature", "surface temperature", s, flags=re.I)
    s = re.sub(r"\b10\s*m\s+wind", "surface wind", s, flags=re.I)

    # Rankings: #1 -> number 1
    s = re.sub(r"#(\d+)\b", r"number \1", s)
    s = re.sub(r"\bpct\b", "percentile", s, flags=re.I)

    # Phonetic cues for non-English meteorological names so British neural TTS pronounces them cleanly
    s = re.sub(r"\bMet\s+[EÉ]ireann\b", "Met Air-uhn", s, flags=re.I)
    s = re.sub(r"\bEl\s+Ni[nñ]o\b", "El Neen-yo", s, flags=re.I)
    s = re.sub(r"\bLa\s+Ni[nñ]a\b", "La Neen-ya", s, flags=re.I)
    s = re.sub(r"\bLos\s+R[ií]os\b", "Los Ree-os", s, flags=re.I)

    # Clean up double periods or extra whitespace
    s = re.sub(r"\.\s*\.", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compose_spoken_script_from_story(story: dict) -> str:
    """Build a natural, well-paced newsreader script if `voice_script` was not supplied explicitly."""
    explicit = (story.get("voice_script") or "").strip()
    if len(explicit) >= 60:
        return normalise_for_newsreader(explicit)

    headline = (story.get("headline") or "").rstrip(".")
    beats = story.get("beats") or []
    parts = [f"{headline}."]
    transitions = [
        "",
        "Looking at how long this has lasted:",
        "In the historical climate record:",
        "And on the role of climate change:",
    ]
    for idx, b in enumerate(beats[:4]):
        txt = (b.get("text") or "").strip()
        if not txt:
            continue
        lead = transitions[idx] if idx < len(transitions) else ""
        parts.append(f"{lead} {txt}".strip())
    return normalise_for_newsreader(" ".join(parts))


async def _synthesise_one_mp3(text: str, out_path: Path, voice: str, rate: str, pitch: str) -> None:
    import edge_tts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp3")
    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await comm.save(str(tmp))
    tmp.replace(out_path)


async def _synthesise_all_async(stories: list[dict], audio_dir: Path) -> list[str]:
    voice = os.environ.get("WEATHER_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
    rate = os.environ.get("WEATHER_TTS_RATE", DEFAULT_RATE).strip() or DEFAULT_RATE
    pitch = os.environ.get("WEATHER_TTS_PITCH", DEFAULT_PITCH).strip() or DEFAULT_PITCH

    audio_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(3)

    async def _worker(idx: int, story: dict) -> str:
        async with sem:
            spoken = compose_spoken_script_from_story(story)
            filename = f"story-{idx}.mp3"
            dest = audio_dir / filename
            await _synthesise_one_mp3(spoken, dest, voice=voice, rate=rate, pitch=pitch)
            rel = f"data/audio/{filename}"
            story["audio_url"] = rel
            print(f"[tts] Synthesised {filename} ({dest.stat().st_size // 1024} KB) using {voice}", file=sys.stderr)
            return rel

    return list(await asyncio.gather(*(_worker(idx, st) for idx, st in enumerate(stories, start=1))))


def build_bulletin_audio(bulletin: dict, out_data_dir: Path) -> dict:
    """Generate MP3 files for every story in `bulletin` under `out_data_dir/audio/` and attach `audio_url`."""
    stories = bulletin.get("stories") or []
    if not stories:
        return bulletin
    audio_dir = out_data_dir / "audio"
    asyncio.run(_synthesise_all_async(stories, audio_dir))
    bulletin["voice"] = os.environ.get("WEATHER_TTS_VOICE", DEFAULT_VOICE)
    return bulletin
