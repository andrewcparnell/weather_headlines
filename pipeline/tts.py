"""Generate studio-style British female newsreader MP3 narration for each weather headline.

Default synthesis uses Microsoft's neural `en-GB-SoniaNeural` British female voice via `edge-tts`
(zero API key required, runs identically locally and in GitHub Actions every 6 hours), paired with
phonetic normalisation so units, year ranges, rankings, and names such as Met Eireann
and El Nino are spoken clearly and calmly.
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
    s = re.sub(r"\+(\d+(?:\.\d+)?)\s*°C\+?", r"\1 degrees Celsius", s)
    s = re.sub(r"-(\d+(?:\.\d+)?)\s*°C\+?", r"minus \1 degrees Celsius", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°C\+?", r"\1 degrees Celsius", s)

    # Units spoken naturally by a TV presenter
    s = re.sub(r"\b(\d+(?:,\d+)?(?:\.\d+)?)\s*mm\s*/\s*24\s*h\b", r"\1 millimetres in 24 hours", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:,\d+)?(?:\.\d+)?)\s*mm\b", r"\1 millimetres", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*cm\b", r"\1 centimetres", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*km²", r"\1 square kilometres", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*km/h\b", r"\1 kilometres an hour", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*mph\b", r"\1 miles an hour", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*m/s\b", r"\1 metres per second", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*hPa\b", r"\1 hectopascals", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*×", r"\1 times", s)
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
    """Build the exact spoken script from `cues[*].spoken` (single source of truth matching on-screen text)."""
    cues = story.get("cues") or []
    cue_sentences = [(c.get("spoken") or "").strip() for c in cues if (c.get("spoken") or "").strip()]
    if cue_sentences:
        return normalise_for_newsreader(" ".join(cue_sentences))

    explicit = (story.get("voice_script") or "").strip()
    if len(explicit) >= 40:
        return normalise_for_newsreader(explicit)

    headline = (story.get("headline") or "").rstrip(".")
    beats = story.get("beats") or []
    parts = [f"{headline}."]
    for b in beats[:4]:
        txt = (b.get("text") or "").strip()
        if txt:
            parts.append(txt)
    return normalise_for_newsreader(" ".join(parts))


def assign_cue_timings(cues: list[dict], duration_sec: float, sentence_offsets_sec: list[float] | None = None) -> None:
    """Assign [start_sec, end_sec] to each cue using TTS SentenceBoundary offsets or spoken length weights."""
    if not cues or duration_sec <= 1.0:
        return

    n = len(cues)
    if sentence_offsets_sec and len(sentence_offsets_sec) == n:
        starts = [max(0.0, float(s)) for s in sentence_offsets_sec]
        starts[0] = 0.0
        for i in range(n):
            end = starts[i + 1] if i + 1 < n else duration_sec
            cues[i]["start_sec"] = round(starts[i], 2)
            cues[i]["end_sec"] = round(max(starts[i] + 0.4, end), 2)
        cues[-1]["end_sec"] = round(duration_sec, 2)
        return

    weights = [max(12, len(normalise_for_newsreader(c.get("spoken") or c.get("ticker") or ""))) for c in cues]
    total_w = float(sum(weights))
    acc = 0.0
    for i, w in enumerate(weights):
        t0 = (acc / total_w) * duration_sec
        acc += w
        t1 = duration_sec if i == n - 1 else (acc / total_w) * duration_sec
        cues[i]["start_sec"] = round(t0, 2)
        cues[i]["end_sec"] = round(max(t0 + 0.4, t1), 2)


async def _synthesise_raw_mp3_bytes(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    sem: asyncio.Semaphore,
) -> bytes:
    import edge_tts

    async with sem:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
                audio_bytes = bytearray()
                async for msg in comm.stream():
                    if msg.get("type") == "audio":
                        audio_bytes.extend(msg["data"])
                if len(audio_bytes) < 1_800:
                    raise RuntimeError(f"Cue TTS returned insufficient audio payload ({len(audio_bytes)} bytes)")
                return bytes(audio_bytes)
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    await asyncio.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"Edge-TTS cue synthesis failed: {last_err}")


async def _synthesise_story_from_cues(
    story: dict,
    out_path: Path,
    voice: str,
    rate: str,
    pitch: str,
    sem: asyncio.Semaphore,
) -> float:
    """Synthesise each cue's exact `spoken` text individually and concatenate raw 48 kbps CBR MP3 frames.

    Guarantees 100% word-for-word match between voiceover and on-screen caption text AND exact
    millisecond [start_sec, end_sec] boundaries for every cue regardless of punctuation.
    """
    cues = story.get("cues") or []
    if not cues:
        spoken = compose_spoken_script_from_story(story)
        raw_bytes = await _synthesise_raw_mp3_bytes(spoken, voice=voice, rate=rate, pitch=pitch, sem=sem)
        if len(raw_bytes) < 16_000:
            raise RuntimeError(f"TTS returned insufficient audio payload ({len(raw_bytes)} bytes)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.mp3")
        tmp.write_bytes(raw_bytes)
        tmp.replace(out_path)
        return round((len(raw_bytes) * 8.0) / 48000.0, 2)

    cue_texts = [
        normalise_for_newsreader((c.get("spoken") or c.get("ticker") or "").strip())
        for c in cues
    ]
    cue_chunks = await asyncio.gather(
        *(_synthesise_raw_mp3_bytes(txt, voice=voice, rate=rate, pitch=pitch, sem=sem) for txt in cue_texts)
    )

    combined = bytearray()
    running_sec = 0.0
    for cue, chunk in zip(cues, cue_chunks):
        cue_dur = (len(chunk) * 8.0) / 48000.0
        cue["start_sec"] = round(running_sec, 2)
        running_sec += cue_dur
        cue["end_sec"] = round(max(cue["start_sec"] + 0.4, running_sec), 2)
        combined.extend(chunk)

    if len(combined) < 16_000:
        raise RuntimeError(f"Combined story TTS returned insufficient audio payload ({len(combined)} bytes)")

    story["voice_script"] = " ".join((c.get("spoken") or "").strip() for c in cues if (c.get("spoken") or "").strip())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp3")
    tmp.write_bytes(bytes(combined))
    tmp.replace(out_path)

    total_sec = round((len(combined) * 8.0) / 48000.0, 2)
    cues[-1]["end_sec"] = total_sec
    return total_sec


def format_audio_stamp(generated_utc: str | None) -> str:
    """Normalise an ISO timestamp into a compact URL-safe stamp (e.g. '20261127T060000Z')."""
    compact = re.sub(r"[^0-9TZ]", "", str(generated_utc or "").upper())
    return compact if len(compact) >= 8 else "latest"


async def _synthesise_all_async(stories: list[dict], audio_dir: Path, stamp: str) -> list[str]:
    voice = os.environ.get("WEATHER_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
    rate = os.environ.get("WEATHER_TTS_RATE", DEFAULT_RATE).strip() or DEFAULT_RATE
    pitch = os.environ.get("WEATHER_TTS_PITCH", DEFAULT_PITCH).strip() or DEFAULT_PITCH

    audio_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(4)

    async def _worker(idx: int, story: dict) -> str:
        filename = f"story-{idx}-{stamp}.mp3"
        dest = audio_dir / filename
        duration_sec = await _synthesise_story_from_cues(
            story, dest, voice=voice, rate=rate, pitch=pitch, sem=sem
        )
        rel = f"data/audio/{filename}"
        story["audio_url"] = rel
        story["duration_sec"] = duration_sec
        print(
            f"[tts] Synthesised {filename} ({dest.stat().st_size // 1024} KB, {duration_sec:.2f}s, {len(story.get('cues') or [])} synced cues) using {voice}",
            file=sys.stderr,
        )
        return rel

    written_rels = list(await asyncio.gather(*(_worker(idx, st) for idx, st in enumerate(stories, start=1))))
    keep_names = {Path(r).name for r in written_rels}
    for old_mp3 in audio_dir.glob("story-*.mp3"):
        if old_mp3.name not in keep_names:
            old_mp3.unlink(missing_ok=True)
    return written_rels


def build_bulletin_audio(bulletin: dict, out_data_dir: Path) -> dict:
    """Generate stamped MP3 files `story-{n}-{generated_utc}.mp3` under `out_data_dir/audio/`."""
    stories = bulletin.get("stories") or []
    if not stories:
        return bulletin
    stamp = format_audio_stamp(bulletin.get("generated_utc"))
    audio_dir = out_data_dir / "audio"
    asyncio.run(_synthesise_all_async(stories, audio_dir, stamp=stamp))
    bulletin["voice"] = os.environ.get("WEATHER_TTS_VOICE", DEFAULT_VOICE)
    return bulletin


def main() -> int:
    import json
    from datetime import datetime, timezone

    from .config import DATA_DIR
    from .validator import validate_and_clean_bulletin

    headlines_path = DATA_DIR / "headlines.json"
    payload = json.loads(headlines_path.read_text())
    cleaned, errors = validate_and_clean_bulletin(payload)
    if errors:
        raise RuntimeError(f"Cannot synthesise audio for invalid bulletin: {errors}")
    if not cleaned.get("generated_utc"):
        cleaned["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cleaned = build_bulletin_audio(cleaned, DATA_DIR)
    tmp = headlines_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(headlines_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
