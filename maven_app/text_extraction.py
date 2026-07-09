"""
MAVEN Text Extraction: pulls on-screen overlay text (via frame OCR) and the
video description from a TikTok URL.
Public entry point: extract_text_url(url) → TextExtractionResult.
"""
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

from tiktok import download_video, ensure_ffmpeg, fetch_metadata, validate_url

_ocr_engine = None  # lazy-loaded on first call to _get_ocr()

MIN_CONFIDENCE    = 0.6   # OCR lines below this are noise
MIN_LINE_CHARS    = 3     # shorter lines are noise
FUZZY_MATCH_RATIO = 0.9   # SequenceMatcher ratio treating two frames as the same overlay
FRAME_WIDTH       = 720   # frames scaled to this width before OCR


class NoTextFoundError(RuntimeError):
    """Raised when neither overlay text nor a description is found."""


@dataclass
class TextExtractionResult:
    description: str
    overlay_segments: List[dict]  # [{"start": float, "end": float, "text": str}, ...]
    text: str                     # description + unique overlay lines → feeds score_text()


def _normalize(text: str) -> str:
    return ' '.join(text.casefold().split())


def _is_junk(text: str, confidence: float, uploader: str = '') -> bool:
    """True for OCR noise and TikTok watermark artifacts (logo, @handle)."""
    t = text.strip()
    if confidence < MIN_CONFIDENCE or len(t) < MIN_LINE_CHARS:
        return True
    if t.startswith('@'):
        return True
    low = t.casefold().lstrip('@')
    if low == 'tiktok':
        return True
    if uploader and low == uploader.casefold().lstrip('@'):
        return True
    return False


def _same_overlay(norm_a: str, norm_b: str) -> bool:
    if norm_a == norm_b:
        return True
    return SequenceMatcher(None, norm_a, norm_b).ratio() >= FUZZY_MATCH_RATIO


def group_overlay_segments(frame_results: List[dict]) -> List[dict]:
    """Merge consecutive frames showing the same overlay into timed segments.

    frame_results: [{'ts': int, 'lines': [(text, confidence), ...]}, ...],
    one entry per sampled frame (1/sec), lines already junk-filtered.
    Frames match when their normalized text is identical or fuzzy-similar
    (ratio >= FUZZY_MATCH_RATIO), absorbing per-frame OCR jitter; the
    highest-confidence variant of the text wins.
    Returns [{'start': float, 'end': float, 'text': str}, ...].
    """
    segments = []
    current = None  # {'start', 'end', 'text', 'conf', 'norm'}
    for frame in frame_results:
        lines = frame.get('lines') or []
        if not lines:
            if current:
                segments.append(current)
                current = None
            continue
        text = ' '.join(t for t, _ in lines)
        conf = sum(c for _, c in lines) / len(lines)
        norm = _normalize(text)
        if current is not None and _same_overlay(current['norm'], norm):
            current['end'] = frame['ts'] + 1
            if conf > current['conf']:
                current.update(text=text, conf=conf, norm=norm)
        else:
            if current:
                segments.append(current)
            current = {'start': frame['ts'], 'end': frame['ts'] + 1,
                       'text': text, 'conf': conf, 'norm': norm}
    if current:
        segments.append(current)
    return [{'start': float(s['start']), 'end': float(s['end']), 'text': s['text']}
            for s in segments]


def _assemble_text(description: str, overlay_segments: List[dict]) -> str:
    """description + each unique overlay line, newline-joined (feeds score_text)."""
    parts = []
    if description:
        parts.append(description)
    seen = set()
    for seg in overlay_segments:
        key = _normalize(seg['text'])
        if key not in seen:
            seen.add(key)
            parts.append(seg['text'])
    return '\n'.join(parts).strip()
