"""
MAVEN Text Extraction: pulls on-screen overlay text (via frame OCR) and the
video description from a supported video URL (TikTok or Instagram Reel).
Public entry point: extract_text_url(url) → TextExtractionResult.
"""
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

from video_source import download_video, ensure_ffmpeg, fetch_metadata, validate_url

_ocr_engine = None  # lazy-loaded on first call to _get_ocr()

MIN_CONFIDENCE    = 0.6   # OCR lines below this are noise
MIN_LINE_CHARS    = 3     # shorter lines are noise
FUZZY_MATCH_RATIO = 0.9   # SequenceMatcher ratio treating two frames as the same overlay
FRAME_WIDTH       = 720   # frames scaled to this width before OCR
MAX_VIDEO_SECONDS = 600   # cap frame sampling; overlays past 10 min are ignored


class NoTextFoundError(RuntimeError):
    """Raised when neither overlay text nor a description is found."""


@dataclass
class TextExtractionResult:
    description: str
    overlay_segments: List[dict]  # [{"start": float, "end": float, "text": str}, ...]
    text: str                     # description + unique overlay lines → feeds score_text()


def _normalize(text: str) -> str:
    return ' '.join(text.casefold().split())


def _is_junk(text: str, confidence: float, uploader: str = '',
             junk_terms: frozenset = frozenset()) -> bool:
    """True for OCR noise and platform watermark artifacts (logo, @handle)."""
    t = text.strip()
    if confidence < MIN_CONFIDENCE or len(t) < MIN_LINE_CHARS:
        return True
    if t.startswith('@'):
        return True
    low = t.casefold().lstrip('@')
    if low in junk_terms:
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


def extract_text_url(url: str) -> TextExtractionResult:
    url, platform = validate_url(url)
    metadata = fetch_metadata(url)
    description = (metadata.get('description') or '').strip()
    uploader = (metadata.get('uploader') or '').strip()

    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = download_video(url, tmp_dir)
        frames = _sample_frames(video_path, tmp_dir)
        frame_results = _ocr_frames(frames, uploader, platform.junk_terms)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    overlay_segments = group_overlay_segments(frame_results)
    text = _assemble_text(description, overlay_segments)
    if not text:
        raise NoTextFoundError('No overlay text or description found in video.')
    return TextExtractionResult(
        description=description,
        overlay_segments=overlay_segments,
        text=text,
    )


def _sample_frames(video_path: Path, tmp_dir: str) -> List[Path]:
    """Extract one frame per second as PNGs scaled to FRAME_WIDTH px wide.

    Frame N (1-based in filenames) corresponds to second N-1 of the video.
    Sampling is capped at the first MAX_VIDEO_SECONDS of the video so an
    unusually long upload can't pin a worker.
    Raises RuntimeError on ffmpeg failure.
    """
    ffmpeg_exe = ensure_ffmpeg() or 'ffmpeg'
    frames_dir = Path(tmp_dir) / 'frames'
    frames_dir.mkdir(exist_ok=True)
    cmd = [
        ffmpeg_exe, '-hide_banner', '-loglevel', 'error',
        '-i', str(video_path),
        '-vf', f'fps=1,scale={FRAME_WIDTH}:-2',
        '-t', str(MAX_VIDEO_SECONDS),
        str(frames_dir / 'frame_%04d.png'),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        msg = result.stderr.strip() or f'ffmpeg exited with code {result.returncode}'
        raise RuntimeError(f'Frame sampling failed: {msg}')
    return sorted(frames_dir.glob('frame_*.png'))


def _get_ocr():
    """Load the RapidOCR engine once at first call; return cached instance."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        print('[MAVEN] Loading RapidOCR engine (one-time)...')
        _ocr_engine = RapidOCR()
        print('[MAVEN] RapidOCR engine ready.')
    return _ocr_engine


def _ocr_frames(frames: List[Path], uploader: str,
                junk_terms: frozenset = frozenset()) -> List[dict]:
    """OCR each frame, junk-filtering lines.

    Returns [{'ts': int, 'lines': [(text, confidence), ...]}, ...] — one entry
    per frame (ts = seconds from video start), ready for group_overlay_segments.
    """
    engine = _get_ocr()
    results = []
    for idx, frame in enumerate(frames):
        raw, _elapsed = engine(str(frame))  # [[box, text, score], ...] or None
        lines = []
        for item in (raw or []):
            text, conf = item[1].strip(), float(item[2])
            if not _is_junk(text, conf, uploader, junk_terms):
                lines.append((text, conf))
        results.append({'ts': idx, 'lines': lines})
    return results
