"""
MAVEN Transcription: downloads TikTok audio and transcribes it with faster-whisper.
Public entry point: transcribe_url(url) → TranscriptResult.
"""
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from video_source import download_audio, ensure_ffmpeg, validate_url

_model = None  # lazy-loaded on first call to _get_model()


class NoSpeechError(RuntimeError):
    """Raised when transcription produces no speech output."""


@dataclass
class TranscriptResult:
    text: str
    segments: List[dict]   # [{"start": float, "end": float, "text": str}, ...]
    duration: float


def transcribe_url(url: str) -> TranscriptResult:
    url, _platform = validate_url(url)
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = download_audio(url, tmp_dir)
        return _transcribe(audio_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_model():
    """Load WhisperModel once at first call; return cached instance thereafter."""
    global _model
    if _model is None:
        ensure_ffmpeg()  # faster-whisper needs ffmpeg for audio decoding
        from faster_whisper import WhisperModel
        print('[MAVEN] Loading Whisper small model (one-time, ~244 MB)...')
        _model = WhisperModel('small', device='cpu', compute_type='int8')
        print('[MAVEN] Whisper model ready.')
    return _model


def _transcribe(audio_path: Path) -> TranscriptResult:
    """Transcribe audio_path. Raises RuntimeError if no speech is detected."""
    model = _get_model()
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)
    segments = []
    texts = []
    for seg in segments_iter:
        segments.append({
            'start': round(seg.start, 2),
            'end':   round(seg.end, 2),
            'text':  seg.text.strip(),
        })
        texts.append(seg.text.strip())
    full_text = ' '.join(t for t in texts if t)
    if not full_text.strip():
        raise NoSpeechError('No speech detected in audio.')
    return TranscriptResult(
        text=full_text,
        segments=segments,
        duration=round(info.duration, 2),
    )
