"""
MAVEN Transcription: downloads TikTok audio and transcribes it with faster-whisper.
Public entry point: transcribe_url(url) → TranscriptResult.
"""
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

_TIKTOK_RE = re.compile(r'https?://(www\.)?tiktok\.com/')
_model = None  # lazy-loaded on first call to _get_model()


@dataclass
class TranscriptResult:
    text: str
    segments: List[dict]   # [{"start": float, "end": float, "text": str}, ...]
    duration: float


def transcribe_url(url: str) -> TranscriptResult:
    if not _TIKTOK_RE.match(url.strip()):
        raise ValueError("URL does not appear to be a TikTok link.")
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = _download_audio(url, tmp_dir)
        return _transcribe(audio_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _download_audio(url: str, tmp_dir: str) -> Path:
    """Download TikTok audio to tmp_dir as mp3. Raises RuntimeError on failure."""
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    result = subprocess.run(
        [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--output', output_template,
            '--no-playlist',
            '--quiet',
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    mp3_files = list(Path(tmp_dir).glob('*.mp3'))
    if not mp3_files:
        raise RuntimeError('Download failed: no audio file produced.')
    return mp3_files[0]


def _get_model():
    raise NotImplementedError


def _transcribe(audio_path: Path) -> TranscriptResult:
    raise NotImplementedError
