"""
MAVEN Transcription: downloads TikTok audio and transcribes it with faster-whisper.
Public entry point: transcribe_url(url) → TranscriptResult.
"""
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

_TIKTOK_RE = re.compile(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/')
_model = None       # lazy-loaded on first call to _get_model()
_ffmpeg_exe = None  # resolved once; '' means fall back to system ffmpeg


def _ensure_ffmpeg() -> str:
    """Return the path to a usable ffmpeg executable and inject its directory into PATH.

    imageio-ffmpeg ships its binary under a version-suffixed name such as
    'ffmpeg-win-x86_64-v7.1.exe' rather than the standard 'ffmpeg.exe' that
    both yt-dlp and faster-whisper look for.  This function copies it to a
    stable temp directory under the standard name so both callers work.

    Returns the full path to the normalized binary, or '' if unavailable.
    """
    global _ffmpeg_exe
    if _ffmpeg_exe is not None:
        return _ffmpeg_exe

    try:
        import platform
        import imageio_ffmpeg

        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not src.exists():
            raise FileNotFoundError(src)

        exe_suffix = '.exe' if platform.system() == 'Windows' else ''
        norm_dir = Path(tempfile.gettempdir()) / 'maven_ffmpeg'
        norm_dir.mkdir(exist_ok=True)
        dst = norm_dir / f'ffmpeg{exe_suffix}'

        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            if not exe_suffix:          # Unix needs execute bit
                dst.chmod(0o755)

        _ffmpeg_exe = str(dst)

        # PATH injection lets faster-whisper find 'ffmpeg' via subprocess
        norm_dir_str = str(norm_dir)
        existing = os.environ.get('PATH', '')
        if norm_dir_str not in existing:
            os.environ['PATH'] = norm_dir_str + os.pathsep + existing

    except Exception:
        _ffmpeg_exe = ''  # fall back to system ffmpeg

    return _ffmpeg_exe


class NoSpeechError(RuntimeError):
    """Raised when transcription produces no speech output."""


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
    ffmpeg_exe = _ensure_ffmpeg()
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'mp3',
        '--output', output_template,
        '--no-playlist',
        '--quiet',
        '--impersonate', 'chrome',
    ]
    if ffmpeg_exe:
        # Pass binary path directly (not parent dir) so yt-dlp uses it regardless
        # of filename — yt-dlp treats a file path as the ffmpeg executable itself.
        cmd += ['--ffmpeg-location', ffmpeg_exe]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    mp3_files = list(Path(tmp_dir).glob('*.mp3'))
    if not mp3_files:
        raise RuntimeError('Download failed: no audio file produced.')
    return mp3_files[0]


def _get_model():
    """Load WhisperModel once at first call; return cached instance thereafter."""
    global _model
    if _model is None:
        _ensure_ffmpeg()  # faster-whisper needs ffmpeg for audio decoding
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
