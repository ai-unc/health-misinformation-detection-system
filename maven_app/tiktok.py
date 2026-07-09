"""
MAVEN TikTok plumbing shared by audio transcription (transcription.py) and
text extraction (text_extraction.py): URL validation, ffmpeg normalization,
and yt-dlp download/metadata helpers.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

TIKTOK_RE = re.compile(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/')
_ffmpeg_exe = None  # resolved once; '' means fall back to system ffmpeg


def validate_url(url: str) -> str:
    """Return the stripped URL, or raise ValueError if it is not a TikTok link."""
    url = url.strip()
    if not TIKTOK_RE.match(url):
        raise ValueError("URL does not appear to be a TikTok link.")
    return url


def ensure_ffmpeg() -> str:
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


def _base_cmd() -> List[str]:
    """Common yt-dlp arguments — every download/metadata call goes through here
    so chrome impersonation and the normalized ffmpeg path are never missed."""
    cmd = ['yt-dlp', '--no-playlist', '--quiet', '--impersonate', 'chrome']
    ffmpeg_exe = ensure_ffmpeg()
    if ffmpeg_exe:
        # Pass binary path directly (not parent dir) so yt-dlp uses it regardless
        # of filename — yt-dlp treats a file path as the ffmpeg executable itself.
        cmd += ['--ffmpeg-location', ffmpeg_exe]
    return cmd


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    return result


def download_audio(url: str, tmp_dir: str) -> Path:
    """Download TikTok audio to tmp_dir as mp3. Raises RuntimeError on failure."""
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    cmd = _base_cmd() + [
        '--extract-audio',
        '--audio-format', 'mp3',
        '--output', output_template,
        url,
    ]
    _run(cmd)
    mp3_files = list(Path(tmp_dir).glob('*.mp3'))
    if not mp3_files:
        raise RuntimeError('Download failed: no audio file produced.')
    return mp3_files[0]


def download_video(url: str, tmp_dir: str) -> Path:
    """Download TikTok video to tmp_dir as mp4. Raises RuntimeError on failure."""
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    cmd = _base_cmd() + [
        '-f', 'mp4',
        '--output', output_template,
        url,
    ]
    _run(cmd)
    mp4_files = list(Path(tmp_dir).glob('*.mp4'))
    if not mp4_files:
        raise RuntimeError('Download failed: no video file produced.')
    return mp4_files[0]


def fetch_metadata(url: str) -> dict:
    """Fetch video metadata (description, uploader, ...) without downloading."""
    cmd = _base_cmd() + ['--dump-json', '--skip-download', url]
    result = _run(cmd)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError('Metadata fetch failed: invalid JSON from yt-dlp.')
