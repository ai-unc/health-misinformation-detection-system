"""
MAVEN shared video-source plumbing used by transcription.py and
text_extraction.py: platform dispatch, URL validation, ffmpeg normalization,
and yt-dlp download/metadata helpers. Per-platform definitions live in
tiktok.py and instagram.py.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

_ffmpeg_exe = None  # resolved once; '' means fall back to system ffmpeg

# yt-dlp stderr fragments (casefolded) that mean Instagram blocked an
# anonymous request rather than the video being genuinely broken.
_INSTAGRAM_BLOCK_SIGNATURES = (
    'login required',
    'rate-limit reached',
    'restricted video',
    'requested content is not available',
)
INSTAGRAM_BLOCK_MESSAGE = ('Instagram requires login or has rate-limited this '
                           'request. Try a public Reel or retry later.')


@dataclass(frozen=True)
class Platform:
    """A supported video platform: URL shape + OCR watermark junk terms."""
    name: str              # slug, e.g. 'tiktok'
    display_name: str      # e.g. 'TikTok'
    url_re: re.Pattern     # matches URLs belonging to this platform
    junk_terms: frozenset  # casefolded watermark strings the OCR filter drops


def _platforms() -> Tuple[Platform, ...]:
    # Imported lazily: platform modules import Platform from here, so a
    # top-level import would be circular.
    from tiktok import TIKTOK
    from instagram import INSTAGRAM
    return (TIKTOK, INSTAGRAM)


def validate_url(url: str) -> Tuple[str, Platform]:
    """Return (stripped URL, matching Platform), or raise ValueError."""
    url = url.strip()
    for platform in _platforms():
        if platform.url_re.match(url):
            return url, platform
    raise ValueError('URL is not a supported TikTok or Instagram Reels link.')


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


def _is_instagram_url(url: str) -> bool:
    from instagram import INSTAGRAM  # lazy: platform modules import Platform from here
    return bool(INSTAGRAM.url_re.match(url))


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        low = stderr.casefold()
        if _is_instagram_url(cmd[-1]) and any(sig in low for sig in _INSTAGRAM_BLOCK_SIGNATURES):
            raise RuntimeError(INSTAGRAM_BLOCK_MESSAGE)
        msg = stderr or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    return result


def download_audio(url: str, tmp_dir: str) -> Path:
    """Download the video's audio track to tmp_dir as mp3. Raises RuntimeError on failure."""
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
    """Download the video to tmp_dir as mp4. Raises RuntimeError on failure."""
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
