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
    'empty media response',
)
INSTAGRAM_BLOCK_MESSAGE = ('Instagram requires login or has rate-limited this '
                           'request. Try a public Reel or retry later.')

MAVEN_IG_COOKIES_ENV = 'MAVEN_IG_COOKIES'
INSTAGRAM_COOKIE_MESSAGE = ('Instagram slideshows require login cookies. '
                            'Export a cookies.txt for instagram.com and set '
                            'MAVEN_IG_COOKIES to its path.')

# gallery-dl stderr fragments (casefolded) that mean Instagram rejected the
# request for lack of (valid) login cookies.
_GALLERY_DL_LOGIN_SIGNATURES = ('redirect to login page', 'login required')

# File extensions download_slideshow keeps; everything else gallery-dl
# produces (mp3 soundtrack, .json metadata sidecars) is filtered out.
_IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp'})


@dataclass(frozen=True)
class Platform:
    """A supported video platform: URL shapes + OCR watermark junk terms."""
    name: str                     # slug, e.g. 'tiktok'
    display_name: str             # e.g. 'TikTok'
    url_re: re.Pattern            # matches URLs belonging to this platform
    slideshow_url_re: re.Pattern  # matches this platform's photo/slideshow posts
    junk_terms: frozenset         # casefolded watermark strings the OCR filter drops

    def is_slideshow(self, url: str) -> bool:
        return bool(self.slideshow_url_re.match(url))


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
    raise ValueError('URL is not a supported TikTok or Instagram link.')


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


def _instagram_cookies() -> str:
    """Path to the operator's Instagram cookies.txt, or '' if not configured."""
    return os.environ.get(MAVEN_IG_COOKIES_ENV, '').strip()


def _base_cmd(url: str) -> List[str]:
    """Common yt-dlp arguments — every download/metadata call goes through here
    so chrome impersonation, the normalized ffmpeg path, and Instagram login
    cookies (MAVEN_IG_COOKIES, when set) are never missed."""
    cmd = ['yt-dlp', '--no-playlist', '--quiet', '--impersonate', 'chrome']
    ffmpeg_exe = ensure_ffmpeg()
    if ffmpeg_exe:
        # Pass binary path directly (not parent dir) so yt-dlp uses it regardless
        # of filename — yt-dlp treats a file path as the ffmpeg executable itself.
        cmd += ['--ffmpeg-location', ffmpeg_exe]
    if _is_instagram_url(url):
        cookies = _instagram_cookies()
        if cookies:
            cmd += ['--cookies', cookies]
    return cmd


def _is_instagram_url(url: str) -> bool:
    from instagram import INSTAGRAM  # lazy: platform modules import Platform from here
    return bool(INSTAGRAM.url_re.match(url))


def _run(cmd: List[str], url: str) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        low = stderr.casefold()
        if _is_instagram_url(url) and any(sig in low for sig in _INSTAGRAM_BLOCK_SIGNATURES):
            raise RuntimeError(INSTAGRAM_BLOCK_MESSAGE)
        msg = stderr or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    return result


def download_audio(url: str, tmp_dir: str) -> Path:
    """Download the video's audio track to tmp_dir as mp3. Raises RuntimeError on failure."""
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    cmd = _base_cmd(url) + [
        '--extract-audio',
        '--audio-format', 'mp3',
        '--output', output_template,
        url,
    ]
    _run(cmd, url)
    mp3_files = list(Path(tmp_dir).glob('*.mp3'))
    if not mp3_files:
        raise RuntimeError('Download failed: no audio file produced.')
    return mp3_files[0]


def download_video(url: str, tmp_dir: str) -> Path:
    """Download the video to tmp_dir as mp4. Raises RuntimeError on failure."""
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    cmd = _base_cmd(url) + [
        '-f', 'mp4',
        '--output', output_template,
        url,
    ]
    _run(cmd, url)
    mp4_files = list(Path(tmp_dir).glob('*.mp4'))
    if not mp4_files:
        raise RuntimeError('Download failed: no video file produced.')
    return mp4_files[0]


def _sidecar_num(image: Path) -> int:
    """Carousel position from an image's --write-metadata JSON sidecar's 'num'
    field, or -1 if the sidecar is missing, unreadable, or 'num' doesn't parse
    as an int. Instagram filenames are media-id based (not carousel order);
    'num' is the only field contractually tied to slide position."""
    sidecar = image.parent / (image.name + '.json')
    if not sidecar.exists():
        return -1
    try:
        data = json.loads(sidecar.read_text(encoding='utf-8'))
        return int(data['num'])
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return -1


def download_slideshow(url: str, tmp_dir: str) -> Tuple[List[Path], dict]:
    """Download a slideshow post's slide images into tmp_dir via gallery-dl.

    Returns (image paths in carousel order, metadata dict from the first
    image's --write-metadata JSON sidecar). Instagram requires login cookies
    (MAVEN_IG_COOKIES); missing or rejected cookies raise RuntimeError with
    INSTAGRAM_COOKIE_MESSAGE. Non-image files (TikTok's mp3 soundtrack,
    sidecars) are filtered out. Images are ordered by their sidecar's 'num'
    field (the only field contractually tied to carousel position — Instagram
    filenames are media-id based, not order-based); images with a missing or
    unparseable sidecar fall back to lexicographic filename order, sorted
    after any image with a valid 'num'. --config-ignore keeps operator-local
    gallery-dl config files (/etc/gallery-dl.conf, ~/.config/gallery-dl/config.json)
    from silently overriding filenames/postprocessors here.
    """
    cmd = ['gallery-dl', '--config-ignore', '-D', tmp_dir, '--write-metadata']
    if _is_instagram_url(url):
        cookies = _instagram_cookies()
        if not cookies:
            raise RuntimeError(INSTAGRAM_COOKIE_MESSAGE)
        cmd += ['--cookies', cookies]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        low = stderr.casefold()
        if _is_instagram_url(url) and any(sig in low for sig in _GALLERY_DL_LOGIN_SIGNATURES):
            raise RuntimeError(INSTAGRAM_COOKIE_MESSAGE)
        msg = stderr or f'gallery-dl exited with code {result.returncode}'
        raise RuntimeError(f'Slideshow download failed: {msg}')

    unordered = [p for p in Path(tmp_dir).iterdir()
                if p.suffix.casefold() in _IMAGE_EXTENSIONS]
    if not unordered:
        raise RuntimeError('No images found in this post — it may be a video '
                           'post rather than a slideshow.')

    def sort_key(image: Path):
        num = _sidecar_num(image)
        return (num if num >= 0 else float('inf'), image.name)

    images = sorted(unordered, key=sort_key)

    metadata = {}
    sidecar = images[0].parent / (images[0].name + '.json')
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            metadata = {}
    return images, metadata


def fetch_metadata(url: str) -> dict:
    """Fetch video metadata (description, uploader, ...) without downloading."""
    cmd = _base_cmd(url) + ['--dump-json', '--skip-download', url]
    result = _run(cmd, url)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError('Metadata fetch failed: invalid JSON from yt-dlp.')
