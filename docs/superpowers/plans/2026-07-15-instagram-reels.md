# Instagram Reels Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instagram Reels URLs work in both extraction modes (Whisper audio transcription and frame-OCR text mode) exactly like TikTok URLs, via a shared `video_source.py` core with thin per-platform modules.

**Architecture:** All yt-dlp/ffmpeg plumbing moves verbatim from `tiktok.py` into a new `video_source.py`, which also defines a frozen `Platform` dataclass and `validate_url(url) -> (url, Platform)` dispatch. `tiktok.py` and a new `instagram.py` shrink to thin platform definitions (regex + OCR junk terms). Consumers (`transcription.py`, `text_extraction.py`) switch imports; flow is unchanged.

**Tech Stack:** Python 3.9, Flask, yt-dlp (`--impersonate chrome` via curl_cffi), faster-whisper, RapidOCR. Tests are plain Python scripts (NOT pytest), run from `maven_app/` as `python tests/test_x.py`.

**Spec:** `docs/superpowers/specs/2026-07-15-instagram-reels-design.md`

## Global Constraints

- Work happens on branch `feature/instagram-reels` (already checked out) in `maven_app/`.
- Python 3.9 compatibility: use `typing.Tuple`/`typing.List`, not `tuple[...]`.
- Public Instagram only — no cookie/credential handling anywhere.
- Instagram accepted URL paths: `/reel/`, `/reels/`, `/share/` (with `www.` optional). `/p/` and `/tv/` must be rejected at validation.
- Exact final error message for unsupported URLs: `URL is not a supported TikTok or Instagram Reels link.`
- Exact Instagram block message: `Instagram requires login or has rate-limited this request. Try a public Reel or retry later.`
- Exact UI placeholder: `TikTok or Instagram Reel URL (optional)`
- Every yt-dlp invocation must keep `--impersonate chrome` (existing tests guard this — do not weaken them).
- Tests are plain scripts with `main()` wrapping stdout in UTF-8; follow the existing style in `tests/test_tiktok.py` exactly (print `✓` lines, `assert` with messages, no pytest).
- Fast tests (`test_video_source.py`, `test_tiktok.py`, `test_instagram.py`) must not import `app` (importing `app` loads PubMedBERT, ~30–60 s).
- All commands below run from `maven_app/` unless a path says otherwise.

---

### Task 1: Extract `video_source.py` shared core; slim `tiktok.py`; update consumers

**Files:**
- Create: `maven_app/video_source.py`
- Create: `maven_app/tests/test_video_source.py`
- Rewrite: `maven_app/tiktok.py` (becomes thin platform definition)
- Rewrite: `maven_app/tests/test_tiktok.py` (plumbing tests move out)
- Modify: `maven_app/transcription.py:11,27-28`
- Modify: `maven_app/text_extraction.py:14,112-113`

**Interfaces:**
- Consumes: existing plumbing in `tiktok.py` (moved verbatim).
- Produces (later tasks rely on these exact names):
  - `video_source.Platform` — frozen dataclass with fields `name: str`, `display_name: str`, `url_re: re.Pattern`, `junk_terms: frozenset`
  - `video_source.validate_url(url: str) -> Tuple[str, Platform]`
  - `video_source._platforms() -> Tuple[Platform, ...]` (Task 2 adds Instagram here)
  - `video_source._run(cmd)` (Task 3 adds stderr translation here)
  - `video_source.ensure_ffmpeg()`, `download_audio(url, tmp_dir)`, `download_video(url, tmp_dir)`, `fetch_metadata(url)` — signatures unchanged from today's `tiktok.py`
  - `tiktok.TIKTOK` — the TikTok `Platform` instance

- [ ] **Step 1: Write the failing tests**

Create `maven_app/tests/test_video_source.py`. Tests 2–4 are today's `test_tiktok.py` tests 2–4 with `tiktok` → `video_source` in imports and patch targets; test 1 is new dispatch coverage.

```python
"""
Tests for video_source.py — URL dispatch, download_audio, download_video,
fetch_metadata. Run from maven_app/:  python tests/test_video_source.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_source
from video_source import (
    Platform,
    download_audio,
    download_video,
    fetch_metadata,
    validate_url,
)


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_validate_url_dispatch():
    print('\n=== TEST 1: validate_url dispatch ===')

    url, platform = validate_url('  https://www.tiktok.com/@user/video/123456  ')
    assert url == 'https://www.tiktok.com/@user/video/123456'
    assert isinstance(platform, Platform)
    assert platform.name == 'tiktok'
    print('  ✓ TikTok URL returns stripped url + tiktok Platform')

    try:
        validate_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ unsupported URL raises ValueError')

    try:
        validate_url('')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ empty URL raises ValueError')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_download_audio(tmp_path):
    print('\n=== TEST 2: download_audio ===')

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        output_tpl = cmd[cmd.index('--output') + 1]
        out_dir = Path(output_tpl).parent
        (out_dir / 'fakevideo.mp3').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('video_source.subprocess.run', side_effect=fake_run), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        result = download_audio('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp3'
    assert result.exists()
    print('  ✓ success path returns mp3 Path')

    # Dependency-trap guard: impersonation must always be present
    cmd = captured['cmd']
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    assert '--extract-audio' in cmd
    print('  ✓ command includes --impersonate chrome and --extract-audio')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            download_audio('https://www.tiktok.com/@user/video/bad', str(tmp_path))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            download_audio('https://www.tiktok.com/@user/video/empty', str(empty_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no audio file produced' in str(e)
            print('  ✓ zero-exit but no mp3 raises RuntimeError')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_download_video(tmp_path):
    print('\n=== TEST 3: download_video ===')

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        output_tpl = cmd[cmd.index('--output') + 1]
        out_dir = Path(output_tpl).parent
        (out_dir / 'fakevideo.mp4').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('video_source.subprocess.run', side_effect=fake_run), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        result = download_video('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp4'
    assert result.exists()
    print('  ✓ success path returns mp4 Path')

    cmd = captured['cmd']
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    assert '-f' in cmd and cmd[cmd.index('-f') + 1] == 'mp4'
    assert '--extract-audio' not in cmd
    print('  ✓ command includes --impersonate chrome and -f mp4, no audio extraction')

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            download_video('https://www.tiktok.com/@user/video/empty', str(empty_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no video file produced' in str(e)
            print('  ✓ zero-exit but no mp4 raises RuntimeError')


# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_fetch_metadata():
    print('\n=== TEST 4: fetch_metadata ===')

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        return MagicMock(returncode=0, stderr='',
                         stdout='{"description": "Raspberry leaf tea!", "uploader": "healthmom"}')

    with patch('video_source.subprocess.run', side_effect=fake_run), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        meta = fetch_metadata('https://www.tiktok.com/@user/video/123')
    assert meta['description'] == 'Raspberry leaf tea!'
    assert meta['uploader'] == 'healthmom'
    print('  ✓ returns parsed metadata dict')

    cmd = captured['cmd']
    assert '--dump-json' in cmd and '--skip-download' in cmd
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    print('  ✓ command includes --dump-json --skip-download --impersonate chrome')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='', stdout='not json')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'invalid JSON' in str(e)
            print('  ✓ invalid JSON raises RuntimeError')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='blocked')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'blocked' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    import tempfile as _tf
    import pathlib as _pl
    with _tf.TemporaryDirectory() as _td:
        _tmp = _pl.Path(_td)
        test_validate_url_dispatch()
        audio_dir = _tmp / 'audio'
        audio_dir.mkdir()
        test_download_audio(audio_dir)
        video_dir = _tmp / 'video'
        video_dir.mkdir()
        test_download_video(video_dir)
    test_fetch_metadata()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `python tests/test_video_source.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'video_source'`

- [ ] **Step 3: Create `video_source.py`**

The plumbing functions (`ensure_ffmpeg`, `_base_cmd`, `_run`, `download_audio`, `download_video`, `fetch_metadata`) are moved **verbatim** from today's `tiktok.py:27-131` — only two docstring words change ("TikTok audio" → "the video's audio track"; "TikTok video" → "the video"). New code is the module docstring, `Platform`, `_platforms`, and `validate_url`.

```python
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
    return (TIKTOK,)


def validate_url(url: str) -> Tuple[str, Platform]:
    """Return (stripped URL, matching Platform), or raise ValueError."""
    url = url.strip()
    for platform in _platforms():
        if platform.url_re.match(url):
            return url, platform
    raise ValueError('URL does not appear to be a TikTok link.')


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        msg = result.stderr.strip() or f'yt-dlp exited with code {result.returncode}'
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
```

- [ ] **Step 4: Rewrite `tiktok.py` as the thin platform definition**

Replace the entire file with:

```python
"""
TikTok platform definition for MAVEN video ingestion.
Shared download/metadata plumbing lives in video_source.py.
"""
import re

from video_source import Platform

TIKTOK = Platform(
    name='tiktok',
    display_name='TikTok',
    url_re=re.compile(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/'),
    junk_terms=frozenset({'tiktok'}),
)
```

- [ ] **Step 5: Update consumers' imports**

In `maven_app/transcription.py` change line 11:

```python
from tiktok import download_audio, ensure_ffmpeg, validate_url
```

to:

```python
from video_source import download_audio, ensure_ffmpeg, validate_url
```

and in `transcribe_url` change:

```python
    url = validate_url(url)
```

to:

```python
    url, _platform = validate_url(url)
```

In `maven_app/text_extraction.py` change line 14:

```python
from tiktok import download_video, ensure_ffmpeg, fetch_metadata, validate_url
```

to:

```python
from video_source import download_video, ensure_ffmpeg, fetch_metadata, validate_url
```

and in `extract_text_url` change:

```python
    url = validate_url(url)
```

to:

```python
    url, platform = validate_url(url)
```

(`platform` is intentionally unused until Task 4 threads it into the OCR junk filter.)

- [ ] **Step 6: Rewrite `tests/test_tiktok.py` for the thin module**

Replace the entire file with:

```python
"""
Tests for tiktok.py — the thin TikTok platform definition.
Plumbing tests live in tests/test_video_source.py.
Run from maven_app/:  python tests/test_tiktok.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiktok import TIKTOK
from video_source import validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_platform_definition():
    print('\n=== TEST 1: TIKTOK platform definition ===')

    assert TIKTOK.name == 'tiktok'
    assert TIKTOK.display_name == 'TikTok'
    assert 'tiktok' in TIKTOK.junk_terms
    print('  ✓ name, display_name, junk_terms as expected')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_url_matching():
    print('\n=== TEST 2: TikTok URL matching ===')

    assert TIKTOK.url_re.match('https://www.tiktok.com/@user/video/123456')
    print('  ✓ standard TikTok URL matches')

    assert TIKTOK.url_re.match('https://vm.tiktok.com/ZMhAbcDef/')
    print('  ✓ vm.tiktok.com short URL matches')

    assert not TIKTOK.url_re.match('https://www.youtube.com/watch?v=abc123')
    print('  ✓ non-TikTok URL does not match')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_dispatch():
    print('\n=== TEST 3: validate_url dispatches to TIKTOK ===')

    url, platform = validate_url('  https://vm.tiktok.com/ZMhAbcDef/  ')
    assert url == 'https://vm.tiktok.com/ZMhAbcDef/'
    assert platform is TIKTOK
    print('  ✓ TikTok URL stripped and dispatched to TIKTOK platform')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_platform_definition()
    test_url_matching()
    test_dispatch()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 7: Run the fast tests**

Run: `python tests/test_video_source.py && python tests/test_tiktok.py`
Expected: both print `ALL TESTS PASSED`

- [ ] **Step 8: Run the consumer test suites (slow — imports app / PubMedBERT)**

Run: `python tests/test_transcription.py && python tests/test_text_extraction.py`
Expected: both print `ALL TESTS PASSED` with no edits needed — they patch names on the consumer modules (`patch.object(transcription, 'download_audio')`), which survive the import move, and the Task-1 `validate_url` keeps the old error message `URL does not appear to be a TikTok link.` that their assertions check.

- [ ] **Step 9: Commit**

```bash
git add video_source.py tiktok.py transcription.py text_extraction.py tests/test_video_source.py tests/test_tiktok.py
git commit -m "refactor: extract shared video-source plumbing from tiktok.py

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Instagram platform definition + dispatch

**Files:**
- Create: `maven_app/instagram.py`
- Create: `maven_app/tests/test_instagram.py`
- Modify: `maven_app/video_source.py` (`_platforms`, `validate_url` message)
- Modify: `maven_app/tests/test_transcription.py:26,141` (message assertions)
- Modify: `maven_app/tests/test_text_extraction.py:213` (message assertion)

**Interfaces:**
- Consumes: `video_source.Platform`, `video_source._platforms()`, `video_source.validate_url` from Task 1.
- Produces: `instagram.INSTAGRAM` — the Instagram `Platform` instance with `name='instagram'`, `junk_terms=frozenset({'instagram', 'reels', 'reel'})`. Final `validate_url` error message: `URL is not a supported TikTok or Instagram Reels link.` (Tasks 3–5 rely on this message and on `INSTAGRAM`.)

- [ ] **Step 1: Write the failing tests**

Create `maven_app/tests/test_instagram.py` (mirrors `test_tiktok.py`):

```python
"""
Tests for instagram.py — the thin Instagram platform definition.
Plumbing tests live in tests/test_video_source.py.
Run from maven_app/:  python tests/test_instagram.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instagram import INSTAGRAM
from video_source import validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_platform_definition():
    print('\n=== TEST 1: INSTAGRAM platform definition ===')

    assert INSTAGRAM.name == 'instagram'
    assert INSTAGRAM.display_name == 'Instagram'
    assert INSTAGRAM.junk_terms == frozenset({'instagram', 'reels', 'reel'})
    print('  ✓ name, display_name, junk_terms as expected')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_url_matching():
    print('\n=== TEST 2: Instagram URL matching ===')

    assert INSTAGRAM.url_re.match('https://www.instagram.com/reel/C8abcDEfGhi/')
    print('  ✓ /reel/ URL matches')

    assert INSTAGRAM.url_re.match('https://www.instagram.com/reels/C8abcDEfGhi/')
    print('  ✓ /reels/ URL matches')

    assert INSTAGRAM.url_re.match('https://instagram.com/reel/C8abcDEfGhi/')
    print('  ✓ URL without www matches')

    assert INSTAGRAM.url_re.match('https://www.instagram.com/share/BAxyz123/')
    print('  ✓ /share/ redirect URL matches')

    assert not INSTAGRAM.url_re.match('https://www.instagram.com/p/C8abcDEfGhi/')
    print('  ✓ /p/ photo-post URL rejected')

    assert not INSTAGRAM.url_re.match('https://www.instagram.com/tv/C8abcDEfGhi/')
    print('  ✓ /tv/ IGTV URL rejected')

    assert not INSTAGRAM.url_re.match('https://www.instagram.com/some_user/')
    print('  ✓ profile URL rejected')

    assert not INSTAGRAM.url_re.match('https://www.tiktok.com/@user/video/123')
    print('  ✓ TikTok URL does not match Instagram')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_dispatch():
    print('\n=== TEST 3: validate_url dispatch + error message ===')

    url, platform = validate_url('  https://www.instagram.com/reel/C8abcDEfGhi/  ')
    assert url == 'https://www.instagram.com/reel/C8abcDEfGhi/'
    assert platform is INSTAGRAM
    print('  ✓ Reel URL stripped and dispatched to INSTAGRAM platform')

    try:
        validate_url('https://www.instagram.com/p/C8abcDEfGhi/')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert str(e) == 'URL is not a supported TikTok or Instagram Reels link.'
        print('  ✓ /p/ URL raises ValueError naming both platforms')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_platform_definition()
    test_url_matching()
    test_dispatch()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python tests/test_instagram.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'instagram'`

- [ ] **Step 3: Create `instagram.py`**

```python
"""
Instagram Reels platform definition for MAVEN video ingestion.
Shared download/metadata plumbing lives in video_source.py.
Public Reels only — login-walled content is surfaced as a friendly error
by video_source._run; there is no cookie/credential handling.
"""
import re

from video_source import Platform

INSTAGRAM = Platform(
    name='instagram',
    display_name='Instagram',
    # /reel/, /reels/, and /share/ redirect links only; /p/ photo posts and
    # /tv/ never reach yt-dlp — they fail validation with the standard error.
    url_re=re.compile(r'https?://(www\.)?instagram\.com/(reels?|share)/'),
    junk_terms=frozenset({'instagram', 'reels', 'reel'}),
)
```

- [ ] **Step 4: Register the platform and finalize the error message**

In `maven_app/video_source.py` replace `_platforms` and the `validate_url` raise:

```python
def _platforms() -> Tuple[Platform, ...]:
    # Imported lazily: platform modules import Platform from here, so a
    # top-level import would be circular.
    from tiktok import TIKTOK
    from instagram import INSTAGRAM
    return (TIKTOK, INSTAGRAM)
```

```python
    raise ValueError('URL is not a supported TikTok or Instagram Reels link.')
```

- [ ] **Step 5: Update the three stale message assertions**

In `maven_app/tests/test_transcription.py`, `test_url_validation` (~line 26):

```python
        assert 'not a supported TikTok or Instagram Reels link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')
```

In `maven_app/tests/test_transcription.py`, `test_flask_transcribe_route` (~line 141):

```python
    assert b'not a supported TikTok or Instagram Reels link' in r.data
    print('  ✓ non-TikTok URL → 400')
```

In `maven_app/tests/test_text_extraction.py`, `test_extract_text_url` (~line 213):

```python
        assert 'not a supported TikTok or Instagram Reels link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')
```

- [ ] **Step 6: Run the fast tests**

Run: `python tests/test_instagram.py && python tests/test_tiktok.py && python tests/test_video_source.py`
Expected: all three print `ALL TESTS PASSED`

- [ ] **Step 7: Run the consumer suites (slow)**

Run: `python tests/test_transcription.py && python tests/test_text_extraction.py`
Expected: both print `ALL TESTS PASSED`

- [ ] **Step 8: Commit**

```bash
git add instagram.py video_source.py tests/test_instagram.py tests/test_transcription.py tests/test_text_extraction.py
git commit -m "feat: add Instagram Reels platform and URL dispatch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Friendly errors for Instagram login-wall / rate-limit

**Files:**
- Modify: `maven_app/video_source.py` (`_run` + two module constants)
- Modify: `maven_app/tests/test_video_source.py` (append TEST 5 + main() call)

**Interfaces:**
- Consumes: `video_source._run` from Task 1, `fetch_metadata` for exercising the path.
- Produces: `video_source.INSTAGRAM_BLOCK_MESSAGE` (exact string below) and module-private `_INSTAGRAM_BLOCK_SIGNATURES`. No signature changes — errors still surface as `RuntimeError`, so `app.py` keeps returning 500 with the message.

- [ ] **Step 1: Write the failing test**

Append to `maven_app/tests/test_video_source.py` before the `# ── MAIN` block, and add a `test_instagram_block_translation()` call to `main()` after `test_fetch_metadata()`. The constant is imported inside the test function so the file still imports cleanly before Step 3 exists:

```python
# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_instagram_block_translation():
    print('\n=== TEST 5: Instagram login-wall/rate-limit translation ===')

    from video_source import INSTAGRAM_BLOCK_MESSAGE

    block_stderrs = [
        'ERROR: [Instagram] C8abc: login required (use --cookies to provide account credentials)',
        'ERROR: [Instagram] C8abc: Instagram API is not granting access: rate-limit reached',
        'ERROR: [Instagram] C8abc: Restricted Video: You must be 18 years old or over',
        'ERROR: [Instagram] C8abc: Requested content is not available',
    ]
    for stderr in block_stderrs:
        with patch('video_source.subprocess.run',
                   return_value=MagicMock(returncode=1, stderr=stderr)), \
             patch.object(video_source, 'ensure_ffmpeg', return_value=''):
            try:
                fetch_metadata('https://www.instagram.com/reel/C8abc/')
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_BLOCK_MESSAGE, f'unexpected: {e}'
    print('  ✓ all four block signatures translated to the friendly message')

    # Non-block failures keep stderr passthrough behavior
    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.instagram.com/reel/C8abc/')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ other failures still pass stderr through')
```

- [ ] **Step 2: Run to verify failure**

Run: `python tests/test_video_source.py`
Expected: FAIL with `ImportError: cannot import name 'INSTAGRAM_BLOCK_MESSAGE'`

- [ ] **Step 3: Implement the translation**

In `maven_app/video_source.py`, add below the `_ffmpeg_exe` global:

```python
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
```

and replace `_run` with:

```python
def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        low = stderr.casefold()
        if any(sig in low for sig in _INSTAGRAM_BLOCK_SIGNATURES):
            raise RuntimeError(INSTAGRAM_BLOCK_MESSAGE)
        msg = stderr or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `python tests/test_video_source.py`
Expected: `ALL TESTS PASSED` (including TEST 5)

- [ ] **Step 5: Commit**

```bash
git add video_source.py tests/test_video_source.py
git commit -m "feat: translate Instagram login-wall errors to a friendly message

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Platform-aware OCR watermark filtering

**Files:**
- Modify: `maven_app/text_extraction.py` (`_is_junk`, `_ocr_frames`, `extract_text_url`)
- Modify: `maven_app/tests/test_text_extraction.py` (`test_is_junk`, `test_ocr_frames`, `test_extract_text_url`)

**Interfaces:**
- Consumes: `Platform.junk_terms` (Tasks 1–2); `validate_url` tuple return (Task 1).
- Produces: `_is_junk(text, confidence, uploader='', junk_terms=frozenset())` and `_ocr_frames(frames, uploader, junk_terms=frozenset())`. Public API `extract_text_url(url)` unchanged.

- [ ] **Step 1: Update the tests (failing first)**

In `maven_app/tests/test_text_extraction.py` replace `test_is_junk` with:

```python
def test_is_junk():
    print('\n=== TEST 1: _is_junk ===')

    TIKTOK_TERMS    = frozenset({'tiktok'})
    INSTAGRAM_TERMS = frozenset({'instagram', 'reels', 'reel'})

    assert _is_junk('Perfectly good caption', 0.3)
    print('  ✓ low-confidence line dropped')

    assert _is_junk('ab', 0.99)
    print('  ✓ line under 3 characters dropped')

    assert _is_junk('@healthmom', 0.99)
    print('  ✓ @handle line dropped')

    assert _is_junk('TikTok', 0.99, junk_terms=TIKTOK_TERMS)
    print('  ✓ bare "TikTok" watermark dropped with TikTok junk terms')

    assert not _is_junk('TikTok', 0.99, junk_terms=INSTAGRAM_TERMS)
    print('  ✓ junk terms are platform-specific, not global')

    assert _is_junk('Instagram', 0.99, junk_terms=INSTAGRAM_TERMS)
    assert _is_junk('Reels', 0.99, junk_terms=INSTAGRAM_TERMS)
    print('  ✓ bare "Instagram"/"Reels" watermarks dropped with Instagram junk terms')

    assert _is_junk('healthmom', 0.99, uploader='healthmom')
    print('  ✓ uploader handle without @ dropped')

    assert not _is_junk('Raspberry leaf tea induces labor', 0.95, junk_terms=TIKTOK_TERMS)
    print('  ✓ normal caption kept')

    assert not _is_junk('I saw this on TikTok yesterday', 0.95, junk_terms=TIKTOK_TERMS)
    print('  ✓ sentence merely containing "tiktok" kept')

    assert not _is_junk('I saw this on Instagram yesterday', 0.95, junk_terms=INSTAGRAM_TERMS)
    print('  ✓ sentence merely containing "instagram" kept')
```

In `test_ocr_frames`, change the engine-call line and add an Instagram-terms case at the end of the function:

```python
    with patch.object(text_extraction, '_get_ocr', return_value=fake_engine):
        results = text_extraction._ocr_frames(frames, 'healthmom',
                                              frozenset({'tiktok'}))
```

```python
    # Instagram junk terms filter the Instagram watermark line
    ig_responses = {
        '/fake/frame_0001.png': ([[box, 'Instagram', 0.99],
                                  [box, 'Castor oil starts labor', 0.95]], 0.1),
    }

    def fake_ig_engine(path):
        return ig_responses[path.replace('\\', '/')]

    with patch.object(text_extraction, '_get_ocr', return_value=fake_ig_engine):
        results = text_extraction._ocr_frames([Path('/fake/frame_0001.png')],
                                              'reelmom',
                                              frozenset({'instagram', 'reels', 'reel'}))
    assert results == [{'ts': 0, 'lines': [('Castor oil starts labor', 0.95)]}]
    print('  ✓ Instagram watermark filtered via platform junk terms')
```

In `test_extract_text_url`, append an Instagram happy-path case at the end of the function (reuses `fake_meta`, `fake_frames`, `fake_frame_results` defined earlier in the function):

```python
    # Instagram Reel URL goes through the same pipeline
    with patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=fake_frame_results):
        result = extract_text_url('https://www.instagram.com/reel/C8abcDEfGhi/')
    assert result.text == 'My pregnancy hack! #fyp\nRaspberry leaf tea'
    print('  ✓ Instagram Reel URL passes validation and composes result')
```

- [ ] **Step 2: Run to verify failure**

Run: `python tests/test_text_extraction.py`
Expected: FAIL — `test_is_junk` asserts `not _is_junk('TikTok', 0.99, junk_terms=INSTAGRAM_TERMS)` but the current implementation drops any bare `tiktok` unconditionally (and `_is_junk`/`_ocr_frames` don't accept `junk_terms` yet, so a `TypeError` on the keyword argument is also an acceptable failure signature).

- [ ] **Step 3: Implement platform-aware filtering**

In `maven_app/text_extraction.py` replace `_is_junk` with:

```python
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
```

Replace `_ocr_frames`'s signature and its `_is_junk` call:

```python
def _ocr_frames(frames: List[Path], uploader: str,
                junk_terms: frozenset = frozenset()) -> List[dict]:
```

```python
            if not _is_junk(text, conf, uploader, junk_terms):
```

In `extract_text_url`, pass the platform's terms through:

```python
        frame_results = _ocr_frames(frames, uploader, platform.junk_terms)
```

- [ ] **Step 4: Run to verify pass (slow — imports app)**

Run: `python tests/test_text_extraction.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add text_extraction.py tests/test_text_extraction.py
git commit -m "feat: platform-aware OCR watermark filtering

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: UI placeholder, docs, full verification

**Files:**
- Modify: `maven_app/templates/index.html:171` (placeholder)
- Modify: `maven_app/tests/test_text_extraction.py` (landing-page assertion in `test_flask_transcribe_modes`)
- Modify: `CLAUDE.md` (Flask App Structure section)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: user-visible copy only; no code interfaces.

- [ ] **Step 1: Write the failing assertion**

In `maven_app/tests/test_text_extraction.py`, `test_flask_transcribe_modes`, replace the landing-page check at the end:

```python
    # Landing page advertises both platforms in the URL input
    r = client.get('/')
    assert r.status_code == 200
    assert b'TikTok or Instagram Reel URL (optional)' in r.data
    print('  ✓ landing page renders with both-platform placeholder')
```

- [ ] **Step 2: Run to verify failure (slow — imports app)**

Run: `python tests/test_text_extraction.py`
Expected: FAIL on the new placeholder assertion (old placeholder is `TikTok URL (optional)`)

- [ ] **Step 3: Update the placeholder**

In `maven_app/templates/index.html` line 171, change:

```html
    placeholder="TikTok URL (optional)"
```

to:

```html
    placeholder="TikTok or Instagram Reel URL (optional)"
```

- [ ] **Step 4: Update CLAUDE.md structure section**

In the `## Flask App Structure` block of `CLAUDE.md`, replace the `tiktok.py` line with:

```
  video_source.py     # Shared plumbing: URL dispatch, ffmpeg, yt-dlp helpers
  tiktok.py           # Thin TikTok platform definition
  instagram.py        # Thin Instagram Reels platform definition
```

and in the two module-description lines below it (`transcription.py`, `text_extraction.py`) leave text unchanged. Also update the tree's comment for `tests/` only if it still says "End-to-end and calibration tests" (leave as-is).

- [ ] **Step 5: Run the complete suite**

Run (from `maven_app/`):

```bash
python tests/test_video_source.py && \
python tests/test_tiktok.py && \
python tests/test_instagram.py && \
python tests/test_transcription.py && \
python tests/test_text_extraction.py && \
python tests/test_flask_e2e.py
```

Expected: every script prints `ALL TESTS PASSED` (or `ALL FLASK E2E TESTS PASSED`).

- [ ] **Step 6: Manual smoke test (network-dependent, best-effort)**

Start the app (`python app.py`), then from another shell:

```bash
curl -s -X POST http://localhost:5000/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://www.instagram.com/p/C8abcDEfGhi/", "mode": "audio"}'
```

Expected: 400 with `URL is not a supported TikTok or Instagram Reels link.` (validation rejects `/p/` without touching the network). If a public Reel URL is available, also verify a real `/reel/` URL returns either a transcript or the friendly Instagram block message — never raw yt-dlp stderr.

- [ ] **Step 7: Commit**

```bash
git add templates/index.html tests/test_text_extraction.py ../CLAUDE.md
git commit -m "feat: advertise Instagram Reels in UI placeholder and docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
