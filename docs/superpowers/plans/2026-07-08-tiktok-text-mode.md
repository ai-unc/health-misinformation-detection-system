# TikTok Text-Extraction Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "text" mode to MAVEN's TikTok feature that extracts on-screen overlay text (frame OCR) and the video description, alongside the existing "audio" transcription mode.

**Architecture:** A new shared `tiktok.py` module holds URL validation, ffmpeg normalization, and yt-dlp helpers. `transcription.py` keeps audio-only logic; new `text_extraction.py` samples frames with ffmpeg, OCRs them with RapidOCR, and dedups overlay text into timed segments. `POST /transcribe` gains a `mode` param (default `"audio"`) with a unified response shape.

**Tech Stack:** Flask, yt-dlp (+curl_cffi impersonation), imageio-ffmpeg, faster-whisper (audio), RapidOCR (`rapidocr-onnxruntime`), plain-script tests.

**Spec:** `docs/superpowers/specs/2026-07-08-tiktok-text-mode-design.md`

## Global Constraints

- `curl_cffi >= 0.10, < 0.15` — yt-dlp rejects 0.15+ with ImportError; never loosen this pin.
- Every yt-dlp invocation MUST include `--impersonate chrome` (TikTok blocks plain requests) — always build commands via `tiktok._base_cmd()`.
- ffmpeg comes from imageio-ffmpeg via `tiktok.ensure_ffmpeg()`, which copies the version-suffixed binary (`ffmpeg-win-x86_64-v7.1.exe`) to `%TEMP%\maven_ffmpeg\ffmpeg.exe` and injects that dir into PATH. Never call `imageio_ffmpeg.get_ffmpeg_exe()` directly in app code.
- Tests are plain Python scripts (NOT pytest). Run from `maven_app/`: `python tests/<file>.py`. Each test file's `main()` wraps stdout in UTF-8: `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`.
- `from app import app` in tests loads PubMedBERT (~30-60s on cold cache) — expected, not a hang.
- Commit messages: plain descriptive sentences (match repo style, e.g. "Fix ffmpeg resolution: …"). NO `Co-Authored-By` trailer, no conventional-commit prefixes.
- Working branch: `feature/tiktok-text-extraction` (already checked out).

---

### Task 1: Shared `tiktok.py` module

Extract the TikTok plumbing from `transcription.py` into a new shared module, and add the two new helpers text mode needs (`download_video`, `fetch_metadata`). This task only CREATES the new module + its tests; `transcription.py` is rewired in Task 2.

**Files:**
- Create: `maven_app/tiktok.py`
- Test: `maven_app/tests/test_tiktok.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces (used by Tasks 2 and 4):
  - `TIKTOK_RE: re.Pattern`
  - `validate_url(url: str) -> str` — returns stripped URL or raises `ValueError("URL does not appear to be a TikTok link.")`
  - `ensure_ffmpeg() -> str` — normalized ffmpeg exe path, `''` if unavailable
  - `download_audio(url: str, tmp_dir: str) -> Path` — mp3 path, raises `RuntimeError` on failure
  - `download_video(url: str, tmp_dir: str) -> Path` — mp4 path, raises `RuntimeError` on failure
  - `fetch_metadata(url: str) -> dict` — parsed yt-dlp `--dump-json` output, raises `RuntimeError` on failure

- [ ] **Step 1: Write the failing test**

Create `maven_app/tests/test_tiktok.py`:

```python
"""
Tests for tiktok.py — URL validation, download_audio, download_video, fetch_metadata.
Run from maven_app/:  python tests/test_tiktok.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktok
from tiktok import download_audio, download_video, fetch_metadata, validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_validate_url():
    print('\n=== TEST 1: validate_url ===')

    try:
        validate_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'does not appear to be a TikTok link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')

    try:
        validate_url('')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ empty URL raises ValueError')

    assert validate_url('https://www.tiktok.com/@user/video/123456') == \
        'https://www.tiktok.com/@user/video/123456'
    print('  ✓ standard TikTok URL passes')

    assert validate_url('  https://vm.tiktok.com/ZMhAbcDef/  ') == \
        'https://vm.tiktok.com/ZMhAbcDef/'
    print('  ✓ vm.tiktok.com short URL passes and is stripped')


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

    with patch('tiktok.subprocess.run', side_effect=fake_run), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        result = download_audio('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp3'
    assert result.exists()
    print('  ✓ success path returns mp3 Path')

    # Dependency-trap guard: impersonation must always be present
    cmd = captured['cmd']
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    assert '--extract-audio' in cmd
    print('  ✓ command includes --impersonate chrome and --extract-audio')

    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        try:
            download_audio('https://www.tiktok.com/@user/video/bad', str(tmp_path))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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

    with patch('tiktok.subprocess.run', side_effect=fake_run), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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
    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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

    with patch('tiktok.subprocess.run', side_effect=fake_run), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        meta = fetch_metadata('https://www.tiktok.com/@user/video/123')
    assert meta['description'] == 'Raspberry leaf tea!'
    assert meta['uploader'] == 'healthmom'
    print('  ✓ returns parsed metadata dict')

    cmd = captured['cmd']
    assert '--dump-json' in cmd and '--skip-download' in cmd
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    print('  ✓ command includes --dump-json --skip-download --impersonate chrome')

    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='', stdout='not json')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'invalid JSON' in str(e)
            print('  ✓ invalid JSON raises RuntimeError')

    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='blocked')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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
        test_validate_url()
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

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_tiktok.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tiktok'`

- [ ] **Step 3: Write the implementation**

Create `maven_app/tiktok.py`. The `ensure_ffmpeg` body is moved VERBATIM from `transcription._ensure_ffmpeg` (only the leading underscore in the name is dropped):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run from `maven_app/`: `python tests/test_tiktok.py`
Expected: PASS — ends with `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add maven_app/tiktok.py maven_app/tests/test_tiktok.py
git commit -m "Add shared tiktok.py module: URL validation, ffmpeg normalization, yt-dlp helpers"
```

---

### Task 2: Rewire `transcription.py` onto `tiktok.py`

Slim `transcription.py` to Whisper-only logic, importing plumbing from `tiktok.py`. Update `test_transcription.py`: TEST 2 (`_download_audio`) is now covered by `test_tiktok.py` and is removed; patch targets change from removed internals to the imported names.

**Files:**
- Modify: `maven_app/transcription.py`
- Modify: `maven_app/tests/test_transcription.py`

**Interfaces:**
- Consumes (from Task 1): `tiktok.validate_url(url) -> str`, `tiktok.download_audio(url, tmp_dir) -> Path`, `tiktok.ensure_ffmpeg() -> str`
- Produces (unchanged public API, used by app.py): `transcribe_url(url: str) -> TranscriptResult`, `TranscriptResult(text, segments, duration)`, `NoSpeechError`

- [ ] **Step 1: Rewrite `transcription.py`**

Replace the entire file with (docstring, `NoSpeechError`, `TranscriptResult`, `_get_model`, `_transcribe` unchanged except imports; `_TIKTOK_RE`, `_ensure_ffmpeg`, `_download_audio` deleted — they live in `tiktok.py` now):

```python
"""
MAVEN Transcription: downloads TikTok audio and transcribes it with faster-whisper.
Public entry point: transcribe_url(url) → TranscriptResult.
"""
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tiktok import download_audio, ensure_ffmpeg, validate_url

_model = None  # lazy-loaded on first call to _get_model()


class NoSpeechError(RuntimeError):
    """Raised when transcription produces no speech output."""


@dataclass
class TranscriptResult:
    text: str
    segments: List[dict]   # [{"start": float, "end": float, "text": str}, ...]
    duration: float


def transcribe_url(url: str) -> TranscriptResult:
    url = validate_url(url)
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
```

- [ ] **Step 2: Update `test_transcription.py`**

Apply these changes (patch target `'_download_audio'` becomes `'download_audio'` — the name `transcription` now imports from `tiktok`):

1. Update the module docstring's first line to:
   `Tests for transcription.py — URL validation, _transcribe, cleanup, Flask route.`
2. Replace the import block (lines 11-14) with:

```python
import transcription
from transcription import TranscriptResult, transcribe_url
from transcription import _transcribe
```

3. In `test_url_validation` (TEST 1), replace both occurrences of
   `patch.object(transcription, '_download_audio', ...)` with
   `patch.object(transcription, 'download_audio', ...)`.
4. Delete the whole `test_download_audio` function (TEST 2) — moved to `tests/test_tiktok.py`.
5. In `test_transcribe_url_cleanup` (TEST 4), replace both occurrences of
   `patch.object(transcription, '_download_audio', ...)` with
   `patch.object(transcription, 'download_audio', ...)`.
6. In `test_flask_transcribe_route` (TEST 5), replace all three occurrences of
   `patch.object(transcription, '_download_audio', ...)` with
   `patch.object(transcription, 'download_audio', ...)`.
7. In `main()`, delete these three lines (TEST 2 is gone):

```python
        dl_dir = _tmp / 'dl_test'
        dl_dir.mkdir()
        test_download_audio(dl_dir)
```

8. Renumber the remaining `=== TEST N ===` banner comments/strings if desired is NOT required — leave numbering as-is to keep the diff minimal.

- [ ] **Step 3: Run both test files to verify they pass**

Run from `maven_app/`:
- `python tests/test_tiktok.py` — Expected: `ALL TESTS PASSED`
- `python tests/test_transcription.py` — Expected: `ALL TESTS PASSED` (TEST 5 loads PubMedBERT, ~30-60s)

- [ ] **Step 4: Commit**

```bash
git add maven_app/transcription.py maven_app/tests/test_transcription.py
git commit -m "Rewire transcription.py onto shared tiktok.py helpers"
```

---

### Task 3: Text-extraction pure functions (filtering + grouping)

The OCR post-processing brain: junk filtering, fuzzy dedup of per-frame OCR results into timed segments, and final text assembly. All pure functions — no network, no OCR engine, fully unit-testable.

**Files:**
- Create: `maven_app/text_extraction.py` (pure functions only; pipeline added in Task 4)
- Test: `maven_app/tests/test_text_extraction.py`

**Interfaces:**
- Consumes: nothing yet (pure functions).
- Produces (used by Task 4's pipeline and its tests):
  - `_is_junk(text: str, confidence: float, uploader: str = '') -> bool`
  - `_normalize(text: str) -> str` — casefolded, whitespace-collapsed
  - `group_overlay_segments(frame_results: List[dict]) -> List[dict]` — input `[{'ts': int, 'lines': [(text, conf), ...]}, ...]`, output `[{'start': float, 'end': float, 'text': str}, ...]`
  - `_assemble_text(description: str, overlay_segments: List[dict]) -> str`
  - Constants: `MIN_CONFIDENCE = 0.6`, `MIN_LINE_CHARS = 3`, `FUZZY_MATCH_RATIO = 0.9`

- [ ] **Step 1: Write the failing test**

Create `maven_app/tests/test_text_extraction.py`:

```python
"""
Tests for text_extraction.py — junk filtering, overlay grouping, text assembly.
Run from maven_app/:  python tests/test_text_extraction.py
(extract_text_url and Flask route tests are added by later tasks.)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_extraction
from text_extraction import (
    _assemble_text,
    _is_junk,
    _normalize,
    group_overlay_segments,
)


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_is_junk():
    print('\n=== TEST 1: _is_junk ===')

    assert _is_junk('Perfectly good caption', 0.3)
    print('  ✓ low-confidence line dropped')

    assert _is_junk('ab', 0.99)
    print('  ✓ line under 3 characters dropped')

    assert _is_junk('@healthmom', 0.99)
    print('  ✓ @handle line dropped')

    assert _is_junk('TikTok', 0.99)
    print('  ✓ bare "TikTok" watermark dropped')

    assert _is_junk('healthmom', 0.99, uploader='healthmom')
    print('  ✓ uploader handle without @ dropped')

    assert not _is_junk('Raspberry leaf tea induces labor', 0.95)
    print('  ✓ normal caption kept')

    assert not _is_junk('I saw this on TikTok yesterday', 0.95)
    print('  ✓ sentence merely containing "tiktok" kept')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_group_overlay_segments():
    print('\n=== TEST 2: group_overlay_segments ===')

    # Identical consecutive frames merge into one segment
    frames = [
        {'ts': 0, 'lines': [('Raspberry leaf tea', 0.9)]},
        {'ts': 1, 'lines': [('Raspberry leaf tea', 0.9)]},
        {'ts': 2, 'lines': [('Raspberry leaf tea', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert segs == [{'start': 0.0, 'end': 3.0, 'text': 'Raspberry leaf tea'}]
    print('  ✓ identical consecutive frames merge into one segment')

    # Fuzzy OCR jitter merges; highest-confidence variant wins
    frames = [
        {'ts': 0, 'lines': [('Raspberry 1eaf tea', 0.7)]},
        {'ts': 1, 'lines': [('Raspberry leaf tea', 0.95)]},
    ]
    segs = group_overlay_segments(frames)
    assert len(segs) == 1
    assert segs[0]['text'] == 'Raspberry leaf tea'
    print('  ✓ fuzzy jitter absorbed; highest-confidence text wins')

    # Empty frame splits segments (same overlay reappearing = new segment)
    frames = [
        {'ts': 0, 'lines': [('Drink this daily', 0.9)]},
        {'ts': 1, 'lines': []},
        {'ts': 2, 'lines': [('Drink this daily', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert len(segs) == 2
    assert segs[0] == {'start': 0.0, 'end': 1.0, 'text': 'Drink this daily'}
    assert segs[1] == {'start': 2.0, 'end': 3.0, 'text': 'Drink this daily'}
    print('  ✓ gap splits into two segments')

    # Different overlays become separate segments
    frames = [
        {'ts': 0, 'lines': [('Claim one', 0.9)]},
        {'ts': 1, 'lines': [('A totally different overlay', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert len(segs) == 2
    print('  ✓ different overlays produce separate segments')

    # Multi-line frames join their lines in order
    frames = [
        {'ts': 0, 'lines': [('Line one', 0.9), ('line two', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert segs[0]['text'] == 'Line one line two'
    print('  ✓ multi-line frame joins lines with a space')

    assert group_overlay_segments([]) == []
    print('  ✓ empty input produces no segments')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_assemble_text():
    print('\n=== TEST 3: _assemble_text ===')

    segs = [
        {'start': 0.0, 'end': 3.0, 'text': 'Raspberry leaf tea'},
        {'start': 5.0, 'end': 8.0, 'text': 'raspberry leaf tea'},   # dup (case)
        {'start': 9.0, 'end': 12.0, 'text': 'Avoid your doctor'},
    ]
    out = _assemble_text('My pregnancy hack! #fyp', segs)
    assert out == 'My pregnancy hack! #fyp\nRaspberry leaf tea\nAvoid your doctor'
    print('  ✓ description + unique overlay lines, duplicates removed')

    out = _assemble_text('', segs)
    assert out == 'Raspberry leaf tea\nAvoid your doctor'
    print('  ✓ empty description omitted')

    assert _assemble_text('', []) == ''
    print('  ✓ nothing found produces empty string')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_is_junk()
    test_group_overlay_segments()
    test_assemble_text()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_text_extraction.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'text_extraction'`

- [ ] **Step 3: Write the implementation**

Create `maven_app/text_extraction.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run from `maven_app/`: `python tests/test_text_extraction.py`
Expected: PASS — ends with `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add maven_app/text_extraction.py maven_app/tests/test_text_extraction.py
git commit -m "Add text_extraction pure functions: OCR junk filter, overlay grouping, text assembly"
```

---

### Task 4: Text-extraction pipeline (`extract_text_url`)

Frame sampling via the normalized ffmpeg, lazy RapidOCR engine, and the orchestrating entry point. Also adds the new dependency to requirements.

**Files:**
- Modify: `maven_app/text_extraction.py` (append pipeline functions)
- Modify: `maven_app/tests/test_text_extraction.py` (append pipeline tests)
- Modify: `maven_app/requirements.txt`

**Interfaces:**
- Consumes (Task 1): `download_video`, `ensure_ffmpeg`, `fetch_metadata`, `validate_url` (already imported in Task 3's file header). Consumes (Task 3): `group_overlay_segments`, `_is_junk`, `_assemble_text`.
- Produces (used by Task 5's route):
  - `extract_text_url(url: str) -> TextExtractionResult` — raises `ValueError` (bad URL), `NoTextFoundError` (nothing found), `RuntimeError` (download/ffmpeg failure)
  - `NoTextFoundError` with message `'No overlay text or description found in video.'`

- [ ] **Step 1: Write the failing tests**

Append to `maven_app/tests/test_text_extraction.py` (before `main()`), and add the new names to the existing `from text_extraction import (...)` block: `NoTextFoundError`, `TextExtractionResult`, `_sample_frames`, `extract_text_url`.

```python
# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_sample_frames(tmp_path):
    print('\n=== TEST 4: _sample_frames ===')

    video = tmp_path / 'video.mp4'
    video.write_bytes(b'\x00' * 100)

    def fake_run(cmd, **kwargs):
        # Last arg is the output pattern; fake ffmpeg writing three frames
        out_pattern = Path(cmd[-1])
        for i in (1, 2, 3):
            (out_pattern.parent / f'frame_{i:04d}.png').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('text_extraction.subprocess.run', side_effect=fake_run), \
         patch.object(text_extraction, 'ensure_ffmpeg', return_value=''):
        frames = _sample_frames(video, str(tmp_path))
    assert [f.name for f in frames] == ['frame_0001.png', 'frame_0002.png', 'frame_0003.png']
    print('  ✓ returns sorted frame paths')

    # ffmpeg failure → RuntimeError
    fail_dir = tmp_path / 'fail'
    fail_dir.mkdir()
    with patch('text_extraction.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='corrupt file')), \
         patch.object(text_extraction, 'ensure_ffmpeg', return_value=''):
        try:
            _sample_frames(video, str(fail_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'corrupt file' in str(e)
            print('  ✓ ffmpeg failure raises RuntimeError containing stderr')


# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_extract_text_url():
    print('\n=== TEST 5: extract_text_url ===')

    # Non-TikTok URL → ValueError
    try:
        extract_text_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'does not appear to be a TikTok link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')

    fake_meta = {'description': 'My pregnancy hack! #fyp', 'uploader': 'healthmom'}
    fake_frames = [Path('/fake/frame_0001.png'), Path('/fake/frame_0002.png')]
    fake_frame_results = [
        {'ts': 0, 'lines': [('Raspberry leaf tea', 0.9)]},
        {'ts': 1, 'lines': [('Raspberry leaf tea', 0.9)]},
    ]

    # Happy path: description + overlays composed into result
    with patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=fake_frame_results):
        result = extract_text_url('https://www.tiktok.com/@user/video/123')
    assert result.description == 'My pregnancy hack! #fyp'
    assert result.overlay_segments == [{'start': 0.0, 'end': 2.0, 'text': 'Raspberry leaf tea'}]
    assert result.text == 'My pregnancy hack! #fyp\nRaspberry leaf tea'
    print('  ✓ happy path composes description + overlay segments + text')

    # Description only (no overlays) still succeeds
    with patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=[]):
        result = extract_text_url('https://www.tiktok.com/@user/video/123')
    assert result.text == 'My pregnancy hack! #fyp'
    assert result.overlay_segments == []
    print('  ✓ description-only video succeeds')

    # Nothing at all → NoTextFoundError
    with patch.object(text_extraction, 'fetch_metadata',
                      return_value={'description': '', 'uploader': 'x'}), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=[]), \
         patch.object(text_extraction, '_ocr_frames', return_value=[]):
        try:
            extract_text_url('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected NoTextFoundError'
        except NoTextFoundError as e:
            assert str(e) == 'No overlay text or description found in video.'
            print('  ✓ empty description + no overlays raises NoTextFoundError')

    # Temp dir cleaned up on success and on failure
    with patch('text_extraction.shutil.rmtree') as mock_rmtree, \
         patch('text_extraction.tempfile.mkdtemp', return_value='/fake/tmp'), \
         patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=fake_frame_results):
        extract_text_url('https://www.tiktok.com/@user/video/123')
    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ temp dir removed after success')

    with patch('text_extraction.shutil.rmtree') as mock_rmtree, \
         patch('text_extraction.tempfile.mkdtemp', return_value='/fake/tmp'), \
         patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video',
                      side_effect=RuntimeError('Download failed: blocked')):
        try:
            extract_text_url('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'blocked' in str(e)
    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ temp dir removed even when download fails')
```

Update `main()` to call the new tests:

```python
def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    import tempfile as _tf
    import pathlib as _pl
    test_is_junk()
    test_group_overlay_segments()
    test_assemble_text()
    with _tf.TemporaryDirectory() as _td:
        test_sample_frames(_pl.Path(_td))
    test_extract_text_url()
    print('\nALL TESTS PASSED')
```

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_text_extraction.py`
Expected: FAIL with `ImportError: cannot import name '_sample_frames'`

- [ ] **Step 3: Write the implementation**

Append to `maven_app/text_extraction.py`:

```python
def extract_text_url(url: str) -> TextExtractionResult:
    url = validate_url(url)
    metadata = fetch_metadata(url)
    description = (metadata.get('description') or '').strip()
    uploader = (metadata.get('uploader') or '').strip()

    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = download_video(url, tmp_dir)
        frames = _sample_frames(video_path, tmp_dir)
        frame_results = _ocr_frames(frames, uploader)
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
    Raises RuntimeError on ffmpeg failure.
    """
    ffmpeg_exe = ensure_ffmpeg() or 'ffmpeg'
    frames_dir = Path(tmp_dir) / 'frames'
    frames_dir.mkdir(exist_ok=True)
    cmd = [
        ffmpeg_exe, '-hide_banner', '-loglevel', 'error',
        '-i', str(video_path),
        '-vf', f'fps=1,scale={FRAME_WIDTH}:-2',
        str(frames_dir / 'frame_%04d.png'),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
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


def _ocr_frames(frames: List[Path], uploader: str) -> List[dict]:
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
            if not _is_junk(text, conf, uploader):
                lines.append((text, conf))
        results.append({'ts': idx, 'lines': lines})
    return results
```

- [ ] **Step 4: Add the dependency**

In `maven_app/requirements.txt`, append after the `imageio-ffmpeg` line:

```
rapidocr-onnxruntime
```

Then install it: `pip install rapidocr-onnxruntime`
Expected: installs cleanly alongside existing deps (pulls `onnxruntime`, `opencv-python` or headless variant, `pyclipper`, `shapely`).

- [ ] **Step 5: Run tests to verify they pass**

Run from `maven_app/`: `python tests/test_text_extraction.py`
Expected: PASS — ends with `ALL TESTS PASSED`

Import smoke check (verifies RapidOCR installs and loads):
`python -c "from text_extraction import _get_ocr; _get_ocr(); print('OCR OK')"`
Expected: `[MAVEN] Loading RapidOCR engine (one-time)...` then `OCR OK`

- [ ] **Step 6: Commit**

```bash
git add maven_app/text_extraction.py maven_app/tests/test_text_extraction.py maven_app/requirements.txt
git commit -m "Add extract_text_url pipeline: frame sampling, RapidOCR, overlay segments"
```

---

### Task 5: `/transcribe` mode parameter + unified response

`mode` param (`"audio"` default, `"text"`), unified response with `text` field replacing `transcript_text`. Route tests for the new behavior; existing audio route test updated for the rename.

**Files:**
- Modify: `maven_app/app.py`
- Modify: `maven_app/tests/test_text_extraction.py` (append route test)
- Modify: `maven_app/tests/test_transcription.py` (TEST 5: `transcript_text` → `text`)

**Interfaces:**
- Consumes (Task 2): `transcribe_url`, `NoSpeechError`. Consumes (Task 4): `extract_text_url`, `NoTextFoundError`.
- Produces (consumed by Task 6's UI): JSON responses
  - audio: `{"mode": "audio", "text": str, "segments": [...], "duration": float}`
  - text: `{"mode": "text", "text": str, "segments": [...], "description": str}`
  - errors: `{"error": str}` with 400 (no URL / bad URL / bad mode), 422 (no speech / no text), 500 (other)

- [ ] **Step 1: Write the failing route test**

Append to `maven_app/tests/test_text_extraction.py` (before `main()`):

```python
# ── TEST 6 ─────────────────────────────────────────────────────────────────────

def test_flask_transcribe_modes():
    print('\n=== TEST 6: /transcribe mode handling ===')
    import json
    from app import app  # loads PubMedBERT — takes ~30-60s on cold cache

    client = app.test_client()

    # Unknown mode → 400
    r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                         'mode': 'video'})
    assert r.status_code == 400, f'Expected 400, got {r.status_code}'
    assert b'Unknown mode' in r.data
    print('  ✓ unknown mode → 400')

    # Missing mode defaults to audio
    fake_audio = MagicMock()
    fake_audio.text = 'Spoken words.'
    fake_audio.segments = [{'start': 0.0, 'end': 2.0, 'text': 'Spoken words.'}]
    fake_audio.duration = 2.0
    with patch('app.transcribe_url', return_value=fake_audio):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1'})
    assert r.status_code == 200, f'Expected 200, got {r.status_code}: {r.data}'
    body = json.loads(r.data)
    assert body['mode'] == 'audio'
    assert body['text'] == 'Spoken words.'
    assert body['duration'] == 2.0
    print('  ✓ missing mode defaults to audio with unified response shape')

    # Text mode → 200 with text-mode shape
    fake_text = TextExtractionResult(
        description='My pregnancy hack! #fyp',
        overlay_segments=[{'start': 0.0, 'end': 2.0, 'text': 'Raspberry leaf tea'}],
        text='My pregnancy hack! #fyp\nRaspberry leaf tea',
    )
    with patch('app.extract_text_url', return_value=fake_text):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                             'mode': 'text'})
    assert r.status_code == 200, f'Expected 200, got {r.status_code}: {r.data}'
    body = json.loads(r.data)
    assert body['mode'] == 'text'
    assert body['text'] == 'My pregnancy hack! #fyp\nRaspberry leaf tea'
    assert body['segments'] == [{'start': 0.0, 'end': 2.0, 'text': 'Raspberry leaf tea'}]
    assert body['description'] == 'My pregnancy hack! #fyp'
    assert 'duration' not in body
    print('  ✓ text mode → 200 with mode/text/segments/description')

    # NoTextFoundError → 422
    with patch('app.extract_text_url',
               side_effect=NoTextFoundError('No overlay text or description found in video.')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                             'mode': 'text'})
    assert r.status_code == 422, f'Expected 422, got {r.status_code}'
    print('  ✓ NoTextFoundError → 422')

    # Generic failure in text mode → 500
    with patch('app.extract_text_url', side_effect=RuntimeError('Download failed: blocked')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                             'mode': 'text'})
    assert r.status_code == 500, f'Expected 500, got {r.status_code}'
    print('  ✓ RuntimeError → 500')

    # Landing page includes the mode toggle (added in the UI task; will pass after it)
    r = client.get('/')
    assert r.status_code == 200
    print('  ✓ landing page renders')
```

Add `test_flask_transcribe_modes()` as the last call in `main()` before the final print.

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_text_extraction.py`
Expected: FAIL in TEST 6 — unknown mode currently returns 200/500 path, and `body['text']` is missing (`transcript_text` today).

- [ ] **Step 3: Update `app.py`**

Add the import near the top, next to the existing transcription import:

```python
from text_extraction import NoTextFoundError, extract_text_url
```

Replace the whole `transcribe()` route function with:

```python
@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()
    mode = (data.get('mode') or 'audio').strip().lower()

    if not url:
        return jsonify({'error': 'No URL provided.'}), 400
    if mode not in ('audio', 'text'):
        return jsonify({'error': f"Unknown mode '{mode}'. Use 'audio' or 'text'."}), 400

    try:
        if mode == 'audio':
            result = transcribe_url(url)
            payload = {
                'mode':     'audio',
                'text':     result.text,
                'segments': result.segments,
                'duration': result.duration,
            }
        else:
            result = extract_text_url(url)
            payload = {
                'mode':        'text',
                'text':        result.text,
                'segments':    result.overlay_segments,
                'description': result.description,
            }
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except (NoSpeechError, NoTextFoundError) as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify(payload)
```

- [ ] **Step 4: Update audio route assertions in `test_transcription.py`**

In `test_flask_transcribe_route` (TEST 5), replace:

```python
    assert body['transcript_text'] == 'Raspberry leaf tea is safe.'
```

with:

```python
    assert body['mode'] == 'audio'
    assert body['text'] == 'Raspberry leaf tea is safe.'
```

and update the trailing print to:

```python
    print('  ✓ valid TikTok URL → 200 with mode, text, segments, duration')
```

- [ ] **Step 5: Run all three test files to verify they pass**

Run from `maven_app/`:
- `python tests/test_tiktok.py` — Expected: `ALL TESTS PASSED`
- `python tests/test_transcription.py` — Expected: `ALL TESTS PASSED`
- `python tests/test_text_extraction.py` — Expected: `ALL TESTS PASSED`

- [ ] **Step 6: Commit**

```bash
git add maven_app/app.py maven_app/tests/test_text_extraction.py maven_app/tests/test_transcription.py
git commit -m "Add mode param to /transcribe: audio (default) or text, with unified response"
```

---

### Task 6: UI mode toggle

Segmented Audio/Text toggle in `index.html`, mode-aware button label ("Transcribe" / "Extract Text"), request includes `mode`, response handling reads the unified `text` field, reference panel handles text mode (no duration).

**Files:**
- Modify: `maven_app/templates/index.html`

**Interfaces:**
- Consumes (Task 5): `POST /transcribe {url, mode}` → `{mode, text, segments, duration?|description?}` / `{error}`.
- Produces: nothing downstream.

- [ ] **Step 1: Add the toggle markup**

In the `<!-- URL bar -->` block (around line 152), insert the toggle as the FIRST child of the `<div class="flex gap-2">`, before the `<input id="url-input" ...>`:

```html
  <div id="mode-toggle" class="flex border border-outline-variant/20" role="group" aria-label="Extraction mode">
    <button
      type="button"
      data-mode="audio"
      aria-pressed="true"
      class="mode-btn px-3 py-3 font-label font-medium uppercase tracking-widest text-[10px] transition-colors bg-surface-container-highest text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-[-2px]"
    >Audio</button>
    <button
      type="button"
      data-mode="text"
      aria-pressed="false"
      class="mode-btn px-3 py-3 font-label font-medium uppercase tracking-widest text-[10px] transition-colors text-outline hover:text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-[-2px]"
    >Text</button>
  </div>
```

- [ ] **Step 2: Add mode state + toggle behavior to the script**

After the `const transcriptSegs = ...` declaration (around line 232), add:

```js
let extractMode = 'audio';
const modeButtons = document.querySelectorAll('#mode-toggle .mode-btn');

function extractButtonLabel() {
    return extractMode === 'audio' ? 'Transcribe' : 'Extract Text';
}

modeButtons.forEach(btn => btn.addEventListener('click', () => {
    if (btn.dataset.mode === extractMode) return;
    extractMode = btn.dataset.mode;
    modeButtons.forEach(b => {
        const active = b.dataset.mode === extractMode;
        b.setAttribute('aria-pressed', String(active));
        b.classList.toggle('bg-surface-container-highest', active);
        b.classList.toggle('text-primary', active);
        b.classList.toggle('text-outline', !active);
    });
    if (!transcribeBtn.querySelector('.shimmer-bar')) {
        transcribeBtn.textContent = extractButtonLabel();
    }
}));
```

- [ ] **Step 3: Send the mode and read the unified response**

In the `transcribeBtn` click handler:
- change the fetch body line to `body: JSON.stringify({ url, mode: extractMode }),`
- replace the response handling `else if`/`else` branches:

```js
        if (data.error) {
            statusLabel.textContent = extractMode === 'audio' ? 'Transcription failed' : 'Extraction failed';
            renderTranscribeError(data.error);
        } else if (!data.text || !data.segments) {
            statusLabel.textContent = extractMode === 'audio' ? 'Transcription failed' : 'Extraction failed';
            renderTranscribeError('Unexpected server response.');
        } else {
            textarea.value = data.text;
            charCount.textContent = `Character Count: ${data.text.length}`;
            renderTranscriptPanel(data);
            statusLabel.textContent = data.mode === 'audio' ? 'Transcript ready' : 'Text extracted';
        }
```

- [ ] **Step 4: Mode-aware loading label and reference panel**

In `setTranscribeLoading`, replace the `else` branch's first two lines with:

```js
        transcribeBtn.innerHTML = extractButtonLabel();
        transcribeBtn.disabled  = !urlInput.value.trim();
```

(The "Retranscribe" label is dropped — the label is now purely mode-driven.)

Also update the initial button markup text at line 163 from `>Transcribe</button>` — no change needed (audio is the default mode), leave as is.

Replace the top of `renderTranscriptPanel` (the `mins`/`secs`/`durStr`/`transcriptMeta` lines) with:

```js
function renderTranscriptPanel(data) {
    if (!data.segments.length) {           // e.g. text mode, description only
        transcriptPanel.hidden = true;
        return;
    }
    if (data.mode === 'text') {
        transcriptMeta.textContent = `${data.segments.length} overlay segments ▾`;
    } else {
        const mins   = Math.floor(data.duration / 60);
        const secs   = Math.round(data.duration % 60);
        const durStr = `${mins}:${String(secs).padStart(2, '0')}`;
        transcriptMeta.textContent = `${durStr} · ${data.segments.length} segments ▾`;
    }
```

(the rest of the function body is unchanged).

- [ ] **Step 5: Verify**

1. Template smoke test from `maven_app/`:
   `python -c "from app import app; r = app.test_client().get('/'); assert b'mode-toggle' in r.data and b'data-mode=\"text\"' in r.data; print('template OK')"`
   Expected: `template OK` (toggle markup present with both mode buttons).
2. Manual check: `python app.py`, open `http://localhost:5000` —
   - toggle shows Audio active; clicking Text highlights it and button reads "Extract Text"
   - with a real TikTok URL in Text mode, extraction returns description+overlays into the textarea (first run downloads OCR models)
3. Re-run `python tests/test_text_extraction.py` — Expected: `ALL TESTS PASSED`

- [ ] **Step 6: Commit**

```bash
git add maven_app/templates/index.html
git commit -m "Add Audio/Text mode toggle to UI with mode-aware extract button and reference panel"
```

---

### Task 7: Notebook section

Add a "TikTok Text Extraction" section to `MAVEN_AI_UNC_SPR2026.ipynb`, mirroring the audio section's style (narrative markdown, install cell, self-contained code cell, demo cell). Insert the 4 new cells immediately AFTER the audio demo cell (currently cell index 7, the one starting `import pandas as pd` with `VIDEO_URL = "REPLACE_ME"`) and BEFORE the `## 1. Text Segmentation` markdown cell.

The notebook variant differs from the app module deliberately: it invokes the imageio-ffmpeg binary by its full path (no name normalization needed when we call the binary ourselves) and reuses the audio section's `_nb_ensure_ffmpeg()` for yt-dlp PATH setup — the markdown notes that the audio section's setup cells must run first.

**Files:**
- Modify: `MAVEN_AI_UNC_SPR2026.ipynb` (use the NotebookEdit tool to insert cells)

**Interfaces:**
- Consumes: notebook-level `_nb_ensure_ffmpeg()` (defined in the audio section) and `score_text()` (defined in pipeline section 4).
- Produces: notebook-level `extract_text_url(url) -> TextExtractionResult`.

- [ ] **Step 1: Insert markdown cell (new cell after audio demo)**

```markdown
## TikTok Text Extraction (Overlays + Description)

Extracts the text a viewer *reads* rather than hears: on-screen overlay captions and the video description. The video is downloaded with `yt-dlp` (chrome impersonation), one frame per second is sampled with ffmpeg, each frame is OCR'd with [RapidOCR](https://github.com/RapidAI/RapidOCR), and consecutive frames showing the same overlay are fuzzy-deduplicated into timed segments. The combined description + overlay text feeds `score_text()` exactly like an audio transcript.

> Run the **TikTok Audio Transcription** setup cells above first — this section reuses its ffmpeg helper.
```

- [ ] **Step 2: Insert install cell**

```python
!pip install yt-dlp "curl_cffi>=0.10,<0.15" imageio-ffmpeg rapidocr-onnxruntime -q
```

- [ ] **Step 3: Insert implementation cell**

```python
import json as _json
import re, shutil, subprocess, tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

_nb_ocr = None  # cached RapidOCR engine

MIN_CONFIDENCE    = 0.6   # OCR lines below this are noise
MIN_LINE_CHARS    = 3     # shorter lines are noise
FUZZY_MATCH_RATIO = 0.9   # frames this similar show the same overlay
FRAME_WIDTH       = 720   # frames scaled to this width before OCR


@dataclass
class TextExtractionResult:
    description: str
    overlay_segments: List[dict]  # [{"start": float, "end": float, "text": str}, ...]
    text: str                     # description + unique overlay lines → feeds score_text()


def _nb_ffmpeg_exe() -> str:
    """Full path to the imageio-ffmpeg binary (invoked directly, no rename needed)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def _normalize(text):
    return ' '.join(text.casefold().split())


def _is_junk(text, confidence, uploader=''):
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


def _same_overlay(norm_a, norm_b):
    return norm_a == norm_b or SequenceMatcher(None, norm_a, norm_b).ratio() >= FUZZY_MATCH_RATIO


def group_overlay_segments(frame_results):
    """Merge consecutive frames showing the same overlay into {start, end, text} segments."""
    segments, current = [], None
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


def _assemble_text(description, overlay_segments):
    parts = [description] if description else []
    seen = set()
    for seg in overlay_segments:
        key = _normalize(seg['text'])
        if key not in seen:
            seen.add(key)
            parts.append(seg['text'])
    return '\n'.join(parts).strip()


def _yt_dlp(args):
    """Run yt-dlp with chrome impersonation (required for TikTok). Raises on failure."""
    _nb_ensure_ffmpeg()  # from the audio section: puts ffmpeg dir on PATH for yt-dlp
    cmd = ['yt-dlp', '--no-playlist', '--quiet', '--impersonate', 'chrome'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    return result


def extract_text_url(url: str) -> TextExtractionResult:
    """Extract overlay text (frame OCR) and description from a TikTok URL."""
    if not re.match(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/', url.strip()):
        raise ValueError("URL does not appear to be a TikTok link.")
    url = url.strip()

    meta = _json.loads(_yt_dlp(['--dump-json', '--skip-download', url]).stdout)
    description = (meta.get('description') or '').strip()
    uploader = (meta.get('uploader') or '').strip()

    global _nb_ocr
    if _nb_ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        print('Loading RapidOCR engine (one-time)...')
        _nb_ocr = RapidOCR()

    tmp_dir = tempfile.mkdtemp()
    try:
        # Download video as mp4
        _yt_dlp(['-f', 'mp4', '--output', str(Path(tmp_dir) / '%(id)s.%(ext)s'), url])
        mp4_files = list(Path(tmp_dir).glob('*.mp4'))
        if not mp4_files:
            raise RuntimeError('Download failed: no video file produced.')

        # Sample one frame per second, scaled to FRAME_WIDTH
        frames_dir = Path(tmp_dir) / 'frames'
        frames_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            [_nb_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
             '-i', str(mp4_files[0]),
             '-vf', f'fps=1,scale={FRAME_WIDTH}:-2',
             str(frames_dir / 'frame_%04d.png')],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Frame sampling failed: {result.stderr.strip()}')

        # OCR each frame, junk-filtering lines
        frame_results = []
        for idx, frame in enumerate(sorted(frames_dir.glob('frame_*.png'))):
            raw, _ = _nb_ocr(str(frame))
            lines = [(item[1].strip(), float(item[2])) for item in (raw or [])
                     if not _is_junk(item[1], float(item[2]), uploader)]
            frame_results.append({'ts': idx, 'lines': lines})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    overlay_segments = group_overlay_segments(frame_results)
    text = _assemble_text(description, overlay_segments)
    if not text:
        raise RuntimeError('No overlay text or description found in video.')
    return TextExtractionResult(description=description,
                                overlay_segments=overlay_segments, text=text)
```

- [ ] **Step 4: Insert demo cell**

```python
import pandas as pd

VIDEO_URL = "REPLACE_ME"  # replace with a real TikTok URL, e.g. https://www.tiktok.com/@user/video/123456

result = extract_text_url(VIDEO_URL)

print(f"Description : {result.description}")
print(f"Overlays    : {len(result.overlay_segments)} segments")
print(f"\nCombined text:\n{result.text}\n")

# Timed overlay reference
display(pd.DataFrame(result.overlay_segments))

# Pass extracted text into MAVEN pipeline
# (run all pipeline cells below first so score_text is defined)
df = score_text(result.text)
display(df)
```

- [ ] **Step 5: Verify notebook integrity**

```bash
python -c "import json; nb = json.load(open('MAVEN_AI_UNC_SPR2026.ipynb', encoding='utf-8')); print(len(nb['cells']), 'cells'); [print(i, c['cell_type'], '|', (''.join(c['source']).strip().splitlines() or ['(empty)'])[0][:80]) for i, c in enumerate(nb['cells'])]"
```

Expected: 20 cells; indices 8-11 are the new markdown/install/implementation/demo cells; `## 1. Text Segmentation` now at index 12.

- [ ] **Step 6: Commit**

```bash
git add MAVEN_AI_UNC_SPR2026.ipynb
git commit -m "Add TikTok text extraction section to MAVEN notebook"
```

---

## Final verification (after all tasks)

From `maven_app/`:

```bash
python tests/test_tiktok.py            # ALL TESTS PASSED
python tests/test_transcription.py     # ALL TESTS PASSED
python tests/test_text_extraction.py   # ALL TESTS PASSED
python tests/test_flask_e2e.py         # pre-existing e2e must still pass
```

Optional live smoke test (network + real TikTok URL required): run `python app.py`, POST `{"url": "<real url>", "mode": "text"}` to `/transcribe`, confirm description + overlay segments in the response.
