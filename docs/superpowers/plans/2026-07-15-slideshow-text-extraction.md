# Slideshow Text Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend MAVEN's text-extraction mode to TikTok `/photo/` and Instagram `/p/` slideshow posts: OCR every slide with the existing RapidOCR machinery, combine with the caption, feed `score_text()`.

**Architecture:** gallery-dl (new dependency) downloads slide images + metadata for both platforms (yt-dlp cannot — verified). A new `slideshow.py` module reuses `text_extraction.py`'s OCR/junk-filter/assembly helpers; `extract_text_url()` dispatches to it for slideshow URLs. Audio mode rejects slideshow URLs with a friendly 400. Instagram requires login cookies supplied via the `MAVEN_IG_COOKIES` env var (path to a Netscape `cookies.txt`).

**Tech Stack:** Python 3.10+, Flask, gallery-dl (new), yt-dlp, rapidocr-onnxruntime.

**Spec:** `docs/superpowers/specs/2026-07-15-slideshow-text-extraction-design.md`

## Global Constraints

- Work happens on branch `feat/slideshow-text-extraction` (already created).
- Tests are **plain Python scripts, not pytest** — run from `maven_app/` as `python tests/test_<name>.py`; each has a `main()` that wraps stdout in UTF-8 (`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`) and prints `ALL TESTS PASSED`.
- Do NOT import `app` in new unit tests — `from app import app` triggers a 30-60s PubMedBERT model load.
- All subprocess calls use `timeout=600`.
- The URL-validation error message changes from `'URL is not a supported TikTok or Instagram Reels link.'` to `'URL is not a supported TikTok or Instagram link.'` (slideshows aren't Reels). Four existing assertions reference it; Task 1 updates them.
- Audio-mode rejection message, exact copy: `'Slideshow posts are supported in Text mode only.'`
- Instagram cookie message, exact copy: `'Instagram slideshows require login cookies. Export a cookies.txt for instagram.com and set MAVEN_IG_COOKIES to its path.'`
- Verified ground truth (2026-07-15, gallery-dl 1.32.6): TikTok slideshow downloads anonymously as `<id>_NN <title> [hash].jpg` files plus one `.mp3`; `--write-metadata` writes `<filename>.json` sidecars; TikTok caption is in sidecar key `desc`, uploader in `author.uniqueId`; Instagram caption key is `description`, uploader `username`. Cookieless Instagram: exit code 4, stderr `[instagram][error] HTTP redirect to login page (...)`.

---

### Task 1: Slideshow URL detection + routing regexes

**Files:**
- Modify: `maven_app/video_source.py:32-55` (Platform dataclass, validate_url)
- Modify: `maven_app/tiktok.py`
- Modify: `maven_app/instagram.py`
- Test: `maven_app/tests/test_slideshow.py` (create)
- Modify: `maven_app/tests/test_instagram.py:43-44,66-71`, `maven_app/tests/test_transcription.py:26,141`, `maven_app/tests/test_text_extraction.py:243`

**Interfaces:**
- Produces: `Platform.slideshow_url_re: re.Pattern` field (required, positioned after `url_re`); `Platform.is_slideshow(url: str) -> bool` method. `validate_url` error text becomes `'URL is not a supported TikTok or Instagram link.'`. Instagram `url_re` now accepts `/p/` links.

- [ ] **Step 1: Write the failing test file**

Create `maven_app/tests/test_slideshow.py`:

```python
"""
Tests for slideshow.py and the slideshow plumbing in video_source.py —
TikTok /photo/ and Instagram /p/ slideshow-post text extraction.
Run from maven_app/:  python tests/test_slideshow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiktok import TIKTOK
from instagram import INSTAGRAM
from video_source import validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_slideshow_url_detection():
    print('\n=== TEST 1: slideshow URL detection ===')

    assert TIKTOK.is_slideshow('https://www.tiktok.com/@glowingwithgracee/photo/7618717295417756941')
    print('  ✓ TikTok /photo/ URL detected as slideshow')

    assert not TIKTOK.is_slideshow('https://www.tiktok.com/@user/video/123')
    print('  ✓ TikTok /video/ URL not a slideshow')

    assert INSTAGRAM.is_slideshow('https://www.instagram.com/p/DRzdgElEf3N/')
    assert INSTAGRAM.is_slideshow('https://instagram.com/p/DRzdgElEf3N/')
    print('  ✓ Instagram /p/ URLs detected as slideshow (with and without www)')

    assert not INSTAGRAM.is_slideshow('https://www.instagram.com/reel/C8abcDEfGhi/')
    assert not INSTAGRAM.is_slideshow('https://www.instagram.com/share/BAxyz123/')
    print('  ✓ Instagram /reel/ and /share/ URLs not slideshows')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_slideshow_urls_validate():
    print('\n=== TEST 2: slideshow URLs pass validate_url ===')

    url, platform = validate_url('  https://www.instagram.com/p/DRzdgElEf3N/  ')
    assert url == 'https://www.instagram.com/p/DRzdgElEf3N/'
    assert platform is INSTAGRAM
    print('  ✓ Instagram /p/ URL validates and dispatches to INSTAGRAM')

    url, platform = validate_url('https://www.tiktok.com/@drtosinofficial/photo/7648626696140229910')
    assert platform is TIKTOK
    print('  ✓ TikTok /photo/ URL validates and dispatches to TIKTOK')

    try:
        validate_url('https://www.instagram.com/tv/C8abcDEfGhi/')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert str(e) == 'URL is not a supported TikTok or Instagram link.'
        print('  ✓ unsupported URL raises updated error message')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_slideshow_url_detection()
    test_slideshow_urls_validate()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd maven_app && python tests/test_slideshow.py`
Expected: FAIL — `AttributeError: 'Platform' object has no attribute 'is_slideshow'` (or TypeError constructing Platform once the field is added but definitions aren't updated).

- [ ] **Step 3: Implement**

In `maven_app/video_source.py`, replace the `Platform` dataclass:

```python
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
```

In `validate_url`, change the raise line to:

```python
    raise ValueError('URL is not a supported TikTok or Instagram link.')
```

Replace `maven_app/tiktok.py`'s definition:

```python
TIKTOK = Platform(
    name='tiktok',
    display_name='TikTok',
    url_re=re.compile(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/'),
    slideshow_url_re=re.compile(r'https?://(www\.)?tiktok\.com/@[^/]+/photo/'),
    junk_terms=frozenset({'tiktok'}),
)
```

Replace `maven_app/instagram.py`'s definition and comment:

```python
INSTAGRAM = Platform(
    name='instagram',
    display_name='Instagram',
    # /reel/, /reels/, and /share/ video links plus /p/ slideshow posts;
    # /tv/ never reaches yt-dlp — it fails validation with the standard error.
    url_re=re.compile(r'https?://(www\.)?instagram\.com/(reels?|share|p)/'),
    slideshow_url_re=re.compile(r'https?://(www\.)?instagram\.com/p/'),
    junk_terms=frozenset({'instagram', 'reels', 'reel'}),
)
```

Also update `instagram.py`'s module docstring (it claims `/p/` posts never reach yt-dlp and that there is no cookie handling):

```python
"""
Instagram platform definition for MAVEN video ingestion.
Shared download/metadata plumbing lives in video_source.py.
Public Reels work anonymously; /p/ slideshow posts additionally require
login cookies via the MAVEN_IG_COOKIES env var (see video_source.py).
"""
```

- [ ] **Step 4: Update the four stale assertions in existing tests**

`tests/test_instagram.py` — TEST 2, replace lines 43-44:

```python
    assert INSTAGRAM.url_re.match('https://www.instagram.com/p/C8abcDEfGhi/')
    print('  ✓ /p/ slideshow-post URL matches')
```

`tests/test_instagram.py` — TEST 3, replace the try/except block (lines 66-71):

```python
    try:
        validate_url('https://www.instagram.com/tv/C8abcDEfGhi/')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert str(e) == 'URL is not a supported TikTok or Instagram link.'
        print('  ✓ /tv/ URL raises ValueError naming both platforms')
```

`tests/test_transcription.py:26` and `tests/test_text_extraction.py:243` — change the expected substring:

```python
        assert 'not a supported TikTok or Instagram link' in str(e)
```

`tests/test_transcription.py:141` — change the expected bytes:

```python
    assert b'not a supported TikTok or Instagram link' in r.data
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd maven_app && python tests/test_slideshow.py && python tests/test_instagram.py && python tests/test_tiktok.py && python tests/test_video_source.py`
Expected: each prints `ALL TESTS PASSED`. (`test_transcription.py` / `test_text_extraction.py` load heavy models; they're run in Task 6.)

- [ ] **Step 6: Commit**

```bash
git add maven_app/video_source.py maven_app/tiktok.py maven_app/instagram.py maven_app/tests/
git commit -m "feat: detect TikTok /photo/ and Instagram /p/ slideshow URLs"
```

---

### Task 2: gallery-dl plumbing — download_slideshow + Instagram cookies

**Files:**
- Modify: `maven_app/video_source.py` (new constants/helpers; `_base_cmd` gains a `url` parameter; three call sites)
- Modify: `maven_app/requirements.txt`
- Test: `maven_app/tests/test_slideshow.py` (extend)

**Interfaces:**
- Consumes: `_is_instagram_url(url)` (exists), `Platform` from Task 1.
- Produces: `download_slideshow(url: str, tmp_dir: str) -> Tuple[List[Path], dict]` — ordered slide image paths + metadata dict from the first image's JSON sidecar; raises `RuntimeError` on failure. Constants `MAVEN_IG_COOKIES_ENV = 'MAVEN_IG_COOKIES'`, `INSTAGRAM_COOKIE_MESSAGE` (exact copy in Global Constraints). `_base_cmd(url)` now takes the target URL.

- [ ] **Step 1: Add gallery-dl to requirements**

In `maven_app/requirements.txt`, add the line:

```
gallery-dl>=1.32
```

Run: `pip install gallery-dl>=1.32` (into the project venv) and verify with `gallery-dl --version` → `1.32.6` or newer.

- [ ] **Step 2: Write the failing tests**

Append to `maven_app/tests/test_slideshow.py` (before `main()`), and add the new imports at the top of the file:

```python
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import video_source
from video_source import INSTAGRAM_COOKIE_MESSAGE, download_slideshow
```

```python
# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_instagram_slideshow_requires_cookies():
    print('\n=== TEST 3: Instagram slideshow without cookies fails fast ===')

    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tempfile.mkdtemp())
        assert False, 'Expected RuntimeError'
    except RuntimeError as e:
        assert str(e) == INSTAGRAM_COOKIE_MESSAGE
        print('  ✓ missing MAVEN_IG_COOKIES raises the friendly cookie message')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved


# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_login_redirect_maps_to_cookie_message():
    print('\n=== TEST 4: gallery-dl login redirect maps to cookie message ===')

    tmp = tempfile.mkdtemp()
    cookie_file = Path(tmp) / 'cookies.txt'
    cookie_file.write_text('# Netscape HTTP Cookie File\n', encoding='utf-8')

    saved_env = os.environ.get('MAVEN_IG_COOKIES')
    os.environ['MAVEN_IG_COOKIES'] = str(cookie_file)
    fake = MagicMock(returncode=4, stdout='',
                     stderr='[instagram][error] HTTP redirect to login page '
                            '(https://www.instagram.com/accounts/login/)')
    try:
        with patch('video_source.subprocess.run', return_value=fake):
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_COOKIE_MESSAGE
                print('  ✓ stale/rejected cookies map to the friendly cookie message')
    finally:
        if saved_env is None:
            os.environ.pop('MAVEN_IG_COOKIES', None)
        else:
            os.environ['MAVEN_IG_COOKIES'] = saved_env


# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_download_slideshow_filters_and_orders():
    print('\n=== TEST 5: download_slideshow filters non-images, orders slides, reads sidecar ===')

    tmp = tempfile.mkdtemp()
    # Simulate gallery-dl output layout (verified 2026-07-15): numbered jpgs,
    # one mp3 soundtrack, one .json sidecar per file.
    names = ['777_01 caption [aa].jpg', '777_02 caption [bb].jpg',
             '777_10 caption [cc].jpg', '777 caption [dd].mp3']
    for n in names:
        (Path(tmp) / n).write_bytes(b'x')
        (Path(tmp) / (n + '.json')).write_text(
            json.dumps({'desc': 'the caption', 'author': {'uniqueId': 'someuser'}}),
            encoding='utf-8')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stdout='', stderr='')):
        images, meta = download_slideshow('https://www.tiktok.com/@u/photo/777', tmp)

    assert [p.name for p in images] == ['777_01 caption [aa].jpg',
                                        '777_02 caption [bb].jpg',
                                        '777_10 caption [cc].jpg']
    print('  ✓ mp3 and .json sidecars excluded; slides in carousel order')
    assert meta['desc'] == 'the caption'
    assert meta['author']['uniqueId'] == 'someuser'
    print('  ✓ metadata read from first image sidecar')
```

Add the three new test calls to `main()` (keep existing ones):

```python
    test_instagram_slideshow_requires_cookies()
    test_login_redirect_maps_to_cookie_message()
    test_download_slideshow_filters_and_orders()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd maven_app && python tests/test_slideshow.py`
Expected: FAIL — `ImportError: cannot import name 'INSTAGRAM_COOKIE_MESSAGE'`.

- [ ] **Step 4: Implement in `video_source.py`**

Add `Tuple` is already imported. Below the existing `INSTAGRAM_BLOCK_MESSAGE` block, add:

```python
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


def _instagram_cookies() -> str:
    """Path to the operator's Instagram cookies.txt, or '' if not configured."""
    return os.environ.get(MAVEN_IG_COOKIES_ENV, '').strip()
```

Change `_base_cmd` to accept the URL and pass cookies to yt-dlp for Instagram (docstring update included):

```python
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
```

Update the three call sites (`download_audio`, `download_video`, `fetch_metadata`) from `_base_cmd()` to `_base_cmd(url)`.

Add `download_slideshow` after `download_video`:

```python
def download_slideshow(url: str, tmp_dir: str) -> Tuple[List[Path], dict]:
    """Download a slideshow post's slide images into tmp_dir via gallery-dl.

    Returns (image paths in carousel order, metadata dict from the first
    image's --write-metadata JSON sidecar). Instagram requires login cookies
    (MAVEN_IG_COOKIES); missing or rejected cookies raise RuntimeError with
    INSTAGRAM_COOKIE_MESSAGE. Non-image files (TikTok's mp3 soundtrack,
    sidecars) are filtered out.
    """
    cmd = ['gallery-dl', '-D', tmp_dir, '--write-metadata']
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

    images = sorted(p for p in Path(tmp_dir).iterdir()
                    if p.suffix.casefold() in _IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError('Slideshow download failed: no images produced.')

    metadata = {}
    sidecar = images[0].parent / (images[0].name + '.json')
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            metadata = {}
    return images, metadata
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd maven_app && python tests/test_slideshow.py && python tests/test_video_source.py`
Expected: `ALL TESTS PASSED` from both.

- [ ] **Step 6: Commit**

```bash
git add maven_app/video_source.py maven_app/requirements.txt maven_app/tests/test_slideshow.py
git commit -m "feat: add gallery-dl slideshow download with Instagram cookie support"
```

---

### Task 3: slideshow.py — OCR slides and assemble text

**Files:**
- Create: `maven_app/slideshow.py`
- Test: `maven_app/tests/test_slideshow.py` (extend)

**Interfaces:**
- Consumes: `download_slideshow(url, tmp_dir)` (Task 2); `_ocr_frames(frames, uploader, junk_terms)`, `_assemble_text(description, overlay_segments)`, `TextExtractionResult`, `NoTextFoundError` from `text_extraction.py` (all exist; `_ocr_frames` accepts arbitrary image paths and returns `[{'ts': int, 'lines': [(text, conf), ...]}, ...]`).
- Produces: `extract_slideshow_text(url: str, platform: Platform) -> TextExtractionResult` (segments shaped `{'slide': int, 'text': str}` in `overlay_segments`); `build_slide_segments(frame_results: List[dict]) -> List[dict]`; `_description_from(metadata: dict) -> str`; `_uploader_from(metadata: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `maven_app/tests/test_slideshow.py` (before `main()`); add the import near the other imports:

```python
from slideshow import _description_from, _uploader_from, build_slide_segments
```

```python
# ── TEST 6 ─────────────────────────────────────────────────────────────────────

def test_build_slide_segments():
    print('\n=== TEST 6: build_slide_segments ===')

    frame_results = [
        {'ts': 0, 'lines': [('First slide claim', 0.9), ('subtitle', 0.8)]},
        {'ts': 1, 'lines': []},
        {'ts': 2, 'lines': [('Third slide claim', 0.95)]},
    ]
    segments = build_slide_segments(frame_results)
    assert segments == [
        {'slide': 1, 'text': 'First slide claim subtitle'},
        {'slide': 3, 'text': 'Third slide claim'},
    ]
    print('  ✓ one segment per non-empty slide, 1-based numbering, blank slides skipped')

    assert build_slide_segments([]) == []
    assert build_slide_segments([{'ts': 0, 'lines': []}]) == []
    print('  ✓ empty and all-blank inputs produce no segments')


# ── TEST 7 ─────────────────────────────────────────────────────────────────────

def test_metadata_normalization():
    print('\n=== TEST 7: gallery-dl metadata normalization ===')

    tiktok_meta = {'desc': 'TikTok caption', 'author': {'uniqueId': 'ttuser'}}
    assert _description_from(tiktok_meta) == 'TikTok caption'
    assert _uploader_from(tiktok_meta) == 'ttuser'
    print('  ✓ TikTok schema: desc + author.uniqueId')

    ig_meta = {'description': 'IG caption', 'username': 'iguser'}
    assert _description_from(ig_meta) == 'IG caption'
    assert _uploader_from(ig_meta) == 'iguser'
    print('  ✓ Instagram schema: description + username')

    assert _description_from({}) == ''
    assert _uploader_from({}) == ''
    print('  ✓ missing fields degrade to empty strings')
```

Add both calls to `main()`:

```python
    test_build_slide_segments()
    test_metadata_normalization()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd maven_app && python tests/test_slideshow.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'slideshow'`.

- [ ] **Step 3: Create `maven_app/slideshow.py`**

```python
"""
MAVEN Slideshow Text Extraction: OCRs every slide of a TikTok /photo/ or
Instagram /p/ slideshow post and combines the results with the post caption.
Dispatched from text_extraction.extract_text_url for slideshow URLs; reuses
its OCR, junk-filter, and text-assembly helpers.
Public entry point: extract_slideshow_text(url, platform) → TextExtractionResult.
"""
import shutil
import tempfile
from typing import List

from text_extraction import (NoTextFoundError, TextExtractionResult,
                             _assemble_text, _ocr_frames)
from video_source import Platform, download_slideshow


def _description_from(metadata: dict) -> str:
    """Caption across gallery-dl schemas: TikTok 'desc', Instagram 'description'."""
    return (metadata.get('desc') or metadata.get('description') or '').strip()


def _uploader_from(metadata: dict) -> str:
    """Uploader handle across gallery-dl schemas: TikTok author.uniqueId,
    Instagram username."""
    author = metadata.get('author') or {}
    return (author.get('uniqueId') or metadata.get('username')
            or metadata.get('uploader') or '').strip()


def build_slide_segments(frame_results: List[dict]) -> List[dict]:
    """One segment per slide that produced OCR text; slide numbers are the
    1-based carousel positions. Unlike video frames there is no cross-frame
    merging — each slide is independent content.

    frame_results: [{'ts': idx, 'lines': [(text, confidence), ...]}, ...]
    Returns [{'slide': int, 'text': str}, ...].
    """
    segments = []
    for frame in frame_results:
        lines = frame.get('lines') or []
        if not lines:
            continue
        segments.append({'slide': frame['ts'] + 1,
                         'text': ' '.join(t for t, _ in lines)})
    return segments


def extract_slideshow_text(url: str, platform: Platform) -> TextExtractionResult:
    tmp_dir = tempfile.mkdtemp()
    try:
        images, metadata = download_slideshow(url, tmp_dir)
        description = _description_from(metadata)
        uploader = _uploader_from(metadata)
        # Slides are OCR'd at native resolution — they are text-dense by
        # design, unlike sampled video frames.
        frame_results = _ocr_frames(images, uploader, platform.junk_terms)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    slide_segments = build_slide_segments(frame_results)
    text = _assemble_text(description, slide_segments)
    if not text:
        raise NoTextFoundError('No slide text or caption found in slideshow.')
    return TextExtractionResult(
        description=description,
        overlay_segments=slide_segments,
        text=text,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd maven_app && python tests/test_slideshow.py`
Expected: `ALL TESTS PASSED`. (Importing `slideshow` pulls in `text_extraction`, which is light — RapidOCR loads lazily.)

- [ ] **Step 5: Commit**

```bash
git add maven_app/slideshow.py maven_app/tests/test_slideshow.py
git commit -m "feat: add slideshow OCR module reusing text-extraction helpers"
```

---

### Task 4: Dispatch — text mode routes to slideshow, audio mode rejects

**Files:**
- Modify: `maven_app/text_extraction.py:113-135` (`extract_text_url`) and its module docstring
- Modify: `maven_app/transcription.py:28-35` (`transcribe_url`) and its module docstring
- Test: `maven_app/tests/test_slideshow.py` (extend)

**Interfaces:**
- Consumes: `extract_slideshow_text(url, platform)` (Task 3), `Platform.is_slideshow` (Task 1).
- Produces: `extract_text_url()` transparently handles slideshow URLs; `transcribe_url()` raises `ValueError('Slideshow posts are supported in Text mode only.')` for them (app.py already maps `ValueError` → HTTP 400; no app.py change).

- [ ] **Step 1: Write the failing tests**

Append to `maven_app/tests/test_slideshow.py` (before `main()`):

```python
# ── TEST 8 ─────────────────────────────────────────────────────────────────────

def test_text_mode_dispatches_to_slideshow():
    print('\n=== TEST 8: extract_text_url dispatches slideshow URLs ===')

    import slideshow
    import text_extraction

    calls = []
    saved = slideshow.extract_slideshow_text
    slideshow.extract_slideshow_text = lambda url, platform: (
        calls.append((url, platform.name)) or 'SENTINEL')
    try:
        result = text_extraction.extract_text_url(
            'https://www.tiktok.com/@u/photo/777')
    finally:
        slideshow.extract_slideshow_text = saved

    assert result == 'SENTINEL'
    assert calls == [('https://www.tiktok.com/@u/photo/777', 'tiktok')]
    print('  ✓ slideshow URL routed to extract_slideshow_text, video path untouched')


# ── TEST 9 ─────────────────────────────────────────────────────────────────────

def test_audio_mode_rejects_slideshows():
    print('\n=== TEST 9: transcribe_url rejects slideshow URLs ===')

    from transcription import transcribe_url

    for url in ('https://www.tiktok.com/@u/photo/777',
                'https://www.instagram.com/p/DRzdgElEf3N/'):
        try:
            transcribe_url(url)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert str(e) == 'Slideshow posts are supported in Text mode only.'
    print('  ✓ both platforms rejected in audio mode with the friendly message')
```

Add both calls to `main()`:

```python
    test_text_mode_dispatches_to_slideshow()
    test_audio_mode_rejects_slideshows()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd maven_app && python tests/test_slideshow.py`
Expected: FAIL — TEST 8 reaches `fetch_metadata` (network call or wrong result); TEST 9 gets a download error instead of `ValueError`. Note: TEST 8's monkeypatch only works because `extract_text_url` will import `slideshow` lazily *per call* (Step 3); the test patches the module attribute the lazy import resolves.

- [ ] **Step 3: Implement**

In `maven_app/text_extraction.py`, add the branch at the top of `extract_text_url`:

```python
def extract_text_url(url: str) -> TextExtractionResult:
    url, platform = validate_url(url)
    if platform.is_slideshow(url):
        # Imported lazily: slideshow.py imports helpers from this module.
        import slideshow
        return slideshow.extract_slideshow_text(url, platform)
    metadata = fetch_metadata(url)
```

(rest of the function unchanged). Update the module docstring's first paragraph:

```python
"""
MAVEN Text Extraction: pulls on-screen overlay text (via frame OCR) and the
video description from a supported video URL (TikTok or Instagram Reel).
Slideshow posts (TikTok /photo/, Instagram /p/) dispatch to slideshow.py.
Public entry point: extract_text_url(url) → TextExtractionResult.
"""
```

In `maven_app/transcription.py`, add the guard in `transcribe_url`:

```python
def transcribe_url(url: str) -> TranscriptResult:
    url, platform = validate_url(url)
    if platform.is_slideshow(url):
        raise ValueError('Slideshow posts are supported in Text mode only.')
    tmp_dir = tempfile.mkdtemp()
```

and append to its module docstring:

```python
"""
MAVEN Transcription: downloads audio from a supported video URL (TikTok or
Instagram Reel) and transcribes it with faster-whisper.
Slideshow posts have no speech track and are rejected with a friendly error.
Public entry point: transcribe_url(url) → TranscriptResult.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd maven_app && python tests/test_slideshow.py`
Expected: `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add maven_app/text_extraction.py maven_app/transcription.py maven_app/tests/test_slideshow.py
git commit -m "feat: route slideshow URLs to slideshow extraction; reject in audio mode"
```

---

### Task 5: UI — slide segment rendering, placeholder, docs

**Files:**
- Modify: `maven_app/templates/index.html:171` (placeholder) and `:357-382` (`renderTranscriptPanel`)
- Modify: `CLAUDE.md` (app structure, dependencies, slideshow notes)

**Interfaces:**
- Consumes: `/transcribe` text-mode payload where `segments` items are either `{start, end, text}` (video) or `{slide, text}` (slideshow).
- Produces: user-visible rendering; no JS API consumed elsewhere.

- [ ] **Step 1: Update the URL placeholder (line 171)**

```html
    placeholder="TikTok or Instagram video / slideshow URL (optional)"
```

- [ ] **Step 2: Update `renderTranscriptPanel`**

Replace the function body's meta-line block and segment map (keep the empty-segments guard and the panel-visibility lines unchanged):

```javascript
function renderTranscriptPanel(data) {
    if (!data.segments.length) {           // e.g. text mode, description only
        transcriptPanel.hidden = true;
        return;
    }
    const isSlideshow = data.segments[0].slide != null;
    if (data.mode === 'text') {
        transcriptMeta.textContent = isSlideshow
            ? `${data.segments.length} slides ▾`
            : `${data.segments.length} overlay segments ▾`;
    } else {
        const mins   = Math.floor(data.duration / 60);
        const secs   = Math.round(data.duration % 60);
        const durStr = `${mins}:${String(secs).padStart(2, '0')}`;
        transcriptMeta.textContent = `${durStr} · ${data.segments.length} segments ▾`;
    }

    transcriptSegs.innerHTML = data.segments.map(seg =>
        `<div class="flex gap-3 items-baseline">
            <span class="font-label text-[9px] text-outline uppercase tracking-tighter whitespace-nowrap">${seg.slide != null ? `Slide ${seg.slide}` : `${fmtTime(seg.start)} – ${fmtTime(seg.end)}`}</span>
            <span class="font-body italic text-[12px] text-on-surface-variant leading-snug">${escapeHtml(seg.text)}</span>
        </div>`
    ).join('');

    transcriptPanel.hidden = false;
    transcriptToggle.setAttribute('aria-expanded', 'false');
    transcriptSegs.hidden = true;
}
```

- [ ] **Step 3: Update CLAUDE.md**

In the **Flask App Structure** block, add after the `text_extraction.py` line:

```
  slideshow.py        # Slideshow posts: per-slide OCR for TikTok /photo/ and Instagram /p/
```

In the **Key Dependencies** table, add:

```
| `gallery-dl` | Slideshow (photo post) image + metadata download |
```

After the **TikTok Transcription — Dependency Gotchas** section, add a new section:

```markdown
## Slideshow Posts (TikTok /photo/, Instagram /p/)

Slideshow posts are supported in **text mode only** — each slide is OCR'd and
combined with the caption. Images are fetched with gallery-dl (yt-dlp cannot
download slideshow images on either platform).

- TikTok slideshows work anonymously.
- Instagram slideshows require login cookies: export a Netscape `cookies.txt`
  for instagram.com (browser extension) and set `MAVEN_IG_COOKIES=/path/to/cookies.txt`
  before starting the app. Without it, Instagram slideshow requests return a
  friendly error. When set, the cookies are also passed to yt-dlp for Instagram
  Reels, which reduces anonymous rate-limit failures.
```

- [ ] **Step 4: Verify template renders**

Run: `cd maven_app && python -c "from flask import Flask; import jinja2; env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates')); env.get_template('index.html'); print('template OK')"`
Expected: `template OK`

- [ ] **Step 5: Commit**

```bash
git add maven_app/templates/index.html CLAUDE.md
git commit -m "feat: render slideshow segments as slides; document slideshow support"
```

---

### Task 6: Live end-to-end tests + full suite

**Files:**
- Test: `maven_app/tests/test_slideshow.py` (extend with live section)

**Interfaces:**
- Consumes: everything above.
- Produces: verified end-to-end behavior against the real example posts.

- [ ] **Step 1: Add the live tests**

Append to `maven_app/tests/test_slideshow.py` (before `main()`):

```python
# ── LIVE TESTS (network) ───────────────────────────────────────────────────────

def test_live_tiktok_slideshow():
    print('\n=== LIVE TEST: TikTok slideshow end-to-end ===')

    from text_extraction import extract_text_url

    result = extract_text_url(
        'https://www.tiktok.com/@glowingwithgracee/photo/7618717295417756941')
    assert result.description, 'expected a caption'
    print(f'  ✓ caption extracted ({len(result.description)} chars)')
    assert result.overlay_segments, 'expected OCR text from slides'
    slides = [seg['slide'] for seg in result.overlay_segments]
    assert slides == sorted(slides) and slides[0] >= 1
    print(f'  ✓ {len(result.overlay_segments)} slide segments in order: {slides}')
    assert result.text.startswith(result.description[:20])
    print('  ✓ assembled text begins with caption')
    preview = result.overlay_segments[0]['text'][:80]
    print(f'  slide 1 preview: {preview!r}')


def test_live_instagram_slideshow():
    print('\n=== LIVE TEST: Instagram slideshow ===')

    import video_source
    from text_extraction import extract_text_url
    from video_source import INSTAGRAM_COOKIE_MESSAGE

    url = 'https://www.instagram.com/p/DRzdgElEf3N/'
    if not video_source._instagram_cookies():
        try:
            extract_text_url(url)
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert str(e) == INSTAGRAM_COOKIE_MESSAGE
        print('  ~ MAVEN_IG_COOKIES not set — verified friendly cookie error; '
              'full extraction SKIPPED')
        return

    result = extract_text_url(url)
    assert result.description, 'expected a caption'
    assert result.overlay_segments, 'expected OCR text from slides'
    print(f'  ✓ caption + {len(result.overlay_segments)} slide segments extracted')
```

Update `main()` to run live tests only when asked, mirroring a light flag convention:

```python
def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_slideshow_url_detection()
    test_slideshow_urls_validate()
    test_instagram_slideshow_requires_cookies()
    test_login_redirect_maps_to_cookie_message()
    test_download_slideshow_filters_and_orders()
    test_build_slide_segments()
    test_metadata_normalization()
    test_text_mode_dispatches_to_slideshow()
    test_audio_mode_rejects_slideshows()
    if '--live' in _sys.argv:
        test_live_tiktok_slideshow()
        test_live_instagram_slideshow()
    else:
        print('\n(live network tests skipped — pass --live to run them)')
    print('\nALL TESTS PASSED')
```

- [ ] **Step 2: Run the live suite**

Run: `cd maven_app && python tests/test_slideshow.py --live`
Expected: `ALL TESTS PASSED`; TikTok live test shows ~12 ordered slide segments; Instagram shows the cookie-error path (or full extraction if `MAVEN_IG_COOKIES` is set).

- [ ] **Step 3: Run every existing test script**

Run from `maven_app/`:

```bash
python tests/test_video_source.py && \
python tests/test_tiktok.py && \
python tests/test_instagram.py && \
python tests/test_slideshow.py && \
python tests/test_text_extraction.py && \
python tests/test_transcription.py
```

Expected: each prints `ALL TESTS PASSED`. (`test_text_extraction.py` / `test_transcription.py` load models and may hit the network; report any failure that also occurs on `main` as pre-existing rather than fixing it here.)

- [ ] **Step 4: Commit**

```bash
git add maven_app/tests/test_slideshow.py
git commit -m "test: add live end-to-end slideshow tests"
```
