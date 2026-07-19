# Instagram Anonymous-First Slideshow Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Instagram `/p/` slideshow posts work with zero configuration by fetching caption + slides anonymously from Instagram's public embed endpoint, demoting `MAVEN_IG_COOKIES` to an optional fallback.

**Architecture:** New `maven_app/instagram_embed.py` fetches `/p/<shortcode>/embed/captioned/` via `curl_cffi` Chrome TLS impersonation, parses the page's `contextJSON` blob (`gql_data.shortcode_media`), and downloads each slide's `display_url` to `tmp_dir/embed/NN.jpg`. `video_source.download_slideshow()` tries this first for Instagram URLs and falls back to the existing gallery-dl + cookies leg only on `EmbedUnavailableError` with cookies configured. Spec: `docs/superpowers/specs/2026-07-18-ig-anonymous-slideshow-design.md`.

**Tech Stack:** Python 3.10+, `curl_cffi` (already pinned `>=0.10,<0.15` in `maven_app/requirements.txt`), stdlib `json`/`re`/`pathlib`, `unittest.mock` in plain-script tests.

## Global Constraints

- **Zero new dependencies** — `curl_cffi` is already in `maven_app/requirements.txt` (yt-dlp constraint).
- Tests are **plain Python scripts, not pytest**: run `cd maven_app && ../.venv/bin/python tests/test_slideshow.py` (add `--live` for network tests).
- All user-facing message text lives in `video_source.py`; `instagram_embed.py` is transport/parse only.
- Exact message strings (verbatim):
  - `INSTAGRAM_COOKIE_MESSAGE` (reworded): `Instagram blocked anonymous access to this post — it may be private, removed, or rate-limited. Retry later, or set MAVEN_IG_COOKIES to a logged-in cookies.txt for instagram.com to access it with your account.`
  - `NO_IMAGES_MESSAGE` (hoisted, text unchanged): `No images found in this post — it may be a video post rather than a slideshow.`
- Windows-safe (pure-Python paths; no shell tricks). Commit after every green test cycle. Do not commit the untracked `TODO.md`.

---

### Task 1: `instagram_embed.py` module + unit tests

**Files:**
- Create: `maven_app/instagram_embed.py`
- Test: `maven_app/tests/test_slideshow.py` (append TESTS 10–12 + fixture helper; register in `main()`)

**Interfaces:**
- Produces (Task 2 relies on these exact names):
  - `instagram_embed.download_slideshow_anonymous(url: str, tmp_dir: str) -> Tuple[List[Path], dict]` — metadata dict is `{'description': <caption str>, 'username': <owner str>}`
  - `instagram_embed.EmbedUnavailableError(RuntimeError)` — fallback-eligible failure
  - `instagram_embed.NotASlideshowError(RuntimeError)` — post is a video; terminal
  - Test seams: `instagram_embed._fetch_embed_page(shortcode: str) -> str`, `instagram_embed._download_image(url: str, dest: Path) -> None`

- [ ] **Step 1: Write the failing tests**

In `maven_app/tests/test_slideshow.py`, add to the import block (after `from slideshow import ...`):

```python
import instagram_embed
```

Add the fixture helper right after the imports:

```python
def _embed_page(media: dict) -> str:
    """Minified embed-page fixture mirroring the real page's JSON-in-string
    shape (verified 2026-07-18), including Instagram's \\/ slash escaping and
    a decoy contextJSON that must be skipped."""
    context = {'context': {'type': media.get('__typename')},
               'gql_data': {'shortcode_media': media}}
    inner = json.dumps(context, separators=(',', ':')).replace('/', '\\/')
    blob = '{"contextJSON":' + json.dumps(inner) + '}'
    return ('<html><head><script>{"contextJSON":"not-json"}</script></head>'
            '<body><script type="application/json">{"require":[[' + blob +
            ']]}</script></body></html>')
```

Append after TEST 9:

```python
# ── TEST 10 ────────────────────────────────────────────────────────────────────

def test_embed_parse_carousel():
    print('\n=== TEST 10: embed page parse — carousel ===')

    media = {
        '__typename': 'GraphSidecar',
        'owner': {'username': 'healthaccount'},
        'edge_media_to_caption': {'edges': [{'node': {'text': 'the caption'}}]},
        'edge_sidecar_to_children': {'edges': [
            {'node': {'display_url': 'https://cdn.example/1.jpg'}},
            {'node': {'display_url': 'https://cdn.example/2.jpg'}},
            {'node': {'display_url': 'https://cdn.example/3.jpg'}},
        ]},
    }
    parsed = instagram_embed._parse_shortcode_media(_embed_page(media))
    assert parsed['__typename'] == 'GraphSidecar'
    assert parsed['owner']['username'] == 'healthaccount'
    print('  ✓ shortcode_media recovered through double-encoded contextJSON')

    urls = instagram_embed._slide_urls(parsed)
    assert urls == ['https://cdn.example/1.jpg', 'https://cdn.example/2.jpg',
                    'https://cdn.example/3.jpg']
    print('  ✓ slide URLs in carousel order with \\/ escapes decoded')


# ── TEST 11 ────────────────────────────────────────────────────────────────────

def test_embed_parse_edge_shapes():
    print('\n=== TEST 11: embed parse — single image, video post, missing data ===')

    single = {'__typename': 'GraphImage', 'owner': {'username': 'u'},
              'edge_media_to_caption': {'edges': []},
              'display_url': 'https://cdn.example/only.jpg'}
    urls = instagram_embed._slide_urls(
        instagram_embed._parse_shortcode_media(_embed_page(single)))
    assert urls == ['https://cdn.example/only.jpg']
    print('  ✓ single GraphImage post yields its one display_url')

    video = {'__typename': 'GraphVideo', 'owner': {'username': 'u'},
             'display_url': 'https://cdn.example/poster.jpg'}
    try:
        instagram_embed._slide_urls(video)
        assert False, 'Expected NotASlideshowError'
    except instagram_embed.NotASlideshowError:
        print('  ✓ GraphVideo post raises NotASlideshowError')

    try:
        instagram_embed._parse_shortcode_media('<html><body>no data</body></html>')
        assert False, 'Expected EmbedUnavailableError'
    except instagram_embed.EmbedUnavailableError:
        print('  ✓ page without contextJSON raises EmbedUnavailableError')


# ── TEST 12 ────────────────────────────────────────────────────────────────────

def test_download_slideshow_anonymous():
    print('\n=== TEST 12: download_slideshow_anonymous writes ordered slides ===')

    media = {
        '__typename': 'GraphSidecar',
        'owner': {'username': 'healthaccount'},
        'edge_media_to_caption': {'edges': [{'node': {'text': 'the caption'}}]},
        'edge_sidecar_to_children': {'edges': [
            {'node': {'display_url': 'https://cdn.example/1.jpg'}},
            {'node': {'display_url': 'https://cdn.example/2.jpg'}},
            {'node': {'display_url': 'https://cdn.example/3.jpg'}},
        ]},
    }
    html = _embed_page(media)
    tmp = tempfile.mkdtemp()

    def fake_download(url, dest):
        dest.write_bytes(url.encode())

    with patch('instagram_embed._fetch_embed_page',
               return_value=html) as mock_fetch, \
         patch('instagram_embed._download_image', side_effect=fake_download):
        images, meta = instagram_embed.download_slideshow_anonymous(
            'https://www.instagram.com/p/ABC123xyz_-/', tmp)

    mock_fetch.assert_called_once_with('ABC123xyz_-')
    print('  ✓ shortcode extracted from URL')
    assert [p.name for p in images] == ['01.jpg', '02.jpg', '03.jpg']
    assert all(p.parent.name == 'embed' for p in images)
    print('  ✓ slides written to tmp_dir/embed/ as zero-padded NN.jpg in order')
    assert [p.read_bytes().decode() for p in images] == [
        'https://cdn.example/1.jpg', 'https://cdn.example/2.jpg',
        'https://cdn.example/3.jpg']
    print('  ✓ each slide downloaded from its carousel-ordered display_url')
    assert meta == {'description': 'the caption', 'username': 'healthaccount'}
    print('  ✓ metadata uses the Instagram schema slideshow.py already reads')
```

In `main()`, add after `test_audio_mode_rejects_slideshows()`:

```python
    test_embed_parse_carousel()
    test_embed_parse_edge_shapes()
    test_download_slideshow_anonymous()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd maven_app && ../.venv/bin/python tests/test_slideshow.py`
Expected: FAIL immediately with `ModuleNotFoundError: No module named 'instagram_embed'`

- [ ] **Step 3: Write the implementation**

Create `maven_app/instagram_embed.py`:

```python
"""
MAVEN anonymous Instagram slideshow ingestion: fetches a /p/ post's caption
and slide images through Instagram's public embed endpoint
(/p/<shortcode>/embed/captioned/) using curl_cffi chrome TLS impersonation —
no login cookies needed for public posts. Transport and parse only: all
user-facing message text lives in video_source.py, which falls back to the
gallery-dl + MAVEN_IG_COOKIES path when EmbedUnavailableError is raised.
Data path verified live 2026-07-18 on 12- and 7-slide public carousels.
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

from curl_cffi import requests as cffi_requests

_EMBED_URL = 'https://www.instagram.com/p/{shortcode}/embed/captioned/'
_SHORTCODE_RE = re.compile(r'instagram\.com/p/([A-Za-z0-9_-]+)')
# contextJSON's value is a JSON-encoded string; capture it with its escape
# sequences intact, then json.loads twice (unescape, then decode).
_CONTEXT_JSON_RE = re.compile(r'"contextJSON"\s*:\s*"((?:\\.|[^"\\])*)"')
_TIMEOUT_SECONDS = 30


class EmbedUnavailableError(RuntimeError):
    """Anonymous embed path can't serve this post (blocked, private, removed,
    markup changed, or a network/download failure). Caller may fall back to
    the cookie-authenticated gallery-dl path."""


class NotASlideshowError(RuntimeError):
    """The post is a GraphVideo — definitively not a slideshow; no fallback
    can change that."""


def _fetch_embed_page(shortcode: str) -> str:
    try:
        resp = cffi_requests.get(_EMBED_URL.format(shortcode=shortcode),
                                 impersonate='chrome',
                                 timeout=_TIMEOUT_SECONDS)
    except Exception as e:
        raise EmbedUnavailableError(f'embed page fetch failed: {e}')
    if resp.status_code != 200:
        raise EmbedUnavailableError(f'embed page HTTP {resp.status_code}')
    return resp.text


def _download_image(url: str, dest: Path) -> None:
    try:
        resp = cffi_requests.get(url, impersonate='chrome',
                                 timeout=_TIMEOUT_SECONDS)
    except Exception as e:
        raise EmbedUnavailableError(f'slide download failed: {e}')
    if resp.status_code != 200 or not resp.content:
        raise EmbedUnavailableError(f'slide download HTTP {resp.status_code}')
    dest.write_bytes(resp.content)


def _parse_shortcode_media(html: str) -> dict:
    """gql_data.shortcode_media from the first parseable contextJSON blob."""
    for match in _CONTEXT_JSON_RE.finditer(html):
        try:
            context = json.loads(json.loads(f'"{match.group(1)}"'))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if not isinstance(context, dict):
            continue
        media = (context.get('gql_data') or {}).get('shortcode_media')
        if media:
            return media
    raise EmbedUnavailableError('no shortcode_media in embed page')


def _slide_urls(media: dict) -> List[str]:
    """display_urls in carousel order. GraphVideo children of a mixed carousel
    contribute their poster frame; a plain GraphVideo post raises
    NotASlideshowError."""
    if media.get('__typename') == 'GraphVideo':
        raise NotASlideshowError('post is a video')
    edges = (media.get('edge_sidecar_to_children') or {}).get('edges') or []
    urls = [(e.get('node') or {}).get('display_url') for e in edges]
    urls = [u for u in urls if u]
    if not urls and media.get('display_url'):
        urls = [media['display_url']]
    if not urls:
        raise EmbedUnavailableError('no slide image URLs in embed data')
    return urls


def download_slideshow_anonymous(url: str, tmp_dir: str) -> Tuple[List[Path], dict]:
    """Anonymous counterpart of video_source.download_slideshow for Instagram
    /p/ posts: returns (slide image paths in carousel order, metadata dict
    with the 'description'/'username' keys slideshow.py already normalizes).
    Slides land in tmp_dir/embed/ so a failed attempt's partial files stay
    out of the gallery-dl fallback's root-level glob of tmp_dir."""
    m = _SHORTCODE_RE.search(url)
    if not m:
        raise EmbedUnavailableError(f'no /p/ shortcode in URL: {url}')
    media = _parse_shortcode_media(_fetch_embed_page(m.group(1)))
    urls = _slide_urls(media)

    caption_edges = (media.get('edge_media_to_caption') or {}).get('edges') or []
    caption = ((caption_edges[0].get('node') or {}).get('text') or ''
               if caption_edges else '')
    owner = (media.get('owner') or {}).get('username') or ''

    embed_dir = Path(tmp_dir) / 'embed'
    embed_dir.mkdir(exist_ok=True)
    images = []
    for idx, slide_url in enumerate(urls, start=1):
        dest = embed_dir / f'{idx:02d}.jpg'
        _download_image(slide_url, dest)
        images.append(dest)
    return images, {'description': caption, 'username': owner}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd maven_app && ../.venv/bin/python tests/test_slideshow.py`
Expected: all tests print ✓ lines, ending `ALL TESTS PASSED` (TESTS 10–12 included)

- [ ] **Step 5: Commit**

```bash
git add maven_app/instagram_embed.py maven_app/tests/test_slideshow.py
git commit -m "feat: anonymous Instagram slideshow ingestion via public embed endpoint"
```

---

### Task 2: Tiered fallback in `video_source.download_slideshow()`

**Files:**
- Modify: `maven_app/video_source.py` (constants ~lines 28–42; `download_slideshow` ~lines 206–257)
- Test: `maven_app/tests/test_slideshow.py` (rework TEST 3; add 3B/3C; patch anonymous path in TESTS 4/5B/5C; register in `main()`)

**Interfaces:**
- Consumes: `instagram_embed.download_slideshow_anonymous`, `EmbedUnavailableError`, `NotASlideshowError` (Task 1).
- Produces: `download_slideshow(url, tmp_dir)` public contract unchanged; new constant `video_source.NO_IMAGES_MESSAGE`; gallery-dl leg hoisted unchanged into `_download_slideshow_gallery_dl(url, tmp_dir)`; `INSTAGRAM_COOKIE_MESSAGE` reworded per Global Constraints.

- [ ] **Step 1: Write the failing tests**

In `maven_app/tests/test_slideshow.py`, extend the `video_source` import to include the new constant:

```python
from video_source import (
    INSTAGRAM_COOKIE_MESSAGE,
    NO_IMAGES_MESSAGE,
    download_slideshow,
    validate_url,
)
```

Replace TEST 3 (`test_instagram_slideshow_requires_cookies`) entirely with:

```python
def test_instagram_anon_failure_without_cookies():
    print('\n=== TEST 3: anonymous failure + no cookies → friendly message, '
          'no gallery-dl call ===')

    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.EmbedUnavailableError('blocked')), \
             patch('video_source.subprocess.run') as mock_run:
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/',
                                   tempfile.mkdtemp())
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_COOKIE_MESSAGE
        mock_run.assert_not_called()
        print('  ✓ EmbedUnavailableError without cookies raises the friendly '
              'message and never invokes gallery-dl')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved


# ── TEST 3B ────────────────────────────────────────────────────────────────────

def test_instagram_anonymous_success_skips_gallery_dl():
    print('\n=== TEST 3B: anonymous success bypasses gallery-dl entirely ===')

    sentinel = ([Path('/x/embed/01.jpg')],
                {'description': 'cap', 'username': 'user'})
    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        with patch('instagram_embed.download_slideshow_anonymous',
                   return_value=sentinel) as mock_anon, \
             patch('video_source.subprocess.run') as mock_run:
            result = download_slideshow(
                'https://www.instagram.com/p/DRzdgElEf3N/', '/x')
        assert result == sentinel
        mock_anon.assert_called_once_with(
            'https://www.instagram.com/p/DRzdgElEf3N/', '/x')
        mock_run.assert_not_called()
        print('  ✓ anonymous result returned as-is; no cookies, no gallery-dl')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved


# ── TEST 3C ────────────────────────────────────────────────────────────────────

def test_instagram_video_post_no_fallback():
    print('\n=== TEST 3C: video post is terminal — no cookie fallback ===')

    tmp = tempfile.mkdtemp()
    cookie_file = Path(tmp) / 'cookies.txt'
    cookie_file.write_text('# Netscape HTTP Cookie File\n', encoding='utf-8')
    saved_env = os.environ.get('MAVEN_IG_COOKIES')
    os.environ['MAVEN_IG_COOKIES'] = str(cookie_file)
    try:
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.NotASlideshowError('video')), \
             patch('video_source.subprocess.run') as mock_run:
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == NO_IMAGES_MESSAGE
        mock_run.assert_not_called()
        print('  ✓ NotASlideshowError maps to the video-post message even '
              'with cookies configured')
    finally:
        if saved_env is None:
            os.environ.pop('MAVEN_IG_COOKIES', None)
        else:
            os.environ['MAVEN_IG_COOKIES'] = saved_env
```

In TESTS 4, 5B, and 5C (`test_login_redirect_maps_to_cookie_message`,
`test_download_slideshow_orders_by_sidecar_num`,
`test_download_slideshow_empty_images_friendly_error`): these exercise the
gallery-dl leg through Instagram URLs, so each needs the anonymous path to
fail first. Add this patch line as the first context manager of each `with`
block that wraps a `download_slideshow(...)` call (keeping the existing
`video_source.subprocess.run` patch):

```python
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.EmbedUnavailableError('blocked')), \
             patch('video_source.subprocess.run', ...existing...):
```

In `main()`, replace the `test_instagram_slideshow_requires_cookies()` line with:

```python
    test_instagram_anon_failure_without_cookies()
    test_instagram_anonymous_success_skips_gallery_dl()
    test_instagram_video_post_no_fallback()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd maven_app && ../.venv/bin/python tests/test_slideshow.py`
Expected: FAIL with `ImportError: cannot import name 'NO_IMAGES_MESSAGE' from 'video_source'`

- [ ] **Step 3: Write the implementation**

In `maven_app/video_source.py`:

Replace the `INSTAGRAM_COOKIE_MESSAGE` assignment (keep `MAVEN_IG_COOKIES_ENV` above it) with:

```python
INSTAGRAM_COOKIE_MESSAGE = ('Instagram blocked anonymous access to this post '
                            '— it may be private, removed, or rate-limited. '
                            'Retry later, or set MAVEN_IG_COOKIES to a '
                            'logged-in cookies.txt for instagram.com to '
                            'access it with your account.')

NO_IMAGES_MESSAGE = ('No images found in this post — it may be a video post '
                     'rather than a slideshow.')
```

Rename the existing `download_slideshow` to `_download_slideshow_gallery_dl`
(body unchanged except the inline no-images string becomes the constant), and
add the tiered public function above it:

```python
def download_slideshow(url: str, tmp_dir: str) -> Tuple[List[Path], dict]:
    """Download a slideshow post's slide images into tmp_dir.

    TikTok goes straight to gallery-dl (anonymous). Instagram tries the
    anonymous embed path first (instagram_embed.py — no cookies needed for
    public posts); on EmbedUnavailableError it falls back to gallery-dl with
    MAVEN_IG_COOKIES when set, else raises INSTAGRAM_COOKIE_MESSAGE. A
    GraphVideo post raises NO_IMAGES_MESSAGE outright — no fallback exists
    that would make it a slideshow.
    Returns (image paths in carousel order, metadata dict).
    """
    if _is_instagram_url(url):
        return _download_slideshow_instagram(url, tmp_dir)
    return _download_slideshow_gallery_dl(url, tmp_dir)


def _download_slideshow_instagram(url: str, tmp_dir: str) -> Tuple[List[Path], dict]:
    import instagram_embed  # lazy, matching the platform-module import pattern
    try:
        return instagram_embed.download_slideshow_anonymous(url, tmp_dir)
    except instagram_embed.NotASlideshowError:
        raise RuntimeError(NO_IMAGES_MESSAGE)
    except instagram_embed.EmbedUnavailableError:
        if not _instagram_cookies():
            raise RuntimeError(INSTAGRAM_COOKIE_MESSAGE)
        return _download_slideshow_gallery_dl(url, tmp_dir)
```

In `_download_slideshow_gallery_dl`, replace the docstring's first paragraph
with "gallery-dl leg of download_slideshow: TikTok always; Instagram only as
the cookie-authenticated fallback." (rest of the docstring unchanged) and
replace the inline no-images literal:

```python
    if not unordered:
        raise RuntimeError(NO_IMAGES_MESSAGE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd maven_app && ../.venv/bin/python tests/test_slideshow.py`
Expected: `ALL TESTS PASSED` (TikTok TEST 5 still passes untouched — no
`--cookies` for TikTok; TESTS 4/5B/5C reach the gallery-dl leg via the
patched anonymous failure)

- [ ] **Step 5: Run the scoring test suite for regressions**

Run: `cd maven_app && ../.venv/bin/python tests/test_scoring.py`
Expected: passes (no video_source consumers broken; ~30–60s PubMedBERT load)

- [ ] **Step 6: Commit**

```bash
git add maven_app/video_source.py maven_app/tests/test_slideshow.py
git commit -m "feat: try anonymous embed path first for Instagram slideshows, cookies as fallback"
```

---

### Task 3: Live verification + docs

**Files:**
- Modify: `maven_app/tests/test_slideshow.py` (flip the Instagram live test to anonymous-first)
- Modify: `CLAUDE.md` (slideshow section, app structure listing, dependency table)

**Interfaces:**
- Consumes: the tiered `download_slideshow` (Task 2) via `text_extraction.extract_text_url`.
- Produces: documentation of the new default; live-verified anonymous path.

- [ ] **Step 1: Flip the live Instagram test**

Replace `test_live_instagram_slideshow` entirely with:

```python
def test_live_instagram_slideshow():
    print('\n=== LIVE TEST: Instagram slideshow (anonymous) ===')

    from text_extraction import extract_text_url

    # Unset cookies for the call to prove the anonymous path works alone.
    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        result = extract_text_url('https://www.instagram.com/p/DRzdgElEf3N/')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved

    assert result.description, 'expected a caption'
    print(f'  ✓ caption extracted anonymously ({len(result.description)} chars)')
    assert result.overlay_segments, 'expected OCR text from slides'
    slides = [seg['slide'] for seg in result.overlay_segments]
    assert slides == sorted(slides) and slides[0] >= 1
    print(f'  ✓ {len(result.overlay_segments)} slide segments in order: {slides}')
    assert result.text.startswith(result.description[:20])
    print('  ✓ assembled text begins with caption')
```

- [ ] **Step 2: Run the full suite including live tests**

Run: `cd maven_app && ../.venv/bin/python tests/test_slideshow.py --live`
Expected: `ALL TESTS PASSED`; live Instagram block prints caption length and
an in-order slide list (12 slides for `DRzdgElEf3N`) with `MAVEN_IG_COOKIES`
unset for the call

- [ ] **Step 3: Update CLAUDE.md**

In the **Slideshow Posts** section: replace the two bullets "TikTok slideshows
work anonymously." and "Instagram slideshows require login cookies: …" (and
retitle the verification bullet) with:

```markdown
- Slideshows on both platforms work anonymously out of the box. Instagram
  caption + slides come from the public embed endpoint
  (`/p/<shortcode>/embed/captioned/`) fetched with curl_cffi Chrome TLS
  impersonation (`maven_app/instagram_embed.py`) — no account or cookies.
  Verified live 2026-07-18 (12-slide and 7-slide public carousels; plain
  curl without TLS impersonation gets a decoy error page, so keep curl_cffi
  healthy).
- `MAVEN_IG_COOKIES` is an optional fallback used only when the anonymous
  path fails (private/removed posts, rate-limiting): export a Netscape
  `cookies.txt` for instagram.com (browser extension, or
  `~/.config/maven/export_instagram_cookies.py` on the dev machine) and set
  `MAVEN_IG_COOKIES=/path/to/cookies.txt`. When set, the cookies are also
  passed to yt-dlp for Instagram Reels, which reduces anonymous rate-limit
  failures. Keep the file `chmod 600` and out of the repo — it is the
  account's live session. The cookie-authenticated path was last verified
  end-to-end 2026-07-16; if fallback requests return the cookie error, the
  session expired — re-export.
```

In the **Flask App Structure** listing, add under `instagram.py`:

```
  instagram_embed.py  # Anonymous Instagram /p/ ingestion via public embed endpoint
```

In the **Key Dependencies** table, update the gallery-dl row and add curl_cffi:

```markdown
| `gallery-dl` | Slideshow image + metadata download (TikTok; Instagram cookie fallback) |
| `curl_cffi` | Chrome TLS impersonation (yt-dlp downloads + anonymous Instagram embed fetch) |
```

- [ ] **Step 4: Re-run unit tests (docs shouldn't break anything, sanity check)**

Run: `cd maven_app && ../.venv/bin/python tests/test_slideshow.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add maven_app/tests/test_slideshow.py CLAUDE.md
git commit -m "test+docs: live-verify anonymous Instagram slideshow path; demote cookies to fallback"
```
