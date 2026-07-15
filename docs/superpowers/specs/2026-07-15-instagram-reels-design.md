# Instagram Reels Support — Design

**Date:** 2026-07-15
**Status:** Approved
**Branch:** feature/instagram-reels

## Goal

Generalize MAVEN's audio-visual extraction (audio transcription + on-screen
text OCR) so Instagram Reels work exactly like TikTok videos: paste a Reel
URL, pick audio or text mode, get text suitable for `score_text()`. The
per-platform code must mirror the existing TikTok structure.

## Decisions

| Question | Decision |
|---|---|
| Mode scope | Both modes — Whisper audio transcription **and** frame-OCR text mode work for Reels. |
| Authentication | Public Reels only. When Instagram login-walls or rate-limits a request, surface a clear, friendly error. No cookie/credential handling. |
| Accepted URLs | `instagram.com/reel/…`, `/reels/…`, `/share/…` (share redirects), `www.` optional. `/p/` photo posts and `/tv/` are rejected at validation. |
| Code layout | Shared core + thin platform modules: new `video_source.py` owns all yt-dlp/ffmpeg plumbing; `tiktok.py` and new `instagram.py` are thin, mirrored platform definitions. |

## Architecture

```
maven_app/
  video_source.py      # NEW — shared plumbing + Platform dataclass + URL dispatch
  tiktok.py            # SLIMMED — thin TikTok platform definition
  instagram.py         # NEW — thin Instagram platform definition (mirrors tiktok.py)
  transcription.py     # imports from video_source; flow unchanged
  text_extraction.py   # imports from video_source; junk filter takes platform terms
  app.py               # unchanged routes
  templates/index.html # placeholder text only
```

### `video_source.py` (moved verbatim from `tiktok.py` unless noted)

- `Platform` dataclass: `name` (slug), `display_name`, `url_re` (compiled
  regex), `junk_terms` (frozenset of watermark strings for the OCR filter).
- `validate_url(url) -> tuple[str, Platform]` — strips the URL, returns it
  with the first platform whose `url_re` matches; raises `ValueError`
  naming both supported platforms when nothing matches. The dispatcher
  imports the thin platform modules inside the function body to avoid a
  circular import (thin modules import `Platform` from this module).
- `ensure_ffmpeg()` — verbatim.
- `_base_cmd()`, `download_audio()`, `download_video()`, `fetch_metadata()`
  — verbatim (yt-dlp handles Instagram natively; `--impersonate chrome`
  stays on every call).
- `_run()` — gains a stderr translation step: when yt-dlp fails **on an
  Instagram URL** (the URL is the last element of every yt-dlp command; gated
  so generic phrases like `login required` in a TikTok failure never
  mistranslate) and stderr matches Instagram's known login-wall/rate-limit
  signatures (case-insensitive substrings such as `login required`,
  `rate-limit reached`, `Restricted Video`,
  `Requested content is not available`), raise
  `RuntimeError` with a friendly message — "Instagram requires login or has
  rate-limited this request. Try a public Reel or retry later." — instead of
  raw stderr. All other failures keep today's behavior (stderr passthrough).

### `tiktok.py` (thin)

```python
TIKTOK = Platform(
    name='tiktok',
    display_name='TikTok',
    url_re=re.compile(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/'),
    junk_terms=frozenset({'tiktok'}),
)
```

### `instagram.py` (thin, mirrors `tiktok.py`)

```python
INSTAGRAM = Platform(
    name='instagram',
    display_name='Instagram',
    url_re=re.compile(r'https?://(www\.)?instagram\.com/(reels?|share)/'),
    junk_terms=frozenset({'instagram', 'reels', 'reel'}),
)
```

`/p/` and `/tv/` URLs fail the regex and get the standard "not a supported
link" `ValueError`, which names Reels as the supported Instagram form.

### `transcription.py` / `text_extraction.py`

Imports switch from `tiktok` to `video_source`; public APIs
(`transcribe_url`, `extract_text_url`) and flow are unchanged. Two touches in
`text_extraction.py`:

- `_is_junk(text, confidence, uploader, junk_terms)` — the hardcoded
  `'tiktok'` watermark check becomes a membership test against the platform's
  `junk_terms`; the `@handle`/uploader logic is unchanged.
- `extract_text_url()` threads the `Platform` returned by `validate_url`
  through to the OCR junk filter.

### `app.py` and UI

- `app.py`: no route changes. `ValueError` from validation and `RuntimeError`
  from downloads already map to 400/500 with their messages.
- `templates/index.html`: URL input placeholder becomes
  "TikTok or Instagram Reel URL (optional)". Mode toggle, buttons, and status
  labels are already platform-generic.

## Error handling summary

| Failure | Behavior |
|---|---|
| Non-TikTok, non-Reel URL | `ValueError` at validation → 400 with message naming both platforms |
| Instagram `/p/` or `/tv/` URL | Same validation `ValueError` (regex does not match) |
| Instagram login-wall / rate-limit | `RuntimeError` with friendly retry message → 500 |
| Other yt-dlp failure | `RuntimeError` with stderr passthrough (unchanged) |
| No speech / no text found | `NoSpeechError` / `NoTextFoundError` → 422 (unchanged) |

## Testing

Plain-script style with mocked `subprocess.run`, no network — same as today.

- `tests/test_video_source.py` — plumbing tests moved from `test_tiktok.py`
  (download commands, impersonation guard, metadata parsing) plus new cases:
  URL dispatch returns the right `Platform`; Instagram login-wall stderr is
  translated to the friendly message; other stderr passes through.
- `tests/test_tiktok.py` — slims to TikTok URL validation cases.
- `tests/test_instagram.py` — mirrors `test_tiktok.py`: accepts `/reel/`,
  `/reels/`, `/share/` (with and without `www.`); rejects `/p/`, `/tv/`,
  and non-Instagram URLs; junk terms present on the platform object.
- `tests/test_transcription.py` / `test_text_extraction.py` — patch targets
  updated for the `video_source` import move; behavior assertions unchanged.

## Out of scope

- Cookie/credential support for login-walled Reels.
- Other platforms (YouTube Shorts, Facebook Reels) — the `Platform` registry
  makes these ~20-line additions later.
- Notebook changes — the notebook documents the scoring pipeline, not the
  URL ingestion plumbing.
