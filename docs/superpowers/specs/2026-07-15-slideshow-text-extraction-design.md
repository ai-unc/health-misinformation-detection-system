# Slideshow Text Extraction — Design

**Date:** 2026-07-15
**Status:** Approved
**Branch:** `feat/slideshow-text-extraction`

## Goal

Extend MAVEN's text-extraction mode to slideshow (photo) posts on TikTok
(`tiktok.com/@user/photo/<id>`) and Instagram (`instagram.com/p/<id>`).
A slideshow post is a carousel of still images with a caption; the health
claims live in the slide images and the caption. MAVEN OCRs every slide,
combines the results with the caption, and feeds the text to the existing
`score_text()` pipeline.

Example posts used for validation:

- https://www.tiktok.com/@glowingwithgracee/photo/7618717295417756941 (12 slides + mp3)
- https://www.tiktok.com/@drtosinofficial/photo/7648626696140229910
- https://www.instagram.com/p/DRzdgElEf3N/ (12-image carousel)
- https://www.instagram.com/p/DVhV_kzGnOf/

## Verified constraints (2026-07-15)

- **yt-dlp cannot fetch slideshow images.** It rejects TikTok `/photo/` URLs
  outright ("Unsupported URL"); with the `/photo/` → `/video/` swap it returns
  metadata + audio but no slide images. Instagram `/p/` carousels enumerate as
  a playlist whose every image entry fails with "No video formats found",
  though the playlist-level caption is returned.
- **gallery-dl fetches TikTok slideshows anonymously** — all slide images, the
  mp3 audio track, and full metadata (tested with gallery-dl 1.32.6).
- **Instagram images are login-walled.** gallery-dl anonymous requests
  redirect to the login page; the public `/embed/captioned/` page is a
  JavaScript shell with no image URLs. Cookie auth is required for Instagram
  slide images.

## Decisions

| Decision | Choice |
|---|---|
| Instagram images | Cookie-based auth (reverses the previous no-credential stance) |
| Cookie source | `MAVEN_IG_COOKIES` env var → path to Netscape-format `cookies.txt` |
| Modes | Text mode only; audio mode on a slideshow URL returns a friendly error |
| Image/metadata fetcher | gallery-dl for both platforms (Approach A) |

## Architecture

### URL routing

- `Platform` (in `video_source.py`) gains a `slideshow_url_re` field:
  TikTok `/photo/`, Instagram `/p/`. A `Platform.is_slideshow(url)` method
  keeps the check in one place.
- Instagram's main `url_re` is extended to accept `/p/` links (currently
  rejected by design).
- **Text mode:** `extract_text_url()` in `text_extraction.py` branches after
  `validate_url()` — slideshow URLs dispatch to the new `slideshow.py`
  module; video URLs follow the existing frame-sampling path unchanged.
  `app.py`'s text-mode handling does not change.
- **Audio mode:** `transcribe_url()` raises
  `ValueError("Slideshow posts are supported in Text mode only.")` for
  slideshow URLs; the existing handler maps `ValueError` to HTTP 400.

### Slideshow data flow

- New `download_slideshow(url, tmp_dir) -> (images, metadata)` helper in
  `video_source.py`, alongside the yt-dlp helpers. One gallery-dl invocation
  downloads into `tmp_dir` with `--write-metadata` (per-file JSON sidecars
  carrying caption and uploader). Non-image files (TikTok's mp3) are filtered
  out by extension. Metadata field names differ per extractor; a small
  per-platform mapping normalizes them to `description` and `uploader`.
  Same 600-second timeout as the yt-dlp calls.
- New `slideshow.py` module reuses the existing OCR machinery:
  - Each slide goes through `_ocr_frames()` (already accepts arbitrary image
    paths) with the platform's `junk_terms` + uploader filtering.
  - Slides are OCR'd at native resolution — no downscaling; slides are
    text-dense by design.
  - No fuzzy cross-frame merging (that absorbs video jitter); each slide
    becomes one segment: `{"slide": N, "text": "..."}` (1-based N).
  - `_assemble_text()` builds the final text (caption + unique slide lines;
    its seen-set dedup handles text repeated across slides).
  - Returns the same `TextExtractionResult` the Flask route and pipeline
    already consume; `overlay_segments` holds the slide-shaped segments.

### Cookies & error handling

- `MAVEN_IG_COOKIES` (path to `cookies.txt`) is passed as `--cookies` to
  gallery-dl for Instagram URLs, and also to the existing yt-dlp Instagram
  calls (harmless; improves Reels reliability). TikTok never uses cookies.
- gallery-dl login-redirect/abort output on an Instagram slideshow maps to a
  friendly message, mirroring the `_INSTAGRAM_BLOCK_SIGNATURES` pattern:
  *"Instagram slideshows require login cookies. Export cookies.txt and set
  MAVEN_IG_COOKIES."*
- If images download but OCR + caption produce nothing, the existing
  `NoTextFoundError` (HTTP 422) fires as today.

### UI & docs

- URL placeholder: "TikTok or Instagram video / slideshow URL".
- Segments with a `slide` field render as "Slide N" instead of a timestamp
  range; the meta line reads "N slides ▾" instead of "N overlay segments ▾".
- Audio mode on a slideshow URL surfaces the server's 400 message; no
  front-end URL sniffing.
- `requirements.txt` gains `gallery-dl`; CLAUDE.md documents slideshow
  support and `MAVEN_IG_COOKIES`.

## Testing

New `tests/test_slideshow.py`, following the existing plain-script convention
(`main()`, UTF-8 stdout wrap on Windows):

- **Unit (no network):** slideshow URL detection for both platforms;
  per-slide segment building from canned OCR output; metadata normalization
  from canned gallery-dl JSON; mp3/non-image filtering; audio-mode rejection
  of slideshow URLs.
- **Live e2e:** the TikTok example post end-to-end (anonymous). The Instagram
  example runs only when `MAVEN_IG_COOKIES` is set; otherwise the test
  verifies the friendly cookie error and reports the skip.

## Out of scope

- Audio-mode transcription of TikTok slideshow soundtracks (TTS voiceovers) —
  explicitly deferred by decision above.
- Cookie handling for any platform other than Instagram.
- Any per-slide image UI (thumbnails, previews); only extracted text is shown.
