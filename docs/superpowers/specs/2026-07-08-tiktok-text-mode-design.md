# TikTok Text-Extraction Mode — Design

**Date:** 2026-07-08
**Status:** Approved
**Branch:** feature/tiktok-text-extraction

## Goal

Add a second TikTok extraction mode to MAVEN. The existing audio transcription
becomes **audio mode**; the new **text mode** extracts on-screen text overlays
(captions rendered in the video) and the video's description, producing text
suitable for `score_text()` — the same downstream path transcripts use today.

## Decisions

| Question | Decision |
|---|---|
| Overlay source | OCR on sampled video frames — catches burned-in text (CapCut etc.) and TikTok-native text stickers (rendered into frames too). No separate metadata-sticker path. |
| OCR engine | RapidOCR (`rapidocr-onnxruntime`) — pip-only, no system binary, CPU-fast. |
| API shape | One endpoint: `POST /transcribe` with `mode: "audio" \| "text"`, defaulting to `"audio"`. |
| Code layout | New shared `tiktok.py` helper module; `transcription.py` keeps audio; new `text_extraction.py` for text mode. |
| Deliverables | Flask app **and** a new notebook section. |

## Architecture

```
maven_app/
  tiktok.py            # NEW — shared TikTok plumbing
  transcription.py     # audio mode (slimmed: Whisper only)
  text_extraction.py   # NEW — text mode (frames → OCR → dedup + description)
  app.py               # /transcribe gains mode param
```

### `tiktok.py` (extracted from `transcription.py`, behavior unchanged)

- `TIKTOK_RE` / `validate_url(url)` — existing URL regex check
- `ensure_ffmpeg()` — the imageio-ffmpeg copy-and-normalize logic, verbatim
  (copies the version-suffixed binary to `%TEMP%\maven_ffmpeg\ffmpeg.exe` and
  injects the directory into `PATH`)
- `download_audio(url, tmp_dir) -> Path` and `download_video(url, tmp_dir) -> Path`
  — both build the same yt-dlp command core
  (`--no-playlist --quiet --impersonate chrome --ffmpeg-location <exe>`);
  the video variant uses `-f mp4` instead of `--extract-audio`
- `fetch_metadata(url) -> dict` — `yt-dlp --dump-json --skip-download` with the
  same impersonation; text mode reads `description` and the uploader handle
  from it

### `transcription.py`

Keeps `transcribe_url()`, `TranscriptResult`, `NoSpeechError`, and the lazy
Whisper loader, importing plumbing from `tiktok.py`. Public API unchanged, so
existing tests and the notebook's audio section stay valid.

### `text_extraction.py`

Public entry point mirroring the audio module:

```python
extract_text_url(url) -> TextExtractionResult(description, overlay_segments, text)
```

plus `NoTextFoundError`. The RapidOCR engine is lazy-loaded on first use, same
pattern as the Whisper model.

### Dependency constraints (respected by construction)

- `curl_cffi >= 0.10, < 0.15` stays pinned (yt-dlp rejects 0.15+)
- All downloads go through the one shared command builder, so
  `--impersonate chrome` and `--ffmpeg-location` are always applied
- Frame extraction calls the `ensure_ffmpeg()`-normalized binary
- New requirement: `rapidocr-onnxruntime`

## Text-extraction pipeline

`extract_text_url(url)` runs in a temp dir (cleaned up in `finally`, same as
audio mode):

1. **Validate URL** via shared regex → `ValueError` if not TikTok.
2. **Fetch metadata** → `description` (may be empty) and uploader handle.
3. **Download video** as mp4 (shared downloader, impersonation applied).
4. **Sample frames** with normalized ffmpeg: `fps=1`, scaled to 720 px width.
   TikTok overlays persist for seconds, so 1 fps cannot miss one; a 3-minute
   video yields ≤ 180 PNGs. Sampling is capped at the first
   `MAX_VIDEO_SECONDS = 600` seconds so a much longer video can't pin a
   worker for many minutes.
5. **OCR each frame** with RapidOCR → `(text line, confidence)` pairs per
   frame; frame index = timestamp in seconds.
6. **Filter junk** per line: confidence < 0.6, lines shorter than 3 characters,
   watermark noise (lines matching `@username`, `TikTok`, or the uploader
   handle from metadata).
7. **Dedup into segments**: consecutive frames showing the same overlay merge
   into `{start, end, text}`. "Same" = casefolded, whitespace-collapsed
   comparison with fuzzy match (`difflib.SequenceMatcher` ratio ≥ 0.9) to
   absorb per-frame OCR jitter; the highest-confidence variant wins.
   Implemented as a **pure function** `group_overlay_segments(frame_results)`
   so it is unit-testable without network or OCR.
8. **Assemble result**:
   - `description` — from metadata
   - `overlay_segments` — `[{start, end, text}, ...]` (same shape as audio segments)
   - `text` — feeds `score_text()`: description + each unique overlay line,
     newline-joined
   - Raise `NoTextFoundError` only if **both** description and overlays are
     empty (→ HTTP 422, mirroring `NoSpeechError`).

**Accepted trade-off:** OCR on stylized video text occasionally misreads
characters. The confidence filter plus fuzzy dedup absorbs most of it, and
PubMedBERT embeddings are robust to small typos, so residual noise is
tolerable for scoring.

## API

`POST /transcribe` accepts `{url, mode}`; `mode` defaults to `"audio"`;
unknown mode → 400. Unified response so the UI has one rendering path:

```json
{ "mode": "audio", "text": "...", "segments": [...], "duration": 123.4 }
{ "mode": "text",  "text": "...", "segments": [...], "description": "..." }
```

The audio response's `transcript_text` field is **renamed to `text`** (the
bundled UI is the only consumer and is updated in the same change).

Error mapping (shared across modes):
- `ValueError` (bad URL, bad mode) → 400
- `NoSpeechError` / `NoTextFoundError` → 422
- anything else → 500 with the exception message

## UI

- Segmented **Audio / Text** toggle next to the URL input
- Button label: "Transcribe" in audio mode, "Extract Text" in text mode
  (existing loading shimmer retained)
- On success, the returned `text` is inserted into the analyze textarea,
  exactly as transcripts are today

## Testing

New plain-script `maven_app/tests/test_text_extraction.py` following the
existing runner pattern (UTF-8 stdout wrapper, run from `maven_app/`):

- URL validation rejects non-TikTok links
- `group_overlay_segments()` unit tests: merging identical consecutive frames,
  fuzzy-jitter absorption, gap splitting, highest-confidence text selection
- Junk filtering: watermark/handle lines and low-confidence lines dropped
- `NoTextFoundError` raised when description and overlays are both empty
- Flask route: bad mode → 400, missing mode defaults to audio, mocked
  extraction result → correct JSON shape

`test_transcription.py` keeps passing since `transcribe_url`'s API is
unchanged; it gains only an import-path check covering the `tiktok.py`
refactor.

## Notebook

New section after the TikTok audio transcription section:

- Install cell: `rapidocr-onnxruntime`, reusing the existing pinned
  `curl_cffi` / `imageio-ffmpeg` installs
- Narrative on the frames → OCR → dedup approach
- The pipeline code
- Demo feeding `extract_text_url(...).text` into `score_text()`

## Out of scope

- Non-TikTok platforms
- A metadata-only sticker extraction path (OCR covers stickers)
- Running audio and text modes in a single request
- GPU acceleration for OCR
