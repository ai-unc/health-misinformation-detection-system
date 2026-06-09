# TikTok Audio Transcription — Design Spec
**Date:** 2026-06-09
**Branch:** `feature/tiktok-transcription`
**Status:** Approved

---

## 1. Overview

Add a TikTok URL intake path to MAVEN. The user pastes a TikTok URL into a URL bar above the existing textarea; clicking "Transcribe" downloads the video audio with yt-dlp, transcribes it with faster-whisper, and populates the textarea with the plain transcript text. The user can edit the text, then clicks "Analyze Manuscript" as normal. The MAVEN pipeline is untouched.

The same transcription logic is also documented as a standalone section in the Jupyter notebook for Colab-based review.

---

## 2. Scope

**In scope:**
- `maven_app/transcription.py` — download + transcription module
- `POST /transcribe` — new Flask route
- `maven_app/templates/index.html` — URL bar, loading state, transcript reference panel
- `maven_app/requirements.txt` — add `faster-whisper`, `yt-dlp`
- `MAVEN_AI_UNC_SPR2026.ipynb` — new "TikTok Audio Transcription" section
- `maven_app/tests/test_transcription.py` — unit tests

**Out of scope:**
- Per-chunk timestamps in MAVEN result cards
- Support for non-TikTok video platforms
- Real-time streaming transcription
- Changes to `pipeline.py` or the `/analyze` route

---

## 3. Architecture & Data Flow

```
[User pastes TikTok URL]
        │
        ▼
  POST /transcribe  { "url": "..." }
        │
        ▼
  transcription.py
    _download_audio(url)   → yt-dlp → temp .mp3
    _transcribe(path)      → faster-whisper small → TranscriptResult
    cleanup temp file
        │
        ▼
  { transcript_text, segments: [{start, end, text}], duration }
        │
        ▼
  Frontend populates textarea with transcript_text
  Collapsible "Transcript Reference" panel shows timestamped segments
        │
        ▼
  [User clicks "Analyze Manuscript"] → existing /analyze flow, unchanged
```

`pipeline.py` and `/analyze` are not modified. By the time text reaches `score_text()`, it is plain text regardless of origin.

---

## 4. `transcription.py` Module

### Public API

```python
@dataclass
class TranscriptResult:
    text: str              # full joined transcript
    segments: list[dict]   # [{"start": float, "end": float, "text": str}, ...]
    duration: float        # total audio duration in seconds

def transcribe_url(url: str) -> TranscriptResult
```

### Internal functions

```python
def _download_audio(url: str) -> Path    # yt-dlp → temp .mp3, raises RuntimeError on failure
def _transcribe(audio_path: Path) -> TranscriptResult   # faster-whisper → TranscriptResult
```

### Model

- **Model:** `faster-whisper` `small` (~244 MB), loaded once at import time (same pattern as PubMedBERT in `pipeline.py`)
- First import triggers a one-time model download to the HuggingFace cache
- Runs on CPU; no GPU required

### URL validation

`transcribe_url` checks the URL against `r'https?://(www\.)?tiktok\.com/'` before attempting download. Raises `ValueError` for non-matching URLs.

### Temp file handling

`_download_audio` writes to `tempfile.mkdtemp()`. `transcribe_url` wraps the full operation in `try/finally` to guarantee temp directory cleanup on both success and failure.

### Error surface

| Condition | Exception |
|---|---|
| Non-TikTok URL | `ValueError` |
| yt-dlp download failure | `RuntimeError` |
| Empty transcript (music-only / silent) | `RuntimeError("No speech detected in audio.")` |
| faster-whisper failure | `RuntimeError` |

---

## 5. Flask Route — `POST /transcribe`

**Request:**
```json
{ "url": "https://www.tiktok.com/@user/video/..." }
```

**Responses:**

| Status | Body |
|---|---|
| `200` | `{ "transcript_text": "...", "segments": [{...}], "duration": 134.6 }` |
| `400` | `{ "error": "No URL provided." }` |
| `400` | `{ "error": "URL does not appear to be a TikTok link." }` |
| `422` | `{ "error": "No speech detected in audio." }` |
| `500` | `{ "error": "<yt-dlp or faster-whisper message>" }` |

The route catches `ValueError` → 400, all `RuntimeError` → 500, with one exception: if `str(exc) == "No speech detected in audio."` it returns 422 instead. `_transcribe` raises that exact string for the empty-transcript case, making the distinction unambiguous.

---

## 6. UI Changes (`index.html`)

### URL bar (always visible above the textarea)

- Compact URL `<input>` + "Transcribe" button row, same full width as the textarea
- Transcribe button is dimmed (`opacity: 0.4`) when the input is empty
- Placeholder text: `"TikTok URL (optional)"`

### Loading state (during transcription)

- Transcribe button shows shimmer animation (matches existing Analyze loading state)
- Textarea and Analyze button disabled while transcribing

### Post-transcription state

- Transcript text fills the textarea (user can edit before analyzing)
- Transcribe button label changes to "Retranscribe"
- A collapsible "Transcript Reference" panel appears between the textarea and the Analyze button, showing timestamped segments (`start – end: text`)
- Character count updates to reflect transcript length

### Error state

- Transcription errors render as a brief text message replacing the shimmer (same visual pattern as `/analyze` errors)

### No changes to existing analyze flow

The Analyze Manuscript button, result cards, and all existing behavior are unchanged.

---

## 7. Notebook Integration (`MAVEN_AI_UNC_SPR2026.ipynb`)

A new **"TikTok Audio Transcription"** section is inserted between "Pipeline Overview" and "Text Segmentation." It contains:

1. **Install cell:** `!pip install faster-whisper yt-dlp`
2. **Function definitions:** inline `download_audio()` and `transcribe_url()` (mirrors `transcription.py` logic; no cross-import from `maven_app/`)
3. **Demo cell:** user sets `VIDEO_URL = "..."` → `result = transcribe_url(VIDEO_URL)` → `df = score_text(result.text)`
4. **Reference cell:** displays `result.segments` as a DataFrame (`start`, `end`, `text` columns)

---

## 8. Testing (`test_transcription.py`)

| Test | Method |
|---|---|
| `_download_audio` with mocked yt-dlp subprocess | Verifies correct CLI args; no network call |
| `transcribe_url` with short `.wav` fixture | Verifies `TranscriptResult` shape; avoids model download in CI |
| `POST /transcribe` — invalid URL | Flask test client → assert 400 |
| `POST /transcribe` — no URL | Flask test client → assert 400 |
| `POST /transcribe` — valid mock | Flask test client → assert 200, correct response keys |

---

## 9. Dependencies

Add to `maven_app/requirements.txt`:
```
faster-whisper
yt-dlp
```

No version pins at design time; pin to latest stable at implementation time.

---

## 10. Open Questions

None — all decisions resolved during design session.
