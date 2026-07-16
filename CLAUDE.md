# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A health misinformation detection system for the UNC Department of Maternal and Child Health (MCH), Spring 2026. MAVEN exists in two forms that share the same underlying pipeline:

- **Flask app** (`maven_app/`) — the runnable inference interface; accepts text input and returns misinformation scores via a web UI
- **Jupyter notebook** (`MAVEN_AI_UNC_SPR2026.ipynb`) — documented walkthrough of the same pipeline; the primary deliverable for Colab-based review and reproducibility

**Colab link:** https://colab.research.google.com/drive/1F4g-JPFI6RhhFU2QdoYmiPrOZwoQNAcX?usp=sharing

**PRD:** `PRD.md` — authoritative source for scope, requirements, and milestones.

## Running the Flask App

Requires **Python 3.10+** — yt-dlp dropped Python 3.9 support after its
2025.10.14 release, and older extractor versions fail against current
Instagram ("empty media response"). The project venv is built with Homebrew
`python3.11`.

```bash
cd maven_app
pip install -r requirements.txt
python app.py
```

The app runs at `http://localhost:5000` and exposes a web UI for real-time text flagging.

## Running the Notebook

The notebook is designed to run on **Google Colab**. Open the Colab link above or upload the `.ipynb` to Colab. Dependencies are installed via `!pip install` cells at the top of each section.

For local development: `jupyter notebook MAVEN_AI_UNC_SPR2026.ipynb`

## Flask App Structure

```
maven_app/
  app.py              # Flask routes and request handling
  pipeline.py         # Shared inference pipeline (chunk → embed → score)
  video_source.py     # Shared plumbing: URL dispatch, ffmpeg, yt-dlp helpers
  tiktok.py           # Thin TikTok platform definition
  instagram.py        # Thin Instagram Reels platform definition
  transcription.py    # Audio mode: TikTok audio → faster-whisper transcript
  text_extraction.py  # Text mode: frame OCR (RapidOCR) + video description
  slideshow.py        # Slideshow posts: per-slide OCR for TikTok /photo/ and Instagram /p/
  requirements.txt
  anchors/            # Authority and misinfo anchor JSON files
  templates/          # Jinja2 HTML templates
  tests/              # End-to-end and calibration tests
```

## Notebook Structure

The notebook documents the pipeline with narrative explanations, organized into four sections:

1. **Pipeline Overview** — architecture diagram and scale-handling strategy
2. **Text Segmentation** — `chunk_text()` dispatches to sentence / paragraph / sliding-window based on token count
3. **PubMedBERT Embeddings** — `embed()` wraps `NeuML/pubmedbert-base-embeddings` for batched encoding at any scale
4. **Misinformation Markers** — `compute_markers()` produces four scored signals per chunk; `score_text()` returns a scored DataFrame with a `flagged` column

## Key Dependencies

| Library | Purpose |
|---|---|
| `sentence-transformers` | PubMedBERT embedding model |
| `scikit-learn` | Isolation Forest anomaly detection |
| `nltk` (punkt) | Sentence tokenization |
| `requests` + `beautifulsoup4` + `lxml` | Web scraping |
| `pandas` | Tabular results (DataFrames) |
| `wikipedia-api` | Wikipedia article fetching |
| `rapidocr-onnxruntime` | Frame OCR for TikTok text mode |
| `gallery-dl` | Slideshow (photo post) image + metadata download |

## Running Tests

Tests are plain Python scripts, not pytest. Run from `maven_app/`:

```bash
cd maven_app
python tests/test_transcription.py
```

`from app import app` in test files triggers PubMedBERT model loading (~30-60s on cold cache).
On Windows, test `main()` functions wrap stdout in UTF-8 to handle Unicode output: `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`.

## TikTok Transcription — Dependency Gotchas

- `curl_cffi` must be `>=0.10,<0.15` — yt-dlp rejects 0.15+ with `ImportError` at import time
- `imageio-ffmpeg` ships its binary as `ffmpeg-win-x86_64-v7.1.exe`, not `ffmpeg.exe` — `tiktok.ensure_ffmpeg()` normalizes this by copying it to `%TEMP%\maven_ffmpeg\ffmpeg.exe` on first use
- TikTok downloads require `--impersonate chrome` (handled automatically); without `curl_cffi` installed, all targets show as unavailable

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

## Pipeline Entry Point

`score_text(text)` in the final code cell is the end-to-end function. It accepts any string, auto-selects a chunking strategy, embeds with PubMedBERT, computes four misinformation markers, and returns a `pd.DataFrame` with columns: `chunk`, `authority_sim`, `misinfo_sim`, `claim_delta`, `isolation_score`, `misinfo_score`, `flagged`.

The `AUTHORITY_ANCHORS` and `MISINFO_ANCHORS` lists in the markers cell are placeholders. Replace them with domain-validated claims before Milestone 1.
