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
  pipeline.py         # Public entry point: chunk → retrieve → verify → P(misinfo)
  embedding.py        # PubMedBERT sentence-embedding model (single load)
  retrieval.py        # Reference-library retrieval + on-topic gate
  verifier.py         # DeBERTa-v3 NLI cross-encoder (stance verification)
  scoring.py          # Feature aggregation, stance, calibrated P(misinfo)
  video_source.py     # Shared plumbing: URL dispatch, ffmpeg, yt-dlp helpers
  tiktok.py           # Thin TikTok platform definition
  instagram.py        # Thin Instagram Reels platform definition
  transcription.py    # Audio mode: TikTok audio → faster-whisper transcript
  text_extraction.py  # Text mode: frame OCR (RapidOCR) + video description
  slideshow.py        # Slideshow posts: per-slide OCR for TikTok /photo/ and Instagram /p/
  requirements.txt
  anchors/            # reference_library.json + embedding cache
  models/             # Calibration head artifact (when fitted)
  templates/          # Jinja2 HTML templates
  tests/              # End-to-end and calibration tests
```

## Notebook Structure

The notebook documents the pipeline with narrative explanations, organized into four sections:

1. **Pipeline Overview** — architecture diagram and scale-handling strategy
2. **Text Segmentation** — `chunk_text()` dispatches to sentence / paragraph / sliding-window based on token count
3. **PubMedBERT Embeddings** — `embed()` wraps `NeuML/pubmedbert-base-embeddings` for batched encoding at any scale
4. **Retrieve-and-Verify Scoring** — retrieves top-k misinfo/authority reference claims per chunk, runs NLI stance verification, and aggregates to `score_text()`'s scored DataFrame with `stance` and `flagged` columns

## Key Dependencies

| Library | Purpose |
|---|---|
| `sentence-transformers` | PubMedBERT embedding model |
| `scikit-learn` | Logistic calibration head |
| `transformers` + `sentencepiece` | DeBERTa-v3 NLI verifier (stance verification) |
| `nltk` (punkt) | Sentence tokenization |
| `pandas` | Tabular results (DataFrames) |
| `rapidocr-onnxruntime` | Frame OCR for TikTok text mode |
| `gallery-dl` | Slideshow (photo post) image + metadata download |

## Running Tests

Tests are plain Python scripts, not pytest. Run from `maven_app/`:

```bash
cd maven_app
python tests/test_scoring.py
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
  for instagram.com (browser extension, or
  `~/.config/maven/export_instagram_cookies.py` on the dev machine, which pulls
  them from Chrome via yt-dlp) and set `MAVEN_IG_COOKIES=/path/to/cookies.txt`
  before starting the app. Without it, Instagram slideshow requests return a
  friendly error. When set, the cookies are also passed to yt-dlp for Instagram
  Reels, which reduces anonymous rate-limit failures. Keep the file `chmod 600`
  and out of the repo — it is the account's live session.
- The cookie-authenticated Instagram path was verified end-to-end on
  2026-07-16 (`python tests/test_slideshow.py --live` with `MAVEN_IG_COOKIES`
  set: caption + 12 ordered slide segments from a real /p/ carousel). New
  deployments/accounts should repeat that one-time live run. If slideshow
  requests start returning the cookie error again, the session expired —
  re-export the cookies.
- TikTok short links (`vm.tiktok.com/...`) to photo posts are not detected as
  slideshows and will fail — use the full `/photo/` URL instead.
- After cloning, run `git config core.hooksPath .githooks` once to enable the
  pre-commit guard that blocks cookie exports from being committed (defense
  beyond .gitignore — catches `git add -f` and renamed cookie files).

## Pipeline Entry Point

`score_text(text)` in `maven_app/pipeline.py` is the end-to-end function:
chunk → PubMedBERT embed → retrieve reference claims (misinfo + authority,
on-topic gate) → DeBERTa-v3 NLI stance verification → P(misinfo) with
threshold τ (heuristic 0.5 until `maven_app/models/calibration_head.joblib`
exists — fit it with `ml/training/fit_calibration.py`). Returns a
DataFrame: `chunk, chunk_mode, misinfo_entail, guidance_contradict,
misinfo_contradict, top_claim_sim, top_auth_sim, stance, scoreable,
misinfo_score, flagged, matched_claim, evidence_correction, misinfo_type,
misinfo_type_confidence`.

The reference library is built by `scripts/build_reference_library.py`
from the domain .docx assets + `ml/data/claim_type_map.json` +
`ml/data/claim_paraphrases.json`. Evaluation lives in `ml/eval/`
(`run_eval.py`, labeled set, reports); training in `ml/training/`.
Env vars: `MAVEN_VERIFIER_PATH` (verifier checkpoint),
`MAVEN_CALIBRATION_PATH` (calibration artifact). Production adopts the
locally fine-tuned `maven-verifier-v1` checkpoint via `MAVEN_VERIFIER_PATH`
— reproduce with `.venv/bin/python ml/training/finetune_verifier.py --train ml/data/nli_pairs.jsonl --out ml/training/checkpoints/maven-verifier-v1 --cpu`
(checkpoint is local-only/gitignored) — see
`ml/eval/reports/2026-07-18-retrieve-verify-finetuned.md`.
