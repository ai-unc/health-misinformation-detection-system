# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A health misinformation detection system for the UNC Department of Maternal and Child Health (MCH), Spring 2026. The primary deliverable is `MAVEN_AI_UNC_SPR2026.ipynb`, a Google Colab notebook.

**Colab link:** https://colab.research.google.com/drive/1F4g-JPFI6RhhFU2QdoYmiPrOZwoQNAcX?usp=sharing

**PRD:** `PRD.md` — authoritative source for scope, requirements, and milestones.

## Running the Notebook

This project is designed to run on **Google Colab**, not locally. Open the Colab link above or upload the `.ipynb` to Colab. Dependencies are installed via `!pip install` cells at the top of each section.

For local development: `jupyter notebook MAVEN_AI_UNC_SPR2026.ipynb`

## Notebook Structure

The notebook contains the maternal health misinformation detection pipeline, organized into four sections:

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

## Pipeline Entry Point

`score_text(text)` in the final code cell is the end-to-end function. It accepts any string, auto-selects a chunking strategy, embeds with PubMedBERT, computes four misinformation markers, and returns a `pd.DataFrame` with columns: `chunk`, `authority_sim`, `misinfo_sim`, `claim_delta`, `isolation_score`, `misinfo_score`, `flagged`.

The `AUTHORITY_ANCHORS` and `MISINFO_ANCHORS` lists in the markers cell are placeholders. Replace them with domain-validated claims before Milestone 1.
