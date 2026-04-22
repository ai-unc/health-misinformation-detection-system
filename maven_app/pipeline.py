"""
MAVEN Pipeline: extracted from MAVEN_AI_UNC_SPR2026.ipynb
Exposes score_text() as the single public entry point.

Anchor sets and the IsolationForest training corpus are loaded from
JSON files in maven_app/anchors/ (produced by scripts/build_anchors.py
from the domain .docx assets). Embedding + fitted-forest artifacts are
cached under maven_app/anchors/_cache/ and rebuilt automatically on first
import after the JSON anchors change (delete the _cache/ dir to force a
rebuild).
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Constants
SENTENCE_THRESHOLD  = 300
PARAGRAPH_THRESHOLD = 3_000
EMBED_DIM           = 768
FLAG_THRESHOLD      = 0.60
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'

# Anchor file paths
_HERE        = Path(__file__).resolve().parent
ANCHOR_DIR   = _HERE / 'anchors'
CACHE_DIR    = ANCHOR_DIR / '_cache'

AUTHORITY_JSON   = ANCHOR_DIR / 'authority_anchors.json'
MISINFO_JSON     = ANCHOR_DIR / 'misinfo_anchors.json'
MISINFO_TYPE_JSON = ANCHOR_DIR / 'misinfo_type_anchors.json'
COMP_PARAS_JSON  = ANCHOR_DIR / 'comprehensive_paragraphs.json'

CACHE_AUTH_EMBS        = CACHE_DIR / 'authority_embs.npy'
CACHE_CLAIM_EMBS       = CACHE_DIR / 'misinfo_claim_embs.npy'
CACHE_TYPE_NPZ         = CACHE_DIR / 'misinfo_type_centroids.npz'
CACHE_ISO_FOREST       = CACHE_DIR / 'iso_forest.joblib'
CACHE_ISO_CALIBRATION  = CACHE_DIR / 'iso_calibration.npy'   # [lo, hi] from training corpus


# Model loading (once at import time)
print(f'[MAVEN] Loading PubMedBERT on {DEVICE}...')
_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', device=DEVICE)
print('[MAVEN] Model loaded.')


# Text segmentation
def _approx_tokens(text: str) -> int:
    return len(text.split())


def _by_sentence(text: str) -> List[str]:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]


def _by_paragraph(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n{2,}', text)]
    return [p for p in paras if len(p) > 40]


def _by_sliding_window(text: str, window: int = 200, stride: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk = ' '.join(words[i: i + window])
        if len(chunk) > 40:
            chunks.append(chunk)
        if i + window >= len(words):
            break
    return chunks


def chunk_text(
    text: str,
    mode: str = 'auto',
    window: int = 200,
    stride: int = 100,
) -> Tuple[List[str], str]:
    if mode == 'auto':
        n = _approx_tokens(text)
        if n < SENTENCE_THRESHOLD:
            mode = 'sentence'
        elif n < PARAGRAPH_THRESHOLD:
            mode = 'paragraph'
        else:
            mode = 'sliding_window'

    dispatch = {
        'sentence':       lambda: _by_sentence(text),
        'paragraph':      lambda: _by_paragraph(text),
        'sliding_window': lambda: _by_sliding_window(text, window, stride),
    }
    return dispatch[mode](), mode


# Embedding
def embed(texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBED_DIM))
    return _model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


# Anchor + cache loading
def _load_anchors():
    """Load all four JSON anchor files into Python objects."""
    if not AUTHORITY_JSON.exists():
        raise FileNotFoundError(
            f'Missing {AUTHORITY_JSON}. Run `python scripts/build_anchors.py` first.'
        )
    authority = json.loads(AUTHORITY_JSON.read_text(encoding='utf-8'))
    misinfo   = json.loads(MISINFO_JSON.read_text(encoding='utf-8'))
    misinfo_types = json.loads(MISINFO_TYPE_JSON.read_text(encoding='utf-8'))
    comp_paras = json.loads(COMP_PARAS_JSON.read_text(encoding='utf-8'))
    return authority, misinfo, misinfo_types, comp_paras


def _cache_complete() -> bool:
    return all(p.exists() for p in (
        CACHE_AUTH_EMBS, CACHE_CLAIM_EMBS, CACHE_TYPE_NPZ,
        CACHE_ISO_FOREST, CACHE_ISO_CALIBRATION,
    ))


def _build_and_persist_cache(authority, misinfo, misinfo_types, comp_paras):
    """Compute embeddings + fit IsolationForest, persist to CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[MAVEN] Embedding {len(authority)} authority anchors...')
    auth_embs = embed(authority)
    np.save(CACHE_AUTH_EMBS, auth_embs)

    print(f'[MAVEN] Embedding {len(misinfo)} misinfo claims...')
    claim_texts = [m['claim'] for m in misinfo]
    claim_embs = embed(claim_texts)
    np.save(CACHE_CLAIM_EMBS, claim_embs)

    print(f'[MAVEN] Embedding {sum(len(v) for v in misinfo_types.values())} misinfo-type seeds across {len(misinfo_types)} categories...')
    type_ids = list(misinfo_types.keys())
    type_centroids = np.zeros((len(type_ids), EMBED_DIM), dtype=np.float32)
    for i, tid in enumerate(type_ids):
        seed_embs = embed(misinfo_types[tid])
        type_centroids[i] = _l2(seed_embs.mean(axis=0))
    np.savez(CACHE_TYPE_NPZ, centroids=type_centroids, type_ids=np.array(type_ids))

    print(f'[MAVEN] Embedding {len(comp_paras)} comprehensive paragraphs (IsoForest training corpus)...')
    comp_embs = embed(comp_paras, show_progress=False)
    print(f'[MAVEN] Fitting IsolationForest on {len(comp_embs)} authoritative-text embeddings...')
    iso = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
    iso.fit(comp_embs)
    joblib.dump(iso, CACHE_ISO_FOREST)

    # Calibrate iso_score against training-corpus distribution so the same
    # raw score maps to the same normalized value regardless of input batch
    # (per-batch min-max would force a single-chunk request to iso_score=0).
    iso_raw_train = -iso.decision_function(comp_embs)
    iso_lo, iso_hi = np.percentile(iso_raw_train, [1, 99])
    np.save(CACHE_ISO_CALIBRATION, np.array([iso_lo, iso_hi], dtype=np.float64))
    print(f'[MAVEN] IsoForest calibration: lo={iso_lo:.4f} hi={iso_hi:.4f}')

    return (
        auth_embs, claim_embs, type_centroids, np.array(type_ids),
        iso, float(iso_lo), float(iso_hi),
    )


def _load_cache():
    auth_embs   = np.load(CACHE_AUTH_EMBS)
    claim_embs  = np.load(CACHE_CLAIM_EMBS)
    type_data   = np.load(CACHE_TYPE_NPZ, allow_pickle=False)
    type_centroids = type_data['centroids']
    type_ids       = type_data['type_ids']
    iso = joblib.load(CACHE_ISO_FOREST)
    iso_lo, iso_hi = np.load(CACHE_ISO_CALIBRATION).tolist()
    return auth_embs, claim_embs, type_centroids, type_ids, iso, float(iso_lo), float(iso_hi)


# Initialize anchors + centroids + forest at import time
print('[MAVEN] Loading anchors from JSON...')
_authority, _misinfo, _misinfo_types, _comp_paras = _load_anchors()
print(f'  authority anchors:    {len(_authority)}')
print(f'  misinfo claim/evidence pairs: {len(_misinfo)}')
print(f'  misinfo type categories: {len(_misinfo_types)}')
print(f'  comprehensive paragraphs (forest corpus): {len(_comp_paras)}')

if _cache_complete():
    print('[MAVEN] Loading cached embeddings + IsoForest...')
    (_authority_embs, _misinfo_claim_embs, _misinfo_type_centroids, _misinfo_type_ids,
     _iso_forest, _iso_lo, _iso_hi) = _load_cache()
else:
    print('[MAVEN] Cache miss — rebuilding (one-time, ~30-60s)...')
    (_authority_embs, _misinfo_claim_embs, _misinfo_type_centroids, _misinfo_type_ids,
     _iso_forest, _iso_lo, _iso_hi) = _build_and_persist_cache(
        _authority, _misinfo, _misinfo_types, _comp_paras,
    )

# Centroids (aggregate over individual anchor embeddings, then re-normalize)
_authority_centroid = _l2(_authority_embs.mean(axis=0))[np.newaxis, :]
_misinfo_centroid   = _l2(_misinfo_claim_embs.mean(axis=0))[np.newaxis, :]

# Convert type_ids back to plain list[str] for downstream lookup
_misinfo_type_ids_list: List[str] = [str(t) for t in _misinfo_type_ids.tolist()]

print('[MAVEN] Anchors ready.')


# Marker computation
def compute_markers(chunk_embs: np.ndarray) -> dict:
    """Compute per-chunk scoring signals.

    All cosine similarities below are dot products (embeddings are L2-normalized
    by `embed()`). Returns a dict with the four scalar markers used by the
    composite score, plus two matrices (per-claim and per-type sims) used by
    `score_text` to enrich flagged rows with explainability fields.
    """
    auth_sim    = (chunk_embs @ _authority_centroid.T).flatten()
    misinfo_sim = (chunk_embs @ _misinfo_centroid.T).flatten()
    claim_delta = auth_sim - misinfo_sim

    # Normalize against the training-corpus iso_raw distribution (cached
    # at build time) instead of per-batch min/max. Per-batch normalization
    # forces a single-chunk request to iso_score=0.0, which would cap the
    # composite score at 0.70 * delta_score and prevent any single-sentence
    # input from ever exceeding FLAG_THRESHOLD.
    iso_raw   = -_iso_forest.decision_function(chunk_embs)
    iso_score = np.clip((iso_raw - _iso_lo) / (_iso_hi - _iso_lo + 1e-10), 0.0, 1.0)

    # Per-claim sims:  shape (n_chunks, n_claims) — used to find the closest
    # documented misinfo claim for each flagged chunk.
    per_claim_sims = chunk_embs @ _misinfo_claim_embs.T

    # Per-type-centroid sims: shape (n_chunks, n_types)
    per_type_sims = chunk_embs @ _misinfo_type_centroids.T

    return {
        'authority_sim':   auth_sim,
        'misinfo_sim':     misinfo_sim,
        'claim_delta':     claim_delta,
        'isolation_score': iso_score,
        'per_claim_sims':  per_claim_sims,
        'per_type_sims':   per_type_sims,
    }


# Public API
def score_text(
    text: str,
    chunk_mode: str = 'auto',
    window: int = 200,
    stride: int = 100,
    batch_size: int = 32,
    flag_threshold: float = FLAG_THRESHOLD,
) -> pd.DataFrame:
    """
    Intake any body of text; return a DataFrame of per-chunk misinformation scores.

    Columns:
        chunk, chunk_mode, authority_sim, misinfo_sim, claim_delta,
        isolation_score, misinfo_score, flagged,
        matched_claim, evidence_correction, misinfo_type, misinfo_type_confidence

    The last four columns are populated only for rows where flagged=True; they
    carry, respectively, the closest documented misinfo CLAIM, its paired
    evidence-based correction, the JGIM-taxonomy category that best fits the
    chunk, and the cosine-similarity confidence of that category assignment.
    All four are None for non-flagged rows.
    """
    chunks, mode_used = chunk_text(text, mode=chunk_mode, window=window, stride=stride)
    if not chunks:
        return pd.DataFrame()

    chunk_embs = embed(chunks, batch_size=batch_size)
    markers    = compute_markers(chunk_embs)

    delta_clipped = np.clip(markers['claim_delta'], -1.0, 1.0)
    delta_score   = (1.0 - delta_clipped) / 2.0
    misinfo_score = 0.70 * delta_score + 0.30 * markers['isolation_score']
    flagged       = misinfo_score >= flag_threshold

    # Per-flagged-row enrichment (kept None elsewhere to keep payload small)
    n = len(chunks)
    matched_claim:        List[Optional[str]]   = [None] * n
    evidence_correction:  List[Optional[str]]   = [None] * n
    misinfo_type:         List[Optional[str]]   = [None] * n
    misinfo_type_conf:    List[Optional[float]] = [None] * n

    if flagged.any():
        per_claim = markers['per_claim_sims']
        per_type  = markers['per_type_sims']
        for i in np.where(flagged)[0]:
            best_claim_idx = int(per_claim[i].argmax())
            matched_claim[i]       = _misinfo[best_claim_idx]['claim']
            evidence_correction[i] = _misinfo[best_claim_idx]['evidence']

            best_type_idx = int(per_type[i].argmax())
            misinfo_type[i]      = _misinfo_type_ids_list[best_type_idx]
            misinfo_type_conf[i] = round(float(per_type[i][best_type_idx]), 4)

    return pd.DataFrame({
        'chunk':                   chunks,
        'chunk_mode':              mode_used,
        'authority_sim':           markers['authority_sim'].round(4),
        'misinfo_sim':             markers['misinfo_sim'].round(4),
        'claim_delta':             markers['claim_delta'].round(4),
        'isolation_score':         markers['isolation_score'].round(4),
        'misinfo_score':           misinfo_score.round(4),
        'flagged':                 flagged,
        'matched_claim':           matched_claim,
        'evidence_correction':     evidence_correction,
        'misinfo_type':            misinfo_type,
        'misinfo_type_confidence': misinfo_type_conf,
    })
