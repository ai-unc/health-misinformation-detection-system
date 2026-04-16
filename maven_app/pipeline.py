"""
MAVEN Pipeline: extracted from MAVEN_AI_UNC_SPR2026.ipynb
Exposes score_text() as the single public entry point.
"""

import re
from typing import List, Tuple

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

# Anchor sets. TODO: replace with domain-validated claims before Milestone 1 
AUTHORITY_ANCHORS: List[str] = [
    'Prenatal folic acid supplementation significantly reduces the risk of neural tube defects.',
    'The CDC and WHO recommend influenza vaccination during any trimester of pregnancy.',
    'Gestational diabetes is managed through diet, exercise, and medication when necessary.',
    'Breastfeeding provides immune protection to infants through maternal antibodies.',
    'Regular prenatal care is associated with reduced maternal and infant mortality.',
    'Preeclampsia is characterized by high blood pressure and requires clinical management.',
    'Group B Streptococcus screening is routinely performed at 36 weeks of gestation.',
    'Maternal pertussis vaccination protects newborns before they can be immunized.',
    'Low-dose aspirin may be recommended to reduce preeclampsia risk in high-risk pregnancies.',
    'Postpartum depression affects approximately 1 in 8 women and is treatable.',
]

MISINFO_ANCHORS: List[str] = [
    'Vaccines during pregnancy cause miscarriage and permanent fetal harm.',
    'Epidurals damage the spine and impair fetal brain development.',
    'Folic acid supplements are unnecessary and potentially toxic to the fetus.',
    'Ultrasounds during pregnancy are dangerous and should be avoided.',
    'C-sections cause long-term immune deficiencies in children.',
    'Herbal teas during pregnancy can replace prenatal vitamins entirely.',
    'Breastfeeding mothers must avoid all medications, including prescribed antibiotics.',
    'Induced labor always leads to worse outcomes than spontaneous labor.',
    'Gestational diabetes can be cured by eating only organic foods.',
    'Natural childbirth without any medical intervention is always the safest option.',
]

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


# Centroids & Isolation Forest (built once) 
def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


print('[MAVEN] Embedding anchors...')
_authority_embs = embed(AUTHORITY_ANCHORS)
_misinfo_embs   = embed(MISINFO_ANCHORS)

_authority_centroid = _l2(_authority_embs.mean(axis=0))[np.newaxis, :]
_misinfo_centroid   = _l2(_misinfo_embs.mean(axis=0))[np.newaxis, :]

_iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
_iso_forest.fit(_authority_embs)
print('[MAVEN] Anchors ready.')


# Marker computation 
def compute_markers(chunk_embs: np.ndarray) -> dict:
    auth_sim    = (chunk_embs @ _authority_centroid.T).flatten()
    misinfo_sim = (chunk_embs @ _misinfo_centroid.T).flatten()
    claim_delta = auth_sim - misinfo_sim

    iso_raw   = -_iso_forest.decision_function(chunk_embs)
    lo, hi    = iso_raw.min(), iso_raw.max()
    iso_score = (iso_raw - lo) / (hi - lo + 1e-10)

    return {
        'authority_sim':   auth_sim,
        'misinfo_sim':     misinfo_sim,
        'claim_delta':     claim_delta,
        'isolation_score': iso_score,
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

    Columns: chunk, chunk_mode, authority_sim, misinfo_sim, claim_delta,
             isolation_score, misinfo_score, flagged
    """
    chunks, mode_used = chunk_text(text, mode=chunk_mode, window=window, stride=stride)
    if not chunks:
        return pd.DataFrame()

    chunk_embs = embed(chunks, batch_size=batch_size)
    markers    = compute_markers(chunk_embs)

    delta_clipped = np.clip(markers['claim_delta'], -1.0, 1.0)
    delta_score   = (1.0 - delta_clipped) / 2.0
    misinfo_score = 0.70 * delta_score + 0.30 * markers['isolation_score']

    return pd.DataFrame({
        'chunk':           chunks,
        'chunk_mode':      mode_used,
        'authority_sim':   markers['authority_sim'].round(4),
        'misinfo_sim':     markers['misinfo_sim'].round(4),
        'claim_delta':     markers['claim_delta'].round(4),
        'isolation_score': markers['isolation_score'].round(4),
        'misinfo_score':   misinfo_score.round(4),
        'flagged':         misinfo_score >= flag_threshold,
    })
