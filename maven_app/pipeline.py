"""
MAVEN Pipeline — retrieve-and-verify scoring.
score_text() is the single public entry point.

Flow: chunk -> embed (PubMedBERT) -> retrieve reference claims ->
NLI stance verification -> calibrated P(misinfo) -> flag + explainability.
Reference data: maven_app/anchors/reference_library.json (build with
scripts/build_reference_library.py; embedding cache under anchors/_cache/).
"""
import re
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

from embedding import embed
import retrieval
import scoring

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SENTENCE_THRESHOLD = 300
PARAGRAPH_THRESHOLD = 3_000


# Text segmentation (unchanged from the legacy pipeline)
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


def chunk_text(text: str, mode: str = 'auto', window: int = 200,
               stride: int = 100) -> Tuple[List[str], str]:
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


# Public API
def score_text(
    text: str,
    chunk_mode: str = 'auto',
    window: int = 200,
    stride: int = 100,
    batch_size: int = 32,
    flag_threshold: float = None,
) -> pd.DataFrame:
    """
    Intake any body of text; return a DataFrame of per-chunk misinformation scores.

    misinfo_score is P(misinfo) — heuristic until a calibration artifact
    exists, calibrated after (see scoring.py). flagged = scoreable and
    misinfo_score >= tau (tau from the calibration artifact, else 0.5;
    flag_threshold overrides both). matched_claim / evidence_correction /
    misinfo_type / misinfo_type_confidence populate only on flagged rows.
    stance is explanatory metadata: asserts_misinfo / contradicts_guidance /
    debunks_misinfo / on_topic_neutral / off_topic.
    """
    chunks, mode_used = chunk_text(text, mode=chunk_mode, window=window, stride=stride)
    if not chunks:
        return pd.DataFrame()

    chunk_embs = embed(chunks, batch_size=batch_size)
    results = scoring.score_chunks(chunks, chunk_embs)
    tau = flag_threshold if flag_threshold is not None else scoring.active_tau()

    rows = []
    for chunk, r in zip(chunks, results):
        flagged = bool(r.scoreable and r.p_misinfo >= tau)
        matched = r.matched if flagged and r.matched is not None else None

        evidence_correction = None
        misinfo_type = None
        misinfo_type_confidence = None
        if matched is not None:
            evidence_correction = matched['correction']
            misinfo_type = matched['type_id']
            misinfo_type_confidence = round(r.misinfo_entail, 4)
        elif flagged and r.contradicted is not None:
            # No cataloged claim was entailed (matched_claim stays None), but
            # a flagged row still needs an explanation: fall back to the
            # authority statement whose contradiction drove the flag. If that
            # authority entry links back to a base misinfo claim, surface its
            # type the same way the matched path does; otherwise leave it None.
            evidence_correction = r.contradicted['text']
            if r.contradicted['parent_id']:
                claim = retrieval.base_entry(r.contradicted)
                misinfo_type = claim['type_id']
                misinfo_type_confidence = round(r.misinfo_entail, 4)

        rows.append({
            'chunk':                   chunk,
            'chunk_mode':              mode_used,
            'misinfo_entail':          round(r.misinfo_entail, 4),
            'guidance_contradict':     round(r.guidance_contradict, 4),
            'misinfo_contradict':      round(r.misinfo_contradict, 4),
            'top_claim_sim':           round(r.top_claim_sim, 4),
            'top_auth_sim':            round(r.top_auth_sim, 4),
            'stance':                  r.stance,
            'scoreable':               r.scoreable,
            'misinfo_score':           round(r.p_misinfo, 4),
            'flagged':                 flagged,
            'matched_claim':           matched['text'] if matched else None,
            'evidence_correction':     evidence_correction,
            'misinfo_type':            misinfo_type,
            'misinfo_type_confidence': misinfo_type_confidence,
        })
    return pd.DataFrame(rows)
