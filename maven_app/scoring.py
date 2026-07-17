"""Feature aggregation, stance derivation, and P(misinfo) for MAVEN chunks.

Heuristic score (until a calibration artifact exists):
    p = clip(max(misinfo_entail, guidance_contradict) - DEBUNK_DAMP * misinfo_contradict, 0, 1)

With a calibration artifact (maven_app/models/calibration_head.joblib,
fitted by ml/training/fit_calibration.py), p comes from the logistic
head and tau from the artifact. The flag decision itself lives in
pipeline.score_text (p >= tau); stance here is explanatory metadata only.
"""
import os
from pathlib import Path
from typing import List, NamedTuple, Optional

import joblib
import numpy as np

import retrieval
import verifier as verifier_mod
from retrieval import base_entry, retrieve

TAU = 0.5                 # heuristic flag threshold (calibration overrides)
ENTAIL_MIN = 0.5          # min prob for a stance to be claimed at all
DOMINANCE_MARGIN = 0.15   # how much contradict must beat entail to call a debunk
DEBUNK_DAMP = 0.5         # how strongly debunking suppresses the heuristic score
PAIR_CAP_ACTIVE = verifier_mod.PAIR_CAP

FEATURES = ['misinfo_entail', 'guidance_contradict', 'misinfo_contradict',
            'top_claim_sim', 'top_auth_sim']

_HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = Path(os.environ.get('MAVEN_CALIBRATION_PATH',
                                       _HERE / 'models' / 'calibration_head.joblib'))


class ChunkScore(NamedTuple):
    p_misinfo: float
    stance: str
    scoreable: bool
    misinfo_entail: float
    guidance_contradict: float
    misinfo_contradict: float
    top_claim_sim: float
    top_auth_sim: float
    matched: Optional[dict]
    contradicted: Optional[dict]


def features_of(cs: ChunkScore) -> List[float]:
    return [getattr(cs, name) for name in FEATURES]


_calibration = None
_calibration_loaded = False


def _get_calibration():
    global _calibration, _calibration_loaded
    if not _calibration_loaded:
        if CALIBRATION_PATH.exists():
            artifact = joblib.load(CALIBRATION_PATH)
            if artifact.get('features') != FEATURES:
                raise RuntimeError(
                    f'Calibration artifact features {artifact.get("features")} '
                    f'!= scoring.FEATURES {FEATURES}; refit with ml/training/fit_calibration.py'
                )
            _calibration = artifact
            print(f'[MAVEN] Calibration head loaded (tau={artifact["tau"]:.3f}).')
        else:
            print('[MAVEN] No calibration artifact; using heuristic score '
                  f'(tau={TAU}).')
        # Marked only on success so a bad artifact keeps raising on every
        # call instead of silently degrading to the heuristic after the
        # first failure. Prints above still happen once per process.
        _calibration_loaded = True
    return _calibration


def active_tau() -> float:
    artifact = _get_calibration()
    return float(artifact['tau']) if artifact else TAU


def _stance(scoreable, e_m, c_m, c_a) -> str:
    if not scoreable:
        return 'off_topic'
    if c_m >= ENTAIL_MIN and c_m > e_m + DOMINANCE_MARGIN:
        return 'debunks_misinfo'
    if e_m >= ENTAIL_MIN:
        return 'asserts_misinfo'
    if c_a >= ENTAIL_MIN:
        return 'contradicts_guidance'
    return 'on_topic_neutral'


def _p_misinfo(features: List[float]) -> float:
    artifact = _get_calibration()
    if artifact is not None:
        return float(artifact['model'].predict_proba(
            np.array(features).reshape(1, -1))[0, 1])
    e_m, c_a, c_m = features[0], features[1], features[2]
    return float(np.clip(max(e_m, c_a) - DEBUNK_DAMP * c_m, 0.0, 1.0))


def _allocate_pairs(chunks, results, cap):
    """Round-robin (premise, hypothesis) allocation under the pair cap.

    Each chunk's candidates are ordered [mis[0], auth[0], mis[1], auth[1], ...];
    round r hands every chunk its r-th candidate (chunks in order) before any
    chunk gets its (r+1)-th, so a tight cap trims candidate depth everywhere
    instead of starving tail chunks of their first pair. Deterministic.
    """
    per_chunk = []
    for ci, r in enumerate(results):
        cands = []
        for i in range(max(len(r.misinfo), len(r.authority))):
            if i < len(r.misinfo):
                cands.append((ci, 'misinfo', r.misinfo[i]))
            if i < len(r.authority):
                cands.append((ci, 'authority', r.authority[i]))
        per_chunk.append(cands)

    pairs, index = [], []  # index[i] = (chunk_idx, kind, Retrieved)
    for rank in range(max((len(c) for c in per_chunk), default=0)):
        for cands in per_chunk:
            if len(pairs) >= cap:
                return pairs, index
            if rank < len(cands):
                ci, kind, cand = cands[rank]
                pairs.append((chunks[ci], cand.entry['text']))
                index.append((ci, kind, cand))
    return pairs, index


def score_chunks(chunks: List[str], chunk_embs: np.ndarray, nli=None) -> List[ChunkScore]:
    if not chunks:
        return []
    results = retrieve(chunk_embs)

    # Latency guard: shrink per-kind k so total pairs stay under the cap.
    n_pairs = sum(len(r.misinfo) + len(r.authority) for r in results)
    if n_pairs > PAIR_CAP_ACTIVE:
        k_eff = max(1, PAIR_CAP_ACTIVE // (2 * len(chunks)))
        results = [type(r)(misinfo=r.misinfo[:k_eff], authority=r.authority[:k_eff])
                   for r in results]

    # k_eff floors at 1 per kind, so enough scoreable chunks can still exceed
    # the cap; the round-robin allocator enforces it by trimming candidate
    # depth everywhere rather than dropping tail chunks wholesale.
    pairs, index = _allocate_pairs(chunks, results, PAIR_CAP_ACTIVE)

    nli = nli or verifier_mod.get_verifier()
    probs = nli.predict(pairs) if pairs else np.empty((0, 3))

    scores: List[ChunkScore] = []
    for ci, r in enumerate(results):
        e_m = c_m = c_a = 0.0
        top_claim_sim = max((c.sim for c in r.misinfo), default=0.0)
        top_auth_sim = max((c.sim for c in r.authority), default=0.0)
        matched: Optional[dict] = None
        contradicted: Optional[dict] = None
        best_entail = -1.0
        best_contra = -1.0
        for (pci, kind, cand), row in zip(index, probs):
            if pci != ci:
                continue
            entail, _, contradict = float(row[0]), float(row[1]), float(row[2])
            if kind == 'misinfo':
                c_m = max(c_m, contradict)
                if entail > best_entail:
                    best_entail = entail
                    matched = base_entry(cand.entry)
                e_m = max(e_m, entail)
            else:
                c_a = max(c_a, contradict)
                # Argmax authority pair: the retrieved authority entry whose
                # contradiction is highest for this chunk. Unlike `matched`,
                # never nulled below — it's the explanation-of-last-resort
                # for a flagged chunk that entailed no cataloged claim.
                if contradict > best_contra:
                    best_contra = contradict
                    contradicted = cand.entry

        scoreable = r.scoreable
        features = [e_m, c_a, c_m, top_claim_sim, top_auth_sim]
        p = _p_misinfo(features) if scoreable else 0.0
        stance = _stance(scoreable, e_m, c_m, c_a)
        if stance != 'asserts_misinfo' and e_m < ENTAIL_MIN:
            matched = None  # only surface a matched claim the chunk plausibly asserts
        scores.append(ChunkScore(
            p_misinfo=p, stance=stance, scoreable=scoreable,
            misinfo_entail=e_m, guidance_contradict=c_a, misinfo_contradict=c_m,
            top_claim_sim=top_claim_sim, top_auth_sim=top_auth_sim,
            matched=matched, contradicted=contradicted,
        ))
    return scores
