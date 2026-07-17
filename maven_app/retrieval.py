"""Reference-library retrieval with an on-topic similarity gate.

Loads reference_library.json + a cached embedding matrix (rebuilt
automatically when the library JSON's hash changes; delete
anchors/_cache/ to force).
"""
import hashlib
import json
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from embedding import embed

K_PER_KIND = 4
TOPIC_FLOOR = 0.47

_HERE = Path(__file__).resolve().parent
LIBRARY_PATH = _HERE / 'anchors' / 'reference_library.json'
CACHE_DIR = _HERE / 'anchors' / '_cache'
CACHE_EMBS = CACHE_DIR / 'reference_embs.npy'
CACHE_META = CACHE_DIR / 'reference_meta.json'


class Retrieved(NamedTuple):
    entry: dict
    sim: float


class RetrievalResult(NamedTuple):
    misinfo: List[Retrieved]
    authority: List[Retrieved]

    @property
    def scoreable(self) -> bool:
        return bool(self.misinfo or self.authority)


def _library_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load():
    if not LIBRARY_PATH.exists():
        raise FileNotFoundError(
            f'Missing {LIBRARY_PATH}. Run `python scripts/build_reference_library.py` first.'
        )
    raw = LIBRARY_PATH.read_bytes()
    lib = json.loads(raw.decode('utf-8'))
    lib_hash = _library_hash(raw)

    if CACHE_EMBS.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text(encoding='utf-8'))
        if meta.get('hash') == lib_hash:
            print('[MAVEN] Loading cached reference embeddings...')
            return lib, np.load(CACHE_EMBS)

    print(f'[MAVEN] Embedding {len(lib)} reference entries (one-time)...')
    embs = embed([e['text'] for e in lib])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_EMBS, embs)
    CACHE_META.write_text(json.dumps({'hash': lib_hash, 'n': len(lib)}),
                          encoding='utf-8')
    return lib, embs


_library, _ref_embs = _load()
_by_id = {e['id']: e for e in _library}
_mis_idx = np.array([i for i, e in enumerate(_library) if e['kind'] == 'misinfo'])
_auth_idx = np.array([i for i, e in enumerate(_library) if e['kind'] == 'authority'])
print(f'[MAVEN] Reference library ready: {len(_mis_idx)} misinfo, {len(_auth_idx)} authority.')


def base_entry(entry: dict) -> dict:
    """Resolve a paraphrase entry to its base claim (identity for base entries)."""
    return _by_id[entry['parent_id']] if entry['parent_id'] else entry


def _top_k(sims: np.ndarray, idx: np.ndarray, k: int, floor: float,
           dedup_parents: bool) -> List[Retrieved]:
    order = idx[np.argsort(-sims[idx])]
    out: List[Retrieved] = []
    seen_parents = set()
    for i in order:
        sim = float(sims[i])
        if sim < floor or len(out) >= k:
            break
        entry = _library[i]
        if dedup_parents:
            parent = base_entry(entry)['id']
            if parent in seen_parents:
                continue
            seen_parents.add(parent)
        out.append(Retrieved(entry, sim))
    return out


def retrieve(chunk_embs: np.ndarray, k: int = K_PER_KIND,
             floor: float = TOPIC_FLOOR) -> List[RetrievalResult]:
    """Top-k reference candidates per chunk, per kind, above the topic floor."""
    if chunk_embs.size == 0:
        return []
    sims_all = chunk_embs @ _ref_embs.T  # embeddings are L2-normalized
    results = []
    for sims in sims_all:
        results.append(RetrievalResult(
            misinfo=_top_k(sims, _mis_idx, k, floor, dedup_parents=True),
            authority=_top_k(sims, _auth_idx, k, floor, dedup_parents=False),
        ))
    return results
