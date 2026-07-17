"""Shared PubMedBERT sentence-embedding model (single load per process)."""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_DIM = 768
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'[MAVEN] Loading PubMedBERT on {DEVICE}...')
_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', device=DEVICE)
print('[MAVEN] Model loaded.')


def embed(texts, batch_size=32, show_progress=False):
    if not texts:
        return np.empty((0, EMBED_DIM))
    return _model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
