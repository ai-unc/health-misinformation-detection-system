"""NLI cross-encoder verifier: stance of a chunk against a reference claim.

predict() returns probs in FIXED column order [entail, neutral, contradict],
mapped from the checkpoint's id2label so fine-tuned checkpoints with a
different internal order keep working. Override the checkpoint with the
MAVEN_VERIFIER_PATH env var (points at a HF model id or local dir).
"""
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CHECKPOINT = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
PAIR_CAP = 256          # max NLI pairs per request (latency guard)
MAX_LENGTH = 256        # token truncation per pair

_COLUMN_FOR_LABEL = {'entailment': 0, 'entail': 0,
                     'neutral': 1,
                     'contradiction': 2, 'contradict': 2}


class Verifier:
    def __init__(self, checkpoint=None, device=None):
        self.checkpoint = (checkpoint
                           or os.environ.get('MAVEN_VERIFIER_PATH')
                           or DEFAULT_CHECKPOINT)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[MAVEN] Loading NLI verifier {self.checkpoint!r} on {self.device}...')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint, torch_dtype=torch.float32).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load NLI verifier {self.checkpoint!r} '
                f'(set MAVEN_VERIFIER_PATH to a valid checkpoint): {exc}'
            ) from exc

        self._col_of = np.empty(self.model.config.num_labels, dtype=int)
        for idx, label in self.model.config.id2label.items():
            key = label.lower()
            if key not in _COLUMN_FOR_LABEL:
                raise RuntimeError(f'Unrecognized NLI label {label!r} in {self.checkpoint!r}')
            self._col_of[int(idx)] = _COLUMN_FOR_LABEL[key]
        print('[MAVEN] Verifier ready.')

    @torch.no_grad()
    def predict(self, pairs, batch_size=16):
        """(premise, hypothesis) pairs -> (n, 3) probs [entail, neutral, contradict]."""
        if not pairs:
            return np.empty((0, 3))
        out = np.empty((len(pairs), 3), dtype=np.float64)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch],
                                 truncation=True, max_length=MAX_LENGTH,
                                 padding=True, return_tensors='pt').to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            for model_idx in range(probs.shape[1]):
                out[start:start + len(batch), self._col_of[model_idx]] = probs[:, model_idx]
        return out


_singleton = None


def get_verifier() -> Verifier:
    global _singleton
    if _singleton is None:
        _singleton = Verifier()
    return _singleton
