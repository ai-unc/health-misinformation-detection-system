"""Unit test for calibration fitting helpers (no models loaded)."""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/training
from fit_calibration import choose_tau, fit_head, has_two_classes, save_artifact

FEATURES = ['misinfo_entail', 'guidance_contradict', 'misinfo_contradict',
            'top_claim_sim', 'top_auth_sim']


def main():
    rng = np.random.default_rng(42)
    n = 200
    # Synthetic world: misinfo rows have high entail, debunk rows high c_m.
    y = rng.integers(0, 2, n)
    X = np.zeros((n, 5))
    X[:, 0] = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.3, n))
    X[:, 1] = rng.uniform(0, 0.3, n)
    X[:, 2] = np.where(y == 1, rng.uniform(0.0, 0.2, n), rng.uniform(0.2, 0.9, n))
    X[:, 3] = rng.uniform(0.4, 0.9, n)
    X[:, 4] = rng.uniform(0.4, 0.9, n)

    model = fit_head(X, y)
    probs = model.predict_proba(X)[:, 1]
    # Separable synthetic data -> strong fit, monotone in entail
    assert ((probs > 0.5) == (y == 1)).mean() > 0.9

    # Single-class guard: main() must refuse to call fit_head when the
    # scored sample has only one label (fit_head itself has no way to
    # produce a meaningful decision boundary from one class).
    assert has_two_classes(y.tolist()) is True
    assert has_two_classes([1, 1, 1, 1]) is False
    assert has_two_classes([0, 0]) is False
    assert has_two_classes([0, 1, 0, 1]) is True

    tau = choose_tau(y, probs)
    assert 0.05 <= tau <= 0.95

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'calibration_head.joblib'
        save_artifact(model, FEATURES, tau, {'n': n}, out)
        artifact = joblib.load(out)
        assert artifact['features'] == FEATURES
        assert artifact['tau'] == tau
        assert artifact['model'].predict_proba(X[:1]).shape == (1, 2)

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
