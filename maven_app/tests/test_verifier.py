"""NLI verifier sanity on canonical perinatal stance triplets.

First run downloads the checkpoint (~700 MB) — allow several minutes.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from verifier import Verifier

CLAIM = "Epidurals damage the baby's brain."
CASES = [
    # (premise, expected argmax column: 0=entail, 1=neutral, 2=contradict)
    ("Epidurals damage the baby's brain and always lead to a c-section.", 0),
    ("No, epidurals do not damage your baby's brain — that is a myth.", 2),
    ("Packing my hospital bag tonight, 38 weeks tomorrow!", 1),
]


def main():
    v = Verifier()
    pairs = [(premise, CLAIM) for premise, _ in CASES]
    t0 = time.time()
    probs = v.predict(pairs)
    elapsed = time.time() - t0
    print(f'  {len(pairs)} pairs in {elapsed:.2f}s ({elapsed / len(pairs):.2f}s/pair)')

    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    for (premise, expected), row in zip(CASES, probs):
        got = int(row.argmax())
        print(f'  [{["ENTAIL", "NEUTRAL", "CONTRA"][got]}] '
              f'e={row[0]:.3f} n={row[1]:.3f} c={row[2]:.3f}  {premise[:60]}')
        assert got == expected, f'{premise[:60]}: expected col {expected}, got {got}'

    assert v.predict([]).shape == (0, 3)
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
