"""Validate the deterministic synthetic NLI pair builder."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'ml' / 'data' / 'nli_pairs.jsonl'


def main():
    result = subprocess.run([sys.executable, str(ROOT / 'ml/data/build_nli_pairs.py')],
                            capture_output=True, text=True)
    print(result.stdout)
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in OUT.read_text(encoding='utf-8').splitlines()]
    labels = Counter(r['label'] for r in rows)
    print('label counts:', dict(labels))

    assert set(labels) == {'entailment', 'neutral', 'contradiction'}
    assert labels['entailment'] >= 70      # >= one per base claim
    assert labels['contradiction'] >= 140  # correction + debunk template per claim
    assert labels['neutral'] >= 70
    for r in rows[:50]:
        assert r['premise'] and r['hypothesis'] and r['premise'] != r['hypothesis']

    # Task 8 carry-in: verify clean-chatter neutral pairs are present
    chatter_neutrals = sum(1 for r in rows
                           if r['label'] == 'neutral' and 'hospital bag' in r['premise'])
    assert chatter_neutrals >= 2, 'clean-chatter neutral pairs missing'

    # determinism
    subprocess.run([sys.executable, str(ROOT / 'ml/data/build_nli_pairs.py')],
                   capture_output=True, text=True, check=True)
    rows2 = [json.loads(line) for line in OUT.read_text(encoding='utf-8').splitlines()]
    assert rows == rows2, 'builder is not deterministic'

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
