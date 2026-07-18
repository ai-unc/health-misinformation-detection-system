"""M1 baseline: logistic regression on PubMedBERT embeddings.

Trains on purely synthetic texts derived from the reference library
(positives: claims + paraphrases + assertion templates; negatives:
corrections + debunk templates + extracted authority statements), then
evaluates through the shared harness. This baseline has no stance
signal — expect it to flag debunks; that is the point of the comparison.

Usage (from repo root, venv active; loads PubMedBERT):
    python ml/training/train_baseline.py \
        [--out ml/eval/reports/2026-07-16-m1-baseline-logreg.md]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ml' / 'eval'))
sys.path.insert(0, str(ROOT / 'ml' / 'data'))
sys.path.insert(0, str(ROOT / 'maven_app'))

from build_nli_pairs import ASSERT_TEMPLATE, DEBUNK_TEMPLATE, _lower_first  # noqa: E402
from harness import evaluate, load_eval_set, render_report  # noqa: E402

LIB_PATH = ROOT / 'maven_app' / 'anchors' / 'reference_library.json'


def build_training_texts():
    lib = json.loads(LIB_PATH.read_text(encoding='utf-8'))
    misinfo = [e for e in lib if e['kind'] == 'misinfo']
    base = [e for e in misinfo if e['parent_id'] is None]
    authority = [e for e in lib if e['kind'] == 'authority']

    positives = [e['text'] for e in misinfo]
    positives += [ASSERT_TEMPLATE.format(claim=_lower_first(e['text'])) for e in base]
    negatives = [e['text'] for e in authority]
    negatives += [DEBUNK_TEMPLATE.format(claim=_lower_first(e['text'])) for e in base]
    return positives, negatives


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(
        ROOT / f'ml/eval/reports/{date.today().isoformat()}-m1-baseline-logreg.md'))
    args = parser.parse_args()

    print('[baseline] importing embedding model...')
    from embedding import embed

    positives, negatives = build_training_texts()
    print(f'[baseline] {len(positives)} positives, {len(negatives)} negatives')
    X = embed(positives + negatives)
    y = np.array([1] * len(positives) + [0] * len(negatives))
    model = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, y)

    def baseline_score_text(text: str) -> pd.DataFrame:
        prob = float(model.predict_proba(embed([text]))[0, 1])
        return pd.DataFrame({'misinfo_score': [prob], 'flagged': [prob >= 0.5]})

    items = load_eval_set(ROOT / 'ml/eval/eval_set.jsonl')
    metrics = evaluate(baseline_score_text, items)
    report = render_report(
        metrics, 'M1 baseline: logistic regression on PubMedBERT embeddings',
        meta={'train': 'synthetic (reference library templates)',
              'note': 'stance-blind by construction — compare debunk flag rate '
                      'against the retrieve-and-verify report'})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding='utf-8')
    print(f"[baseline] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
          f"F1={metrics['f1']:.3f} -> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
