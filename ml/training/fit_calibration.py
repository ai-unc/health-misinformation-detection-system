"""Fit the logistic calibration head + flag threshold on the calibration split.

Usage (from repo root, venv active; loads all models):
    python ml/training/fit_calibration.py \
        [--eval-set ml/eval/eval_set.jsonl] \
        [--out maven_app/models/calibration_head.joblib]

Uses ONLY items whose id hashes into the calibration split (harness.split_of)
and which are not marked "disputed". Doc-level features = the features of the
chunk with max misinfo_score. Rerun whenever the eval set grows; report
test-split metrics separately via ml/eval/run_eval.py --split test.

IMPORTANT: A calibrated head can flag rows where both misinfo_entail and
guidance_contradict are low, producing flagged rows with an empty evidence_correction.
Before adopting a fitted artifact, verify flagged calibration-split rows keep
non-empty evidence_correction to avoid breaking the heuristic's guarantee.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ml' / 'eval'))
sys.path.insert(0, str(ROOT / 'maven_app'))

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def fit_head(X, y) -> LogisticRegression:
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(np.asarray(X), np.asarray(y))
    return model


def choose_tau(y_true, probs) -> float:
    """Threshold maximizing F1 on the calibration split."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    best_tau, best_f1 = 0.5, -1.0
    for tau in TAUS:
        pred = probs >= tau
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return best_tau


def save_artifact(model, features, tau, meta, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': features, 'tau': tau, 'meta': meta},
                out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-set', default=str(ROOT / 'ml/eval/eval_set.jsonl'))
    parser.add_argument('--out', default=str(ROOT / 'maven_app/models/calibration_head.joblib'))
    args = parser.parse_args()

    from harness import load_eval_set, split_of
    items = [it for it in load_eval_set(args.eval_set)
             if split_of(it['id']) == 'calibration' and not it.get('disputed')]
    if len(items) < 12:
        sys.exit(f'Only {len(items)} calibration items - need at least 12. '
                 'Grow the eval set before fitting.')
    print(f'[fit_calibration] {len(items)} calibration items; importing pipeline...')

    import scoring
    from pipeline import score_text

    X, y = [], []
    for item in items:
        df = score_text(item['text'])
        if df.empty:
            continue
        best = df.iloc[df['misinfo_score'].idxmax()]
        X.append([best[name] for name in scoring.FEATURES])
        y.append(1 if item['label'] == 'misinfo' else 0)

    if len(y) < 12:
        sys.exit(f'Only {len(y)} items scored (of {len(items)} candidates) - need at least 12. Grow the eval set (ml/eval/LABELING_GUIDE.md).')

    model = fit_head(X, y)
    probs = model.predict_proba(np.asarray(X))[:, 1]
    tau = choose_tau(y, probs)
    save_artifact(model, scoring.FEATURES, tau,
                  {'fitted_on': date.today().isoformat(), 'n': len(y),
                   'eval_set': args.eval_set},
                  args.out)
    print(f'[fit_calibration] wrote {args.out} (n={len(y)}, tau={tau})')
    print('[fit_calibration] Restart the app / rerun eval to pick it up. '
          'Report metrics with: python ml/eval/run_eval.py --split test --out ...')
    # Adoption check: remind operator to verify flagged calibration rows have non-empty evidence_correction.
    print('[fit_calibration] ** Adoption check: verify flagged calibration-split rows retain '
          'non-empty evidence_correction before deploying. Calibration may break the heuristic\'s '
          'guarantee that all flagged rows have supporting evidence. **')
    return 0


if __name__ == '__main__':
    sys.exit(main())
