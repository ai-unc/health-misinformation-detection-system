"""Scorer-agnostic evaluation harness for MAVEN misinformation scorers.

A scorer is any callable with the `score_text` shape: (text: str) ->
pd.DataFrame with at least `misinfo_score` (float) and `flagged` (bool)
columns. Doc-level aggregation: max chunk score / any chunk flagged.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

REQUIRED_FIELDS = ('id', 'text', 'label', 'stance', 'source')
VALID_LABELS = ('misinfo', 'not_misinfo')
VALID_STANCES = ('asserts_misinfo', 'debunks_misinfo', 'accurate', 'neutral', 'off_topic')
CALIBRATION_FRACTION = 0.4


def load_eval_set(path):
    items = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f'{path}:{line_no}: missing field {field!r}')
        if item['label'] not in VALID_LABELS:
            raise ValueError(f'{path}:{line_no}: bad label {item["label"]!r}')
        if item['stance'] not in VALID_STANCES:
            raise ValueError(f'{path}:{line_no}: bad stance {item["stance"]!r}')
        items.append(item)
    ids = [it['id'] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path}: duplicate ids')
    return items


def split_of(item_id: str) -> str:
    """Deterministic 40/60 calibration/test split keyed on item id."""
    bucket = int(hashlib.sha256(item_id.encode('utf-8')).hexdigest(), 16) % 100
    return 'calibration' if bucket < int(CALIBRATION_FRACTION * 100) else 'test'


def doc_score(df):
    """Aggregate a per-chunk score_text DataFrame to one doc-level score."""
    if df.empty:
        return 0.0, False
    return float(df['misinfo_score'].max()), bool(df['flagged'].any())


def evaluate(score_fn, items):
    y_true, y_prob, y_flag, latencies, rows = [], [], [], [], []
    for item in items:
        t0 = time.time()
        df = score_fn(item['text'])
        latencies.append(time.time() - t0)
        prob, flagged = doc_score(df)
        truth = 1 if item['label'] == 'misinfo' else 0
        y_true.append(truth)
        y_prob.append(prob)
        y_flag.append(1 if flagged else 0)
        rows.append({'id': item['id'], 'stance': item['stance'],
                     'true': truth, 'prob': prob, 'flagged': bool(flagged)})

    y_true_a = np.array(y_true)
    y_flag_a = np.array(y_flag)
    tp = int(((y_true_a == 1) & (y_flag_a == 1)).sum())
    fp = int(((y_true_a == 0) & (y_flag_a == 1)).sum())
    fn = int(((y_true_a == 1) & (y_flag_a == 0)).sum())
    tn = int(((y_true_a == 0) & (y_flag_a == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    pr_auc = (float(average_precision_score(y_true_a, np.array(y_prob)))
              if len(set(y_true)) > 1 else float('nan'))

    per_stance = {}
    for stance in sorted({r['stance'] for r in rows}):
        sub = [r for r in rows if r['stance'] == stance]
        flagged_n = sum(r['flagged'] for r in sub)
        per_stance[stance] = {'n': len(sub), 'flagged': flagged_n,
                              'flag_rate': flagged_n / len(sub)}

    return {'n': len(items), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': (tp + tn) / len(items) if items else 0.0,
            'pr_auc': pr_auc,
            'mean_latency_s': float(np.mean(latencies)) if latencies else 0.0,
            'per_stance': per_stance, 'rows': rows}


def render_report(metrics, title, meta=None):
    lines = [f'# {title}', '']
    for key, value in (meta or {}).items():
        lines.append(f'- **{key}:** {value}')
    lines += [
        '',
        '| Metric | Value |',
        '|---|---|',
        f"| N | {metrics['n']} |",
        f"| Precision | {metrics['precision']:.3f} |",
        f"| Recall | {metrics['recall']:.3f} |",
        f"| F1 | {metrics['f1']:.3f} |",
        f"| Accuracy | {metrics['accuracy']:.3f} |",
        f"| PR-AUC | {metrics['pr_auc']:.3f} |",
        f"| Mean latency (s) | {metrics['mean_latency_s']:.2f} |",
        '',
        '## Confusion matrix (flagged as misinfo)',
        '',
        '| | pred misinfo | pred not |',
        '|---|---|---|',
        f"| true misinfo | {metrics['tp']} | {metrics['fn']} |",
        f"| true not | {metrics['fp']} | {metrics['tn']} |",
        '',
        '## Per-stance flag rates',
        '',
        '| Gold stance | n | flagged | flag rate |',
        '|---|---|---|---|',
    ]
    for stance, s in metrics['per_stance'].items():
        lines.append(f"| {stance} | {s['n']} | {s['flagged']} | {s['flag_rate']:.2f} |")
    lines += ['', '## Per-item results', '',
              '| id | gold stance | true | prob | flagged |', '|---|---|---|---|---|']
    for r in metrics['rows']:
        lines.append(f"| {r['id']} | {r['stance']} | {r['true']} | {r['prob']:.3f} | {r['flagged']} |")
    lines.append('')
    return '\n'.join(lines)
