# ML Backend Rework (Retrieve-and-Verify) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MAVEN's centroid-cosine + IsolationForest scoring with a retrieve-and-verify pipeline (PubMedBERT retrieval → DeBERTa-v3 NLI stance verification → calibrated P(misinfo)), with an evaluation harness that measures every change against the current system.

**Architecture:** Embeddings only *retrieve* candidate reference claims; an NLI cross-encoder decides *stance* (entails misinfo claim / contradicts authority guidance / debunks misinfo); a calibrated head turns features into P(misinfo). The eval harness + baseline metrics land first so all later phases are measured claims.

**Tech Stack:** Python 3.11, sentence-transformers (PubMedBERT `NeuML/pubmedbert-base-embeddings`), transformers + sentencepiece (NLI verifier `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), scikit-learn (logistic calibration head), pandas, Flask. Spec: `docs/superpowers/specs/2026-07-16-ml-backend-rework-design.md`.

## Global Constraints

- Python 3.10+ (project venv is Homebrew python3.11; activate `venv/` if present).
- No LLM calls at inference time. Core scorer is local, free, reproducible.
- CPU inference must work; soft latency budget ≤15 s for a typical transcript.
- Tests are **plain Python scripts, not pytest** (house style). `maven_app` tests run from `maven_app/`: `python tests/test_x.py`. New `ml/` tests run from the repo root: `python ml/eval/tests/test_x.py`.
- Importing `pipeline` (or `app`) loads PubMedBERT (~30–60 s cold). The first verifier use downloads ~700 MB from Hugging Face (one-time).
- Retrieval defaults: `K_PER_KIND = 4`, `TOPIC_FLOOR = 0.45`. Verifier: `PAIR_CAP = 256`. Heuristic threshold `TAU = 0.5` until calibration artifact exists.
- Stance vocabulary (runtime): `asserts_misinfo / contradicts_guidance / debunks_misinfo / on_topic_neutral / off_topic`. Gold-label stance vocabulary (eval set): `asserts_misinfo / debunks_misinfo / accurate / neutral / off_topic`.
- Env vars: `MAVEN_VERIFIER_PATH` (verifier checkpoint override), `MAVEN_CALIBRATION_PATH` (calibration artifact override).
- The flag comes ONLY from `P(misinfo) ≥ τ`; stance is explanatory metadata.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (omitted from snippets below for brevity — always append it).
- Work happens on the existing `ml-backend-rework` branch.

---

### Task 1: Evaluation harness library

**Files:**
- Create: `ml/eval/harness.py`
- Test: `ml/eval/tests/test_harness.py`

**Interfaces:**
- Consumes: nothing (pure functions; pandas/numpy/sklearn already in venv).
- Produces (used by Tasks 2, 3, 10, 12):
  - `load_eval_set(path) -> list[dict]` — validated JSONL items
  - `split_of(item_id: str) -> str` — deterministic `'calibration' | 'test'` (40/60)
  - `doc_score(df: pd.DataFrame) -> tuple[float, bool]` — doc-level (max `misinfo_score`, any `flagged`)
  - `evaluate(score_fn, items) -> dict` — metrics dict with keys `n, tp, fp, fn, tn, precision, recall, f1, accuracy, pr_auc, mean_latency_s, per_stance, rows`
  - `render_report(metrics: dict, title: str, meta: dict | None) -> str` — markdown

- [ ] **Step 1: Write the failing test**

Create `ml/eval/tests/test_harness.py`:

```python
"""Harness unit test with stub scorers (no models loaded)."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/eval
from harness import doc_score, evaluate, load_eval_set, render_report, split_of


def _write_set(items) -> str:
    f = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False, encoding='utf-8')
    for it in items:
        f.write(json.dumps(it) + '\n')
    f.close()
    return f.name


ITEMS = [
    {'id': 'a1', 'text': 'misinfo one', 'label': 'misinfo', 'stance': 'asserts_misinfo', 'source': 'synthetic-seed'},
    {'id': 'a2', 'text': 'misinfo two', 'label': 'misinfo', 'stance': 'asserts_misinfo', 'source': 'synthetic-seed'},
    {'id': 'b1', 'text': 'fine one', 'label': 'not_misinfo', 'stance': 'accurate', 'source': 'synthetic-seed'},
    {'id': 'b2', 'text': 'fine two', 'label': 'not_misinfo', 'stance': 'debunks_misinfo', 'source': 'synthetic-seed'},
]


def _stub_scorer(flag_map):
    def score_fn(text):
        p, fl = flag_map[text]
        return pd.DataFrame({'misinfo_score': [p], 'flagged': [fl]})
    return score_fn


def main():
    # load_eval_set validates
    path = _write_set(ITEMS)
    items = load_eval_set(path)
    assert len(items) == 4

    # bad label rejected
    bad = _write_set([dict(ITEMS[0], id='x', label='nope')])
    try:
        load_eval_set(bad)
        raise AssertionError('bad label accepted')
    except ValueError:
        pass

    # duplicate id rejected
    dup = _write_set([ITEMS[0], ITEMS[0]])
    try:
        load_eval_set(dup)
        raise AssertionError('duplicate id accepted')
    except ValueError:
        pass

    # split is deterministic and both splits exist over many ids
    assert split_of('a1') == split_of('a1')
    splits = {split_of(f'id-{i}') for i in range(200)}
    assert splits == {'calibration', 'test'}

    # doc_score
    df = pd.DataFrame({'misinfo_score': [0.2, 0.9], 'flagged': [False, True]})
    assert doc_score(df) == (0.9, True)
    assert doc_score(pd.DataFrame()) == (0.0, False)

    # perfect scorer -> perfect metrics
    perfect = _stub_scorer({
        'misinfo one': (0.9, True), 'misinfo two': (0.8, True),
        'fine one': (0.1, False), 'fine two': (0.2, False),
    })
    m = evaluate(perfect, items)
    assert m['tp'] == 2 and m['tn'] == 2 and m['fp'] == 0 and m['fn'] == 0
    assert m['precision'] == 1.0 and m['recall'] == 1.0 and m['f1'] == 1.0
    assert m['per_stance']['debunks_misinfo']['flag_rate'] == 0.0

    # stance-blind scorer flags the debunk -> fp recorded per stance
    blind = _stub_scorer({
        'misinfo one': (0.9, True), 'misinfo two': (0.8, True),
        'fine one': (0.1, False), 'fine two': (0.9, True),
    })
    m2 = evaluate(blind, items)
    assert m2['fp'] == 1
    assert m2['per_stance']['debunks_misinfo']['flag_rate'] == 1.0

    # report renders
    report = render_report(m2, 'stub run', meta={'note': 'test'})
    assert 'stub run' in report and 'debunks_misinfo' in report and 'F1' in report

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from repo root: `python ml/eval/tests/test_harness.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness'`

- [ ] **Step 3: Write the implementation**

Create `ml/eval/harness.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run from repo root: `python ml/eval/tests/test_harness.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add ml/eval/harness.py ml/eval/tests/test_harness.py
git commit -m "feat(ml): add scorer-agnostic evaluation harness"
```

---

### Task 2: Seed eval set + labeling guide

**Files:**
- Create: `ml/eval/eval_set.jsonl`
- Create: `ml/eval/LABELING_GUIDE.md`

**Interfaces:**
- Consumes: `load_eval_set` from Task 1 (validation).
- Produces: `ml/eval/eval_set.jsonl` — the labeled set consumed by Tasks 3, 10, 12. Items with `source: "synthetic-seed"`; the team later appends `source: "hand-labeled"` items (target 200–300 total per spec).

- [ ] **Step 1: Write the seed eval set**

Create `ml/eval/eval_set.jsonl` with exactly these 24 lines (one JSON object per line):

```json
{"id": "seed-001", "text": "Epidurals cause permanent damage to your baby's brain — hospitals just won't admit it.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": "casual register"}
{"id": "seed-002", "text": "You have to eat for two now, mama! Double portions every meal or baby won't grow.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": "casual register"}
{"id": "seed-003", "text": "A recent review demonstrates that routine ultrasound exposure meaningfully increases autism risk in the third trimester.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": "clinical register — register-confound probe"}
{"id": "seed-004", "text": "Formula is basically poison — any amount permanently ruins your baby's gut microbiome.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": ""}
{"id": "seed-005", "text": "Home birth is statistically safer than hospital birth for every pregnancy, full stop.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": ""}
{"id": "seed-006", "text": "Morning sickness means your body is rejecting toxins from the vaccines you got as a child.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": "inaccurate mechanism"}
{"id": "seed-007", "text": "If you get a flu shot while pregnant your baby will develop allergies — my midwife confirmed this.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": ""}
{"id": "seed-008", "text": "Inductions are only scheduled so OBs can bill more before the holidays; there's never a medical reason.", "label": "misinfo", "stance": "asserts_misinfo", "source": "synthetic-seed", "notes": "undermines trust"}
{"id": "seed-009", "text": "ACOG recommends low-dose aspirin starting at 12 weeks for patients at high risk of preeclampsia.", "label": "not_misinfo", "stance": "accurate", "source": "synthetic-seed", "notes": "clinical register"}
{"id": "seed-010", "text": "girl the nausea is so rough in the first trimester but small frequent snacks + ginger tea genuinely helped me survive lol", "label": "not_misinfo", "stance": "accurate", "source": "synthetic-seed", "notes": "casual register — register-confound probe"}
{"id": "seed-011", "text": "You only need about 200 to 300 extra calories a day in the second and third trimesters — a small snack, not double portions.", "label": "not_misinfo", "stance": "accurate", "source": "synthetic-seed", "notes": ""}
{"id": "seed-012", "text": "Low-mercury fish like salmon two or three times a week is actually encouraged during pregnancy for the omega-3s.", "label": "not_misinfo", "stance": "accurate", "source": "synthetic-seed", "notes": ""}
{"id": "seed-013", "text": "Postpartum depression is common and treatable — talk to your provider; you're not failing anyone by asking for help.", "label": "not_misinfo", "stance": "accurate", "source": "synthetic-seed", "notes": ""}
{"id": "seed-014", "text": "The Tdap vaccine during pregnancy passes protective antibodies to your newborn before they can be vaccinated themselves.", "label": "not_misinfo", "stance": "accurate", "source": "synthetic-seed", "notes": ""}
{"id": "seed-015", "text": "No, epidurals do NOT damage your baby's brain — that myth keeps scaring people out of safe pain relief. Let's debunk it.", "label": "not_misinfo", "stance": "debunks_misinfo", "source": "synthetic-seed", "notes": "hard negative"}
{"id": "seed-016", "text": "Seeing the 'eating for two' thing again — you don't need double calories, you need about one extra snack a day. Myth busted.", "label": "not_misinfo", "stance": "debunks_misinfo", "source": "synthetic-seed", "notes": "hard negative"}
{"id": "seed-017", "text": "PSA: the claim that flu shots cause allergies in babies is false. Flu vaccination in pregnancy protects both of you.", "label": "not_misinfo", "stance": "debunks_misinfo", "source": "synthetic-seed", "notes": "hard negative"}
{"id": "seed-018", "text": "Formula is not poison. Fed is best when breastfeeding isn't working — the microbiome differences are small and temporary.", "label": "not_misinfo", "stance": "debunks_misinfo", "source": "synthetic-seed", "notes": "hard negative"}
{"id": "seed-019", "text": "Packing my hospital bag tonight! 38 weeks tomorrow. Any last-minute must-haves I'm forgetting?", "label": "not_misinfo", "stance": "neutral", "source": "synthetic-seed", "notes": "on-topic, no claim"}
{"id": "seed-020", "text": "Our birth class covered breathing techniques and when to call the midwife. Feeling a bit more ready.", "label": "not_misinfo", "stance": "neutral", "source": "synthetic-seed", "notes": ""}
{"id": "seed-021", "text": "Baby kicked so hard during the ultrasound today the tech laughed.", "label": "not_misinfo", "stance": "neutral", "source": "synthetic-seed", "notes": ""}
{"id": "seed-022", "text": "Top 5 budget standing desks for your home office in 2026 — number 3 surprised me.", "label": "not_misinfo", "stance": "off_topic", "source": "synthetic-seed", "notes": ""}
{"id": "seed-023", "text": "The new season of the baking show is so good, the bread week finale had me stressed.", "label": "not_misinfo", "stance": "off_topic", "source": "synthetic-seed", "notes": ""}
{"id": "seed-024", "text": "Traffic on I-40 was brutal this morning; it took me an hour to get to campus.", "label": "not_misinfo", "stance": "off_topic", "source": "synthetic-seed", "notes": ""}
```

- [ ] **Step 2: Write the labeling guide**

Create `ml/eval/LABELING_GUIDE.md`:

```markdown
# MAVEN Eval-Set Labeling Guide

This guide doubles as the PRD D3 annotation guidelines (review with
Dr. Bazzano before large-scale labeling).

## What goes in the set

Real TikTok/Instagram transcripts, OCR text, and captions about
perinatal / maternal / reproductive health, plus curated hard cases.
Target: 200–300 items total. Keep items short (a caption, a transcript,
one post) — the unit a MAVEN user would score.

## File format

`ml/eval/eval_set.jsonl` — one JSON object per line:

| Field | Values |
|---|---|
| `id` | unique, stable (e.g. `hl-041`). Never reuse or renumber — the calibration/test split hashes this id. |
| `text` | the raw text, unedited |
| `label` | `misinfo` \| `not_misinfo` |
| `stance` | `asserts_misinfo` \| `debunks_misinfo` \| `accurate` \| `neutral` \| `off_topic` |
| `source` | `synthetic-seed` \| `hand-labeled` |
| `notes` | free text (why it's hard, where it came from) |

## Label definitions

- **misinfo** — the text asserts, endorses, or instructs based on a
  health claim that contradicts current clinical guidance (ACOG, WHO,
  CDC, AAP...). The claim must be *made*, not merely mentioned.
- **not_misinfo** — everything else, including:
  - `debunks_misinfo`: quotes/mentions a false claim in order to refute
    it. These are the most important negatives — label generously.
  - `accurate`: consistent with guidance (any register — casual counts).
  - `neutral`: on-topic but no verifiable health claim (lived experience,
    questions, logistics).
  - `off_topic`: not about perinatal/maternal health.

## Edge rules

- Personal experience ("my labor took 30 hours") is `neutral` unless it
  generalizes into advice that contradicts guidance.
- Emerging/contested science: if major guidelines disagree with the
  claim today, label `misinfo`; if genuinely unsettled, `neutral` + note.
- Sarcasm/jokes: label by what a reasonable reader would take away.
- When two annotators disagree after discussion, keep the item with a
  note recording both views; exclude it from calibration by adding
  `"disputed": true`.

## Split discipline

The 40/60 calibration/test split is derived from `id` hashes
(`harness.split_of`). Never tune thresholds, prompts, or weights on the
test split. Report test-split numbers only.
```

- [ ] **Step 3: Validate the set loads**

Run from repo root:
`python -c "import sys; sys.path.insert(0, 'ml/eval'); from harness import load_eval_set; items = load_eval_set('ml/eval/eval_set.jsonl'); print(len(items), 'items OK')"`
Expected: `24 items OK`

- [ ] **Step 4: Commit**

```bash
git add ml/eval/eval_set.jsonl ml/eval/LABELING_GUIDE.md
git commit -m "feat(ml): add seed eval set and labeling guide"
```

---

### Task 3: run_eval CLI + baseline metrics of the current system

**Files:**
- Create: `ml/eval/run_eval.py`
- Create: `ml/eval/reports/2026-07-16-baseline-legacy.md` (generated)

**Interfaces:**
- Consumes: `harness.py` (Task 1), `eval_set.jsonl` (Task 2), `maven_app/pipeline.py` `score_text(text) -> pd.DataFrame` (current legacy implementation).
- Produces: `run_eval.py` CLI reused verbatim in Task 12. Baseline report locked into git BEFORE the pipeline is reworked.

- [ ] **Step 1: Write the CLI**

Create `ml/eval/run_eval.py`:

```python
"""Run the eval harness against the current maven_app pipeline.

Usage (from repo root, venv active):
    python ml/eval/run_eval.py --out ml/eval/reports/<name>.md \
        [--eval-set ml/eval/eval_set.jsonl] [--split all|calibration|test] \
        [--title "..."]

Note: importing the pipeline loads PubMedBERT (~30-60 s cold start).
"""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))          # ml/eval
sys.path.insert(0, str(ROOT / 'maven_app'))

from harness import evaluate, load_eval_set, render_report, split_of  # noqa: E402


def git_rev() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:
        return 'unknown'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-set', default=str(ROOT / 'ml/eval/eval_set.jsonl'))
    parser.add_argument('--split', default='all', choices=['all', 'calibration', 'test'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    items = load_eval_set(args.eval_set)
    if args.split != 'all':
        items = [it for it in items if split_of(it['id']) == args.split]
    print(f'[run_eval] {len(items)} items (split={args.split})')

    print('[run_eval] importing pipeline (loads models)...')
    from pipeline import score_text  # deferred: heavy import

    metrics = evaluate(score_text, items)
    title = args.title or f'MAVEN eval — {date.today().isoformat()}'
    report = render_report(metrics, title, meta={
        'git commit': git_rev(),
        'eval set': args.eval_set,
        'split': args.split,
        'items': len(items),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding='utf-8')
    print(f'[run_eval] wrote {out}')
    print(f"[run_eval] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
          f"F1={metrics['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Run the baseline**

Run from repo root (venv active):

```bash
python ml/eval/run_eval.py \
    --out ml/eval/reports/2026-07-16-baseline-legacy.md \
    --title "Baseline: legacy centroid+IsoForest scorer (pre-rework)"
```

Expected: report file written; console prints metrics. **Inspect the report** — expectation from the spec's failure analysis: high flag rate on `debunks_misinfo` rows (stance blindness) and flags on casual `accurate`/`neutral` rows (register confound). Whatever the numbers are, they are the baseline; do not tune anything.

- [ ] **Step 3: Commit**

```bash
git add ml/eval/run_eval.py ml/eval/reports/2026-07-16-baseline-legacy.md
git commit -m "feat(ml): add run_eval CLI; lock in legacy-scorer baseline metrics"
```

---

### Task 4: Reference library builder + claim type map

**Files:**
- Create: `scripts/build_reference_library.py`
- Create: `ml/data/claim_type_map.json`
- Create: `ml/data/claim_paraphrases.json`
- Create: `ml/data/misinfo_types.json` (copy of `maven_app/anchors/misinfo_type_anchors.json` — preserves the hand-curated seeds before the original is deleted in Task 13)
- Create (generated): `maven_app/anchors/reference_library.json`
- Test: `maven_app/tests/test_reference_library.py`

**Interfaces:**
- Consumes: `scripts/build_anchors.py` (port `parse_misinfo_pairs`, `style_name`, `is_heading`, `normalize`, and the path constants — the file exists in the repo; copy the function bodies verbatim), source docx files under `perinatal_knowledge_base/dept_mch_feed/`.
- Produces: `maven_app/anchors/reference_library.json` — a JSON list of entries, each:

```json
{
  "id": "mis-001",
  "text": "You must 'eat for two' during pregnancy.",
  "kind": "misinfo",
  "domain": "antenatal",
  "type_id": "not_aligned_with_guidelines",
  "correction": "No. Caloric needs remain unchanged...",
  "parent_id": null
}
```

  - `kind`: `"misinfo"` or `"authority"`. Authority entries have `type_id: null`, `correction: null`.
  - Paraphrase entries have `parent_id` set to their base claim's id and inherit its `type_id`/`correction`.
  - Consumed by Tasks 5 (retrieval), 11 (NLI pair builder).

- [ ] **Step 1: Write the failing test**

Create `maven_app/tests/test_reference_library.py`:

```python
"""Validate the generated reference library schema and content quality."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LIB_PATH = Path(__file__).resolve().parent.parent / 'anchors' / 'reference_library.json'
VALID_TYPES = {
    'unattributed_risk', 'not_aligned_with_guidelines',
    'promotes_alternative_medicine', 'exaggerated_risk',
    'discourages_evidence_based', 'undermines_medical_trust',
    'inaccurate_biological_mechanism', 'other',
}
ORG_NAME_LINE = re.compile(r'^[A-Z]{2,}(\s*[—–-]|\s+is\s+the\b)')


def main():
    assert LIB_PATH.exists(), f'missing {LIB_PATH}; run scripts/build_reference_library.py'
    lib = json.loads(LIB_PATH.read_text(encoding='utf-8'))

    ids = [e['id'] for e in lib]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    by_id = {e['id']: e for e in lib}

    misinfo = [e for e in lib if e['kind'] == 'misinfo']
    authority = [e for e in lib if e['kind'] == 'authority']
    base_misinfo = [e for e in misinfo if e['parent_id'] is None]
    print(f'entries: {len(lib)}  misinfo base: {len(base_misinfo)}  '
          f'misinfo total: {len(misinfo)}  authority: {len(authority)}')

    assert len(base_misinfo) >= 70, 'expected the ~74 curated claims'
    assert len(authority) >= 100, 'expected corrections + extracted sentences'

    for e in lib:
        assert e['kind'] in ('misinfo', 'authority')
        assert isinstance(e['text'], str) and len(e['text'].split()) >= 4, e['id']
        if e['kind'] == 'misinfo':
            assert e['type_id'] in VALID_TYPES, f"{e['id']}: bad type {e['type_id']}"
            assert e['correction'], f"{e['id']}: misinfo entry missing correction"
        else:
            assert e['type_id'] is None and e['correction'] is None
        if e['parent_id'] is not None:
            parent = by_id[e['parent_id']]
            assert parent['parent_id'] is None, 'paraphrase of a paraphrase'
            assert e['type_id'] == parent['type_id']
            assert e['correction'] == parent['correction']

    # Authority entries must be real statements, not org-name lines.
    # Corrections are curated content and exempt from the word bounds.
    correction_texts = {e['correction'] for e in misinfo}
    for e in authority:
        if e['text'] in correction_texts:
            continue
        words = e['text'].split()
        assert 8 <= len(words) <= 60, f"{e['id']}: {len(words)} words"
        assert not ORG_NAME_LINE.match(e['text']), f"{e['id']}: org-name line: {e['text'][:60]}"
        assert 'Edition' not in e['text'] and 'Resource for AI' not in e['text']

    # Every base misinfo claim's correction must appear as an authority entry.
    auth_texts = {e['text'] for e in authority}
    missing = [e['id'] for e in base_misinfo if e['correction'] not in auth_texts]
    assert not missing, f'corrections missing from authority set: {missing[:5]}'

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_reference_library.py`
Expected: FAIL with `missing .../reference_library.json`

- [ ] **Step 3: Create the claim type map**

The 8 JGIM-taxonomy types (from the hand-curated seeds; keep this table in the file header):

| type_id | meaning |
|---|---|
| `unattributed_risk` | vague hidden dangers, "they won't tell you" |
| `not_aligned_with_guidelines` | contradicts ACOG/WHO/CDC/AAP guidance |
| `promotes_alternative_medicine` | pushes unproven alternatives over standard care |
| `exaggerated_risk` | a real risk inflated far beyond evidence |
| `discourages_evidence_based` | discourages recommended screening/treatment/vaccines |
| `undermines_medical_trust` | claims providers/system act on hidden motives |
| `inaccurate_biological_mechanism` | false physiology or mechanism |
| `other` | none of the above |

Create `ml/data/claim_type_map.json` by reading every claim in `maven_app/anchors/misinfo_anchors.json` (in order) and assigning the best-fitting type. Claim ids are `mis-001`, `mis-002`, ... in file order. Format (include the claim text so drift is detectable):

```json
{
  "_status": "draft — needs team review with Dr. Bazzano",
  "claims": [
    {"id": "mis-001", "claim": "You must 'eat for two' during pregnancy.", "type_id": "not_aligned_with_guidelines"},
    {"id": "mis-002", "claim": "All seafood must be avoided during pregnancy.", "type_id": "exaggerated_risk"}
  ]
}
```

Cover ALL claims in the file (one entry per claim, no omissions). Judgment calls are fine — the `_status` field flags the file for team review.

- [ ] **Step 4: Create the seed paraphrase file**

Create `ml/data/claim_paraphrases.json`. Structure: map of base-claim id → list of assertion paraphrases (each becomes a retrievable library entry with `parent_id`). Seed with hand-written paraphrases for the three claims below; bulk LLM-assisted expansion is a later offline step (documented in the file header comment via a `"_note"` key):

```json
{
  "_note": "Assertion paraphrases of base claims (casual + clinical register). Expand offline (LLM-assisted, human-reviewed) — every paraphrase must ASSERT the claim, never debunk it. Keys are base claim ids from reference_library.json.",
  "mis-001": [
    "pregnant mamas need double portions at every meal, you're literally feeding two people now",
    "Nutritional requirements during gestation necessitate a twofold increase in caloric intake."
  ],
  "mis-002": [
    "seafood is completely off the menu for nine months, no exceptions, it's not worth the risk",
    "Current prenatal protocols require the total elimination of all fish and shellfish from the maternal diet."
  ],
  "mis-003": [
    "even one sip of coffee while pregnant is playing russian roulette with your baby",
    "Complete abstinence from caffeine is medically mandated throughout pregnancy."
  ]
}
```

(If `mis-002`/`mis-003` texts don't match the actual second/third claims in `misinfo_anchors.json`, adjust the paraphrases to assert whatever those claims actually say — paraphrases must match their parent.)

- [ ] **Step 5: Preserve the type seeds**

Copy `maven_app/anchors/misinfo_type_anchors.json` to `ml/data/misinfo_types.json` unchanged (the original is deleted in Task 13; the seeds remain useful reference/training material).

- [ ] **Step 6: Write the builder**

Create `scripts/build_reference_library.py`:

```python
"""
build_reference_library.py — Build the retrieve-and-verify reference library.

Reads:
  - perinatal_knowledge_base/dept_mch_feed/perinatal_misinformation.docx
  - perinatal_knowledge_base/dept_mch_feed/perinatal_comprehensive.docx
  - ml/data/claim_type_map.json        (claim id -> JGIM type)
  - ml/data/claim_paraphrases.json     (claim id -> assertion paraphrases)

Writes:
  - maven_app/anchors/reference_library.json

Entry schema:
  {id, text, kind: misinfo|authority, domain, type_id, correction, parent_id}

Idempotent. Re-run when any input changes. Replaces scripts/build_anchors.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

ROOT = Path(__file__).resolve().parents[1]
FEED_DIR = ROOT / 'perinatal_knowledge_base' / 'dept_mch_feed'
MISINFO_DOCX = FEED_DIR / 'perinatal_misinformation.docx'
COMPRE_DOCX = FEED_DIR / 'perinatal_comprehensive.docx'
TYPE_MAP_PATH = ROOT / 'ml' / 'data' / 'claim_type_map.json'
PARAPHRASE_PATH = ROOT / 'ml' / 'data' / 'claim_paraphrases.json'
OUT_PATH = ROOT / 'maven_app' / 'anchors' / 'reference_library.json'

# --- Ported verbatim from scripts/build_anchors.py: ---------------------
# style_name(), is_heading(), normalize(), parse_misinfo_pairs(), and the
# GUIDELINE_RE / SECTION_LETTER_RE / SECTION_TO_DOMAIN constants.
# Copy those function bodies here unchanged (build_anchors.py is in the
# repo until Task 13). Do NOT port extract_authority_sentences() or
# extract_comprehensive_paragraphs() — replaced below.
# -------------------------------------------------------------------------

ORG_NAME_LINE = re.compile(r'^[A-Z]{2,}(\s*[—–-]|\s+is\s+the\b)')
JUNK_MARKERS = ('Edition', 'Resource for AI', '|')
MIN_WORDS, MAX_WORDS = 8, 60


def extract_authority_statements(doc_path) -> list[str]:
    """Declarative sentences from the comprehensive docx, junk-filtered."""
    from docx import Document
    doc = Document(str(doc_path))
    sentences: list[str] = []
    seen = set()
    for para in doc.paragraphs:
        if is_heading(para):
            continue
        text = normalize(para.text)
        if not text or any(m in text for m in JUNK_MARKERS):
            continue
        for sent in sent_tokenize(text):
            sent = sent.strip()
            words = sent.split()
            if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                continue
            if not sent.endswith('.'):
                continue
            if ORG_NAME_LINE.match(sent):
                continue
            if sent.upper() == sent:  # ALL-CAPS headings
                continue
            if sent not in seen:
                seen.add(sent)
                sentences.append(sent)
    return sentences


def main() -> int:
    pairs = parse_misinfo_pairs(MISINFO_DOCX)  # [{claim, evidence, domain}, ...]

    type_map_doc = json.loads(TYPE_MAP_PATH.read_text(encoding='utf-8'))
    type_map = {c['id']: c for c in type_map_doc['claims']}
    paraphrases = json.loads(PARAPHRASE_PATH.read_text(encoding='utf-8'))

    entries = []

    # Base misinfo claims
    for i, pair in enumerate(pairs, 1):
        cid = f'mis-{i:03d}'
        mapped = type_map.get(cid)
        if mapped is None:
            sys.exit(f'ERROR: {cid} missing from claim_type_map.json')
        if normalize(mapped['claim']) != normalize(pair['claim']):
            sys.exit(f'ERROR: {cid} claim text drift — regenerate claim_type_map.json\n'
                     f'  map: {mapped["claim"]!r}\n  docx: {pair["claim"]!r}')
        entries.append({'id': cid, 'text': pair['claim'], 'kind': 'misinfo',
                        'domain': pair['domain'], 'type_id': mapped['type_id'],
                        'correction': pair['evidence'], 'parent_id': None})

    # Paraphrase expansions
    by_id = {e['id']: e for e in entries}
    for parent_id, texts in paraphrases.items():
        if parent_id.startswith('_'):
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            sys.exit(f'ERROR: paraphrase parent {parent_id} not found')
        for j, text in enumerate(texts, 1):
            entries.append({'id': f'{parent_id}-p{j}', 'text': text,
                            'kind': 'misinfo', 'domain': parent['domain'],
                            'type_id': parent['type_id'],
                            'correction': parent['correction'],
                            'parent_id': parent_id})

    # Authority: every correction is an authority statement...
    auth_texts, seen = [], set()
    for pair in pairs:
        ev = normalize(pair['evidence'])
        if ev and ev not in seen:
            seen.add(ev)
            auth_texts.append(pair['evidence'])
    # ...plus extracted declarative sentences from the comprehensive docx.
    for sent in extract_authority_statements(COMPRE_DOCX):
        if sent not in seen:
            seen.add(sent)
            auth_texts.append(sent)

    for i, text in enumerate(auth_texts, 1):
        entries.append({'id': f'auth-{i:03d}', 'text': text, 'kind': 'authority',
                        'domain': None, 'type_id': None, 'correction': None,
                        'parent_id': None})

    OUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    n_mis = sum(1 for e in entries if e['kind'] == 'misinfo')
    n_base = sum(1 for e in entries if e['kind'] == 'misinfo' and e['parent_id'] is None)
    n_auth = len(entries) - n_mis
    print(f'Wrote {OUT_PATH}')
    print(f'  misinfo: {n_mis} ({n_base} base + {n_mis - n_base} paraphrases)')
    print(f'  authority: {n_auth}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

Note: the correction texts skip the word-count/junk filters deliberately — they are curated content and always kept (the test already exempts them from the word bounds). Only *extracted* sentences pass through the filters.

- [ ] **Step 7: Build and run the test**

```bash
python scripts/build_reference_library.py     # from repo root
cd maven_app && python tests/test_reference_library.py
```

Expected: builder prints counts (~74 base misinfo + 6 paraphrases; authority ≥ 100); test prints `ALL TESTS PASSED`. If the authority extraction yields junk (inspect 10 random authority entries by eye), tighten `JUNK_MARKERS`/filters rather than accepting noise.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_reference_library.py ml/data/claim_type_map.json \
    ml/data/claim_paraphrases.json ml/data/misinfo_types.json \
    maven_app/anchors/reference_library.json maven_app/tests/test_reference_library.py
git commit -m "feat(ml): build retrieve-and-verify reference library with typed claims"
```

---

### Task 5: Embedding module + retrieval with on-topic gate

**Files:**
- Create: `maven_app/embedding.py`
- Create: `maven_app/retrieval.py`
- Test: `maven_app/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `reference_library.json` (Task 4).
- Produces (used by Tasks 7, 8):
  - `embedding.embed(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray` — L2-normalized (n, 768). `embedding.EMBED_DIM = 768`. (Extracted so both `retrieval` and `pipeline` can use it without a circular import.)
  - `retrieval.Retrieved` — NamedTuple `(entry: dict, sim: float)`
  - `retrieval.RetrievalResult` — NamedTuple `(misinfo: list[Retrieved], authority: list[Retrieved])` with property `scoreable: bool`
  - `retrieval.retrieve(chunk_embs: np.ndarray, k: int = K_PER_KIND, floor: float = TOPIC_FLOOR) -> list[RetrievalResult]`
  - `retrieval.K_PER_KIND = 4`, `retrieval.TOPIC_FLOOR = 0.45`
  - `retrieval.base_entry(entry: dict) -> dict` — resolves a paraphrase to its base claim entry

- [ ] **Step 1: Write the failing test**

Create `maven_app/tests/test_retrieval.py`:

```python
"""Retrieval sanity: known probes hit the right claims; off-topic gates out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding import embed
from retrieval import TOPIC_FLOOR, base_entry, retrieve


def main():
    probes = [
        "Epidurals damage the baby's brain and always lead to a c-section.",
        'you gotta eat double portions now mama, you are eating for two!',
        'Top 5 budget standing desks for your home office in 2026.',
    ]
    embs = embed(probes)
    results = retrieve(embs)
    assert len(results) == 3

    epidural, eating, desks = results

    # On-topic misinfo probes retrieve misinfo candidates above the floor
    assert epidural.scoreable and len(epidural.misinfo) >= 1
    assert all(r.sim >= TOPIC_FLOOR for r in epidural.misinfo)
    top_texts = ' | '.join(r.entry['text'].lower() for r in epidural.misinfo)
    assert 'epidural' in top_texts, f'epidural claim not retrieved: {top_texts[:200]}'

    assert eating.scoreable
    eating_texts = ' | '.join(r.entry['text'].lower() for r in eating.misinfo)
    assert 'two' in eating_texts or 'eat' in eating_texts

    # Candidates are deduped by base claim (no two paraphrases of one parent)
    parents = [base_entry(r.entry)['id'] for r in eating.misinfo]
    assert len(parents) == len(set(parents)), f'duplicate parents: {parents}'

    # Off-topic probe gates out entirely
    assert not desks.scoreable, (
        f'off-topic probe retrieved: mis={[(r.entry["id"], round(r.sim, 3)) for r in desks.misinfo]} '
        f'auth={[(r.entry["id"], round(r.sim, 3)) for r in desks.authority]}'
    )

    # base_entry resolves paraphrases and is identity on base entries
    for r in epidural.misinfo:
        base = base_entry(r.entry)
        assert base['parent_id'] is None and base['correction']

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_retrieval.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'embedding'`

- [ ] **Step 3: Write embedding.py**

Create `maven_app/embedding.py` (this code MOVES out of `pipeline.py` in Task 8; until then it is duplicated, which is fine — the model weights are cached by HF):

```python
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
```

- [ ] **Step 4: Write retrieval.py**

Create `maven_app/retrieval.py`:

```python
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
TOPIC_FLOOR = 0.45

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
```

- [ ] **Step 5: Run test to verify it passes**

Run from `maven_app/`: `python tests/test_retrieval.py`
Expected: `ALL TESTS PASSED` (first run embeds the library once, ~30 s).

If the off-topic assertion fails because the desks probe retrieves something at ≥0.45: print the offending sims, and raise `TOPIC_FLOOR` in steps of 0.02 (max 0.55) until the three probes behave; note the chosen floor in the commit message. Do NOT lower it below 0.45 to make on-topic probes pass — if those fail, the library content is the problem (inspect it).

- [ ] **Step 6: Commit**

```bash
git add maven_app/embedding.py maven_app/retrieval.py maven_app/tests/test_retrieval.py
git commit -m "feat: add embedding module and reference retrieval with on-topic gate"
```

---

### Task 6: NLI verifier wrapper

**Files:**
- Create: `maven_app/verifier.py`
- Modify: `maven_app/requirements.txt`
- Test: `maven_app/tests/test_verifier.py`

**Interfaces:**
- Consumes: nothing internal (transformers + torch).
- Produces (used by Task 7):
  - `verifier.Verifier(checkpoint: str | None = None, device: str | None = None)` — loads from `checkpoint` arg, else `MAVEN_VERIFIER_PATH` env, else `DEFAULT_CHECKPOINT`
  - `Verifier.predict(pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray` — shape (n, 3), softmax probs in FIXED column order `[entail, neutral, contradict]` regardless of the checkpoint's internal label order; `(premise, hypothesis)` pairs
  - `verifier.get_verifier() -> Verifier` — lazy module-level singleton
  - `verifier.PAIR_CAP = 256`, `verifier.DEFAULT_CHECKPOINT = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'`

- [ ] **Step 1: Add dependencies**

In `maven_app/requirements.txt`, after the `torch>=2.2` line, add:

```
transformers>=4.40
sentencepiece>=0.1.99
protobuf
```

Then install: `pip install "transformers>=4.40" "sentencepiece>=0.1.99" protobuf`

- [ ] **Step 2: Write the failing test**

Create `maven_app/tests/test_verifier.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_verifier.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'verifier'`

- [ ] **Step 4: Write the implementation**

Create `maven_app/verifier.py`:

```python
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
                self.checkpoint).to(self.device).eval()
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
```

- [ ] **Step 5: Run test to verify it passes**

Run from `maven_app/`: `python tests/test_verifier.py`
Expected: `ALL TESTS PASSED`, with per-pair CPU latency printed (expect roughly 0.1–0.4 s/pair unbatched-equivalent; note the number).

- [ ] **Step 6: Commit**

```bash
git add maven_app/verifier.py maven_app/tests/test_verifier.py maven_app/requirements.txt
git commit -m "feat: add DeBERTa-v3 NLI verifier wrapper with fixed label order"
```

---

### Task 7: Scoring module (features, stance, heuristic P(misinfo), calibration hook)

**Files:**
- Create: `maven_app/scoring.py`
- Test: `maven_app/tests/test_scoring_unit.py`

**Interfaces:**
- Consumes: `retrieval.retrieve/base_entry/K_PER_KIND`, `verifier.get_verifier/Verifier/PAIR_CAP` (Tasks 5–6).
- Produces (used by Task 8):
  - `scoring.ChunkScore` — NamedTuple with fields `p_misinfo: float, stance: str, scoreable: bool, misinfo_entail: float, guidance_contradict: float, misinfo_contradict: float, top_claim_sim: float, top_auth_sim: float, matched: dict | None` (`matched` is a BASE misinfo library entry)
  - `scoring.score_chunks(chunks: list[str], chunk_embs: np.ndarray, nli=None) -> list[ChunkScore]` (`nli` injectable for tests; defaults to `get_verifier()`)
  - `scoring.active_tau() -> float` — calibration artifact's τ if present, else `TAU = 0.5`
  - `scoring.FEATURES = ['misinfo_entail', 'guidance_contradict', 'misinfo_contradict', 'top_claim_sim', 'top_auth_sim']` (single source of truth; Task 10 imports it)
  - `scoring.features_of(cs: ChunkScore) -> list[float]` — feature vector in `FEATURES` order
  - Constants: `TAU = 0.5`, `ENTAIL_MIN = 0.5`, `DOMINANCE_MARGIN = 0.15`, `DEBUNK_DAMP = 0.5`
  - Calibration artifact path: `maven_app/models/calibration_head.joblib`, env override `MAVEN_CALIBRATION_PATH`; artifact dict `{'model', 'features', 'tau', 'meta'}` (Task 10 writes it)

- [ ] **Step 1: Write the failing test**

Create `maven_app/tests/test_scoring_unit.py`. A `StubNLI` returns canned probabilities keyed on hypothesis text, so no model loads (retrieval + embedding DO load — unavoidable, they're module-level):

```python
"""scoring.score_chunks unit test with a stubbed NLI verifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from embedding import embed
import scoring
from scoring import ChunkScore, features_of, score_chunks


class StubNLI:
    """Returns canned [entail, neutral, contradict] per (premise, hypothesis)."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def predict(self, pairs, batch_size=16):
        self.calls.append(len(pairs))
        out = np.zeros((len(pairs), 3))
        for i, (premise, hypothesis) in enumerate(pairs):
            row = None
            for (p_key, h_key), probs in self.table.items():
                if p_key in premise and (h_key is None or h_key in hypothesis):
                    row = probs
                    break
            out[i] = row if row is not None else (0.05, 0.90, 0.05)
        return out


ASSERT_TEXT = "Epidurals damage the baby's brain and always lead to a c-section."
DEBUNK_TEXT = "No, epidurals do not damage your baby's brain — that is a myth."
NOVEL_TEXT = "Low-dose aspirin is useless for preeclampsia prevention, skip it."
OFFTOPIC_TEXT = 'Top 5 budget standing desks for your home office in 2026.'


def main():
    chunks = [ASSERT_TEXT, DEBUNK_TEXT, NOVEL_TEXT, OFFTOPIC_TEXT]
    embs = embed(chunks)

    stub = StubNLI({
        ('Epidurals damage', 'pidural'): (0.92, 0.06, 0.02),   # asserts the claim
        ('No, epidurals', 'pidural'): (0.02, 0.08, 0.90),      # contradicts the claim
        ('aspirin is useless', 'aspirin'): (0.02, 0.10, 0.88), # contradicts guidance
    })

    results = score_chunks(chunks, embs, nli=stub)
    assert len(results) == 4
    r_assert, r_debunk, r_novel, r_off = results

    # 1) asserting misinfo: high p, asserts stance, matched base claim populated
    assert r_assert.stance == 'asserts_misinfo', r_assert
    assert r_assert.p_misinfo >= 0.8, r_assert.p_misinfo
    assert r_assert.matched is not None and r_assert.matched['parent_id'] is None
    assert r_assert.matched['correction']

    # 2) debunk: low p (damped), debunk stance
    assert r_debunk.stance == 'debunks_misinfo', r_debunk
    assert r_debunk.p_misinfo <= 0.25, r_debunk.p_misinfo

    # 3) contradicts guidance: high p even with no misinfo entailment
    #    (aspirin guidance lives in the authority set via corrections/extraction)
    if r_novel.scoreable and r_novel.guidance_contradict > 0.5:
        assert r_novel.p_misinfo >= 0.5
        assert r_novel.stance == 'contradicts_guidance', r_novel

    # 4) off-topic: not scoreable, p == 0, no NLI pairs were spent on it
    assert not r_off.scoreable and r_off.p_misinfo == 0.0
    assert r_off.stance == 'off_topic'

    # feature vector matches FEATURES order
    f = features_of(r_assert)
    assert len(f) == len(scoring.FEATURES)
    assert f[0] == r_assert.misinfo_entail

    # pair cap: force a tiny cap and confirm total pairs respect it
    old_cap = scoring.PAIR_CAP_ACTIVE
    scoring.PAIR_CAP_ACTIVE = 4
    try:
        stub2 = StubNLI({})
        score_chunks(chunks, embs, nli=stub2)
        assert sum(stub2.calls) <= 4, stub2.calls
    finally:
        scoring.PAIR_CAP_ACTIVE = old_cap

    # heuristic tau without calibration artifact
    assert scoring.active_tau() == scoring.TAU or isinstance(scoring.active_tau(), float)

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_scoring_unit.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring'`

- [ ] **Step 3: Write the implementation**

Create `maven_app/scoring.py`:

```python
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


def features_of(cs: ChunkScore) -> List[float]:
    return [getattr(cs, name) for name in FEATURES]


_calibration = None
_calibration_loaded = False


def _get_calibration():
    global _calibration, _calibration_loaded
    if not _calibration_loaded:
        _calibration_loaded = True
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
    return _calibration


def active_tau() -> float:
    artifact = _get_calibration()
    return float(artifact['tau']) if artifact else TAU


def _stance(scoreable, e_m, c_m, c_a) -> str:
    if not scoreable:
        return 'off_topic'
    if c_m >= ENTAIL_MIN and c_m > e_m + DOMINANCE_MARGIN:
        return 'debunks_misinfo'
    if e_m >= ENTAIL_MIN and e_m >= c_a:
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

    pairs, index = [], []  # index[i] = (chunk_idx, kind, Retrieved)
    for ci, (chunk, r) in enumerate(zip(chunks, results)):
        for cand in r.misinfo:
            pairs.append((chunk, cand.entry['text']))
            index.append((ci, 'misinfo', cand))
        for cand in r.authority:
            pairs.append((chunk, cand.entry['text']))
            index.append((ci, 'authority', cand))

    nli = nli or verifier_mod.get_verifier()
    probs = nli.predict(pairs) if pairs else np.empty((0, 3))

    scores: List[ChunkScore] = []
    for ci, r in enumerate(results):
        e_m = c_m = c_a = 0.0
        top_claim_sim = max((c.sim for c in r.misinfo), default=0.0)
        top_auth_sim = max((c.sim for c in r.authority), default=0.0)
        matched: Optional[dict] = None
        best_entail = -1.0
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
            matched=matched,
        ))
    return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run from `maven_app/`: `python tests/test_scoring_unit.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add maven_app/scoring.py maven_app/tests/test_scoring_unit.py
git commit -m "feat: add stance derivation and calibratable P(misinfo) scoring"
```

---

### Task 8: Rework pipeline.py + end-to-end acceptance test

**Files:**
- Rewrite: `maven_app/pipeline.py`
- Create: `maven_app/tests/test_scoring.py` (end-to-end acceptance, real models)
- Delete: `maven_app/tests/test_iso_calibration.py`

**Interfaces:**
- Consumes: `embedding.embed`, `scoring.score_chunks/active_tau` (Tasks 5, 7).
- Produces: `score_text(text, chunk_mode='auto', window=200, stride=100, batch_size=32, flag_threshold=None) -> pd.DataFrame` — the public API consumed by `app.py`, tests, and the eval harness. Columns, in order: `chunk, chunk_mode, misinfo_entail, guidance_contradict, misinfo_contradict, top_claim_sim, top_auth_sim, stance, scoreable, misinfo_score, flagged, matched_claim, evidence_correction, misinfo_type, misinfo_type_confidence`. `misinfo_score` = P(misinfo); `flagged` = `scoreable and misinfo_score >= tau`; the four `matched_*`/`misinfo_type*` fields populate only on flagged rows (None otherwise); `misinfo_type_confidence` = the matched claim's entailment probability.

- [ ] **Step 1: Write the failing acceptance test**

Create `maven_app/tests/test_scoring.py` (replaces `test_iso_calibration.py` as the pipeline acceptance test; loads all models — expect ~1–2 min on cold caches):

```python
"""End-to-end acceptance: the spec's five stance cases against score_text."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import score_text

EXPECTED_COLUMNS = [
    'chunk', 'chunk_mode', 'misinfo_entail', 'guidance_contradict',
    'misinfo_contradict', 'top_claim_sim', 'top_auth_sim', 'stance',
    'scoreable', 'misinfo_score', 'flagged', 'matched_claim',
    'evidence_correction', 'misinfo_type', 'misinfo_type_confidence',
]


def _one(text):
    t0 = time.time()
    df = score_text(text)
    elapsed = time.time() - t0
    assert len(df) == 1, f'expected 1 chunk, got {len(df)}'
    row = df.iloc[0].to_dict()
    print(f"  [{row['stance']:>20}] p={row['misinfo_score']:.3f} "
          f"flagged={row['flagged']} ({elapsed:.1f}s)  {text[:60]}")
    return row


def main():
    print('\n=== 1) misinfo, casual register -> flagged ===')
    r = _one("Epidurals damage the baby's brain and always lead to a c-section.")
    assert list(r.keys()) == EXPECTED_COLUMNS
    assert r['flagged'], r
    assert r['stance'] == 'asserts_misinfo'
    assert r['matched_claim'] and r['evidence_correction']
    assert r['misinfo_type'] is not None
    assert isinstance(r['misinfo_type_confidence'], float)

    print('\n=== 2) misinfo, clinical register -> flagged (register-confound probe) ===')
    r = _one('Peer-reviewed evidence establishes that epidural analgesia produces '
             'permanent neurological damage in neonates.')
    assert r['flagged'], r

    print('\n=== 3) debunk -> NOT flagged (stance-blindness probe) ===')
    r = _one('No, epidurals do NOT damage your baby\'s brain — that myth keeps '
             'scaring people out of safe pain relief.')
    assert not r['flagged'], r
    assert r['stance'] == 'debunks_misinfo', r
    assert r['matched_claim'] is None

    print('\n=== 4) accurate + casual -> NOT flagged (register-confound probe) ===')
    r = _one('girl the nausea is rough but small frequent snacks and ginger tea '
             'genuinely helped me survive the first trimester lol')
    assert not r['flagged'], r

    print('\n=== 5) off-topic -> not scoreable, never flagged ===')
    r = _one('Top 5 budget standing desks for your home office in 2026 — '
             'number 3 surprised me.')
    assert not r['scoreable'] and not r['flagged']
    assert r['stance'] == 'off_topic'
    assert r['misinfo_score'] == 0.0

    print('\n=== multi-chunk mixed text keeps per-chunk behavior ===')
    df = score_text(
        'Doctors push induction for billing convenience, not for patient safety. '
        'ACOG recommends low-dose aspirin from 12 weeks for high-risk patients. '
        'I packed my hospital bag last night and I am so excited.'
    )
    print(df[['stance', 'misinfo_score', 'flagged']].to_string())
    assert len(df) >= 2
    assert df['misinfo_score'].between(0, 1).all()
    assert not df.iloc[-1]['flagged']  # the hospital-bag sentence must never flag

    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `maven_app/`: `python tests/test_scoring.py`
Expected: FAIL — current `score_text` returns the old columns (`assert list(r.keys()) == EXPECTED_COLUMNS` fails).

- [ ] **Step 3: Rewrite pipeline.py**

Replace the entire contents of `maven_app/pipeline.py` with:

```python
"""
MAVEN Pipeline — retrieve-and-verify scoring.
score_text() is the single public entry point.

Flow: chunk -> embed (PubMedBERT) -> retrieve reference claims ->
NLI stance verification -> calibrated P(misinfo) -> flag + explainability.
Reference data: maven_app/anchors/reference_library.json (build with
scripts/build_reference_library.py; embedding cache under anchors/_cache/).
"""
import re
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

from embedding import embed
import scoring

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SENTENCE_THRESHOLD = 300
PARAGRAPH_THRESHOLD = 3_000


# Text segmentation (unchanged from the legacy pipeline)
def _approx_tokens(text: str) -> int:
    return len(text.split())


def _by_sentence(text: str) -> List[str]:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]


def _by_paragraph(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n{2,}', text)]
    return [p for p in paras if len(p) > 40]


def _by_sliding_window(text: str, window: int = 200, stride: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk = ' '.join(words[i: i + window])
        if len(chunk) > 40:
            chunks.append(chunk)
        if i + window >= len(words):
            break
    return chunks


def chunk_text(text: str, mode: str = 'auto', window: int = 200,
               stride: int = 100) -> Tuple[List[str], str]:
    if mode == 'auto':
        n = _approx_tokens(text)
        if n < SENTENCE_THRESHOLD:
            mode = 'sentence'
        elif n < PARAGRAPH_THRESHOLD:
            mode = 'paragraph'
        else:
            mode = 'sliding_window'

    dispatch = {
        'sentence':       lambda: _by_sentence(text),
        'paragraph':      lambda: _by_paragraph(text),
        'sliding_window': lambda: _by_sliding_window(text, window, stride),
    }
    return dispatch[mode](), mode


# Public API
def score_text(
    text: str,
    chunk_mode: str = 'auto',
    window: int = 200,
    stride: int = 100,
    batch_size: int = 32,
    flag_threshold: float = None,
) -> pd.DataFrame:
    """
    Intake any body of text; return a DataFrame of per-chunk misinformation scores.

    misinfo_score is P(misinfo) — heuristic until a calibration artifact
    exists, calibrated after (see scoring.py). flagged = scoreable and
    misinfo_score >= tau (tau from the calibration artifact, else 0.5;
    flag_threshold overrides both). matched_claim / evidence_correction /
    misinfo_type / misinfo_type_confidence populate only on flagged rows.
    stance is explanatory metadata: asserts_misinfo / contradicts_guidance /
    debunks_misinfo / on_topic_neutral / off_topic.
    """
    chunks, mode_used = chunk_text(text, mode=chunk_mode, window=window, stride=stride)
    if not chunks:
        return pd.DataFrame()

    chunk_embs = embed(chunks, batch_size=batch_size)
    results = scoring.score_chunks(chunks, chunk_embs)
    tau = flag_threshold if flag_threshold is not None else scoring.active_tau()

    rows = []
    for chunk, r in zip(chunks, results):
        flagged = bool(r.scoreable and r.p_misinfo >= tau)
        matched = r.matched if flagged and r.matched is not None else None
        rows.append({
            'chunk':                   chunk,
            'chunk_mode':              mode_used,
            'misinfo_entail':          round(r.misinfo_entail, 4),
            'guidance_contradict':     round(r.guidance_contradict, 4),
            'misinfo_contradict':      round(r.misinfo_contradict, 4),
            'top_claim_sim':           round(r.top_claim_sim, 4),
            'top_auth_sim':            round(r.top_auth_sim, 4),
            'stance':                  r.stance,
            'scoreable':               r.scoreable,
            'misinfo_score':           round(r.p_misinfo, 4),
            'flagged':                 flagged,
            'matched_claim':           matched['text'] if matched else None,
            'evidence_correction':     matched['correction'] if matched else None,
            'misinfo_type':            matched['type_id'] if matched else None,
            'misinfo_type_confidence': round(r.misinfo_entail, 4) if matched else None,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Delete the superseded test**

```bash
git rm maven_app/tests/test_iso_calibration.py
```

- [ ] **Step 5: Run the acceptance test**

Run from `maven_app/`: `python tests/test_scoring.py`
Expected: `ALL TESTS PASSED`.

Debugging guidance if a case fails (do NOT loosen the test):
- Case 3 (debunk flags): inspect the printed row — if `misinfo_contradict` is high but p is still ≥ τ, the heuristic damp is losing to `guidance_contradict`; check whether the debunk text also contradicts an *authority* statement (it shouldn't — debunks agree with authority). If it does, the retrieved authority statement is probably itself a restated myth; inspect and fix the library extraction filters (Task 4).
- Case 4 (casual accurate flags): check which claim was entailed; if the NLI genuinely entails a misinfo claim from accurate text, record it as a known verifier weakness for the fine-tune (Task 11) and check whether `top_claim_sim` was marginal — raising `TOPIC_FLOOR` by 0.02 is acceptable, retuning the heuristic constants is not (that's Task 10's job with data).
- Case 5 (off-topic scoreable): raise `TOPIC_FLOOR` per Task 5 Step 5 rules.

- [ ] **Step 6: Confirm the other pipeline consumers still import**

Run from `maven_app/`: `python -c "from pipeline import score_text, chunk_text; print('imports OK')"`
Expected: `imports OK` (model + library load logs, then the message).

- [ ] **Step 7: Commit**

```bash
git add maven_app/pipeline.py maven_app/tests/test_scoring.py
git commit -m "feat!: replace centroid+IsoForest scoring with retrieve-and-verify pipeline"
```

---

### Task 9: Flask app + template + e2e test update

**Files:**
- Modify: `maven_app/app.py` (one addition)
- Modify: `maven_app/templates/index.html` (`renderScorePills`, ~lines 551–574)
- Modify: `maven_app/tests/test_flask_e2e.py` (expected columns + prints)

**Interfaces:**
- Consumes: the new `score_text` DataFrame columns (Task 8).
- Produces: `/analyze` JSON rows now carry the new columns; UI pills show ENTAIL / CONTRA / SIM / stance.

- [ ] **Step 1: app.py — cast the new numpy bool**

In `maven_app/app.py`, in the `analyze()` loop, after `row['flagged'] = bool(row['flagged'])`, add:

```python
        row['scoreable'] = bool(row['scoreable'])
```

(`_OPTIONAL_FIELDS` and the summary block need no change — `misinfo_score` and `flagged` keep their names.)

- [ ] **Step 2: Template — replace the score pills**

In `maven_app/templates/index.html`, replace the whole `renderScorePills` function (currently lines 551–574, the version using `claim_delta`/`authority_sim`/`misinfo_sim`/`isolation_score`) with:

```javascript
function renderScorePills(chunk) {
    const flagged      = chunk.flagged;
    const misinfoColor = flagged ? 'text-secondary' : 'text-tertiary';

    const STANCE_SHORT = {
        asserts_misinfo:      'ASSERT',
        contradicts_guidance: 'CONTRA-G',
        debunks_misinfo:      'DEBUNK',
        on_topic_neutral:     'NEUTRAL',
        off_topic:            'OFF-TOPIC',
    };
    const stanceLabel = STANCE_SHORT[chunk.stance] ?? '—';
    const stanceColor = chunk.stance === 'asserts_misinfo' ? 'text-secondary'
                      : chunk.stance === 'debunks_misinfo' ? 'text-tertiary-fixed-dim'
                      : 'text-on-surface';

    const pill = (label, value, valueClass = 'text-on-surface') => `
        <div class="flex flex-col items-center bg-surface-container p-2 border border-outline-variant/10">
            <span class="font-label text-[9px] text-outline uppercase tracking-tighter">${label}</span>
            <span class="font-label text-xs font-bold ${valueClass}">${value}</span>
        </div>`;

    return `
        <div class="grid grid-cols-4 gap-2 mb-6">
            ${pill('ENTAIL', fmtPct(chunk.misinfo_entail), misinfoColor)}
            ${pill('CONTRA', fmtPct(chunk.guidance_contradict))}
            ${pill('SIM',    fmtPct(chunk.top_claim_sim))}
            ${pill('STANCE', stanceLabel, stanceColor)}
        </div>`;
}
```

Then search the template for any other reference to the removed columns (`claim_delta`, `authority_sim`, `misinfo_sim`, `isolation_score`) — as of this writing `renderScorePills` is the only consumer; if others appear, update them the same way.

- [ ] **Step 3: Update the e2e test's column expectations**

In `maven_app/tests/test_flask_e2e.py`: replace the expected-columns list (currently around lines 102–103, listing `'chunk', 'chunk_mode', 'authority_sim', 'misinfo_sim', 'claim_delta', 'isolation_score', 'misinfo_score', 'flagged', ...`) with:

```python
        'chunk', 'chunk_mode', 'misinfo_entail', 'guidance_contradict',
        'misinfo_contradict', 'top_claim_sim', 'top_auth_sim', 'stance',
        'scoreable', 'misinfo_score', 'flagged',
```

and replace the `isolation_score` debug print (around line 60) with:

```python
    print(f'  stance: {row["stance"]}  entail: {row["misinfo_entail"]:.4f}')
```

Adjust any other assertions in that file that reference removed columns the same way (keep the test's intent, swap the column names).

- [ ] **Step 4: Run the e2e test**

Run from `maven_app/`: `python tests/test_flask_e2e.py`
Expected: passes (its `main()` prints per-case results and a final success line).

- [ ] **Step 5: Manual smoke check of the UI**

```bash
cd maven_app && python app.py
```

Open `http://localhost:5000`, paste "Epidurals damage the baby's brain and always lead to a c-section." — expect a flagged card with ENTAIL/CONTRA/SIM/STANCE pills and the matched claim + correction panel. Then paste the debunk sentence from Task 8 case 3 — expect NOT flagged, stance pill DEBUNK. Ctrl-C the server.

- [ ] **Step 6: Commit**

```bash
git add maven_app/app.py maven_app/templates/index.html maven_app/tests/test_flask_e2e.py
git commit -m "feat: surface stance and NLI signals in API and UI"
```

---

### Task 10: Calibration fitting script

**Files:**
- Create: `ml/training/fit_calibration.py`
- Test: `ml/training/tests/test_fit_calibration.py`

**Interfaces:**
- Consumes: `harness.load_eval_set/split_of` (Task 1), `scoring.FEATURES/features_of` and `pipeline.score_text` (Tasks 7–8).
- Produces: `maven_app/models/calibration_head.joblib` — dict `{'model': LogisticRegression, 'features': list[str], 'tau': float, 'meta': dict}`; `scoring._get_calibration()` picks it up on next process start. Pure helpers for tests: `fit_head(X, y) -> LogisticRegression`, `choose_tau(y_true, probs) -> float`.

- [ ] **Step 1: Write the failing test**

Create `ml/training/tests/test_fit_calibration.py`:

```python
"""Unit test for calibration fitting helpers (no models loaded)."""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/training
from fit_calibration import choose_tau, fit_head, save_artifact

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
```

- [ ] **Step 2: Run test to verify it fails**

Run from repo root: `python ml/training/tests/test_fit_calibration.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'fit_calibration'`

- [ ] **Step 3: Write the implementation**

Create `ml/training/fit_calibration.py`:

```python
"""Fit the logistic calibration head + flag threshold on the calibration split.

Usage (from repo root, venv active; loads all models):
    python ml/training/fit_calibration.py \
        [--eval-set ml/eval/eval_set.jsonl] \
        [--out maven_app/models/calibration_head.joblib]

Uses ONLY items whose id hashes into the calibration split (harness.split_of)
and which are not marked "disputed". Doc-level features = the features of the
chunk with max misinfo_score. Rerun whenever the eval set grows; report
test-split metrics separately via ml/eval/run_eval.py --split test.
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
        sys.exit(f'Only {len(items)} calibration items — need at least 12. '
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run from repo root: `python ml/training/tests/test_fit_calibration.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Fit on the seed calibration split and inspect (do NOT commit the artifact yet)**

```bash
python ml/training/fit_calibration.py
```

Expected: either a fitted artifact (if ≥12 of the 24 seed ids hash into calibration) or the "grow the eval set" exit — both fine. If it fits, sanity-check: rerun `python ml/eval/run_eval.py --split test --out /tmp/calibrated-check.md` and confirm F1 didn't collapse versus the heuristic run. **Decision rule:** commit `maven_app/models/calibration_head.joblib` only when fitted on ≥12 items AND test-split F1 ≥ the heuristic's; the seed set is small, so leaving the heuristic active (no artifact) is the expected outcome until the team's hand-labeled items land.

- [ ] **Step 6: Commit**

```bash
git add ml/training/fit_calibration.py ml/training/tests/test_fit_calibration.py
git commit -m "feat(ml): add calibration head fitting with F1-optimal threshold"
```

---

### Task 11: Synthetic NLI pairs + verifier fine-tune script

**Files:**
- Create: `ml/data/build_nli_pairs.py`
- Create: `ml/training/finetune_verifier.py`
- Create: `ml/training/README.md`
- Test: `ml/data/tests/test_build_nli_pairs.py`

**Interfaces:**
- Consumes: `maven_app/anchors/reference_library.json` (Task 4).
- Produces: `ml/data/nli_pairs.jsonl` (generated, committed) — lines `{"premise", "hypothesis", "label"}` with label ∈ `entailment|neutral|contradiction`; the fine-tune script consumes it plus optional extra JSONL files (HealthVer/SciFact conversions, documented in README). Output checkpoint is adopted via `MAVEN_VERIFIER_PATH` only if it beats zero-shot on the test split.

- [ ] **Step 1: Write the failing test**

Create `ml/data/tests/test_build_nli_pairs.py`:

```python
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

    # determinism
    subprocess.run([sys.executable, str(ROOT / 'ml/data/build_nli_pairs.py')],
                   capture_output=True, text=True, check=True)
    rows2 = [json.loads(line) for line in OUT.read_text(encoding='utf-8').splitlines()]
    assert rows == rows2, 'builder is not deterministic'

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run from repo root: `python ml/data/tests/test_build_nli_pairs.py`
Expected: FAIL (`build_nli_pairs.py` doesn't exist; returncode assertion trips).

- [ ] **Step 3: Write the pair builder**

Create `ml/data/build_nli_pairs.py`:

```python
"""Deterministic synthetic NLI pairs from the reference library.

Per base misinfo claim C with correction E:
  entailment:    (each paraphrase of C, C)  and  (assertion template of C, C)
  contradiction: (E, C)                     and  (debunk template of C, C)
  neutral:       (C, next claim C' from a different domain)

Output: ml/data/nli_pairs.jsonl  {"premise", "hypothesis", "label"}
LLM-assisted expansion happens upstream in claim_paraphrases.json; this
script stays deterministic so the dataset is reproducible.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = ROOT / 'maven_app' / 'anchors' / 'reference_library.json'
OUT_PATH = ROOT / 'ml' / 'data' / 'nli_pairs.jsonl'

ASSERT_TEMPLATE = 'Listen, this is something they never tell you: {claim}'
DEBUNK_TEMPLATE = "You may have heard that {claim} That is a myth — it has been thoroughly debunked."


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def main() -> int:
    lib = json.loads(LIB_PATH.read_text(encoding='utf-8'))
    base = [e for e in lib if e['kind'] == 'misinfo' and e['parent_id'] is None]
    paraphrases = {}
    for e in lib:
        if e['kind'] == 'misinfo' and e['parent_id']:
            paraphrases.setdefault(e['parent_id'], []).append(e['text'])

    pairs = []
    for i, entry in enumerate(base):
        claim = entry['text']
        pairs.append({'premise': ASSERT_TEMPLATE.format(claim=_lower_first(claim)),
                      'hypothesis': claim, 'label': 'entailment'})
        for para in paraphrases.get(entry['id'], []):
            pairs.append({'premise': para, 'hypothesis': claim, 'label': 'entailment'})
        pairs.append({'premise': entry['correction'], 'hypothesis': claim,
                      'label': 'contradiction'})
        pairs.append({'premise': DEBUNK_TEMPLATE.format(claim=_lower_first(claim)),
                      'hypothesis': claim, 'label': 'contradiction'})
        # neutral: pair with the next claim from a different domain (wrap around)
        for other in base[i + 1:] + base[:i]:
            if other['domain'] != entry['domain']:
                pairs.append({'premise': claim, 'hypothesis': other['text'],
                              'label': 'neutral'})
                break

    OUT_PATH.write_text('\n'.join(json.dumps(p, ensure_ascii=False) for p in pairs) + '\n',
                        encoding='utf-8')
    print(f'Wrote {len(pairs)} pairs to {OUT_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run from repo root: `python ml/data/tests/test_build_nli_pairs.py`
Expected: `ALL TESTS PASSED` with label counts printed.

- [ ] **Step 5: Write the fine-tune script + README**

Create `ml/training/finetune_verifier.py`:

```python
"""Fine-tune the NLI verifier on synthetic perinatal pairs (+ optional extras).

Designed for Colab GPU. Colab setup cell:
    !pip install "transformers>=4.40" datasets accelerate sentencepiece
Then:
    python ml/training/finetune_verifier.py \
        --train ml/data/nli_pairs.jsonl [path/to/healthver.jsonl ...] \
        --out ml/training/checkpoints/maven-verifier-v1 \
        [--base MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli] \
        [--epochs 2] [--lr 2e-5] [--batch 16]

Every --train file is JSONL with {"premise", "hypothesis", "label"} where
label is entailment|neutral|contradiction. 10% is held out for eval.
Adoption gate (do this manually after training):
  MAVEN_VERIFIER_PATH=<out dir> python ml/eval/run_eval.py --split test --out ...
  Adopt only if test-split F1 beats the zero-shot run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

DEFAULT_BASE = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, label2id, max_length=256):
        self.enc = tokenizer([r['premise'] for r in rows],
                             [r['hypothesis'] for r in rows],
                             truncation=True, max_length=max_length,
                             padding='max_length')
        self.labels = [label2id[r['label']] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', default=DEFAULT_BASE)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base)
    label2id = {label.lower(): int(idx)
                for idx, label in model.config.id2label.items()}
    # tolerate 'entail'/'contradiction' naming variants in the config
    for want, alts in (('entailment', ('entail',)), ('contradiction', ('contradict',))):
        if want not in label2id:
            for alt in alts:
                if alt in label2id:
                    label2id[want] = label2id[alt]
    missing = {'entailment', 'neutral', 'contradiction'} - set(label2id)
    if missing:
        raise SystemExit(f'base checkpoint labels missing {missing}: {model.config.id2label}')

    rows = load_rows(args.train)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(rows))
    cut = max(1, int(0.1 * len(rows)))
    eval_rows = [rows[i] for i in order[:cut]]
    train_rows = [rows[i] for i in order[cut:]]
    print(f'train={len(train_rows)} eval={len(eval_rows)}')

    def accuracy(eval_pred):
        logits, labels = eval_pred
        return {'accuracy': float((logits.argmax(-1) == labels).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            learning_rate=args.lr, per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch, eval_strategy='epoch',
            save_strategy='epoch', save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model='accuracy',
            logging_steps=50, report_to=[],
        ),
        train_dataset=PairDataset(train_rows, tokenizer, label2id),
        eval_dataset=PairDataset(eval_rows, tokenizer, label2id),
        compute_metrics=accuracy,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f'Saved fine-tuned verifier to {args.out}')
    print('Adoption gate: MAVEN_VERIFIER_PATH=<out> python ml/eval/run_eval.py '
          '--split test --out ml/eval/reports/<date>-finetuned.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

Create `ml/training/README.md`:

```markdown
# Verifier Training & Calibration

## Fine-tuning the NLI verifier (PRD M2)

1. Generate synthetic pairs (deterministic, committed):
   `python ml/data/build_nli_pairs.py`
2. Optional public data (recommended): convert HealthVer
   (https://github.com/sarrouti/HealthVer) and/or SciFact
   (https://github.com/allenai/scifact) claim-evidence pairs to the same
   JSONL schema ({"premise", "hypothesis", "label"}) and pass them as
   extra --train files. Map SUPPORTS→entailment, REFUTES→contradiction,
   NOINFO/NEI→neutral.
3. Train on Colab GPU (see finetune_verifier.py docstring for the pip cell):
   `python ml/training/finetune_verifier.py --train ml/data/nli_pairs.jsonl --out ml/training/checkpoints/maven-verifier-v1`
4. Adoption gate — measure before adopting:
   `MAVEN_VERIFIER_PATH=ml/training/checkpoints/maven-verifier-v1 python ml/eval/run_eval.py --split test --out ml/eval/reports/<date>-finetuned.md`
   Adopt (export MAVEN_VERIFIER_PATH in the app environment) only if
   test-split F1 beats the zero-shot report. Keep checkpoints out of git.

## Calibration head (PRD E4)

Once the hand-labeled eval set has grown (see ml/eval/LABELING_GUIDE.md):
`python ml/training/fit_calibration.py` — fits on the calibration split,
picks the F1-optimal tau, writes maven_app/models/calibration_head.joblib
(commit it; a few KB). Report test-split metrics via run_eval.
```

- [ ] **Step 6: Syntax-check the training script (no GPU run)**

Run from repo root: `python -c "import ast; ast.parse(open('ml/training/finetune_verifier.py').read()); print('syntax OK')"`
Expected: `syntax OK` (the actual training run is a manual Colab step per the spec).

- [ ] **Step 7: Commit**

```bash
git add ml/data/build_nli_pairs.py ml/data/nli_pairs.jsonl \
    ml/data/tests/test_build_nli_pairs.py ml/training/finetune_verifier.py \
    ml/training/README.md
git commit -m "feat(ml): add synthetic NLI pair builder and verifier fine-tune script"
```

---

### Task 12: Measured comparison — new system vs. baseline

**Files:**
- Create (generated): `ml/eval/reports/2026-07-16-retrieve-verify-zeroshot.md`

**Interfaces:**
- Consumes: `run_eval.py` (Task 3), reworked pipeline (Tasks 8–9).
- Produces: the committed A/B evidence for PRD M4/E2.

- [ ] **Step 1: Run the harness against the new pipeline**

From repo root:

```bash
python ml/eval/run_eval.py \
    --out ml/eval/reports/2026-07-16-retrieve-verify-zeroshot.md \
    --title "Retrieve-and-verify (zero-shot NLI, heuristic aggregation)"
```

- [ ] **Step 2: Compare against the baseline report**

Open both `ml/eval/reports/2026-07-16-baseline-legacy.md` and the new report. Acceptance gates (on the seed set):

- `debunks_misinfo` flag rate: must be **lower** than baseline (target 0).
- `off_topic` flag rate: must be 0.
- F1: must be ≥ baseline.
- Mean latency: must be ≤ 15 s/item.

Append a `## Comparison vs baseline` section to the new report (hand-written, 3–6 lines) stating each gate and pass/fail. If a gate fails, STOP — debug per Task 8 Step 5 guidance before committing; do not commit a failing comparison as done.

- [ ] **Step 3: Commit**

```bash
git add ml/eval/reports/2026-07-16-retrieve-verify-zeroshot.md
git commit -m "docs(ml): record retrieve-and-verify vs legacy baseline comparison"
```

---

### Task 13: Notebook, CLAUDE.md, and legacy cleanup

**Files:**
- Modify: `MAVEN_AI_UNC_SPR2026.ipynb` (Section 4 replacement)
- Modify: `CLAUDE.md`
- Delete: `scripts/build_anchors.py`, `maven_app/anchors/authority_anchors.json`, `maven_app/anchors/misinfo_anchors.json`, `maven_app/anchors/misinfo_type_anchors.json`, `maven_app/anchors/comprehensive_paragraphs.json`, and stale `maven_app/anchors/_cache/` legacy files

**Interfaces:**
- Consumes: everything prior.
- Produces: docs/deliverables consistent with the new backend.

- [ ] **Step 1: Delete legacy files**

```bash
git rm scripts/build_anchors.py \
    maven_app/anchors/authority_anchors.json \
    maven_app/anchors/misinfo_anchors.json \
    maven_app/anchors/misinfo_type_anchors.json \
    maven_app/anchors/comprehensive_paragraphs.json
rm -f maven_app/anchors/_cache/authority_embs.npy \
    maven_app/anchors/_cache/misinfo_claim_embs.npy \
    maven_app/anchors/_cache/misinfo_type_centroids.npz \
    maven_app/anchors/_cache/iso_forest.joblib \
    maven_app/anchors/_cache/iso_calibration.npy
```

**Caution:** `build_reference_library.py` reads `misinfo_anchors.json`? It does NOT (it reads the docx) — but `ml/data/claim_type_map.json` creation referenced it. Both survive as generated/curated artifacts; nothing imports the deleted files. Verify:

```bash
grep -rn "misinfo_anchors\|authority_anchors\|comprehensive_paragraphs\|misinfo_type_anchors\|build_anchors" \
    maven_app/ scripts/ ml/ --include="*.py" | grep -v test_reference_library
```

Expected: no hits (if `test_reference_library.py` or others reference deleted files, fix them now).

- [ ] **Step 2: Re-run the full maven_app test suite**

From `maven_app/`:

```bash
python tests/test_reference_library.py && \
python tests/test_retrieval.py && \
python tests/test_verifier.py && \
python tests/test_scoring_unit.py && \
python tests/test_scoring.py && \
python tests/test_flask_e2e.py
```

Expected: all print their success lines.

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`:

a. Replace the `maven_app/` structure block's `pipeline.py` line and add the new modules:

```
  app.py              # Flask routes and request handling
  pipeline.py         # Public entry point: chunk → retrieve → verify → P(misinfo)
  embedding.py        # PubMedBERT sentence-embedding model (single load)
  retrieval.py        # Reference-library retrieval + on-topic gate
  verifier.py         # DeBERTa-v3 NLI cross-encoder (stance verification)
  scoring.py          # Feature aggregation, stance, calibrated P(misinfo)
```

and add after the `anchors/` line:

```
  models/             # Calibration head artifact (when fitted)
```

b. Replace the `anchors/` line's comment with `# reference_library.json + embedding cache`.

c. Add to the Key Dependencies table:

```
| `transformers` + `sentencepiece` | DeBERTa-v3 NLI verifier (stance verification) |
```

and delete the table rows for libraries no longer used by the pipeline if present (Isolation Forest reference under scikit-learn stays — scikit-learn now powers the calibration head; update its Purpose cell to `Logistic calibration head`).

d. Replace the "Pipeline Entry Point" section's text with:

```markdown
## Pipeline Entry Point

`score_text(text)` in `maven_app/pipeline.py` is the end-to-end function:
chunk → PubMedBERT embed → retrieve reference claims (misinfo + authority,
on-topic gate) → DeBERTa-v3 NLI stance verification → P(misinfo) with
threshold τ (heuristic 0.5 until `maven_app/models/calibration_head.joblib`
exists — fit it with `ml/training/fit_calibration.py`). Returns a
DataFrame: `chunk, chunk_mode, misinfo_entail, guidance_contradict,
misinfo_contradict, top_claim_sim, top_auth_sim, stance, scoreable,
misinfo_score, flagged, matched_claim, evidence_correction, misinfo_type,
misinfo_type_confidence`.

The reference library is built by `scripts/build_reference_library.py`
from the domain .docx assets + `ml/data/claim_type_map.json` +
`ml/data/claim_paraphrases.json`. Evaluation lives in `ml/eval/`
(`run_eval.py`, labeled set, reports); training in `ml/training/`.
Env vars: `MAVEN_VERIFIER_PATH` (verifier checkpoint),
`MAVEN_CALIBRATION_PATH` (calibration artifact).
```

(Also delete the now-obsolete final paragraph about `AUTHORITY_ANCHORS` / `MISINFO_ANCHORS` placeholders.)

e. In "Running Tests", change the example to `python tests/test_scoring.py`.

- [ ] **Step 4: Update the notebook**

Using NotebookEdit on `MAVEN_AI_UNC_SPR2026.ipynb`: replace the Section 4 ("Misinformation Markers") cells with a "Retrieve-and-Verify Scoring" section. Cell sequence:

1. **Markdown** — narrative: why embedding proximity can't detect misinformation (stance blindness, register confound — reuse the spec's "Why the current approach fails" bullets), and the retrieve→verify→aggregate flow diagram from the spec (as a fenced code block).
2. **Code** — `!pip install "transformers>=4.40" sentencepiece` (append to the existing install cell for the section if one exists, else new cell).
3. **Code** — inline demo reference library: a ~10-entry Python list of dicts using entries copied verbatim from `maven_app/anchors/reference_library.json` (6 misinfo base claims with corrections + 4 authority statements), with a markdown note that the app uses the full JSON library built by `scripts/build_reference_library.py`.
4. **Code** — retrieval: copy `retrieve`/`_top_k`/`base_entry` from `maven_app/retrieval.py`, adapted to embed the inline library directly (no file cache) — i.e. `ref_embs = embed([e['text'] for e in LIBRARY])` using the notebook's existing `embed()` from Section 3.
5. **Code** — verifier: copy the `Verifier` class from `maven_app/verifier.py` verbatim.
6. **Code** — scoring: copy `_stance`, `_p_misinfo` (heuristic branch only), and a simplified `score_chunks` + `score_text` from `maven_app/scoring.py`/`pipeline.py` (drop the calibration-artifact branch; keep constants and the pair cap).
7. **Code** — demo: score three texts (the Task 8 assert/debunk/off-topic cases) and display the DataFrame, with a markdown cell interpreting the stance column and why the debunk is not flagged.

Keep Sections 1–3 (overview, chunking, embeddings) unchanged. Run the notebook top-to-bottom locally (`jupyter nbconvert --to notebook --execute MAVEN_AI_UNC_SPR2026.ipynb --output /tmp/nb-check.ipynb --ExecutePreprocessor.timeout=1200`) — expected: executes cleanly (first run downloads the verifier).

- [ ] **Step 5: Commit**

```bash
git add MAVEN_AI_UNC_SPR2026.ipynb CLAUDE.md
git add -u  # stages the deletions
git commit -m "docs: document retrieve-and-verify backend; remove legacy anchor system"
```

---

### Task 14: M1 baseline classifier (logistic regression on embeddings)

**Files:**
- Create: `ml/training/train_baseline.py`
- Create (generated): `ml/eval/reports/2026-07-16-m1-baseline-logreg.md`

**Interfaces:**
- Consumes: `reference_library.json` (Task 4), `build_nli_pairs.ASSERT_TEMPLATE/DEBUNK_TEMPLATE/_lower_first` (Task 11), `embedding.embed` (Task 5), `harness.evaluate/load_eval_set/render_report` (Task 1).
- Produces: the PRD M1 baseline comparator — a purely synthetic-trained logistic regression scored through the same harness. No runtime integration; it exists for the metrics table (and to demonstrate that an embedding classifier stays stance-blind).

- [ ] **Step 1: Write the script**

Create `ml/training/train_baseline.py`:

```python
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
```

- [ ] **Step 2: Run it**

From repo root: `python ml/training/train_baseline.py --out ml/eval/reports/2026-07-16-m1-baseline-logreg.md`
Expected: report written. Inspect: the `debunks_misinfo` flag rate is expected to be HIGH (stance-blind classifier) — that contrast versus Task 12's report is the M4 architecture-comparison evidence.

- [ ] **Step 3: Commit**

```bash
git add ml/training/train_baseline.py ml/eval/reports/2026-07-16-m1-baseline-logreg.md
git commit -m "feat(ml): add M1 logistic-regression baseline with harness report"
```

---

## Post-plan follow-ups (not tasks — team/process work)

- Team hand-labels real items into `ml/eval/eval_set.jsonl` per `LABELING_GUIDE.md` (target 200–300), then: refit calibration (Task 10 Step 5's decision rule), re-run Task 12's comparison on `--split test`.
- Colab GPU fine-tune run per `ml/training/README.md`; adopt via `MAVEN_VERIFIER_PATH` only if it beats zero-shot on the test split.
- Review `ml/data/claim_type_map.json` (`_status: draft`) with Dr. Bazzano.
- Optional experiment: A/B PubMedBERT vs BGE/GTE retriever through the harness.
