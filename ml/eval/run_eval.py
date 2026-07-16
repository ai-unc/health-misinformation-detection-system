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
