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
