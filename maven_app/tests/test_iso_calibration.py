"""Regression test: single-sentence misinfo must flag.

Previously the per-batch min-max iso_score normalization forced a single-chunk
request to iso_score=0.0, capping misinfo_score at 0.70 * delta_score and
preventing any single-sentence input from ever crossing FLAG_THRESHOLD=0.60.
After the calibration fix, iso_score is normalized against the training-corpus
distribution, so a single-sentence call should now flag overt misinfo and
populate the four explainability fields.
"""
import json
import sys
import time
from pathlib import Path

# Allow running from any cwd: maven_app/ (parent dir) holds pipeline.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import score_text


def _row(df, i):
    return df.iloc[i].to_dict()


def main():
    print('\n=== Single-sentence misinfo (the bug case) ===')
    t0 = time.time()
    df = score_text("Epidurals damage the baby's brain and always lead to a c-section.")
    print(f'  elapsed: {time.time() - t0:.2f}s')
    print(f'  rows: {len(df)}')
    r0 = _row(df, 0)
    print(f'  chunk:           {r0["chunk"][:70]}')
    print(f'  isolation_score: {r0["isolation_score"]:.4f}')
    print(f'  claim_delta:     {r0["claim_delta"]:.4f}')
    print(f'  misinfo_score:   {r0["misinfo_score"]:.4f}')
    print(f'  flagged:         {r0["flagged"]}')
    if r0['flagged']:
        print(f'  matched_claim:       {r0["matched_claim"]}')
        print(f'  evidence_correction: {str(r0["evidence_correction"])[:80]}...')
        print(f'  misinfo_type:        {r0["misinfo_type"]} ({r0["misinfo_type_confidence"]})')
    assert r0['flagged'], (
        f'REGRESSION: single-sentence misinfo no longer flags '
        f'(misinfo_score={r0["misinfo_score"]:.4f}, threshold=0.60)'
    )
    assert r0['matched_claim'] is not None
    assert r0['evidence_correction'] is not None
    assert r0['misinfo_type'] is not None
    assert isinstance(r0['misinfo_type_confidence'], float)
    print('  PASS: single-sentence misinfo flagged with all four explainability fields')

    print('\n=== Single-sentence authority (must NOT flag) ===')
    df2 = score_text(
        'ACOG recommends low-dose aspirin from 12 weeks for high-risk patients to reduce preeclampsia risk.'
    )
    r1 = _row(df2, 0)
    print(f'  isolation_score: {r1["isolation_score"]:.4f}')
    print(f'  misinfo_score:   {r1["misinfo_score"]:.4f}')
    print(f'  flagged:         {r1["flagged"]}')
    assert not r1['flagged'], (
        f'False positive: single authority sentence flagged '
        f'(misinfo_score={r1["misinfo_score"]:.4f})'
    )
    assert r1['matched_claim'] is None
    print('  PASS: authority sentence not flagged, no enrichment fields populated')

    print('\n=== Multi-chunk batch (regression check; iso_score should still be in [0,1]) ===')
    df3 = score_text(
        "Trust your body — it knows what to do. "
        "Doctors push induction for billing convenience, not for patient safety. "
        "ACOG recommends low-dose aspirin from 12 weeks for high-risk patients."
    )
    print(f'  rows: {len(df3)}')
    for i, r in df3.iterrows():
        print(f'  [{i}] iso={r["isolation_score"]:.4f}  delta={r["claim_delta"]:+.4f}  '
              f'score={r["misinfo_score"]:.4f}  flagged={r["flagged"]}')
    assert (df3['isolation_score'] >= 0.0).all() and (df3['isolation_score'] <= 1.0).all()
    print('  PASS: iso_score remains within [0, 1] across multi-chunk batch')

    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
