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
