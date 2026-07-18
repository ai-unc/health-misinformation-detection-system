"""End-to-end Flask test.

Boots the Flask app in-process (no separate server) using app.test_client()
and exercises the contract: JSON round-trip, optional-field handling for
flagged vs non-flagged rows (including the authority-side explanation
fallback), single-sentence misinfo (the previously broken regression), and
per-request latency.
"""
import json
import sys
import time
from pathlib import Path

# Allow running from any cwd: maven_app/ (parent dir) holds app.py and pipeline.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app


def _post(client, text):
    r = client.post('/analyze', json={'text': text})
    return r.status_code, r.get_json()


def main():
    client = app.test_client()

    print('\n=== TEST 1: mixed flagged + non-flagged narrative (multi-chunk) ===')
    t0 = time.time()
    status, body = _post(
        client,
        "Doctors push induction for billing convenience, not for patient safety. "
        "Trust your body — it knows what to do."
    )
    print(f'  status: {status}  elapsed: {time.time() - t0:.2f}s')
    raw = json.dumps(body)
    print(f'  raw JSON head: {raw[:200]}...')

    # Strict JSON round-trip
    parsed = json.loads(raw)
    print(f'  STRICT JSON round-trip: OK  ({len(parsed["chunks"])} chunks)')

    optional = ('matched_claim', 'evidence_correction', 'misinfo_type', 'misinfo_type_confidence')
    for c in body['chunks']:
        flagged = c['flagged']
        all_null = all(c[k] is None for k in optional)
        print(f'  chunk: {c["chunk"][:60]}')
        print(f'    flagged={flagged}  type={c["misinfo_type"]}  conf={c["misinfo_type_confidence"]}')
        print(f'    matched={None if c["matched_claim"] is None else c["matched_claim"][:60]}')
        if flagged:
            # Real invariant, not a strict all-or-nothing dichotomy: the
            # Task 8 authority-side fallback (pipeline.py ~96-123) can flag
            # a row via a contradicted authority entry with no cataloged
            # claim entailed, so matched_claim stays null while
            # evidence_correction is still set from the authority text
            # (and misinfo_type/misinfo_type_confidence stay null too,
            # unless that authority entry carries a parent_id). The
            # matched-claim path is the only one that sets all four
            # together. So: every flagged row ships a non-empty
            # evidence_correction (the test_scoring.py acceptance
            # invariant); matched_claim MAY be null; when matched_claim IS
            # non-null, misinfo_type/misinfo_type_confidence must be set
            # alongside it.
            assert c['evidence_correction'], f'Flagged row missing evidence_correction: {c}'
            if c['matched_claim'] is not None:
                assert c['misinfo_type'] is not None, f'Matched row missing misinfo_type: {c}'
                assert c['misinfo_type_confidence'] is not None, f'Matched row missing misinfo_type_confidence: {c}'
        else:
            assert all_null, f'Non-flagged row leaked NaN: {c}'

    print('\n=== TEST 2: single-sentence misinfo (regression — must flag now) ===')
    status, body = _post(client, "Epidurals damage the baby's brain and always lead to a c-section.")
    print(f'  status: {status}  flagged: {body["summary"]["flagged"]}/{body["summary"]["total"]}')
    assert status == 200
    row = body['chunks'][0]
    print(f'  stance: {row["stance"]}  entail: {row["misinfo_entail"]:.4f}')
    print(f'  misinfo_score:   {row["misinfo_score"]:.4f}')
    print(f'  flagged:         {row["flagged"]}')
    assert row['flagged'], (
        f'REGRESSION: single-sentence misinfo via Flask not flagged '
        f'(score={row["misinfo_score"]:.4f})'
    )
    assert row['matched_claim'] is not None
    assert row['evidence_correction'] is not None
    assert row['misinfo_type'] is not None
    print(f'  matched_claim: {row["matched_claim"]}')
    print(f'  misinfo_type:  {row["misinfo_type"]} (conf {row["misinfo_type_confidence"]})')

    print('\n=== TEST 3: empty body (must 400) ===')
    status, body = _post(client, '')
    print(f'  status: {status}  body: {body}')
    assert status == 400

    print('\n=== TEST 4: tiny body (must 422 — no scorable chunks) ===')
    status, body = _post(client, 'hi')
    print(f'  status: {status}  body: {body}')
    assert status == 422

    print('\n=== TEST 5: latency (3 calls, 4-sentence body, target <5s/call) ===')
    body4 = (
        "Adequate prenatal care reduces maternal mortality. "
        "Folic acid prevents neural tube defects according to CDC guidance. "
        "Trust your body — it knows what to do. "
        "Epidurals damage the baby's brain."
    )
    times = []
    for _ in range(3):
        t = time.time()
        s, _ = _post(client, body4)
        assert s == 200
        times.append(time.time() - t)
    total = sum(times)
    print(f'  3x analyze: {total:.2f}s total ({total/3:.2f}s avg, max {max(times):.2f}s)')
    assert max(times) < 5.0, f'Latency regression: max={max(times):.2f}s exceeds 5s'

    print('\n=== TEST 6: response key contract (no keys removed/renamed) ===')
    expected = {
        'chunk', 'chunk_mode', 'misinfo_entail', 'guidance_contradict',
        'misinfo_contradict', 'top_claim_sim', 'top_auth_sim', 'stance',
        'scoreable', 'misinfo_score', 'flagged',
        'matched_claim', 'evidence_correction', 'misinfo_type', 'misinfo_type_confidence',
    }
    status, body = _post(client, 'Folic acid is essential during early pregnancy.')
    actual = set(body['chunks'][0].keys())
    print(f'  expected: {sorted(expected)}')
    print(f'  actual:   {sorted(actual)}')
    print(f'  missing:  {expected - actual}')
    print(f'  extra:    {actual - expected}')
    assert actual == expected

    print('\n=== TEST 7: authority-side fallback (flagged, matched_claim null) ===')
    # Same probe as test_scoring.py acceptance case 2 (register-confound
    # probe): low misinfo_entail keeps matched_claim null, but the high
    # guidance_contradict still flags the row, so pipeline.py's fallback
    # (~113-123) must supply evidence_correction from the contradicted
    # authority entry.
    status, body = _post(
        client,
        'Peer-reviewed evidence establishes that epidural analgesia produces '
        'permanent neurological damage in neonates.'
    )
    print(f'  status: {status}  flagged: {body["summary"]["flagged"]}/{body["summary"]["total"]}')
    assert status == 200
    flagged_rows = [c for c in body['chunks'] if c['flagged']]
    assert flagged_rows, f'expected at least one flagged chunk: {body["chunks"]}'
    row = flagged_rows[0]
    print(f'  stance: {row["stance"]}  entail: {row["misinfo_entail"]:.4f}  '
          f'guidance_contradict: {row["guidance_contradict"]:.4f}')
    print(f'  misinfo_score:       {row["misinfo_score"]:.4f}')
    print(f'  matched_claim:       {row["matched_claim"]}')
    print(f'  evidence_correction: {row["evidence_correction"]}')
    assert isinstance(row['evidence_correction'], str) and row['evidence_correction'], (
        f'Fallback row missing evidence_correction: {row}'
    )
    # matched_claim pinned to None: observed via this same test client at
    # HEAD (aac15a9) -- tied to the current reference library + NLI
    # checkpoint, not asserted as a permanent structural guarantee. Stance
    # deliberately left unpinned (mirrors test_scoring.py case 2): the
    # shape of the fallback is the contract, not the exact stance string.
    assert row['matched_claim'] is None, f'expected fallback path (matched_claim None), got: {row}'

    print('\nALL FLASK E2E TESTS PASSED')


if __name__ == '__main__':
    main()
