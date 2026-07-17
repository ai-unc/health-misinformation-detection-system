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
        assert isinstance(e['text'], str) and len(e['text'].split()) >= 3, e['id']
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
