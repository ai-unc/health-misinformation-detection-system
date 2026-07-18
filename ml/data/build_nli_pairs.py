"""Deterministic synthetic NLI pairs from the reference library.

Per base misinfo claim C with correction E:
  entailment:    (each paraphrase of C, C)  and  (assertion template of C, C)
  contradiction: (E, C)                     and  (debunk template of C, C)
  neutral:       (C, next claim C' from a different domain) + clean perinatal chatter vs claims/guidance

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

# Task 8 carry-in: the zero-shot checkpoint's contradiction channel fires
# loosely on everyday perinatal chatter. Pair clean neutral statements
# against claims and authority guidance so fine-tuning learns them as neutral.
NEUTRAL_STATEMENTS = [
    "I packed my hospital bag last night and double-checked the car seat.",
    "We toured the birth center on Saturday and met two of the midwives.",
    "My glucose test is scheduled for Thursday morning.",
    "The nursery furniture finally arrived and the crib is set up.",
    "I felt the baby kick during the movie last night.",
    "We are still deciding between two names for our daughter.",
    "My partner installed the infant car seat and had it inspected.",
    "The lactation class at the hospital was rescheduled to next week.",
    "I ordered a pregnancy pillow and it has helped me sleep.",
    "Our doula sent over a checklist for the third trimester.",
]


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def main() -> int:
    lib = json.loads(LIB_PATH.read_text(encoding='utf-8'))
    base = [e for e in lib if e['kind'] == 'misinfo' and e['parent_id'] is None]
    paraphrases = {}
    for e in lib:
        if e['kind'] == 'misinfo' and e['parent_id']:
            paraphrases.setdefault(e['parent_id'], []).append(e['text'])

    authority = [e for e in lib if e['kind'] == 'authority']

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

    for j, stmt in enumerate(NEUTRAL_STATEMENTS):
        pairs.append({'premise': stmt, 'hypothesis': base[j % len(base)]['text'],
                      'label': 'neutral'})
        pairs.append({'premise': stmt, 'hypothesis': authority[j % len(authority)]['text'],
                      'label': 'neutral'})

    OUT_PATH.write_text('\n'.join(json.dumps(p, ensure_ascii=False) for p in pairs) + '\n',
                        encoding='utf-8')
    print(f'Wrote {len(pairs)} pairs to {OUT_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
