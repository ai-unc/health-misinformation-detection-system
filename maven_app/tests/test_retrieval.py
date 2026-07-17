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
