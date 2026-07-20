"""Lexical content gate: non-sentence chunks (hashtag blocks, mention runs,
URLs, emoji) must be non_content / non-scoreable and never reach the verifier.

Regression for the hashtag false positive: '#firsttrimester #pregnancy'
scored 0.98 because it cleared the topic floor and the NLI verifier emits
spurious contradiction for non-propositional premises (base checkpoint and
maven-verifier-v1 both do). The gate must stop these chunks before NLI,
regardless of verifier behavior — proven here with an adversarial stub that
contradicts everything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from embedding import embed
from scoring import has_scoreable_content, score_chunks

HASHTAG_EXACT   = '#firsttrimester #pregnancy'
HASHTAG_BLOCK   = '#firsttrimester #pregnancy #momlife #babybump #newmom #morningsickness'
HASHTAG_TAIL    = '😅 #firsttrimester #pregnancy #caffeinefree #momtok #fyp'
MENTION_RUN     = '@pregnancycoach @momlife.daily @firsttrimestertips credit: @obgyn.talks'
URL_SCHEMED     = 'https://linktr.ee/pregnancycoach https://www.instagram.com/p/DLxyz123abc/'
URL_HEALTH_SLUG = 'https://www.healthline.com/health/pregnancy/first-trimester-morning-sickness'
URL_BARE        = 'linktr.ee/pregnancycoach'
EMOJI_ONLY      = '🤰👶💕✨🥰🥰🥰💯💯💯🍼🍼🤰👶💕✨🥰🥰🥰💯💯💯🍼🍼'

SENTENCE        = 'No coffee for me this week!'
HEADLINE_3W     = 'Epidurals are dangerous.'
MIXED_RESIDUE   = 'Vaccines cause autism #pregnancy #firsttrimester'
NUMERIC_CLAIM   = '200 mg caffeine max per day'
# Deliberate pass-throughs: real words, so the lexical gate must NOT fire —
# gating pipe-joined text would false-negative legit OCR slides
# ('Myth: ... | Fact: ...'). The topic floor / verifier remain their defense.
PIPE_NOISE      = 'pregnancy | first trimester | week 8 | baby | tips'
NAV_TEXT        = 'PART 1 OF 3 SWIPE FOR MORE ➡️➡️➡️ FOLLOW FOR PART 2'

ASSERT_TEXT = "Epidurals damage the baby's brain and always lead to a c-section."


class AdversarialNLI:
    """Worst-case verifier: high contradiction for every pair it sees."""

    def __init__(self):
        self.seen_pairs = []

    def predict(self, pairs, batch_size=16):
        self.seen_pairs.extend(pairs)
        return np.tile((0.02, 0.03, 0.95), (len(pairs), 1))


def main():
    # --- unit: gate fires on junk ---
    for text in (HASHTAG_EXACT, HASHTAG_BLOCK, HASHTAG_TAIL, MENTION_RUN,
                 URL_SCHEMED, URL_HEALTH_SLUG, URL_BARE, EMOJI_ONLY):
        assert not has_scoreable_content(text), f'gate must fire: {text!r}'

    # --- unit: gate must NOT fire on real content ---
    for text in (SENTENCE, HEADLINE_3W, MIXED_RESIDUE, NUMERIC_CLAIM,
                 PIPE_NOISE, NAV_TEXT):
        assert has_scoreable_content(text), f'gate must not fire: {text!r}'

    # --- integration: gated chunks never reach the verifier ---
    chunks = [HASHTAG_BLOCK, MENTION_RUN, ASSERT_TEXT]
    embs = embed(chunks)
    nli = AdversarialNLI()
    r_tag, r_mention, r_assert = score_chunks(chunks, embs, nli=nli)

    for r, text in ((r_tag, HASHTAG_BLOCK), (r_mention, MENTION_RUN)):
        assert r.stance == 'non_content', (text, r.stance)
        assert not r.scoreable, (text, r)
        assert r.p_misinfo == 0.0, (text, r.p_misinfo)
        assert r.top_claim_sim == 0.0 and r.top_auth_sim == 0.0, (text, r)
        assert r.matched is None and r.contradicted is None, (text, r)

    # even an always-contradict verifier cannot flag them: no pair was spent
    seen_premises = {premise for premise, _ in nli.seen_pairs}
    assert HASHTAG_BLOCK not in seen_premises, 'hashtag chunk reached the verifier'
    assert MENTION_RUN not in seen_premises, 'mention chunk reached the verifier'

    # The real chunk still flows through scoring in the same call. The
    # always-contradict stub hits its misinfo pairs too, so debunk damping
    # yields p = 0.95 - 0.5*0.95 — the point is p > 0, unlike the gated
    # chunks' hard 0.0.
    assert r_assert.scoreable, r_assert
    assert r_assert.stance != 'non_content', r_assert
    assert r_assert.p_misinfo > 0.0, r_assert.p_misinfo
    assert ASSERT_TEXT in seen_premises, 'real chunk must reach the verifier'

    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
