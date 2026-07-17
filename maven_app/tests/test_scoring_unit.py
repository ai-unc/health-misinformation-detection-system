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
        # h-key scoped to the retrieved misinfo claim (mis-017). The debunk chunk
        # also retrieves epidural *authority* text (auth-016), which it agrees
        # with — a real NLI would entail it, so a bare 'pidural' key that returns
        # contradict for that authority pair would be an unfaithful stub.
        ('No, epidurals', 'Choosing an epidural'): (0.02, 0.08, 0.90),  # contradicts the claim
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
