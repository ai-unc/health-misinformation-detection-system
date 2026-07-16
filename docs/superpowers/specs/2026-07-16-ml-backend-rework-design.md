# ML Backend Rework: Retrieve-and-Verify — Design

**Date:** 2026-07-16
**Status:** Approved
**Branch:** `ml-backend-rework`

## Goal

Replace MAVEN's embedding-proximity scoring (centroid cosine similarity +
IsolationForest) with a retrieve-and-verify architecture: PubMedBERT
retrieval over a curated claim/evidence library, an NLI cross-encoder that
determines *stance*, and a calibrated head that outputs a real
P(misinfo). Build the evaluation harness first so every change is a
measured claim against the current system.

## Why the current approach fails

All four markers are functions of embedding proximity, and embedding
geometry encodes **topic and register, not truth or stance**:

- **Stance blindness.** "Epidurals damage babies' brains" and "Myth:
  epidurals do NOT damage babies' brains" embed nearly identically. The
  system cannot distinguish asserting misinfo from debunking it, so
  debunking content — what MCH researchers produce — gets flagged.
- **Register confound.** `claim_delta` (70% of the score) mostly measures
  "clinical prose vs. casual prose." False claims in clinical language
  pass; true claims in casual language get flagged.
- **Centroid collapse.** 119 authority anchors (many are bare org names)
  and 74 misinfo claims spanning 16 domains are each averaged into a
  single vector, destroying topical structure.
- **IsolationForest ≈ noise.** Trained on 281 embeddings from one docx
  (including title-page fragments); its signal is "unlike this one
  document," which fires on casual register, not misinfo. It is 30% of
  the composite score.
- **No probability semantics, no evaluation.** `0.70·delta + 0.30·iso ≥
  0.60` is uncalibrated; the repo has zero labeled data and zero metrics,
  leaving PRD M1–M5 and E1–E4 unsatisfied.

**Worth keeping:** the 74 domain-curated claim→evidence-correction pairs,
the JGIM 8-type taxonomy, PubMedBERT as a *retriever*, chunking, caching,
and retrieval-based explainability.

## Decisions (constraints locked with user)

| Decision | Choice |
|---|---|
| LLM at inference | None. Core scorer is local, free, reproducible. LLMs allowed offline for data generation only. |
| Labels | Synthetic training data from the claim library + public datasets; small hand-labeled real eval set (~200–300 items) for honest metrics. |
| Latency | Soft budget: typical transcript scored in ≤15 s on CPU (download/transcription already dominates). Base-size verifier acceptable. |
| Architecture | Retrieve-and-verify (Approach B). Fine-tuned end-to-end classifier built only as a baseline comparator. |
| Runtime deletions | IsolationForest and centroid `claim_delta` leave the runtime. Iso survives only as an ablation feature in the harness until numbers justify dropping it. |

## Architecture

### Inference data flow

```
text ──▶ chunk_text (unchanged)
     ──▶ PubMedBERT embed (unchanged)
     ──▶ RETRIEVE  top-k nearest misinfo claims + top-k authority statements
         │         similarity floor = on-topic gate
         │         below floor → scoreable=False, never flagged
     ──▶ VERIFY    NLI cross-encoder, one batched pass over all
         │         (chunk, reference) pairs:
         │           P(chunk entails misinfo claim)       → asserts misinfo
         │           P(chunk contradicts authority stmt)  → novel-claim path
         │           P(chunk contradicts misinfo claim)   → debunk, protected
     ──▶ AGGREGATE calibrated logistic head over features → P(misinfo)
     ──▶ flag = P(misinfo) ≥ τ  +  stance  +  matched claim/correction/type
```

### Components

**`maven_app/retrieval.py`** — loads the reference library + cached
embeddings; `retrieve(chunk_embs)` returns top-k candidates per chunk from
each of the two reference kinds, with the similarity floor applied.
Defaults: k=4 per kind; floor starts at 0.45 and is tuned on the
calibration split in Phase 4.

**Reference library** (replaces the four anchor JSONs) — entries carry
`{id, text, kind: misinfo|authority, domain, type_id, paired_correction}`:

- *Misinfo claims:* the 74 curated claims, expanded with assertion
  paraphrases (casual + clinical register), each keeping its paired
  correction and domain. `type_id` (JGIM 8-type taxonomy) does not exist
  in the source docx; it is added as a one-time curation task — LLM-assisted
  mapping of the 74 claims to the 8 types, human-reviewed, stored in the
  library JSON. Paraphrases inherit the parent claim's type and correction.
- *Authority statements:* declarative sentences extracted from
  `perinatal_comprehensive.docx` with junk filtering (headers, org-name
  lines, title fragments dropped), **plus the 74 evidence-corrections
  themselves**, which are already ideal authority statements. Bare org
  names are gone.

**`maven_app/verifier.py`** — wraps the NLI cross-encoder. Ships zero-shot
from `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; checkpoint path is
env-overridable (`MAVEN_VERIFIER_PATH`) so the fine-tuned model drops in
without code changes. One batched forward pass for all pairs in a
request. Latency guard: cap total NLI pairs per request (~256), reducing
k proportionally rather than dropping chunks.

**`maven_app/scoring.py`** — per-chunk feature vector
`[max misinfo-entailment, max guidance-contradiction, their retrieval
sims]`; calibrated logistic head (fitted offline on the calibration
split, persisted via joblib with version metadata) → P(misinfo); stance
label: `asserts_misinfo / contradicts_guidance / debunks_misinfo /
on_topic_neutral / off_topic`. Stance derivation (precedence order):
`off_topic` if not scoreable; else `debunks_misinfo` if max
P(contradict | misinfo claim) exceeds both entailment signals;
else `asserts_misinfo` if max P(entail | misinfo claim) is the dominant
signal; else `contradicts_guidance` if max P(contradict | authority stmt)
dominates; else `on_topic_neutral`. The dominance margins are constants
in scoring.py, sanity-checked on the calibration split. Stance is
explanatory metadata; the flag itself comes only from P(misinfo) ≥ τ.

**`pipeline.py`** — `score_text()` remains the single public entry point
with the same DataFrame shape. Column changes:

- Removed: `authority_sim`, `misinfo_sim`, `claim_delta`, `isolation_score`
- Added: `misinfo_entail`, `guidance_contradict`, `top_claim_sim`,
  `stance`, `scoreable`
- `misinfo_score` now *is* P(misinfo); `flagged` = P(misinfo) ≥ τ
- Kept: `chunk`, `chunk_mode`, `matched_claim`, `evidence_correction`,
  `misinfo_type`, `misinfo_type_confidence`

`matched_claim` becomes "the claim the chunk *entailed*" rather than "the
claim it is near," so the shown correction addresses what the text
asserts. `misinfo_type` comes from the matched claim's curated
domain/type instead of nearest-centroid guessing. `app.py` and templates
get the small matching update.

## Data and evaluation

### Repo layout

Training/eval assets live in a new top-level `ml/` directory so
`maven_app/` stays a lean inference app:

```
ml/
  data/       dataset builders (synthetic generation, public loaders)
  eval/       labeled sets (JSONL) + scorer-agnostic harness
  training/   verifier fine-tune script (Colab GPU)
```

### Evaluation harness (built FIRST)

A scorer-agnostic runner: takes any `score_text`-shaped callable and a
labeled JSONL set, emits precision/recall/F1, PR-AUC, confusion matrix,
per-stance recall (especially **debunk false-positive rate**), and
latency. The first run scores the *current* cosine system to lock in
baseline numbers (PRD E1/E2, M4 comparison).

### Labeled eval set

~200–300 items, hand-labeled by the team: real TikTok/IG transcripts and
captions plus curated hard cases. Each item:
`{text, label: misinfo|not_misinfo, stance, source, notes}`.
Split ~40% calibration (fits the logistic head and threshold τ) / 60%
test (reported numbers; never fitted on). A short labeling guide doc
doubles as PRD D3 annotation guidelines.

### Synthetic training data

Offline LLM-assisted generation, versioned JSON, never at inference. From
each of the 74 claim/evidence pairs:

- casual + clinical assertion paraphrases (positives)
- debunk rewrites (hard negatives)
- the correction itself + paraphrases (negatives)
- off-topic filler (negatives)

Used for: paraphrase-expanding the retrieval library, fine-tuning the
verifier, and training the M1 baseline (logistic regression on PubMedBERT
embeddings).

### Verifier fine-tuning (PRD M2)

DeBERTa-v3-base NLI fine-tuned on HealthVer + SciFact (public
health-domain entail/contradict pairs) + synthetic perinatal NLI pairs.
Runs on Colab GPU via `ml/training/`. The zero-shot checkpoint ships
first; the fine-tuned checkpoint replaces it only if it beats zero-shot
on the test split.

## Error handling

- No retrieval hit above the similarity floor → `scoreable=False`,
  `stance=off_topic`, P(misinfo)=0.0, never flagged.
- Verifier checkpoint missing/unloadable → clear startup error naming
  `MAVEN_VERIFIER_PATH`, same pattern as the current anchor errors.
- Pair-count overflow → reduce k proportionally; all chunks still scored.
- Empty/short chunk handling unchanged from current pipeline.

## Testing

- `tests/test_scoring.py` replaces `test_iso_calibration.py` with
  executable acceptance cases:
  - misinfo asserted casually → flagged
  - misinfo asserted in clinical register → flagged
  - debunk of a known claim → NOT flagged
  - accurate-but-casual statement → NOT flagged
  - off-topic text → `scoreable=False`
- Flask e2e (`test_flask_e2e.py`) updated for the new columns.
- The eval harness is the real gate: each rollout phase lands only if it
  beats the previous phase's numbers on the test split.

## Rollout phases

1. **Harness + baseline** — eval harness, labeled-set scaffolding +
   labeling guide, baseline metrics of the current cosine system.
2. **Reference library** — authority-statement extraction, claim
   paraphrase expansion, new library format + embedding cache.
3. **Retrieve + zero-shot verify** — retrieval.py, verifier.py, heuristic
   aggregation; measured against baseline.
4. **Calibration** — logistic head + threshold tuning on the calibration
   split (PRD E4).
5. **Fine-tune verifier** — Colab GPU training; adopt only if it beats
   zero-shot (PRD M2, M4).
6. **Integration** — pipeline.py/app.py/templates/notebook updates, docs,
   scaling-proposal notes.

Optional experiment once the harness exists (cheap): A/B PubMedBERT vs. a
general-purpose retriever (BGE/GTE) for the retrieval stage, since
social-media register is out-of-domain for PubMed-trained embeddings.

## Out of scope

- LLM calls at inference time (revisit post-semester if ever).
- Real-time monitoring, multilingual support (PRD out-of-scope).
- Ensemble of classifier + retrieve-and-verify (Approach C) — revisit
  only if Phase 5 metrics disappoint.
- Notebook restructure beyond documenting the new pipeline sections.
