# Retrieve-and-verify (fine-tuned NLI maven-verifier-v1, heuristic aggregation)

- **git commit:** 7275164
- **eval set:** /Users/tdnguye6/Desktop/projects/hmd-ai@unc/health-misinformation-detection-system/ml/eval/eval_set.jsonl
- **split:** all
- **items:** 24
- **verifier checkpoint:** run with `MAVEN_VERIFIER_PATH=ml/training/checkpoints/maven-verifier-v1` (fine-tuned `maven-verifier-v1`). The checkpoint is local-only and gitignored (not committed); regenerate it with:
  `.venv/bin/python ml/training/finetune_verifier.py --train ml/data/nli_pairs.jsonl --out ml/training/checkpoints/maven-verifier-v1 --cpu`
  (deterministic pairs, seeded trainer — reproducible from a clean checkout).

| Metric | Value |
|---|---|
| N | 24 |
| Precision | 0.667 |
| Recall | 0.250 |
| F1 | 0.364 |
| Accuracy | 0.708 |
| PR-AUC | 0.512 |
| Mean latency (s) | 0.27 |

## Confusion matrix (flagged as misinfo)

| | pred misinfo | pred not |
|---|---|---|
| true misinfo | 2 | 6 |
| true not | 1 | 15 |

## Per-stance flag rates

| Gold stance | n | flagged | flag rate |
|---|---|---|---|
| accurate | 6 | 1 | 0.17 |
| asserts_misinfo | 8 | 2 | 0.25 |
| debunks_misinfo | 4 | 0 | 0.00 |
| neutral | 3 | 0 | 0.00 |
| off_topic | 3 | 0 | 0.00 |

## Per-item results

| id | gold stance | true | prob | flagged |
|---|---|---|---|---|
| seed-001 | asserts_misinfo | 1 | 0.286 | False |
| seed-002 | asserts_misinfo | 1 | 0.429 | False |
| seed-003 | asserts_misinfo | 1 | 0.568 | True |
| seed-004 | asserts_misinfo | 1 | 0.000 | False |
| seed-005 | asserts_misinfo | 1 | 0.000 | False |
| seed-006 | asserts_misinfo | 1 | 0.000 | False |
| seed-007 | asserts_misinfo | 1 | 0.505 | True |
| seed-008 | asserts_misinfo | 1 | 0.000 | False |
| seed-009 | accurate | 0 | 0.509 | True |
| seed-010 | accurate | 0 | 0.277 | False |
| seed-011 | accurate | 0 | 0.000 | False |
| seed-012 | accurate | 0 | 0.000 | False |
| seed-013 | accurate | 0 | 0.000 | False |
| seed-014 | accurate | 0 | 0.000 | False |
| seed-015 | debunks_misinfo | 0 | 0.361 | False |
| seed-016 | debunks_misinfo | 0 | 0.000 | False |
| seed-017 | debunks_misinfo | 0 | 0.293 | False |
| seed-018 | debunks_misinfo | 0 | 0.499 | False |
| seed-019 | neutral | 0 | 0.030 | False |
| seed-020 | neutral | 0 | 0.000 | False |
| seed-021 | neutral | 0 | 0.034 | False |
| seed-022 | off_topic | 0 | 0.000 | False |
| seed-023 | off_topic | 0 | 0.000 | False |
| seed-024 | off_topic | 0 | 0.005 | False |

## Comparison vs baseline

| # | Gate | Baseline (legacy) | Measured (fine-tuned) | Result |
|---|---|---|---|---|
| 1 | `debunks_misinfo` flag rate lower than baseline (target 0) | 100% (4/4) | 0.00 (0/4) | PASS |
| 2 | `off_topic` flag rate exactly 0 | 67% (2/3) | 0.00 (0/3) | PASS |
| 3 | F1 ≥ baseline | 0.333 | 0.364 | PASS |
| 4 | Mean latency ≤ 15 s/item | 0.06 s | 0.27 s | PASS |

All four gates pass. F1 (0.364) and PR-AUC (0.512) also both beat the zero-shot run (F1 0.000, PR-AUC 0.425 — see `2026-07-16-retrieve-verify-zeroshot.md`).

## Known residual gaps

- 4 of the 8 `asserts_misinfo` seed myths have no covering claim in the misinfo reference library yet; growing library coverage is post-plan work.
- seed-002: a street-register phrasing of a covered myth still scores sub-τ (0.429) — entailment degrades under informal phrasing.
- seed-009: an `accurate` item false-positives at 0.509, just over τ.
- seed-018: a `debunks_misinfo` item scores 0.499, just under τ — close to flipping the other way.
