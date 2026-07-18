# Retrieve-and-verify (zero-shot NLI, heuristic aggregation)

- **git commit:** 7275164
- **eval set:** /Users/tdnguye6/Desktop/projects/hmd-ai@unc/health-misinformation-detection-system/ml/eval/eval_set.jsonl
- **split:** all
- **items:** 24

| Metric | Value |
|---|---|
| N | 24 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| Accuracy | 0.667 |
| PR-AUC | 0.425 |
| Mean latency (s) | 0.31 |

## Confusion matrix (flagged as misinfo)

| | pred misinfo | pred not |
|---|---|---|
| true misinfo | 0 | 8 |
| true not | 0 | 16 |

## Per-stance flag rates

| Gold stance | n | flagged | flag rate |
|---|---|---|---|
| accurate | 6 | 0 | 0.00 |
| asserts_misinfo | 8 | 0 | 0.00 |
| debunks_misinfo | 4 | 0 | 0.00 |
| neutral | 3 | 0 | 0.00 |
| off_topic | 3 | 0 | 0.00 |

## Per-item results

| id | gold stance | true | prob | flagged |
|---|---|---|---|---|
| seed-001 | asserts_misinfo | 1 | 0.000 | False |
| seed-002 | asserts_misinfo | 1 | 0.000 | False |
| seed-003 | asserts_misinfo | 1 | 0.416 | False |
| seed-004 | asserts_misinfo | 1 | 0.000 | False |
| seed-005 | asserts_misinfo | 1 | 0.000 | False |
| seed-006 | asserts_misinfo | 1 | 0.000 | False |
| seed-007 | asserts_misinfo | 1 | 0.047 | False |
| seed-008 | asserts_misinfo | 1 | 0.000 | False |
| seed-009 | accurate | 0 | 0.112 | False |
| seed-010 | accurate | 0 | 0.000 | False |
| seed-011 | accurate | 0 | 0.000 | False |
| seed-012 | accurate | 0 | 0.000 | False |
| seed-013 | accurate | 0 | 0.000 | False |
| seed-014 | accurate | 0 | 0.000 | False |
| seed-015 | debunks_misinfo | 0 | 0.000 | False |
| seed-016 | debunks_misinfo | 0 | 0.000 | False |
| seed-017 | debunks_misinfo | 0 | 0.000 | False |
| seed-018 | debunks_misinfo | 0 | 0.129 | False |
| seed-019 | neutral | 0 | 0.162 | False |
| seed-020 | neutral | 0 | 0.000 | False |
| seed-021 | neutral | 0 | 0.000 | False |
| seed-022 | off_topic | 0 | 0.000 | False |
| seed-023 | off_topic | 0 | 0.000 | False |
| seed-024 | off_topic | 0 | 0.003 | False |

## Comparison vs baseline

| # | Gate | Baseline (legacy) | Measured (zero-shot) | Result |
|---|---|---|---|---|
| 1 | `debunks_misinfo` flag rate lower than baseline (target 0) | 100% (4/4) | 0.00 (0/4) | PASS |
| 2 | `off_topic` flag rate exactly 0 | 67% (2/3) | 0.00 (0/3) | PASS |
| 3 | F1 ≥ baseline | 0.333 | 0.000 | FAIL |
| 4 | Mean latency ≤ 15 s/item | 0.06 s | 0.31 s | PASS |

Zero-shot NLI is insufficient at this operating point; see 2026-07-18-retrieve-verify-finetuned.md for the fine-tuned run that passes all gates (M4 evidence).
