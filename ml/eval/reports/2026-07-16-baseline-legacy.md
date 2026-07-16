# Baseline: legacy centroid+IsoForest scorer (pre-rework)

- **git commit:** 04082be
- **eval set:** /Users/tdnguye6/Desktop/projects/hmd-ai@unc/health-misinformation-detection-system/ml/eval/eval_set.jsonl
- **split:** all
- **items:** 24

| Metric | Value |
|---|---|
| N | 24 |
| Precision | 0.250 |
| Recall | 0.500 |
| F1 | 0.333 |
| Accuracy | 0.333 |
| PR-AUC | 0.301 |
| Mean latency (s) | 0.06 |

## Confusion matrix (flagged as misinfo)

| | pred misinfo | pred not |
|---|---|---|
| true misinfo | 4 | 4 |
| true not | 12 | 4 |

## Per-stance flag rates

| Gold stance | n | flagged | flag rate |
|---|---|---|---|
| accurate | 6 | 3 | 0.50 |
| asserts_misinfo | 8 | 4 | 0.50 |
| debunks_misinfo | 4 | 4 | 1.00 |
| neutral | 3 | 3 | 1.00 |
| off_topic | 3 | 2 | 0.67 |

## Per-item results

| id | gold stance | true | prob | flagged |
|---|---|---|---|---|
| seed-001 | asserts_misinfo | 1 | 0.589 | False |
| seed-002 | asserts_misinfo | 1 | 0.654 | True |
| seed-003 | asserts_misinfo | 1 | 0.512 | False |
| seed-004 | asserts_misinfo | 1 | 0.649 | True |
| seed-005 | asserts_misinfo | 1 | 0.558 | False |
| seed-006 | asserts_misinfo | 1 | 0.692 | True |
| seed-007 | asserts_misinfo | 1 | 0.550 | False |
| seed-008 | asserts_misinfo | 1 | 0.610 | True |
| seed-009 | accurate | 0 | 0.510 | False |
| seed-010 | accurate | 0 | 0.616 | True |
| seed-011 | accurate | 0 | 0.664 | True |
| seed-012 | accurate | 0 | 0.548 | False |
| seed-013 | accurate | 0 | 0.560 | False |
| seed-014 | accurate | 0 | 0.660 | True |
| seed-015 | debunks_misinfo | 0 | 0.602 | True |
| seed-016 | debunks_misinfo | 0 | 0.721 | True |
| seed-017 | debunks_misinfo | 0 | 0.633 | True |
| seed-018 | debunks_misinfo | 0 | 0.632 | True |
| seed-019 | neutral | 0 | 0.640 | True |
| seed-020 | neutral | 0 | 0.717 | True |
| seed-021 | neutral | 0 | 0.619 | True |
| seed-022 | off_topic | 0 | 0.528 | False |
| seed-023 | off_topic | 0 | 0.661 | True |
| seed-024 | off_topic | 0 | 0.722 | True |
