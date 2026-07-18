# M1 baseline: logistic regression on PubMedBERT embeddings

- **train:** synthetic (reference library templates)
- **note:** stance-blind by construction — compare debunk flag rate against the retrieve-and-verify report

| Metric | Value |
|---|---|
| N | 24 |
| Precision | 0.389 |
| Recall | 0.875 |
| F1 | 0.538 |
| Accuracy | 0.500 |
| PR-AUC | 0.514 |
| Mean latency (s) | 0.03 |

## Confusion matrix (flagged as misinfo)

| | pred misinfo | pred not |
|---|---|---|
| true misinfo | 7 | 1 |
| true not | 11 | 5 |

## Per-stance flag rates

| Gold stance | n | flagged | flag rate |
|---|---|---|---|
| accurate | 6 | 5 | 0.83 |
| asserts_misinfo | 8 | 7 | 0.88 |
| debunks_misinfo | 4 | 3 | 0.75 |
| neutral | 3 | 2 | 0.67 |
| off_topic | 3 | 1 | 0.33 |

## Per-item results

| id | gold stance | true | prob | flagged |
|---|---|---|---|---|
| seed-001 | asserts_misinfo | 1 | 0.855 | True |
| seed-002 | asserts_misinfo | 1 | 0.751 | True |
| seed-003 | asserts_misinfo | 1 | 0.349 | False |
| seed-004 | asserts_misinfo | 1 | 0.819 | True |
| seed-005 | asserts_misinfo | 1 | 0.631 | True |
| seed-006 | asserts_misinfo | 1 | 0.779 | True |
| seed-007 | asserts_misinfo | 1 | 0.712 | True |
| seed-008 | asserts_misinfo | 1 | 0.703 | True |
| seed-009 | accurate | 0 | 0.347 | False |
| seed-010 | accurate | 0 | 0.665 | True |
| seed-011 | accurate | 0 | 0.795 | True |
| seed-012 | accurate | 0 | 0.674 | True |
| seed-013 | accurate | 0 | 0.740 | True |
| seed-014 | accurate | 0 | 0.900 | True |
| seed-015 | debunks_misinfo | 0 | 0.697 | True |
| seed-016 | debunks_misinfo | 0 | 0.724 | True |
| seed-017 | debunks_misinfo | 0 | 0.477 | False |
| seed-018 | debunks_misinfo | 0 | 0.802 | True |
| seed-019 | neutral | 0 | 0.406 | False |
| seed-020 | neutral | 0 | 0.586 | True |
| seed-021 | neutral | 0 | 0.502 | True |
| seed-022 | off_topic | 0 | 0.346 | False |
| seed-023 | off_topic | 0 | 0.452 | False |
| seed-024 | off_topic | 0 | 0.573 | True |

## Reading this vs the retrieve-and-verify reports

This baseline's F1 (0.538) exceeds the adopted retrieve-and-verify system's F1
(0.364, see `2026-07-18-retrieve-verify-finetuned.md`), but only by scoring
stance-blind embedding proximity: it flags 3/4 debunks and 5/6 accurate items,
at precision 0.389. Per-stance flag rates, not cross-report F1, are the M4
evidence — they show this baseline flagging the exact content (debunks,
accurate posts) the retrieve-and-verify system exists to leave unflagged.
