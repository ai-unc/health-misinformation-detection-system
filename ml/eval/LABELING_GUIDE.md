# MAVEN Eval-Set Labeling Guide

This guide doubles as the PRD D3 annotation guidelines (review with
Dr. Bazzano before large-scale labeling).

## What goes in the set

Real TikTok/Instagram transcripts, OCR text, and captions about
perinatal / maternal / reproductive health, plus curated hard cases.
Target: 200–300 items total. Keep items short (a caption, a transcript,
one post) — the unit a MAVEN user would score.

## File format

`ml/eval/eval_set.jsonl` — one JSON object per line:

| Field | Values |
|---|---|
| `id` | unique, stable (e.g. `hl-041`). Never reuse or renumber — the calibration/test split hashes this id. |
| `text` | the raw text, unedited |
| `label` | `misinfo` \| `not_misinfo` |
| `stance` | `asserts_misinfo` \| `debunks_misinfo` \| `accurate` \| `neutral` \| `off_topic` |
| `source` | `synthetic-seed` \| `hand-labeled` |
| `notes` | free text (why it's hard, where it came from) |

## Label definitions

- **misinfo** — the text asserts, endorses, or instructs based on a
  health claim that contradicts current clinical guidance (ACOG, WHO,
  CDC, AAP...). The claim must be *made*, not merely mentioned.
- **not_misinfo** — everything else, including:
  - `debunks_misinfo`: quotes/mentions a false claim in order to refute
    it. These are the most important negatives — label generously.
  - `accurate`: consistent with guidance (any register — casual counts).
  - `neutral`: on-topic but no verifiable health claim (lived experience,
    questions, logistics).
  - `off_topic`: not about perinatal/maternal health.

## Edge rules

- Personal experience ("my labor took 30 hours") is `neutral` unless it
  generalizes into advice that contradicts guidance.
- Emerging/contested science: if major guidelines disagree with the
  claim today, label `misinfo`; if genuinely unsettled, `neutral` + note.
- Sarcasm/jokes: label by what a reasonable reader would take away.
- When two annotators disagree after discussion, keep the item with a
  note recording both views; exclude it from calibration by adding
  `"disputed": true`.

## Split discipline

The 40/60 calibration/test split is derived from `id` hashes
(`harness.split_of`). Never tune thresholds, prompts, or weights on the
test split. Report test-split numbers only.
