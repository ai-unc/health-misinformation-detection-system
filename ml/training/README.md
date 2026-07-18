# Verifier Training & Calibration

## Fine-tuning the NLI verifier (PRD M2)

1. Generate synthetic pairs (deterministic, committed):
   `python ml/data/build_nli_pairs.py`
2. Optional public data (recommended): convert HealthVer
   (https://github.com/sarrouti/HealthVer) and/or SciFact
   (https://github.com/allenai/scifact) claim-evidence pairs to the same
   JSONL schema ({"premise", "hypothesis", "label"}) and pass them as
   extra --train files. Map SUPPORTS→entailment, REFUTES→contradiction,
   NOINFO/NEI→neutral.
3. Train (Colab GPU, or locally with `--cpu`; see finetune_verifier.py docstring for the pip cell):
   `python ml/training/finetune_verifier.py --train ml/data/nli_pairs.jsonl --out ml/training/checkpoints/maven-verifier-v1 --cpu`
4. Adoption gate — measure before adopting:
   `MAVEN_VERIFIER_PATH=ml/training/checkpoints/maven-verifier-v1 python ml/eval/run_eval.py --split test --out ml/eval/reports/<date>-finetuned.md`
   Adopt (export MAVEN_VERIFIER_PATH in the app environment) only if
   test-split F1 beats the zero-shot report. Keep checkpoints out of git.

## Calibration head (PRD E4)

Once the hand-labeled eval set has grown (see ml/eval/LABELING_GUIDE.md):
`python ml/training/fit_calibration.py` — fits on the calibration split,
picks the F1-optimal tau, writes maven_app/models/calibration_head.joblib
(commit it; a few KB). Report test-split metrics via run_eval.
