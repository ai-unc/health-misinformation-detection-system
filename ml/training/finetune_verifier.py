"""Fine-tune the NLI verifier on synthetic perinatal pairs (+ optional extras).

Designed for Colab GPU. Colab setup cell:
    !pip install "transformers>=4.40" datasets accelerate sentencepiece
Then:
    python ml/training/finetune_verifier.py \
        --train ml/data/nli_pairs.jsonl [path/to/healthver.jsonl ...] \
        --out ml/training/checkpoints/maven-verifier-v1 \
        [--base MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli] \
        [--epochs 2] [--lr 2e-5] [--batch 16]

Every --train file is JSONL with {"premise", "hypothesis", "label"} where
label is entailment|neutral|contradiction. 10% is held out for eval.
Adoption gate (do this manually after training):
  MAVEN_VERIFIER_PATH=<out dir> python ml/eval/run_eval.py --split test --out ...
  Adopt only if test-split F1 beats the zero-shot run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

DEFAULT_BASE = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, label2id, max_length=256):
        self.enc = tokenizer([r['premise'] for r in rows],
                             [r['hypothesis'] for r in rows],
                             truncation=True, max_length=max_length,
                             padding='max_length')
        self.labels = [label2id[r['label']] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', default=DEFAULT_BASE)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--cpu', action='store_true',
                        help='force CPU training (DeBERTa-v3 NaNs on Apple MPS)')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    # The base checkpoint ships fp16 weights; training in pure fp16 NaNs out
    # (same pin maven_app/verifier.py carries for inference).
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, dtype=torch.float32)
    label2id = {label.lower(): int(idx)
                for idx, label in model.config.id2label.items()}
    # tolerate 'entail'/'contradiction' naming variants in the config
    for want, alts in (('entailment', ('entail',)), ('contradiction', ('contradict',))):
        if want not in label2id:
            for alt in alts:
                if alt in label2id:
                    label2id[want] = label2id[alt]
    missing = {'entailment', 'neutral', 'contradiction'} - set(label2id)
    if missing:
        raise SystemExit(f'base checkpoint labels missing {missing}: {model.config.id2label}')

    rows = load_rows(args.train)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(rows))
    cut = max(1, int(0.1 * len(rows)))
    eval_rows = [rows[i] for i in order[:cut]]
    train_rows = [rows[i] for i in order[cut:]]
    print(f'train={len(train_rows)} eval={len(eval_rows)}')

    def accuracy(eval_pred):
        logits, labels = eval_pred
        return {'accuracy': float((logits.argmax(-1) == labels).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            learning_rate=args.lr, per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch, eval_strategy='epoch',
            save_strategy='epoch', save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model='accuracy',
            logging_steps=50, report_to=[], use_cpu=args.cpu,
        ),
        train_dataset=PairDataset(train_rows, tokenizer, label2id),
        eval_dataset=PairDataset(eval_rows, tokenizer, label2id),
        compute_metrics=accuracy,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f'Saved fine-tuned verifier to {args.out}')
    print('Adoption gate: MAVEN_VERIFIER_PATH=<out> python ml/eval/run_eval.py '
          '--split test --out ml/eval/reports/<date>-finetuned.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
