#!/usr/bin/env python
"""Ablation: fine-tune DistilBERT end-to-end instead of freezing it.

The main text pipeline freezes DistilBERT and trains only a BiLSTM head (see
``models/common/text_features.py`` for why). This script tests that decision by
fine-tuning the whole 66 M-parameter backbone on the identical split, so the report
can state the result instead of asserting it.

    python scripts/ablation_finetune_text.py --data-dir data/tess

Writes ``Results/metrics/metrics_text_finetuned.json`` (validation + test) and
``Results/metrics/history_text_finetuned.csv``. The fine-tuned checkpoint is
deliberately *not* committed: it is ~265 MB, and the main text result does not depend
on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models.common import config, data as dm, train_utils as TU  # noqa: E402

VARIANT = "text_finetuned"


class FineTunedTextClassifier(nn.Module):
    """DistilBERT (trainable) + masked mean pooling + linear head."""

    def __init__(self, backbone: str = config.TEXT_BACKBONE,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = config.DROPOUT):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(backbone)
        hidden = int(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def tokenize(texts: list[str], max_tokens: int) -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TEXT_BACKBONE)
    enc = tok(list(texts), padding="max_length", truncation=True,
              max_length=max_tokens, return_tensors="np")
    return {"input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    for d in config.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
    TU.set_seed(args.seed)
    print(f"[{VARIANT}] torch threads: {TU.configure_threads()}", flush=True)

    df = dm.resolve_paths(dm.load_split(args.seed), args.data_dir)
    parts = {s: dm.subset(df, s) for s in ("train", "val", "test")}
    print(f"[{VARIANT}] split: train {len(parts['train'])} / val {len(parts['val'])}"
          f" / test {len(parts['test'])}")

    t0 = time.time()
    feats = {s: tokenize(list(p["transcript"]), config.TEXT_MAX_TOKENS)
             for s, p in parts.items()}
    print(f"[{VARIANT}] tokenised  [{time.time() - t0:.1f}s]")

    loaders = {
        s: TU.make_loader(
            TU.ArrayDataset(feats[s], parts[s]["label"].to_numpy()),
            batch_size=args.batch_size, shuffle=(s == "train"), seed=args.seed)
        for s in ("train", "val")
    }

    model = FineTunedTextClassifier()
    n_params = sum(p.numel() for p in model.parameters())

    def forward_fn(m, batch, return_rep: bool = False):
        return m(batch["input_ids"], batch["attention_mask"])

    print(f"[{VARIANT}] fine-tuning {n_params / 1e6:.1f} M parameters "
          f"(lr {args.lr}, patience {args.patience})", flush=True)
    trainer = TU.Trainer(model, forward_fn, tag=VARIANT, lr=args.lr,
                         max_epochs=args.epochs, patience=args.patience)
    history = trainer.fit(loaders["train"], loaders["val"])
    history.to_csv(config.METRICS_DIR / f"history_{VARIANT}.csv", index=False)

    test_loader = TU.make_loader(
        TU.ArrayDataset(feats["test"], parts["test"]["label"].to_numpy()),
        batch_size=args.batch_size, shuffle=False)
    val_out = TU.evaluate(model, forward_fn, loaders["val"], torch.device("cpu"))
    test_out = TU.evaluate(model, forward_fn, test_loader, torch.device("cpu"))

    TU.save_json(
        {"variant": VARIANT, "seed": args.seed, "ablation_of": "text",
         "backbone": config.TEXT_BACKBONE, "frozen_backbone": False,
         "n_params": n_params, "best_epoch": trainer.best_epoch,
         "epochs_run": int(history["epoch"].max()),
         "validation_accuracy": val_out["metrics"]["accuracy"],
         "validation_macro_f1": val_out["metrics"]["macro_f1"],
         "peak_train_accuracy": float(history["train_acc"].max()),
         **test_out["metrics"]},
        config.METRICS_DIR / f"metrics_{VARIANT}.json")

    print(f"[{VARIANT}] test accuracy {test_out['metrics']['accuracy']:.4f} "
          f"macro F1 {test_out['metrics']['macro_f1']:.4f} "
          f"| peak train acc {float(history['train_acc'].max()):.4f}")
    print(f"[{VARIANT}] wrote {config.METRICS_DIR / f'metrics_{VARIANT}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
