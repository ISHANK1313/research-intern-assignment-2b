#!/usr/bin/env python
"""Text-only emotion recognition: DistilBERT embeddings -> BiLSTM -> MLP.

Pipeline (assignment Figure, blocks 1-3 and 5):

    preprocess      transcript built from the file name ("Say the word <x>.")
    features        frozen DistilBERT base-uncased token embeddings (768-d per token)
    contextual model 2-layer BiLSTM (hidden 128) over the token sequence, masked mean pool
    classifier      MLP -> 7 logits, cross-entropy

    python -m models.text_pipeline.train --data-dir data/tess

Expect this variant to land close to chance, and see Section C of the report for
the structural reason why.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.common import (config, data as data_mod, nets, text_features,  # noqa: E402
                           train_utils)

VARIANT = "text"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--hidden", type=int, default=config.TEXT_HIDDEN)
    p.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def build_masked_arrays(emb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {"text": emb.astype(np.float32), "mask": mask.astype(bool)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for d in config.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
    train_utils.set_seed(args.seed)
    print(f"[{VARIANT}] torch threads: {train_utils.configure_threads()}", flush=True)
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    df = data_mod.resolve_paths(data_mod.load_split(args.seed), args.data_dir)
    train_df = data_mod.subset(df, "train")
    val_df = data_mod.subset(df, "val")
    print(f"[{VARIANT}] split: train {len(train_df)} / val {len(val_df)}")
    print(f"[{VARIANT}] unique transcripts in train split: "
          f"{train_df['transcript'].nunique()}")

    t0 = time.time()
    print(f"[{VARIANT}] computing DistilBERT embeddings ({config.TEXT_BACKBONE})...")
    tr_emb, tr_mask = text_features.load_or_compute(
        list(train_df["transcript"]), cache_dir)
    va_emb, va_mask = text_features.load_or_compute(
        list(val_df["transcript"]), cache_dir)
    embed_dim = tr_emb.shape[-1]
    print(f"[{VARIANT}] embeddings {tr_emb.shape} | token length "
          f"{tr_emb.shape[1]}   [{time.time() - t0:.1f}s]")

    arrays = {"train": build_masked_arrays(tr_emb, tr_mask),
              "val": build_masked_arrays(va_emb, va_mask)}
    loaders = {
        name: train_utils.make_loader(
            train_utils.ArrayDataset(arrays[name], sub["label"].to_numpy()),
            batch_size=args.batch_size, shuffle=(name == "train"), seed=args.seed)
        for name, sub in (("train", train_df), ("val", val_df))
    }

    model = nets.TextClassifier(
        in_dim=embed_dim, hidden=args.hidden, num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{VARIANT}] TextClassifier | {n_params:,} trainable parameters"
          f" (DistilBERT frozen)")

    trainer = train_utils.Trainer(
        model, train_utils.text_forward, tag=VARIANT, lr=args.lr,
        max_epochs=args.epochs, patience=args.patience)
    history = trainer.fit(loaders["train"], loaders["val"])

    tag = VARIANT
    history.to_csv(config.METRICS_DIR / f"history_{tag}.csv", index=False)
    arch = {"in_dim": int(embed_dim), "hidden": args.hidden,
            "num_classes": config.NUM_CLASSES, "dropout": config.DROPOUT,
            "num_layers": 2}
    train_utils.save_checkpoint(
        model, config.CHECKPOINT_DIR / f"{tag}.pt",
        extra={"arch": arch, "variant": VARIANT, "seed": args.seed,
               "backbone": config.TEXT_BACKBONE,
               "frozen_backbone": True,
               "best_epoch": trainer.best_epoch,
               "best_val_macro_f1": trainer.best_val_f1})

    val_out = train_utils.evaluate(model, train_utils.text_forward,
                                  loaders["val"], torch.device("cpu"))
    train_utils.save_json(
        {"variant": VARIANT, "split": "val", "seed": args.seed,
         "best_epoch": trainer.best_epoch, "epochs_run": int(history["epoch"].max()),
         "n_params": n_params, "backbone": config.TEXT_BACKBONE,
         "frozen_backbone": True,
         "unique_train_transcripts": int(train_df["transcript"].nunique()),
         **val_out["metrics"]},
        config.METRICS_DIR / f"metrics_{tag}_val.json")

    print(f"[{VARIANT}] " + train_utils.format_metrics(val_out["metrics"],
                                                      "validation metrics:"))
    print(f"[{VARIANT}] wrote checkpoint -> {config.CHECKPOINT_DIR / f'{tag}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
