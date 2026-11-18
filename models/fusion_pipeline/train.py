#!/usr/bin/env python
"""Multimodal emotion recognition: speech + text late fusion.

Pipeline (assignment Figure, blocks 1-5):

    preprocess      as in the speech pipeline; transcript as in the text pipeline
    features        log-mel (speech) + frozen DistilBERT embeddings (text)
    temporal model  2-layer BiLSTM over mel frames      -> 256-d speech representation
    contextual model 2-layer BiLSTM over word embeddings -> 256-d text representation
    fusion          concatenate the two pooled representations -> 512-d fused vector
    classifier      MLP over the fused vector -> 7 logits, cross-entropy

Both encoders are trained jointly with the classifier, so the speech branch can
specialise for the multimodal setting. The DistilBERT backbone stays frozen (see
``models/common/text_features.py`` for why).

    python -m models.fusion_pipeline.train --data-dir data/tess
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.common import (config, data as data_mod, features, nets,  # noqa: E402
                           text_features, train_utils)

VARIANT = "fusion"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--hidden", type=int, default=config.SPEECH_HIDDEN)
    p.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def build_arrays(df, cache_dir: Path | None, norm: tuple[float, float] | None):
    """Speech + text arrays for one split. Returns (arrays, norm_stats)."""
    mel, valid = features.build_speech_features(df, cache_dir)
    if norm is None:
        norm = features.fit_norm_stats(mel)
    emb, mask = text_features.load_or_compute(list(df["transcript"]), cache_dir)
    arrays = {
        "speech": features.normalize(mel, *norm),
        "speech_mask": features.speech_mask(valid),
        "text": emb.astype(np.float32),
        "mask": mask.astype(bool),
    }
    return arrays, norm


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

    t0 = time.time()
    print(f"[{VARIANT}] building speech + text features...")
    tr_arrays, norm = build_arrays(train_df, cache_dir, None)
    va_arrays, _ = build_arrays(val_df, cache_dir, norm)
    text_dim = tr_arrays["text"].shape[-1]
    print(f"[{VARIANT}] norm stats (train split): mean {norm[0]:.3f} std {norm[1]:.3f}"
          f"   [{time.time() - t0:.1f}s]")

    loaders = {
        name: train_utils.make_loader(
            train_utils.ArrayDataset(arr, sub["label"].to_numpy()),
            batch_size=args.batch_size, shuffle=(name == "train"), seed=args.seed)
        for name, arr, sub in (("train", tr_arrays, train_df),
                               ("val", va_arrays, val_df))
    }

    model = nets.FusionClassifier(
        n_mels=config.N_MELS, text_in_dim=text_dim, hidden=args.hidden,
        num_classes=config.NUM_CLASSES, dropout=config.DROPOUT, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{VARIANT}] FusionClassifier | {n_params:,} trainable parameters"
          f" | fused rep {model.rep_dim}-d")

    trainer = train_utils.Trainer(
        model, train_utils.fusion_forward, tag=VARIANT, lr=args.lr,
        max_epochs=args.epochs, patience=args.patience)
    history = trainer.fit(loaders["train"], loaders["val"])

    history.to_csv(config.METRICS_DIR / f"history_{VARIANT}.csv", index=False,
                   lineterminator="\n")
    arch = {"n_mels": config.N_MELS, "text_in_dim": int(text_dim),
            "hidden": args.hidden, "num_classes": config.NUM_CLASSES,
            "dropout": config.DROPOUT, "num_layers": 2}
    train_utils.save_checkpoint(
        model, config.CHECKPOINT_DIR / f"{VARIANT}.pt",
        extra={"arch": arch, "norm": {"mean": norm[0], "std": norm[1]},
               "variant": VARIANT, "seed": args.seed,
               "best_epoch": trainer.best_epoch,
               "best_val_macro_f1": trainer.best_val_f1})

    val_out = train_utils.evaluate(model, train_utils.fusion_forward,
                                  loaders["val"], torch.device("cpu"))
    train_utils.save_json(
        {"variant": VARIANT, "split": "val", "seed": args.seed,
         "best_epoch": trainer.best_epoch, "epochs_run": int(history["epoch"].max()),
         "n_params": n_params, "rep_dim": int(model.rep_dim),
         "norm": {"mean": norm[0], "std": norm[1]},
         **val_out["metrics"]},
        config.METRICS_DIR / f"metrics_{VARIANT}_val.json")

    print(f"[{VARIANT}] " + train_utils.format_metrics(val_out["metrics"],
                                                      "validation metrics:"))
    print(f"[{VARIANT}] wrote checkpoint -> {config.CHECKPOINT_DIR / f'{VARIANT}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
