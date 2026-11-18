#!/usr/bin/env python
"""Speech-only emotion recognition: log-mel spectrogram -> BiLSTM -> MLP.

Pipeline (assignment Figure, blocks 1-3 and 5):

    preprocess      resample 16 kHz, trim silence, pad/truncate to 3.0 s
    features        log-mel 128 bands, 1024-pt FFT, 256-sample hop -> (128, 188)
    temporal model  2-layer BiLSTM (hidden 128) over the 188 frames, masked mean pool
    classifier      MLP -> 7 logits, cross-entropy

Run from the repository root:

    python -m models.speech_pipeline.train --data-dir data/tess

Outputs land in ``Results/``: training history, validation metrics and the
checkpoint consumed by ``test.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.common import config, data as data_mod, features, nets, train_utils  # noqa: E402

VARIANT = "speech"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    p.add_argument("--no-cache", action="store_true",
                   help="ignore/regen the on-disk feature cache")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for d in config.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
    train_utils.set_seed(args.seed)
    print(f"[{VARIANT}] torch threads: {train_utils.configure_threads()}", flush=True)
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    # --- data ---------------------------------------------------------------
    df = data_mod.resolve_paths(data_mod.load_split(args.seed), args.data_dir)
    train_df = data_mod.subset(df, "train")
    val_df = data_mod.subset(df, "val")
    print(f"[{VARIANT}] split: train {len(train_df)} / val {len(val_df)}")

    t0 = time.time()
    print(f"[{VARIANT}] extracting log-mel features...")
    tr_mel, tr_valid = features.build_speech_features(train_df, cache_dir)
    va_mel, va_valid = features.build_speech_features(val_df, cache_dir)

    # Normalisation statistics come from the training split only.
    mean, std = features.fit_norm_stats(tr_mel)
    print(f"[{VARIANT}] log-mel norm stats (train split): mean {mean:.3f} std {std:.3f}"
          f"   [{time.time() - t0:.1f}s]")

    arrays = {}
    for name, mel, valid in (("train", tr_mel, tr_valid),
                             ("val", va_mel, va_valid)):
        arrays[name] = {
            "speech": features.normalize(mel, mean, std),
            "speech_mask": features.speech_mask(valid),
        }

    loaders = {
        name: train_utils.make_loader(
            train_utils.ArrayDataset(arrays[name], sub["label"].to_numpy()),
            batch_size=args.batch_size, shuffle=(name == "train"), seed=args.seed)
        for name, sub in (("train", train_df), ("val", val_df))
    }

    # --- model --------------------------------------------------------------
    model = nets.SpeechClassifier(
        n_mels=config.N_MELS, hidden=config.SPEECH_HIDDEN,
        num_classes=config.NUM_CLASSES, dropout=config.DROPOUT, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{VARIANT}] SpeechClassifier | {n_params:,} trainable parameters")

    trainer = train_utils.Trainer(
        model, train_utils.speech_forward, tag=VARIANT, lr=args.lr,
        max_epochs=args.epochs, patience=args.patience)
    history = trainer.fit(loaders["train"], loaders["val"])

    # --- persist ------------------------------------------------------------
    history.to_csv(config.METRICS_DIR / f"history_{VARIANT}.csv", index=False)
    train_utils.save_checkpoint(
        model, config.CHECKPOINT_DIR / f"{VARIANT}.pt",
        extra={"norm": {"mean": mean, "std": std}, "variant": VARIANT,
               "seed": args.seed, "best_epoch": trainer.best_epoch,
               "best_val_macro_f1": trainer.best_val_f1,
               "arch": {"n_mels": config.N_MELS, "hidden": config.SPEECH_HIDDEN,
                        "num_layers": 2, "num_classes": config.NUM_CLASSES,
                        "dropout": config.DROPOUT}})

    val_out = train_utils.evaluate(model, train_utils.speech_forward,
                                   loaders["val"], torch.device("cpu"))
    train_utils.save_json(
        {"variant": VARIANT, "split": "val", "seed": args.seed,
         "best_epoch": trainer.best_epoch,
         "epochs_run": int(history["epoch"].max()),
         "n_params": n_params,
         "norm": {"mean": mean, "std": std},
         **val_out["metrics"]},
        config.METRICS_DIR / f"metrics_{VARIANT}_val.json")

    print(f"[{VARIANT}] " + train_utils.format_metrics(val_out["metrics"],
                                                      "validation metrics:"))
    print(f"[{VARIANT}] wrote checkpoint -> {config.CHECKPOINT_DIR / f'{VARIANT}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
