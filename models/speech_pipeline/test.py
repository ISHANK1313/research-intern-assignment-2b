#!/usr/bin/env python
"""Evaluate the trained speech-only model on the canonical test split.

Reproduces the numbers saved in ``Results/metrics/metrics_speech.json`` from the
checkpoint in ``Results/checkpoints/speech.pt`` and the raw audio, and re-writes
the temporal-block representation used for the t-SNE plot.

    python -m models.speech_pipeline.test --data-dir data/tess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from models.common import config, data as data_mod, features, nets, train_utils  # noqa: E402

VARIANT = "speech"
REP_NAME = "temporal_block"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    p.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / f"{VARIANT}.pt"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_utils.set_seed(args.seed)
    train_utils.configure_threads()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}\n"
            "Train the model first:  python -m models.speech_pipeline.train --data-dir <dir>")
    ckpt = train_utils.load_checkpoint(ckpt_path)
    mean, std = ckpt["norm"]["mean"], ckpt["norm"]["std"]

    df = data_mod.resolve_paths(data_mod.load_split(args.seed), args.data_dir)
    test_df = data_mod.subset(df, "test")
    print(f"[{VARIANT}] test utterances: {len(test_df)}")

    mel, valid = features.build_speech_features(test_df, Path(args.cache_dir))
    arrays = {
        "speech": features.normalize(mel, mean, std),
        "speech_mask": features.speech_mask(valid),
    }
    loader = train_utils.make_loader(
        train_utils.ArrayDataset(arrays, test_df["label"].to_numpy()),
        batch_size=args.batch_size, shuffle=False)

    model = nets.SpeechClassifier(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])

    out = train_utils.evaluate(model, train_utils.speech_forward, loader,
                               torch.device("cpu"), collect_reps=True,
                               rep_keys=(VARIANT,))
    payload = train_utils.save_test_artifacts(
        VARIANT, test_df, out, out["reps"][VARIANT], REP_NAME,
        extra={"best_epoch": ckpt.get("best_epoch"),
               "n_params": sum(p.numel() for p in model.parameters()),
               "norm": {"mean": mean, "std": std}})

    print(train_utils.format_metrics(payload, f"[{VARIANT}] test metrics:"))
    print(f"[{VARIANT}] wrote {config.METRICS_DIR / f'metrics_{VARIANT}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
