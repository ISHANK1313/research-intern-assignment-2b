#!/usr/bin/env python
"""Evaluate the trained multimodal (fusion) model on the canonical test split.

    python -m models.fusion_pipeline.test --data-dir data/tess

Besides the fused representation, this also stores the two branch representations
produced *inside* the fusion model (``reps_fusion_speech_branch_test.npy`` and
``reps_fusion_text_branch_test.npy``) so the report can compare the jointly trained
branches against the single-modality models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.common import (config, data as data_mod, features, nets,  # noqa: E402
                           text_features, train_utils)

VARIANT = "fusion"
REP_NAME = "fusion_block"


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
            "Train the model first:  python -m models.fusion_pipeline.train --data-dir <dir>")
    ckpt = train_utils.load_checkpoint(ckpt_path)
    norm = (ckpt["norm"]["mean"], ckpt["norm"]["std"])

    df = data_mod.resolve_paths(data_mod.load_split(args.seed), args.data_dir)
    test_df = data_mod.subset(df, "test")
    print(f"[{VARIANT}] test utterances: {len(test_df)}")

    mel, valid = features.build_speech_features(test_df, Path(args.cache_dir))
    emb, mask = text_features.load_or_compute(
        list(test_df["transcript"]), Path(args.cache_dir))
    arrays = {
        "speech": features.normalize(mel, *norm),
        "speech_mask": features.speech_mask(valid),
        "text": emb.astype(np.float32),
        "mask": mask.astype(bool),
    }
    loader = train_utils.make_loader(
        train_utils.ArrayDataset(arrays, test_df["label"].to_numpy()),
        batch_size=args.batch_size, shuffle=False)

    model = nets.FusionClassifier(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])

    out = train_utils.evaluate(model, train_utils.fusion_forward, loader,
                               torch.device("cpu"), collect_reps=True,
                               rep_keys=("fused", "speech", "text"))
    payload = train_utils.save_test_artifacts(
        VARIANT, test_df, out, out["reps"]["fused"], REP_NAME,
        extra={"best_epoch": ckpt.get("best_epoch"),
               "n_params": sum(p.numel() for p in model.parameters()),
               "norm": {"mean": norm[0], "std": norm[1]}})

    # Extra: the branches as trained jointly inside the fusion model.
    np.save(config.REPS_DIR / f"reps_{VARIANT}_speech_branch_test.npy",
            out["reps"]["speech"].astype(np.float32))
    np.save(config.REPS_DIR / f"reps_{VARIANT}_text_branch_test.npy",
            out["reps"]["text"].astype(np.float32))

    print(train_utils.format_metrics(payload, f"[{VARIANT}] test metrics:"))
    print(f"[{VARIANT}] wrote {config.METRICS_DIR / f'metrics_{VARIANT}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
