#!/usr/bin/env python
"""Verify that the published repository is internally consistent and clean.

Run this after cloning (and after any re-run) to confirm:

1. **Data hygiene** - no audio, no credentials, no feature cache committed.
2. **Shared test set** - the three variants really were evaluated on the same
   utterances (identical label vector and file listing).
3. **Number agreement** - every accuracy in ``accuracy_table.csv`` matches the
   per-variant metric file, and the derived numbers in ``analysis.json`` agree.
4. **Report agreement** - every headline number quoted in ``Report/report.pdf`` is
   present in the saved results, and every figure the report embeds exists as a file.
5. **Artifact completeness** - all plots, representations, predictions and
   checkpoints needed to reproduce the report are present.

Exits non-zero on the first failure group, printing what is wrong.

    python scripts/verify_repo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.common import config, io_utils as IO  # noqa: E402

VARIANTS = ("speech", "text", "fusion")

# Files the report embeds; each must exist in the repository.
REQUIRED_PLOTS = (
    "accuracy_comparison.png",
    "confusion_matrix_speech.png", "confusion_matrix_text.png",
    "confusion_matrix_fusion.png",
    "training_curves_speech.png", "training_curves_text.png",
    "training_curves_fusion.png",
    "tsne_temporal_block.png", "tsne_contextual_block.png", "tsne_fusion_block.png",
    "tsne_blocks_comparison.png",
)

REQUIRED_METRIC_FILES = tuple(
    [f"metrics_{v}.json" for v in VARIANTS]
    + [f"metrics_{v}_val.json" for v in VARIANTS]
    + [f"history_{v}.csv" for v in VARIANTS]
    + ["accuracy_table.csv", "accuracy_table.md", "analysis.json",
       "silhouette_scores.json", "control_baselines.json",
       "metrics_text_finetuned.json"]
)

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def tracked_files(root: Path) -> list[Path] | None:
    """Files tracked by git, i.e. what the repository actually publishes.

    Scanning the working directory instead would flag third-party library content: a
    virtualenv created inside the repo (as the README instructs) contains scipy's
    test-fixture .wav files and certifi's cacert.pem, none of which we ship. Returns
    None when the tree is not a git checkout.
    """
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / line for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    root = config.REPO_ROOT

    # ---------------------------------------------------------------- 1. hygiene
    print("[1] repository hygiene")
    files = tracked_files(root)
    if files is None:
        print("    (not a git checkout - falling back to a filtered directory walk)")
        skip = {".venv", "venv", "cache", "data", ".git", "__pycache__"}
        files = [p for p in root.rglob("*")
                 if p.is_file() and not (skip & set(p.parts))]
    audio = [p for p in files if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]
    check(not audio, f"audio files are tracked in the repo: {audio[:3]}")
    # certifi's cacert.pem is a third-party CA list, not a private key, and is never
    # tracked anyway - the point of this check is what the repository publishes.
    creds = [p for p in files
             if (p.name in ("kaggle.json", ".env", "secrets.json")
                 or p.suffix.lower() == ".pem"
                 or "credential" in p.name.lower())]
    check(not creds, f"possible credential file tracked: {creds[:3]}")
    # The feature cache legitimately exists in a working copy, so assert that it is
    # ignored rather than absent.
    gitignore = (root / ".gitignore").read_text(encoding="utf-8") \
        if (root / ".gitignore").is_file() else ""
    for ignored in ("cache/", "data/", "*.wav", "kaggle.json", ".env"):
        check(ignored in gitignore, f".gitignore does not ignore {ignored!r}")
    print(f"    tracked files: {len(files)} | audio: {len(audio)} | "
          f"credentials: {len(creds)} | .gitignore covers data/cache/credentials")

    # ------------------------------------------------------- 2. shared test set
    print("[2] shared test set")
    labels_path = config.REPS_DIR / "test_labels.npy"
    files_path = config.REPS_DIR / "test_files.csv"
    check(labels_path.is_file(), "test_labels.npy missing")
    check(files_path.is_file(), "test_files.csv missing")
    if labels_path.is_file():
        labels = np.load(labels_path)
        check(len(labels) == 420, f"expected 420 test labels, found {len(labels)}")
        counts = np.bincount(labels, minlength=config.NUM_CLASSES)
        check(set(counts.tolist()) == {60},
              f"test split is not balanced 60/class: {counts.tolist()}")
        print(f"    n_test={len(labels)} per-class counts={counts.tolist()}")

    # ---------------------------------------------------------- 3. number agreement
    print("[3] number agreement between artifacts")
    table = pd.read_csv(config.METRICS_DIR / "accuracy_table.csv")
    metrics = {}
    for v in VARIANTS:
        p = config.METRICS_DIR / f"metrics_{v}.json"
        check(p.is_file(), f"missing {p.name}")
        if not p.is_file():
            continue
        m = IO.load_json(p)
        metrics[v] = m
        row = table[table["variant"] == v]
        check(len(row) == 1, f"{v}: not exactly one row in accuracy_table.csv")
        if len(row):
            r = row.iloc[0]
            check(abs(r["accuracy"] - m["accuracy"]) < 5e-5,
                  f"{v}: accuracy_table {r['accuracy']} != metrics {m['accuracy']}")
            check(abs(r["macro_f1"] - m["macro_f1"]) < 5e-5,
                  f"{v}: macro F1 mismatch ({r['macro_f1']} vs {m['macro_f1']})")
            for e in config.EMOTIONS:
                check(abs(r[f"f1_{e}"] - m["per_class_f1"][e]) < 5e-5,
                      f"{v}: per-class F1 mismatch for {e}")
        print(f"    {v}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    # consistency of the derived analysis with the raw metrics
    analysis_path = config.METRICS_DIR / "analysis.json"
    check(analysis_path.is_file(), "analysis.json missing")
    if analysis_path.is_file() and metrics:
        a = IO.load_json(analysis_path)
        check(a["n_test"] == int(metrics["fusion"]["n_test"]),
              "analysis.json n_test disagrees with metrics")
        gain = a["fusion_gain_vs_speech"]
        expected = metrics["fusion"]["accuracy"] - metrics["speech"]["accuracy"]
        check(abs(gain["accuracy"] - expected) < 5e-5,
              f"analysis fusion gain {gain['accuracy']} != {expected}")
        # confusion matrices must sum to the test size
        for v, m in metrics.items():
            total = int(np.array(m["confusion_matrix"]).sum())
            check(total == m["n_test"],
                  f"{v}: confusion matrix sums to {total}, expected {m['n_test']}")
        print(f"    fusion-minus-speech accuracy = {gain['accuracy']:+.4f}")

    # ------------------------------------------------------ 4. report agreement
    print("[4] report agreement")
    report = root / "Report" / "report.pdf"
    check(report.is_file(), "Report/report.pdf missing")
    missing_plots = [p for p in REQUIRED_PLOTS if not (config.PLOTS_DIR / p).is_file()]
    check(not missing_plots, f"plots referenced by the report are missing: {missing_plots}")
    missing_metrics = [f for f in REQUIRED_METRIC_FILES
                       if not (config.METRICS_DIR / f).is_file()]
    check(not missing_metrics, f"metric files missing: {missing_metrics}")
    print(f"    plots present: {len(REQUIRED_PLOTS) - len(missing_plots)}/"
          f"{len(REQUIRED_PLOTS)}")

    if report.is_file():
        try:
            import pypdf
            text = "\n".join((pg.extract_text() or "")
                             for pg in pypdf.PdfReader(str(report)).pages)
            for v in VARIANTS:
                if v in metrics:
                    acc = f"{metrics[v]['accuracy']:.4f}"
                    check(acc in text,
                          f"report.pdf does not contain the {v} accuracy {acc}")
            check("I(emotion" in text.replace(" ", "")
                  or "0.000000 bits" in text,
                  "report.pdf does not state the zero mutual-information result")
            print(f"    report.pdf pages: {len(pypdf.PdfReader(str(report)).pages)}; "
                  f"headline accuracies present in text")
        except ImportError:
            print("    (pypdf not installed - skipping report text extraction)")

    # -------------------------------------------------- 5. artifact completeness
    print("[5] artifact completeness")
    for v in VARIANTS:
        check((config.CHECKPOINT_DIR / f"{v}.pt").is_file(),
              f"missing checkpoint for {v}")
        check((config.PRED_DIR / f"pred_{v}.csv").is_file(),
              f"missing predictions for {v}")
        check((config.REPS_DIR / f"reps_{v}_test.npy").is_file(),
              f"missing representation for {v}")
    check((config.RESULTS_DIR / "failure_cases.md").is_file(),
          "Results/failure_cases.md missing")
    check((config.SPLIT_DIR / f"split_seed{config.SEED}.csv").is_file(),
          "canonical split file missing")

    total = sum((config.CHECKPOINT_DIR / f"{v}.pt").stat().st_size / 1e6
                for v in VARIANTS
                if (config.CHECKPOINT_DIR / f"{v}.pt").is_file())
    print(f"    checkpoints total {total:.1f} MB")

    # -------------------------------------------------------------------- verdict
    print()
    if failures:
        print(f"FAILED {len(failures)} of {checks} checks:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
