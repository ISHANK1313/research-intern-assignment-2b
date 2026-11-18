#!/usr/bin/env python
"""Summarise the three variants and derive the analysis used by the report.

Every number the report quotes is computed here from the artifacts written by the
pipelines (``Results/metrics/*.json``, ``Results/predictions/*.csv``,
``Results/representations/*.npy``) and written to
``Results/metrics/analysis.json`` plus human-readable tables.

    python scripts/summarize_results.py

Outputs
-------
Results/metrics/accuracy_table.csv / .md   headline 3-variant comparison
Results/metrics/analysis.json              all derived analysis (single source of truth)
Results/failure_cases.md                   fusion failure cases with the evidence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.common import config, features, io_utils as IO  # noqa: E402

VARIANTS = ("speech", "text", "fusion")
SHORT = {"speech": "Speech only", "text": "Text only", "fusion": "Multimodal (fusion)"}
N_ANALYSIS_CASES = 5


def load_metrics(variant: str) -> dict:
    return IO.load_json(config.METRICS_DIR / f"metrics_{variant}.json")


def build_accuracy_table() -> tuple[pd.DataFrame, str]:
    rows = []
    for v in VARIANTS:
        m = load_metrics(v)
        row = {"variant": v, "model": SHORT[v], "accuracy": round(m["accuracy"], 4),
               "macro_f1": round(m["macro_f1"], 4),
               "weighted_f1": round(m["weighted_f1"], 4),
               "rep_dim": m.get("rep_dim"), "n_test": m.get("n_test")}
        for e in config.EMOTIONS:
            row[f"f1_{e}"] = round(m["per_class_f1"][e], 4)
        rows.append(row)
    df = pd.DataFrame(rows)

    header = "| Variant | Accuracy | Macro F1 | Weighted F1 | " + \
             " | ".join(config.EMOTIONS) + " |"
    sep = "|" + "---|" * (4 + len(config.EMOTIONS))
    lines = [header, sep]
    for _, r in df.iterrows():
        f1s = " | ".join(f"{r[f'f1_{e}']:.3f}" for e in config.EMOTIONS)
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['weighted_f1']:.4f} | {f1s} |")
    md = "\n".join(lines)
    return df, md


def easiest_hardest(df: pd.DataFrame) -> dict:
    out = {}
    for _, r in df.iterrows():
        f1 = {e: r[f"f1_{e}"] for e in config.EMOTIONS}
        ranked = sorted(f1.items(), key=lambda kv: kv[1], reverse=True)
        out[r["variant"]] = {
            "ranking": [{"emotion": e, "f1": round(v, 4)} for e, v in ranked],
            "easiest": ranked[0][0], "easiest_f1": round(ranked[0][1], 4),
            "hardest": ranked[-1][0], "hardest_f1": round(ranked[-1][1], 4),
        }
    return out


def top_confusions(variant: str, top_k: int = 6) -> list[dict]:
    m = load_metrics(variant)
    cm = np.array(m["confusion_matrix"], dtype=float)
    total = cm.sum()
    pairs = []
    for i, true_e in enumerate(config.EMOTIONS):
        for j, pred_e in enumerate(config.EMOTIONS):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append({"true": true_e, "pred": pred_e, "count": int(cm[i, j]),
                          "share_of_all": round(float(cm[i, j] / total), 4),
                          "share_of_true_class": round(float(cm[i, j] / cm[i].sum()), 4)})
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top_k]


def _test_valid_samples() -> dict[str, int]:
    """Trimmed sample count per test file, read from the cached features."""
    listing = pd.read_csv(config.REPS_DIR / "test_files.csv")
    out = {}
    for name in listing["file_name"]:
        p = features.cache_path(config.CACHE_DIR, name)
        if p.is_file():
            with np.load(p) as d:
                out[name] = int(d["valid_samples"])
    return out


def duration_analysis(preds: dict[str, pd.DataFrame], valid: dict[str, int]) -> dict:
    base = preds["fusion"].copy()
    base["dur_s"] = base["file_name"].map(lambda n: valid.get(n, np.nan) / config.SAMPLE_RATE)
    base = base.dropna(subset=["dur_s"])
    if base.empty:
        return {}
    edges = [0, 1.5, 2.0, 2.5, 5.0]
    labels = ["<1.5s", "1.5-2.0s", "2.0-2.5s", ">=2.5s"]
    base["bucket"] = pd.cut(base["dur_s"], bins=edges, labels=labels, right=False)

    out = {"overall_mean_duration_s": round(float(base["dur_s"].mean()), 3),
           "overall_median_duration_s": round(float(base["dur_s"].median()), 3),
           "buckets": {}}
    for label in labels:
        key = label
        idx = base.index[base["bucket"] == label]
        if len(idx) == 0:
            continue
        entry = {"n": int(len(idx)),
                 "mean_duration_s": round(float(base.loc[idx, "dur_s"].mean()), 3)}
        for v, df in preds.items():
            merged = df.set_index("file_name").reindex(base.loc[idx, "file_name"])
            entry[v] = round(float(merged["correct"].mean()), 4)
        out["buckets"][key] = entry
    return out


def speaker_analysis(preds: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for v, df in preds.items():
        per = df.groupby("speaker")["correct"].agg(["mean", "size"])
        out[v] = {spk: {"accuracy": round(float(r["mean"]), 4), "n": int(r["size"])}
                  for spk, r in per.iterrows()}
    return out


def fusion_gain(df: pd.DataFrame) -> dict:
    def get(v, key):
        return float(df.loc[df["variant"] == v, key].iloc[0])
    gain = {"accuracy": round(get("fusion", "accuracy") - get("speech", "accuracy"), 4),
            "macro_f1": round(get("fusion", "macro_f1") - get("speech", "macro_f1"), 4),
            "vs_text_accuracy": round(get("fusion", "accuracy") - get("text", "accuracy"), 4),
            "per_emotion_f1_delta_vs_speech":
                {e: round(get("fusion", f"f1_{e}") - get("speech", f"f1_{e}"), 4)
                 for e in config.EMOTIONS}}
    worst = min(gain["per_emotion_f1_delta_vs_speech"].items(), key=lambda kv: kv[1])
    best = max(gain["per_emotion_f1_delta_vs_speech"].items(), key=lambda kv: kv[1])
    gain["largest_gain_emotion"] = best[0]
    gain["largest_gain_value"] = best[1]
    gain["largest_loss_emotion"] = worst[0]
    gain["largest_loss_value"] = worst[1]
    return gain


def _case_entry(row: pd.Series, preds: dict[str, pd.DataFrame],
                provenance: str) -> dict:
    entry = {
        "provenance": provenance,
        "file_name": row["file_name"],
        "speaker": row["speaker"],
        "transcript": row["transcript"],
        "true_emotion": row["true_emotion"],
        "pred_emotion": row["pred_emotion"],
        "fusion_confidence": round(float(row["confidence"]), 4),
        "fusion_prob_true": round(float(row["true_prob"]), 4),
        "runner_up": _runner_up(row, config.EMOTIONS),
    }
    for v in ("speech", "text"):
        other = preds[v][preds[v]["file_name"] == row["file_name"]]
        if not other.empty:
            o = other.iloc[0]
            entry[f"{v}_pred"] = o["pred_emotion"]
            entry[f"{v}_correct"] = bool(o["correct"])
            entry[f"{v}_confidence"] = round(float(o["confidence"]), 4)
    return entry


def pick_failure_cases(preds: dict[str, pd.DataFrame]) -> list[dict]:
    """Build up to N_ANALYSIS_CASES cases, labelled by provenance.

    The multimodal model turns out to make a single test error, so a list of 3-5
    *failures* cannot be filled honestly. Rather than pad it with fabricated cases,
    the list is completed with the model's lowest-margin *correct* predictions
    (near-misses) and the speech-only model's failures. Every row carries its
    provenance so the report never implies the model erred more often than it did.
    """
    fusion = preds["fusion"]
    cases: list[dict] = []
    used: set[str] = set()

    # 1) genuine fusion failures, most confident first, spread over true emotions
    fails = fusion[~fusion["correct"]].sort_values("confidence", ascending=False)
    seen_emotions: set[str] = set()
    for _, row in fails.iterrows():
        if row["true_emotion"] in seen_emotions:
            continue
        cases.append(_case_entry(row, preds, "fusion error (test split)"))
        used.add(row["file_name"]); seen_emotions.add(row["true_emotion"])
        if len(cases) >= N_ANALYSIS_CASES:
            return cases
    for _, row in fails.iterrows():
        if len(cases) >= N_ANALYSIS_CASES:
            return cases
        if row["file_name"] not in used:
            cases.append(_case_entry(row, preds, "fusion error (test split)"))
            used.add(row["file_name"])

    # 2) lowest-margin correct predictions: the hard-but-lucky cases
    correct = fusion[fusion["correct"]].copy()
    correct["margin"] = correct["true_prob"] - correct.apply(
        lambda r: max(float(r[f"p_{e}"]) for e in config.EMOTIONS
                      if e != r["pred_emotion"]), axis=1)
    for _, row in correct.sort_values("margin").iterrows():
        if len(cases) >= N_ANALYSIS_CASES:
            return cases
        if row["file_name"] not in used:
            cases.append(_case_entry(row, preds, "fusion near-miss (correct, lowest margin)"))
            used.add(row["file_name"])

    # 3) speech-only failures, for comparison
    sp = preds["speech"]
    for _, row in sp[~sp["correct"]].sort_values("confidence", ascending=False).iterrows():
        if len(cases) >= N_ANALYSIS_CASES:
            return cases
        if row["file_name"] not in used:
            cases.append(_case_entry(row, preds, "speech-only error (for comparison)"))
            used.add(row["file_name"])
    return cases


def _runner_up(row: pd.Series, emotions: tuple[str, ...]) -> dict:
    probs = {e: float(row[f"p_{e}"]) for e in emotions}
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    second = ordered[1] if len(ordered) > 1 else ordered[0]
    return {"emotion": second[0], "prob": round(second[1], 4)}


def failure_cases_markdown(cases: list[dict]) -> str:
    if not cases:
        return "No fusion misclassifications on the test split."
    lines = [
        "| # | Case type | File | True | Predicted (fusion) | Conf. | P(true) | Speech-only | Text-only |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(cases, 1):
        sp = f"{c.get('speech_pred', '-')}"
        if c.get("speech_correct") is not None:
            sp += " (ok)" if c["speech_correct"] else " (wrong)"
        lines.append(
            f"| {i} | {c['provenance']} | `{c['file_name']}` | {c['true_emotion']} | "
            f"{c['pred_emotion']} | {c['fusion_confidence']:.3f} | "
            f"{c['fusion_prob_true']:.3f} | {sp} | {c.get('text_pred', '-')} |")
    return "\n".join(lines)


def mcnemar_pairwise(preds: dict[str, pd.DataFrame]) -> dict:
    """Exact McNemar tests between variants, paired on the same test utterances.

    Necessary because the raw accuracy difference between the best two variants is a
    single utterance out of 420: without a paired test it would be easy to present
    that as a real improvement.
    """
    from scipy.stats import binomtest

    out = {}
    for a, b in (("speech", "fusion"), ("speech", "text"), ("text", "fusion")):
        pa = preds[a].set_index("file_name")["correct"]
        pb = preds[b].set_index("file_name")["correct"]
        common = pa.index.intersection(pb.index)
        pa, pb = pa.loc[common], pb.loc[common]
        n10 = int(((~pa) & pb).sum())      # a wrong, b right
        n01 = int((pa & (~pb)).sum())      # a right, b wrong
        n = n10 + n01
        p = float(binomtest(min(n10, n01), n, 0.5).pvalue) if n > 0 else 1.0
        out[f"{a}_vs_{b}"] = {
            "n_paired": int(len(common)),
            f"{a}_wrong_{b}_right": n10,
            f"{a}_right_{b}_wrong": n01,
            "accuracy_delta": round(float(pb.mean() - pa.mean()), 6),
            "mcnemar_exact_p": round(p, 4),
            "significant_at_0.05": bool(p < 0.05),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    missing = [v for v in VARIANTS
               if not (config.METRICS_DIR / f"metrics_{v}.json").is_file()]
    if missing:
        print(f"error: no test metrics for {missing}. Run every test.py first.",
              file=sys.stderr)
        return 1

    preds = {v: pd.read_csv(config.PRED_DIR / f"pred_{v}.csv") for v in VARIANTS}
    df, md = build_accuracy_table()

    df.to_csv(config.METRICS_DIR / "accuracy_table.csv", index=False)
    (config.METRICS_DIR / "accuracy_table.md").write_text(md + "\n", encoding="utf-8")

    analysis = {
        "seed": args.seed,
        "n_test": int(load_metrics("fusion")["n_test"]),
        "accuracy_table": json.loads(df.to_json(orient="records")),
        "easiest_hardest": easiest_hardest(df),
        "top_confusions": {v: top_confusions(v) for v in VARIANTS},
        "speaker_breakdown": speaker_analysis(preds),
        "duration_analysis": duration_analysis(preds, _test_valid_samples()),
        "fusion_gain_vs_speech": fusion_gain(df),
        "significance_tests": mcnemar_pairwise(preds),
        "failure_cases": pick_failure_cases(preds),
    }
    sil_path = config.METRICS_DIR / "silhouette_scores.json"
    if sil_path.is_file():
        analysis["silhouette_scores"] = IO.load_json(sil_path)

    IO.save_json(analysis, config.METRICS_DIR / "analysis.json")

    (config.RESULTS_DIR / "failure_cases.md").write_text(
        "# Fusion model failure cases\n\n"
        f"Most confident misclassifications of the multimodal model on the shared "
        f"test split (n={analysis['n_test']}).\n\n"
        + failure_cases_markdown(analysis["failure_cases"]) + "\n",
        encoding="utf-8")

    print(md)
    print(f"\nwrote {config.METRICS_DIR / 'accuracy_table.csv'}")
    print(f"wrote {config.METRICS_DIR / 'analysis.json'}")
    print(f"wrote {config.RESULTS_DIR / 'failure_cases.md'}")
    print(f"\nfusion minus speech-only accuracy: "
          f"{analysis['fusion_gain_vs_speech']['accuracy']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
