#!/usr/bin/env python
"""Control experiments that validate the experimental setup and bound the text task.

Four independent checks, none of which uses the deep models:

1. **Alignment control.** Identify the *speaker* from summarised log-mel features with
   plain logistic regression. This is trivially learnable from TESS audio, so a score
   near 1.0 proves the split, the feature extraction and the label vector are correctly
   aligned. If the plumbing were broken this check would fail loudly.

2. **Independent non-deep speech baseline.** Classify emotion from the same summarised
   log-mel features with logistic regression. Gives a floor that the BiLSTM must beat
   and shows how much emotion signal is present in the acoustics at all.

3. **Information-theoretic ceiling for the text modality.** Computes the mutual
   information I(emotion ; transcript) in bits from the dataset itself. This is the
   cleanest possible statement of how much emotion information the text carries,
   independent of any model.

4. **Bayes-optimal text-only accuracy.** Predicts, for each test utterance, the class
   with highest conditional probability given its transcript, estimated from the
   training split. This is the best any text-only classifier can possibly do on this
   data, so it is the correct yardstick for the neural text pipeline.

    python scripts/control_baselines.py --data-dir data/tess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.common import config, data as dm, features as F, train_utils as TU  # noqa: E402


def summarise_mel(mels: np.ndarray) -> np.ndarray:
    """Cheap fixed-size summary of each log-mel patch: per-band mean and std."""
    return np.concatenate([mels.mean(axis=2), mels.std(axis=2)], axis=1)


def conditional_entropy_and_mi(transcripts: np.ndarray, labels: np.ndarray) -> dict:
    """H(emotion | transcript) and I(emotion ; transcript), in bits."""
    counts = pd.crosstab(pd.Series(transcripts, name="t"),
                         pd.Series(labels, name="c")).to_numpy(dtype=float)
    n = counts.sum()
    p_tc = counts / n
    p_t = p_tc.sum(axis=1)
    p_c = p_tc.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # H(C | T) = -sum_t p(t) sum_c p(c|t) log2 p(c|t)
        cond = np.divide(p_tc, p_t[:, None], out=np.zeros_like(p_tc), where=p_t[:, None] > 0)
        h_c_given_t = -float(np.nansum(p_t[:, None] * cond * np.log2(np.where(cond > 0, cond, 1))))

        # I(C;T) = sum p(c,t) log2( p(c,t) / (p(c) p(t)) )
        denom = p_c[None, :] * p_t[:, None]
        ratio = np.divide(p_tc, denom, out=np.ones_like(p_tc), where=denom > 0)
        mi = float(np.nansum(p_tc * np.log2(np.where(ratio > 0, ratio, 1))))

    return {
        "n_utterances": int(n),
        "n_unique_transcripts": int(counts.shape[0]),
        "h_emotion_bits": round(float(-np.sum(p_c * np.log2(np.where(p_c > 0, p_c, 1)))), 6),
        "h_emotion_given_transcript_bits": round(h_c_given_t, 6),
        "mutual_information_bits": round(mi, 6),
        "emotions_per_transcript_min": int((counts > 0).sum(axis=1).min()),
        "emotions_per_transcript_max": int((counts > 0).sum(axis=1).max()),
        "utterances_per_transcript_min": int(counts.sum(axis=1).min()),
        "utterances_per_transcript_max": int(counts.sum(axis=1).max()),
    }


def bayes_optimal_text_accuracy(train_t, train_y, test_t, test_y, n_classes: int) -> dict:
    """Best achievable text-only accuracy: argmax_c P(c | transcript) from train."""
    counts = pd.crosstab(pd.Series(train_t), pd.Series(train_y))
    counts = counts.reindex(columns=range(n_classes), fill_value=0)
    probs = (counts.to_numpy(dtype=float) + 1.0)          # Laplace smoothing
    probs = probs / probs.sum(axis=1, keepdims=True)
    best = {t: int(np.argmax(p)) for t, p in zip(counts.index, probs)}

    preds, covered = [], 0
    for t in test_t:
        if t in best:
            preds.append(best[t]); covered += 1
        else:
            preds.append(0)                               # unseen transcript -> majority
    preds = np.array(preds)
    return {
        "bayes_optimal_test_accuracy": round(float((preds == test_y).mean()), 6),
        "test_transcripts_seen_in_train": int(covered),
        "n_test": int(len(test_y)),
        "chance_accuracy": round(1.0 / n_classes, 6),
    }


def split_artifact_check(split_df) -> dict:
    """Quantify the word/emotion split artefact described in report Section C.5.

    TESS is a balanced design: every (target word, emotion) cell holds exactly two
    files (one per speaker). Stratifying the split on emotion therefore splits those
    two files across subsets, so wherever the training split holds both of a word's
    files for an emotion, the test split holds none of them. A text model that
    memorises the training word -> majority-emotion mapping then predicts, for every
    word, an emotion with zero test examples.
    """
    tr = dm.subset(split_df, "train")
    te = dm.subset(split_df, "test")
    cell = split_df.groupby(["word", "emotion"]).size()
    trc = tr.groupby(["word", "emotion"]).size().unstack(fill_value=0)
    tec = te.groupby(["word", "emotion"]).size().unstack(fill_value=0)

    majority = trc.idxmax(axis=1)
    absent = sum(1 for w, e in majority.items()
                 if w not in tec.index or int(tec.loc[w, e]) == 0)

    train_vals = pd.Series(trc.to_numpy().ravel()).value_counts().to_dict()
    test_vals = pd.Series(tec.to_numpy().ravel()).value_counts().to_dict()
    return {
        "files_per_word_emotion_cell": (sorted(set(cell.unique().tolist()))
                                        if len(set(cell.unique())) > 1
                                        else int(cell.iloc[0])),
        "train_count_per_cell_distribution": {int(k): int(v) for k, v in train_vals.items()},
        "test_count_per_cell_distribution": {int(k): int(v) for k, v in test_vals.items()},
        "words_total": int(len(majority)),
        "words_whose_train_majority_emotion_is_absent_from_test": int(absent),
        "note": "A text model that memorises the training word->emotion mapping is "
                "therefore guaranteed to be wrong on test; this is why the "
                "Bayes-optimal text predictor scores 0.0 rather than chance.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train_utils_seed = args.seed
    TU.set_seed(train_utils_seed)

    df = dm.resolve_paths(dm.load_split(args.seed), args.data_dir)
    tr, te = dm.subset(df, "train"), dm.subset(df, "test")

    print("[control] extracting log-mel for train/test ...")
    m_tr, _ = F.build_speech_features(tr, config.CACHE_DIR, progress=False)
    m_te, _ = F.build_speech_features(te, config.CACHE_DIR, progress=False)
    X_tr, X_te = summarise_mel(m_tr), summarise_mel(m_te)

    print("[control] 1) speaker-identification alignment control ...")
    y_tr_spk = (tr["speaker"] == "OAF").astype(int).to_numpy()
    y_te_spk = (te["speaker"] == "OAF").astype(int).to_numpy()
    clf = LogisticRegression(max_iter=3000).fit(X_tr, y_tr_spk)
    speaker_acc = float(accuracy_score(y_te_spk, clf.predict(X_te)))

    print("[control] 2) logistic-regression speech baseline ...")
    y_tr, y_te = tr["label"].to_numpy(), te["label"].to_numpy()
    clf2 = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)
    p = clf2.predict(X_te)
    lr_speech = {
        "test_accuracy": round(float(accuracy_score(y_te, p)), 6),
        "test_macro_f1": round(float(f1_score(y_te, p, average="macro", zero_division=0)), 6),
        "train_accuracy": round(float(accuracy_score(y_tr, clf2.predict(X_tr))), 6),
    }

    print("[control] 3) information-theoretic text ceiling ...")
    info = conditional_entropy_and_mi(df["transcript"].to_numpy(), df["label"].to_numpy())

    print("[control] 4) Bayes-optimal text-only accuracy ...")
    bayes = bayes_optimal_text_accuracy(tr["transcript"].to_numpy(), y_tr,
                                        te["transcript"].to_numpy(), y_te,
                                        config.NUM_CLASSES)

    payload = {
        "seed": args.seed,
        "alignment_control": {
            "task": "speaker identification from summarised log-mel",
            "test_accuracy": speaker_acc,
            "note": "must be ~1.0; a lower value would indicate a mis-aligned split "
                    "or feature pipeline.",
        },
        "speech_baseline_logreg": {
            "features": "per-band mean+std of the log-mel patch",
            **lr_speech,
        },
        "text_information": info,
        "bayes_optimal_text": bayes,
        "split_artifact": split_artifact_check(df),
    }
    TU.save_json(payload, config.METRICS_DIR / "control_baselines.json")

    print(f"\n  alignment control (speaker ID)      : {speaker_acc:.4f}  (expect ~1.0)")
    print(f"  logreg speech baseline (test acc)   : {lr_speech['test_accuracy']:.4f}")
    print(f"  I(emotion ; transcript)             : {info['mutual_information_bits']:.6f} bits")
    print(f"  H(emotion | transcript)             : "
          f"{info['h_emotion_given_transcript_bits']:.6f} bits "
          f"(uniform = {info['h_emotion_bits']:.6f})")
    print(f"  Bayes-optimal text-only test acc    : "
          f"{bayes['bayes_optimal_test_accuracy']:.4f}")
    print(f"\nwrote {config.METRICS_DIR / 'control_baselines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
