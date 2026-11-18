"""TESS dataset discovery, label parsing and the canonical train/val/test split."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


@dataclass(frozen=True)
class Utterance:
    path: Path
    speaker: str
    word: str
    emotion: str
    emotion_idx: int
    transcript: str

    @property
    def file_name(self) -> str:
        return self.path.name


# TESS file names look like ``OAF_back_angry.wav`` (target word in the middle,
# emotion last, speaker first). ``ps`` is the abbreviation used for
# "pleasant surprise".
_SPEAKER_RE = re.compile(r"^(OA|OAF|YAF)(?=_|$)", re.IGNORECASE)


def parse_file_name(name: str) -> tuple[str, str, str]:
    """Parse ``speaker, word, emotion`` out of a TESS file name.

    Raises ValueError if the emotion token is not one of the seven TESS classes.
    """
    stem = Path(name).stem
    tokens = stem.split("_")
    if len(tokens) < 3:
        raise ValueError(f"cannot parse TESS file name: {name!r}")

    raw_speaker = tokens[0].upper()
    speaker = config.SPEAKER_ALIASES.get(raw_speaker, raw_speaker)

    # "pleasant_surprise" would be the only label containing an underscore, so try
    # the two-token form first and fall back to the single-token form.
    for n_emo in (2, 1):
        candidate = "_".join(tokens[-n_emo:]).lower()
        if candidate in config.FILENAME_EMOTION_MAP:
            word = "_".join(tokens[1:-n_emo])
            if not word:
                raise ValueError(f"empty target word in {name!r}")
            return speaker, word, config.FILENAME_EMOTION_MAP[candidate]
    raise ValueError(f"unknown emotion label in file name: {name!r}")


def build_index(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Walk `data_dir` and return a dataframe describing every TESS utterance.

    Duplicate-protection: the Kaggle archive ships the whole corpus *twice* -- once
    at the archive root and once inside a nested directory of the same name -- so a
    naive ``rglob`` finds 5 600 files instead of 2 800. Those copies are byte
    identical (checked by size for every group and by MD5 on a sample), so we keep
    exactly one file per name, choosing the shallowest path. Skipping this step
    would put the same recordings in both train and test and silently inflate every
    reported number.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data dir not found: {data_dir}\n"
            "Run:  python scripts/download_tess.py --data-dir <dir>"
        )

    # group by file name, keep the shallowest path
    best: dict[str, Path] = {}
    for wav in sorted(data_dir.rglob("*.wav")):
        prev = best.get(wav.name)
        if prev is None or len(wav.parts) < len(prev.parts):
            best[wav.name] = wav

    rows, skipped = [], 0
    for name, wav in sorted(best.items()):
        try:
            speaker, word, emotion = parse_file_name(name)
        except ValueError:
            skipped += 1
            continue
        rows.append({
            # Stored relative to the data directory so the committed split file is
            # portable across machines and accounts.
            "rel_path": wav.relative_to(data_dir).as_posix(),
            "file_name": name,
            "speaker": speaker,
            "word": word,
            "emotion": emotion,
            "label": config.EMOTION_TO_IDX[emotion],
            "transcript": config.TRANSCRIPT_TEMPLATE.format(word=word),
        })

    if not rows:
        raise RuntimeError(f"no parsable TESS .wav files under {data_dir}")
    if verbose:
        if skipped:
            print(f"[data] skipped {skipped} unparsable file name(s)")
        print(f"[data] {len(rows)} unique utterances "
              f"(de-duplicated from {len(list(data_dir.rglob('*.wav')))} found)")

    df = pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)
    return df


def make_split(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified train/val/test split on the emotion label.

    Stratifying on the label keeps the seven classes balanced across the three
    subsets, and the fixed `seed` makes the split byte-for-byte reproducible, so
    every pipeline is evaluated on exactly the same test utterances.
    """
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_FRAC + config.TEST_FRAC,
        stratify=df["label"], random_state=seed, shuffle=True,
    )
    rel_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["label"], random_state=seed, shuffle=True,
    )
    out = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ]).sort_values("file_name").reset_index(drop=True)
    return out


def split_path(seed: int = config.SEED) -> Path:
    return config.SPLIT_DIR / f"split_seed{seed}.csv"


def write_split(data_dir: Path, seed: int = config.SEED) -> pd.DataFrame:
    """Build the index, split it, and persist the canonical split CSV."""
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_split(build_index(data_dir), seed=seed)
    out = split_path(seed)
    df.to_csv(out, index=False)
    return df


def load_split(seed: int = config.SEED) -> pd.DataFrame:
    """Load the canonical split CSV, creating it only if it does not exist yet."""
    out = split_path(seed)
    if not out.is_file():
        raise FileNotFoundError(
            f"canonical split not found: {out}\n"
            "Run:  python -m models.common.data build --data-dir <dir>"
        )
    return pd.read_csv(out)


def subset(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def resolve_paths(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach an absolute ``path`` column by resolving ``rel_path`` under `data_dir`."""
    data_dir = Path(data_dir)
    out = df.copy()
    out["path"] = [str((data_dir / rel).resolve()) for rel in out["rel_path"]]
    missing = [p for p in out["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio file(s) from the split are missing under {data_dir} "
            f"(e.g. {missing[0]}).\n"
            "Fetch the dataset first:  python scripts/download_tess.py --data-dir <dir>"
        )
    return out


def describe_split(df: pd.DataFrame) -> str:
    lines = [f"total utterances: {len(df)}"]
    for split in ("train", "val", "test"):
        part = subset(df, split)
        lines.append(f"  {split:5s}: {len(part):5d}")
    lines.append("per-class counts (train/val/test):")
    for i, emotion in enumerate(config.EMOTIONS):
        counts = [len(subset(df, s)[lambda d: d["label"] == i]) for s in ("train", "val", "test")]
        lines.append(f"  {emotion:17s} {counts[0]:5d} {counts[1]:5d} {counts[2]:5d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build/refresh the canonical TESS split.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="index the dataset and write the split CSV")
    b.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    b.add_argument("--seed", type=int, default=config.SEED)
    s = sub.add_parser("show", help="print the existing split")
    s.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        df = write_split(Path(args.data_dir), seed=args.seed)
        print(f"wrote {split_path(args.seed)}")
    else:
        df = load_split(seed=args.seed)
    print(describe_split(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
