"""Speech preprocessing and log-mel feature extraction.

Decisions (justified in the report, Section A):

* **Resample to 16 kHz mono.** The corpus is not rate-uniform: measured with
  ``soundfile``, TESS ships at two original rates (96 000 Hz and 24 414 Hz). Keeping
  the raw rate would make the mel filterbank mean different things per speaker, so
  everything is resampled to a single 16 kHz.
* **Trim leading/trailing silence.** Clips carry inconsistent silence padding that
  would otherwise be modelled as signal.
* **Fix the length to 3.0 s.** The longest file in the corpus is 2.985 s, so nothing
  is truncated, and a BiLSTM batch needs a rectangular tensor.
* **Absolute-scale log-mel + a *training-split* global mean/std.** Peak-normalising
  each utterance to 0 dB would destroy the loudness cue that carries much of the
  emotional signal (anger is simply louder). One global statistic keeps those
  between-utterance level differences while putting inputs on a good scale. Fitting
  the statistic on the training split only keeps the test set untouched.
* **Carry the trimmed length through.** Because utterances are zero-padded to 3.0 s,
  the padded frames must not be averaged into the utterance representation, so each
  example also yields a frame-level validity mask.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from . import config


def load_waveform(path: str | Path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load a mono waveform at `sr`, resampling if necessary."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32)


def trim_silence(y: np.ndarray, top_db: float = config.TRIM_TOP_DB) -> np.ndarray:
    """Remove leading/trailing silence. Falls back to the input if over-trimmed."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    # `trim` can return a very short remnant for unusual clips; keep at least 0.2 s
    # so the frame count stays positive.
    if trimmed.size < int(0.2 * config.SAMPLE_RATE):
        return y
    return np.asarray(trimmed, dtype=np.float32)


def fix_length(y: np.ndarray, n: int = config.FIXED_SAMPLES) -> np.ndarray:
    """Pad with zeros or truncate so the waveform is exactly `n` samples."""
    if y.size >= n:
        return y[:n]
    return np.pad(y, (0, n - y.size), mode="constant")


def log_mel(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Absolute-scale log-mel spectrogram of a fixed-length waveform.

    Returns (N_MELS, N_FRAMES) in dB relative to 1.0, so the absolute energy level
    of the utterance survives for the later global scaling.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH, n_mels=config.N_MELS,
        fmin=config.FMIN, fmax=config.FMAX, power=2.0, center=True,
    )
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)


def preprocess(path: str | Path) -> tuple[np.ndarray, int]:
    """Load -> trim -> fix length -> log-mel.

    Returns ``(mel (N_MELS, N_FRAMES), valid_samples)`` where `valid_samples` is the
    number of samples that were actual audio before zero padding.
    """
    y = load_waveform(path)
    y = trim_silence(y)
    valid = int(y.size)
    y = fix_length(y)
    return log_mel(y), valid


def frames_for_samples(n_samples: int, n_frames: int = config.N_FRAMES) -> int:
    """Number of mel frames covering `n_samples` with center-padded STFT framing."""
    return int(min(n_frames, max(1, 1 + n_samples // config.HOP_LENGTH)))


def speech_mask(valid_samples: np.ndarray,
                n_frames: int = config.N_FRAMES) -> np.ndarray:
    """Boolean (N, n_frames) mask: True where a frame contains real audio."""
    n_valid = np.array([frames_for_samples(int(s), n_frames) for s in valid_samples])
    idx = np.arange(n_frames)[None, :]
    return idx < n_valid[:, None]


def cache_path(cache_dir: Path, file_name: str) -> Path:
    return Path(cache_dir) / "mel" / f"{Path(file_name).stem}.npz"


def load_or_compute_mel(path: str | Path, file_name: str,
                        cache_dir: Path | None = None) -> tuple[np.ndarray, int]:
    """Return ``(mel, valid_samples)`` for one file, using an on-disk cache.

    The cache holds derived data only (~270 MB) and is git-ignored; it exists purely
    to keep repeated training runs fast.
    """
    if cache_dir is None:
        return preprocess(path)

    cache_file = cache_path(cache_dir, file_name)
    if cache_file.is_file():
        with np.load(cache_file) as data:
            arr, valid = data["mel"], int(data["valid_samples"])
        if arr.shape == (config.N_MELS, config.N_FRAMES):
            return arr, valid

    arr, valid = preprocess(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, mel=arr, valid_samples=np.int64(valid))
    return arr, valid


def fit_norm_stats(mels: np.ndarray) -> tuple[float, float]:
    """Mean/std of log-mel values, computed over the *training* split only."""
    return float(mels.mean()), float(mels.std() + 1e-6)


def normalize(mel: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((mel - mean) / std).astype(np.float32)


def build_speech_features(df, cache_dir: Path | None = None,
                          progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-mel features for every row of `df`.

    Returns ``(mels (N, N_MELS, N_FRAMES) float32, valid_samples (N,) int64)`` in
    original dB scale; scaling is applied afterwards so that the mean/std can be
    fitted on the training split only.
    """
    n = len(df)
    mels = np.empty((n, config.N_MELS, config.N_FRAMES), dtype=np.float32)
    valid = np.empty(n, dtype=np.int64)
    for i, (path, file_name) in enumerate(zip(df["path"], df["file_name"])):
        mel, n_valid = load_or_compute_mel(path, file_name, cache_dir)
        mels[i] = mel
        valid[i] = n_valid
        if progress and (i + 1) % 200 == 0:
            print(f"\r  log-mel {i + 1}/{n}", end="", flush=True)
    if progress:
        print(f"\r  log-mel {n}/{n}")
    return mels, valid
