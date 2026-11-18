"""Shared configuration for all three emotion-recognition pipelines.

Every pipeline (speech / text / fusion) imports these constants so that the
preprocessing, the class order and the train/val/test split are identical across
variants. Keeping this in one place is what makes the three-way comparison
meaningful: any difference in the reported numbers comes from the model, not
from the data preparation.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ------------------------------------------------------------------------ dataset
# The seven emotional categories of TESS, in a fixed canonical order.
# Every class index in every pipeline refers to this list.
EMOTIONS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "pleasant_surprise",
    "sadness",
    "neutral",
)
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# Raw filename tokens -> canonical emotion label.
# NOTE: in the actual TESS file names "pleasant surprise" is abbreviated to ``ps``
# (e.g. ``OAF_back_ps.wav``), not spelled out. Verified against all 2 800 files.
FILENAME_EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "ps": "pleasant_surprise",
    "pleasant_surprise": "pleasant_surprise",  # tolerated spelling
}

# The older-adult speaker appears as both ``OAF`` and, in a single file, ``OA``.
SPEAKER_ALIASES = {"OA": "OAF", "OAF": "OAF", "YAF": "YAF"}

# TESS utterances are carrier phrases of the form "Say the word <target>".
TRANSCRIPT_TEMPLATE = "Say the word {word}."

# ------------------------------------------------------------------------- splits
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ------------------------------------------------------------------ speech (audio)
SAMPLE_RATE = 16_000
DURATION_SEC = 3.0                      # fixed length after silence trimming
FIXED_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
TRIM_TOP_DB = 30                        # librosa silence-trim threshold

# ----------------------------------------------------------------- log-mel features
N_MELS = 128
N_FFT = 1024
# 32 ms hop with a 64 ms window (50 % overlap). A 16 ms hop was measured to double
# the BiLSTM training time on CPU (188 frames vs 94) without addressing a time scale
# that matters for emotion: the relevant cues (pitch contour, energy envelope,
# rhythm) evolve over hundreds of milliseconds, not tens.
HOP_LENGTH = 512
WIN_LENGTH = 1024
FMIN = 20
FMAX = 8000                             # Nyquist at 16 kHz
# Number of mel frames produced by the settings above for FIXED_SAMPLES.
N_FRAMES = 1 + FIXED_SAMPLES // HOP_LENGTH

# ------------------------------------------------------------------ text (DistilBERT)
TEXT_BACKBONE = "distilbert-base-uncased"
TEXT_MAX_TOKENS = 16

# -------------------------------------------------------------------------- models
SPEECH_HIDDEN = 128                     # BiLSTM hidden size per direction -> 256-d rep
TEXT_HIDDEN = 128                       # BiLSTM hidden size per direction -> 256-d rep
SPEECH_REP_DIM = 2 * SPEECH_HIDDEN
TEXT_REP_DIM = 2 * TEXT_HIDDEN
FUSION_REP_DIM = SPEECH_REP_DIM + TEXT_REP_DIM
DROPOUT = 0.3

# ----------------------------------------------------------------------- training
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 10                           # early stopping on validation macro-F1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "Results"
SPLIT_DIR = RESULTS_DIR / "splits"
PLOTS_DIR = RESULTS_DIR / "plots"
REPS_DIR = RESULTS_DIR / "representations"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CACHE_DIR = REPO_ROOT / "cache"

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "tess"


def all_dirs() -> tuple[Path, ...]:
    return (RESULTS_DIR, SPLIT_DIR, PLOTS_DIR, REPS_DIR, METRICS_DIR, PRED_DIR,
            CHECKPOINT_DIR, CACHE_DIR)
