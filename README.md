# Multimodal Emotion Recognition on TESS

Assignment 2b — one emotion-recognition system built in three variants and compared on
an identical test set:

| # | Variant | Input | Encoder blocks |
|---|---------|-------|----------------|
| 1 | **Speech only** | log-mel spectrogram | 2-layer BiLSTM over frames |
| 2 | **Text only** | transcript | frozen DistilBERT embeddings + 2-layer BiLSTM over tokens |
| 3 | **Multimodal** | log-mel **+** transcript | both branches concatenated, MLP classifier |

Dataset: [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) —
2 female speakers, 2,800 utterances, 7 emotions, 200 carrier-phrase target words.

<!-- RESULTS_TABLE -->
Test set: **420 utterances** (60 per emotion), identical for all three variants.
Chance level for 7 balanced classes is 1/7 = **0.1429**.

| Variant | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Speech only | **0.9952** | 0.9953 | 0.9953 |
| Text only | 0.1429 | 0.0357 | 0.0357 |
| Multimodal (fusion) | **0.9976** | 0.9976 | 0.9976 |

Per-emotion F1:

| Variant | anger | disgust | fear | happiness | pleasant surprise | sadness | neutral |
|---|---|---|---|---|---|---|---|
| Speech only | 1.000 | 0.992 | 1.000 | 0.992 | 0.984 | 1.000 | 1.000 |
| Text only | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 |
| Multimodal | 1.000 | 1.000 | 1.000 | 0.992 | 0.992 | 1.000 | 1.000 |

Cluster separability of the three learned representations (silhouette on the test
split): speech temporal block **0.618**, text contextual block **−0.036**, fusion
block **0.684**.

### How to read this honestly

* **Multimodal does not significantly beat speech-only.** The gap is one utterance out
  of 420 (+0.0024 accuracy); an exact McNemar test on the paired predictions gives
  **p = 1.00**. The fusion architecture is implemented correctly — this corpus simply
  gives the text branch nothing to add.
* **Text-only is at chance for a structural reason, not a modelling failure.**
  I(emotion ; transcript) = **0.000000 bits**: TESS has only 200 distinct transcripts
  and every one of them occurs with all 7 emotions, so the best achievable text-only
  accuracy is exactly 1/7. The report also documents a splitting trap that makes the
  empirically "optimal" text strategy score *below* chance.
* **Speech carries essentially all of the signal.** A logistic regression on summary
  mel statistics alone already reaches 0.9929, which confirms the data path is sound
  and licenses the reading of the text result as a property of the data.
<!-- /RESULTS_TABLE -->

---

## Architecture

Following the assignment's functional blocks:

```
                    ┌──────────────── SPEECH BRANCH ────────────────┐
  wav ──► resample 16 kHz ──► trim silence ──► pad/truncate 3.0 s     │
        ──► log-mel 128 bands, n_fft 1024, hop 512  (128 × 94)        │
        ──► 2-layer BiLSTM (hidden 128)  ──► masked mean pool ──► 256-d ┐
                    └───────────────────────────────────────────────┘ │
                                                                      ├─► concat 512-d
                    ┌───────────────── TEXT BRANCH ────────────────┐ │   ──► MLP(128)
  file name ──► transcript "Say the word <x>."                       │ │   ──► softmax(7)
        ──► DistilBERT-base-uncased token embeddings (frozen, 768-d)  │ │
        ──► 2-layer BiLSTM (hidden 128)  ──► masked mean pool ──► 256-d ┘
                    └───────────────────────────────────────────────┘
```

| Block | Choice |
|-------|--------|
| Speech preprocessing | 16 kHz mono, silence trim (30 dB), fixed 3.0 s, true length kept as a frame mask |
| Text preprocessing | transcript rebuilt from the file name, WordPiece tokenisation, max 16 sub-tokens |
| Speech features | log-mel, 128 bands, 64 ms window, **32 ms hop** → 94 frames; absolute dB scaled by train-split statistics |
| Text features | DistilBERT token embeddings (6 layers, 768-d), **frozen** |
| Temporal modelling | 2-layer BiLSTM, hidden 128 per direction → 256-d |
| Contextual modelling | 2-layer BiLSTM over token embeddings → 256-d |
| Fusion | concatenation of the two pooled vectors → 512-d (both encoders trained jointly) |
| Classifier | MLP Linear(512→128) → GELU → dropout 0.3 → Linear(128→7), cross-entropy |

Full justification for every decision, including the ones that were deliberately *not*
taken, is in [`Report/report.pdf`](Report/report.pdf) Section A.

## Repository layout

```
.
├── models/
│   ├── common/                 shared, non-duplicated infrastructure
│   │   ├── config.py           all hyper-parameters, paths, class order
│   │   ├── data.py             TESS discovery, label parsing, canonical split
│   │   ├── features.py         audio preprocessing + log-mel
│   │   ├── text_features.py    DistilBERT embedding extraction (+cache)
│   │   ├── nets.py             encoder / fusion / classifier blocks
│   │   ├── train_utils.py      seeding, metrics, training loop, artifact I/O
│   │   └── io_utils.py         JSON helpers (kept torch-free)
│   ├── speech_pipeline/        train.py, test.py
│   ├── text_pipeline/          train.py, test.py
│   └── fusion_pipeline/        train.py, test.py
├── scripts/
│   ├── download_tess.py        fetches the dataset (never committed)
│   ├── control_baselines.py    setup-validation controls + information-theoretic bound
│   ├── ablation_finetune_text.py   fine-tuned DistilBERT ablation (Section B.4)
│   ├── make_plots.py           all report figures
│   ├── summarize_results.py    comparison table + analysis.json + failure cases
│   ├── verify_repo.py          consistency checks (numbers, plots, hygiene)
│   └── make_report.py          builds Report/report.pdf
├── Results/                    metrics, plots, representations, predictions
├── Report/report.pdf           the written report (Sections A, B, C)
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ is required. Everything runs on CPU; no GPU is needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# This project is CPU-only. A plain `pip install torch` pulls the default PyPI wheel,
# which bundles ~2.5 GB of unused CUDA libraries — install the CPU build first:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(Installing `-r requirements.txt` on its own also works; it is just a much larger
download.)

### Get the dataset

The raw audio is **not** in this repository. Fetch it with the provided script:

```bash
python scripts/download_tess.py --data-dir data/tess
```

The Kaggle endpoint this uses is public and needs **no account or API key**. If it ever
stops working, download the archive manually and extract it yourself, or point the
pipelines at an existing extraction with `--data-dir`.

Two things the download script handles that matter for correctness:

* the published archive contains the whole corpus **twice** (5,600 files). The loader
  de-duplicates to the **2,800 unique utterances**; without this the same recordings
  would appear in both train and test.
* the "pleasant surprise" emotion is abbreviated **`ps`** in the file names
  (`OAF_back_ps.wav`), not spelled out.

## Usage

Each pipeline is a train/test pair. All three read the same committed split file
(`Results/splits/split_seed42.csv`), which is what makes the comparison valid.

```bash
# 1. build the canonical stratified split (writes Results/splits/split_seed42.csv)
python -m models.common.data build --data-dir data/tess

# 2. speech-only
python -m models.speech_pipeline.train --data-dir data/tess
python -m models.speech_pipeline.test  --data-dir data/tess

# 3. text-only
python -m models.text_pipeline.train   --data-dir data/tess
python -m models.text_pipeline.test    --data-dir data/tess

# 4. multimodal
python -m models.fusion_pipeline.train --data-dir data/tess
python -m models.fusion_pipeline.test  --data-dir data/tess

# 5. figures, tables and the PDF report
python scripts/make_plots.py
python scripts/summarize_results.py
python scripts/make_report.py

# 6. verify internal consistency (numbers vs report, plots, hygiene)
python scripts/verify_repo.py
```

`verify_repo.py` re-derives every headline number from the saved artifacts, asserts
that the three variants share one test set, checks that all figures the report embeds
exist, and confirms no audio or credentials are present. It prints
`ALL n CHECKS PASSED` on success.

`test.py` loads the committed checkpoint from `Results/checkpoints/`, so step 2–4's
`test.py` reproduces the published numbers **without retraining**. The first text/fusion
run downloads `distilbert-base-uncased` (~265 MB) from the Hugging Face Hub into the
local HF cache; that model is a dependency, not a repository artifact.

## Results

Everything under `Results/`:

| Path | Contents |
|------|----------|
| `metrics/accuracy_table.csv` / `.md` | the 3-variant comparison table |
| `metrics/metrics_<variant>.json` | full test metrics per variant (accuracy, macro/weighted F1, per-class F1/precision/recall, confusion matrix) |
| `metrics/analysis.json` | every derived number quoted in the report |
| `metrics/history_<variant>.csv` | per-epoch training curves |
| `metrics/silhouette_scores.json` | cluster separability per representation block |
| `plots/` | confusion matrices, training curves, t-SNE plots, comparisons |
| `representations/` | the learned representations + the shared test labels/listing |
| `predictions/` | per-utterance predictions with probabilities, used for error analysis |
| `checkpoints/` | trained weights (a few MB each) |
| `failure_cases.md` | the fusion model's real failure cases |

## Key findings

1. **Speech carries essentially all of the signal.** TESS emotion is encoded in *how*
   the carrier phrase is spoken, not in what is said. A logistic regression on summary
   mel statistics alone already reaches **0.9929**, and the BiLSTM reaches **0.9952**.
2. **Text-only is at chance, and that is structural, not a bug.** TESS has only
   **200 distinct transcripts**, and **every one of them occurs with all 7 emotion
   labels** — the target word is balanced across emotions by the recording protocol.
   Measured directly from the data, **I(emotion ; transcript) = 0.000000 bits** and
   H(emotion | transcript) = log₂7 exactly. Identical text is labelled with 7 different
   classes, so the best achievable text-only accuracy is exactly 1/7 ≈ 0.143 — which is
   what the model scores. Any text-only score well above that would indicate leakage
   rather than recognition.
3. **This is not an under-trained model.** The frozen-embedding model cannot fit its own
   training set (train accuracy stays at chance), and neither can a fully fine-tuned
   66 M-parameter DistilBERT — because there is no input→label mapping to learn. See the
   ablation in Section B.4 of the report.
4. **There is also a splitting trap.** Because every *(word, emotion)* cell contains
   exactly two files, stratifying on emotion means the emotion a word is most associated
   with in training is absent from test for **all 200 words**. A text model that
   memorises the training word→emotion mapping therefore scores **0.0**, below chance.
   The report documents this rather than presenting it as a result.
5. **Fusion is bounded by the same ceiling.** One modality is uninformative by
   construction, so a large multimodal gain is not available here. The measured
   difference over speech-only is **one utterance** (p = 1.00 by exact McNemar).
   The report states this plainly instead of dressing it up.

## Reproducibility

* One global seed (`SEED = 42` in `models/common/config.py`) controls the split, weight
  initialisation, dropout and batch shuffling.
* The split is stratified on emotion: **1,960 train / 420 val / 420 test**, exactly 60
  test utterances per emotion.
* Normalisation statistics are fitted on the **training split only**; early stopping
  uses the validation split; the test split is used once, for reporting.
* De-duplication of the archive (`build_index`) is required for the numbers to be
  meaningful and is asserted in code, not just documented.

### Verified versions

Developed and executed with: Python 3.11.9, torch 2.14.1+cpu, transformers 5.19.0,
librosa 0.11.0, soundfile 0.14.0, scikit-learn 1.4.0, numpy 1.26.4, pandas 3.0.3,
matplotlib 3.11.2, reportlab 4.4.9 — CPU only, no CUDA.

## Notes and limitations

* Trained on CPU with modest model sizes; no hyper-parameter sweep beyond the choices
  documented in the report.
* Only two speakers are represented, so speaker-independent generalisation is not
  tested; the report does break accuracy down by speaker.
* Attention-based fusion was considered but **not** implemented. No attention results
  are claimed anywhere.
* No data augmentation was used.

## Dataset credit

TESS was created by K. Dupuis and M. K. Pichora-Fuller at the University of Toronto and
is mirrored on Kaggle as
[`ejlok1/toronto-emotional-speech-set-tess`](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).
Please refer to the original dataset page for its licence and citation before
redistributing the audio. This repository contains **code, results and analysis only** —
no audio files are redistributed.
