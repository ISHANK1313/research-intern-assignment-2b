# Results

Every number, figure and table in `Report/report.pdf` is produced from the files in this
directory. Nothing here is hand-edited: re-running the pipelines and then
`scripts/make_plots.py` + `scripts/summarize_results.py` regenerates all of it.

## Layout

| Path | Produced by | Contents |
|------|-------------|----------|
| `splits/split_seed42.csv` | `models/common/data.py` | the canonical stratified split shared by all three variants (1,960 / 420 / 420). Paths are stored relative to the data directory so the file is portable. |
| `metrics/metrics_<variant>.json` | each `test.py` | test accuracy, macro/weighted F1, per-class F1/precision/recall/support and the full confusion matrix |
| `metrics/metrics_<variant>_val.json` | each `train.py` | validation metrics at the best epoch, plus parameter count and training metadata |
| `metrics/history_<variant>.csv` | each `train.py` | per-epoch train/val loss, accuracy and macro F1 — the training curves |
| `metrics/accuracy_table.csv` / `.md` | `scripts/summarize_results.py` | the three-variant comparison table |
| `metrics/analysis.json` | `scripts/summarize_results.py` | every derived figure quoted in the report (easiest/hardest emotions, confusions, speaker and duration breakdowns, fusion deltas, failure cases) |
| `metrics/silhouette_scores.json` | `scripts/make_plots.py` | cluster separability per representation block |
| `metrics/probs_<variant>.npy` | each `test.py` | (420, 7) class probabilities on the test split |
| `predictions/pred_<variant>.csv` | each `test.py` | per-utterance predictions with all class probabilities — the basis of the error analysis |
| `representations/reps_<variant>_test.npy` | each `test.py` | the learned representation per block (speech temporal 256-d, text contextual 256-d, fusion 512-d) |
| `representations/test_labels.npy`, `test_files.csv` | first `test.py` to run | shared ground truth for the test split; later runs assert these match, so the three variants provably share one test set |
| `checkpoints/<variant>.pt` | each `train.py` | trained weights plus the architecture and normalisation statistics (committed so `test.py` reproduces results without retraining) |
| `plots/*.png` | `scripts/make_plots.py` | confusion matrices, training curves, t-SNE plots, accuracy comparisons |
| `failure_cases.md` | `scripts/summarize_results.py` | the fusion model's real failure cases with cross-variant evidence |

## Note on the shared test set

`test_labels.npy` and `test_files.csv` are written once and *verified* by every
subsequent `test.py`. If a variant were ever evaluated on a different test set, the run
fails with an explicit error instead of silently producing an incomparable number.
