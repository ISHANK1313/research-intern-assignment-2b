"""Shared training / evaluation utilities.

One loop serves all three pipelines, so differences in the reported metrics come
from the architecture and the modality, never from the training procedure.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset

from . import config


# --------------------------------------------------------------------------- seeds
def set_seed(seed: int = config.SEED) -> None:
    """Seed every RNG we rely on so runs are reproducible on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_threads(n_threads: int | None = None) -> int:
    """Use all available cores for the CPU LSTM/matmul kernels.

    The default of 2 intra-op threads leaves most of the machine idle on the small
    BiLSTMs used here, which dominates wall-clock time.
    """
    n = n_threads or (os.cpu_count() or 2)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except RuntimeError:
        # already initialised by an earlier parallel region; harmless
        pass
    return torch.get_num_threads()


# ------------------------------------------------------------------------ datasets
class ArrayDataset(Dataset):
    """Dataset over a dict of equal-length numpy arrays plus integer labels."""

    def __init__(self, arrays: dict[str, np.ndarray], labels: np.ndarray):
        self.keys = list(arrays)
        # ascontiguousarray copies when needed, which also sidesteps PyTorch's
        # "non-writable tensor" warning for arrays that came from np.load.
        self.arrays = {k: torch.as_tensor(np.ascontiguousarray(v)) for k, v in arrays.items()}
        self.labels = torch.as_tensor(np.array(labels, dtype=np.int64, copy=True))
        n = len(self.labels)
        for k, v in self.arrays.items():
            if len(v) != n:
                raise ValueError(f"array {k!r} has {len(v)} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.arrays.items()}
        item["label"] = self.labels[idx]
        return item


def make_loader(ds: Dataset, batch_size: int = config.BATCH_SIZE,
                shuffle: bool = False, seed: int = config.SEED) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=gen if shuffle else None)


# ------------------------------------------------------------------------- metrics
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro/weighted F1, per-class F1 and the confusion matrix."""
    labels = list(range(config.NUM_CLASSES))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels,
                              zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels,
                                zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {e: float(v) for e, v in zip(config.EMOTIONS, per_class)},
        "per_class_precision": {e: float(v) for e, v in zip(config.EMOTIONS, precision)},
        "per_class_recall": {e: float(v) for e, v in zip(config.EMOTIONS, recall)},
        "per_class_support": {e: int(v) for e, v in zip(config.EMOTIONS, support)},
        "confusion_matrix": cm.tolist(),
    }


def metrics_to_row(metrics: dict) -> dict:
    """Flatten metrics into a CSV-friendly row."""
    row = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
           "weighted_f1": metrics["weighted_f1"]}
    for emotion in config.EMOTIONS:
        row[f"f1_{emotion}"] = metrics["per_class_f1"][emotion]
    return row


def format_metrics(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  accuracy   : {metrics['accuracy']:.4f}")
    lines.append(f"  macro F1   : {metrics['macro_f1']:.4f}")
    lines.append(f"  weighted F1: {metrics['weighted_f1']:.4f}")
    lines.append("  per-class F1 (n):")
    for emotion in config.EMOTIONS:
        lines.append(f"    {emotion:17s} {metrics['per_class_f1'][emotion]:.4f}"
                     f"  (n={metrics['per_class_support'][emotion]})")
    return "\n".join(lines)


# ------------------------------------------------------------------------ forward fns
ForwardFn = Callable[..., torch.Tensor]


def speech_forward(model, batch, return_rep: bool = False):
    return model(batch["speech"], batch["speech_mask"], return_rep=return_rep)


def text_forward(model, batch, return_rep: bool = False):
    return model(batch["text"], batch["mask"], return_rep=return_rep)


def fusion_forward(model, batch, return_rep: bool = False):
    return model(batch["speech"], batch["text"], batch["mask"],
                 batch["speech_mask"], return_rep=return_rep)


# ------------------------------------------------------------------------- training
@dataclass
class History:
    rows: list[dict] = field(default_factory=list)

    def append(self, **kw) -> None:
        self.rows.append(kw)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def evaluate(model: nn.Module, forward_fn: ForwardFn, loader: DataLoader,
             device: torch.device, collect_reps: bool = False,
             rep_keys: tuple[str, ...] = ()) -> dict:
    """Run the model over `loader` and return predictions (+ reps if requested)."""
    model.eval()
    all_logits, all_labels = [], []
    rep_store: dict[str, list[np.ndarray]] = {k: [] for k in rep_keys}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if collect_reps:
                logits, reps = forward_fn(model, batch, return_rep=True)
                if isinstance(reps, dict):
                    for k in rep_keys:
                        rep_store[k].append(reps[k].cpu().numpy())
                else:
                    rep_store[rep_keys[0]].append(reps.cpu().numpy())
            else:
                logits = forward_fn(model, batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    out = {
        "y_true": y_true,
        "y_pred": probs.argmax(axis=1),
        "probs": probs,
        "logits": logits.numpy(),
        "loss": float(nn.functional.cross_entropy(logits, torch.cat(all_labels)).item()),
        "metrics": compute_metrics(y_true, probs.argmax(axis=1)),
    }
    if collect_reps:
        out["reps"] = {k: np.concatenate(v, axis=0) for k, v in rep_store.items()}
    return out


class Trainer:
    """AdamW + cross-entropy, early stopping on validation macro-F1."""

    def __init__(self, model: nn.Module, forward_fn: ForwardFn,
                 device: torch.device | None = None,
                 lr: float = config.LEARNING_RATE,
                 weight_decay: float = config.WEIGHT_DECAY,
                 grad_clip: float = config.GRAD_CLIP,
                 max_epochs: int = config.MAX_EPOCHS,
                 patience: int = config.PATIENCE,
                 tag: str = "model", verbose: bool = True):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.forward_fn = forward_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.tag = tag
        self.verbose = verbose
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        self.history = History()
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.best_state: dict | None = None

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss, n, preds, trues = 0.0, 0, [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = self.forward_fn(self.model, batch)
                loss = self.criterion(logits, batch["label"])
            if train:
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
            n += len(batch["label"])
            preds.append(logits.detach().argmax(dim=-1).cpu().numpy())
            trues.append(batch["label"].cpu().numpy())
        return total_loss / max(n, 1), np.concatenate(trues), np.concatenate(preds)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        epochs_without_improvement = 0
        for epoch in range(1, self.max_epochs + 1):
            tr_loss, tr_true, tr_pred = self._run_epoch(train_loader, train=True)
            va_loss, va_true, va_pred = self._run_epoch(val_loader, train=False)
            tr_acc = float(accuracy_score(tr_true, tr_pred))
            va_acc = float(accuracy_score(va_true, va_pred))
            va_f1 = float(f1_score(va_true, va_pred, average="macro",
                                   labels=list(range(config.NUM_CLASSES)),
                                   zero_division=0))
            self.history.append(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                                train_macro_f1=float(f1_score(
                                    tr_true, tr_pred, average="macro",
                                    labels=list(range(config.NUM_CLASSES)),
                                    zero_division=0)),
                                val_loss=va_loss, val_acc=va_acc, val_macro_f1=va_f1)

            improved = va_f1 > self.best_val_f1
            if improved:
                self.best_val_f1, self.best_epoch = va_f1, epoch
                self.best_state = {k: v.detach().clone()
                                   for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                mark = " *" if improved else ""
                print(f"  epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}"
                      f" | val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}{mark}",
                      flush=True)

            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch} "
                          f"(no val improvement for {self.patience} epochs)", flush=True)
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.verbose:
            print(f"  best epoch {self.best_epoch} (val macro-F1 {self.best_val_f1:.4f})",
                  flush=True)
        return self.history.to_frame()


# ------------------------------------------------------------------------------ I/O
# Re-exported from io_utils so existing callers keep working, while the torch-free
# plotting/report scripts can import io_utils directly.
from .io_utils import load_json, save_json  # noqa: E402,F401


def save_checkpoint(model: nn.Module, path: Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_test_artifacts(variant: str, test_df: pd.DataFrame, out: dict,
                        rep: np.ndarray, rep_name: str,
                        extra: dict | None = None) -> dict:
    """Persist everything a variant's test run produces, in a uniform layout.

    Also writes (or verifies) the *shared* test-set label vector and file listing.
    Every variant runs against the same canonical test split, so these two files
    must be identical for all three; a mismatch means the comparison is invalid and
    is raised loudly rather than silently overwritten.
    """
    y_true = np.asarray(out["y_true"])
    y_pred = np.asarray(out["y_pred"])
    probs = np.asarray(out["probs"])

    config.REPS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = config.REPS_DIR / "test_labels.npy"
    if labels_path.is_file():
        if not np.array_equal(np.load(labels_path), y_true):
            raise RuntimeError(
                "test label vector differs from the one already saved - the test "
                "split changed between runs, so the variants are not comparable.")
    else:
        np.save(labels_path, y_true)

    files_path = config.REPS_DIR / "test_files.csv"
    cols = ["file_name", "rel_path", "speaker", "word", "emotion", "label", "transcript"]
    listing = test_df[cols].reset_index(drop=True)
    if files_path.is_file():
        prev = pd.read_csv(files_path)
        if list(prev["file_name"]) != list(listing["file_name"]):
            raise RuntimeError(
                "test file listing differs from the one already saved - the test "
                "split changed between runs, so the variants are not comparable.")
    else:
        listing.to_csv(files_path, index=False, lineterminator="\n")

    np.save(config.METRICS_DIR / f"probs_{variant}.npy", probs)
    np.save(config.REPS_DIR / f"reps_{variant}_test.npy", rep.astype(np.float32))

    pred_df = pd.DataFrame({
        "file_name": listing["file_name"],
        "speaker": listing["speaker"],
        "word": listing["word"],
        "transcript": listing["transcript"],
        "true_emotion": [config.EMOTIONS[i] for i in y_true],
        "pred_emotion": [config.EMOTIONS[i] for i in y_pred],
        "correct": y_true == y_pred,
        "confidence": probs.max(axis=1),
        "true_prob": probs[np.arange(len(y_true)), y_true],
    })
    for i, emotion in enumerate(config.EMOTIONS):
        pred_df[f"p_{emotion}"] = probs[:, i]
    pred_df.to_csv(config.PRED_DIR / f"pred_{variant}.csv", index=False,
                   lineterminator="\n")

    payload = {"variant": variant, "split": "test", "rep_name": rep_name,
               "rep_dim": int(rep.shape[1]), "n_test": int(len(y_true)),
               **out["metrics"]}
    if extra:
        payload.update(extra)
    save_json(payload, config.METRICS_DIR / f"metrics_{variant}.json")
    return payload
