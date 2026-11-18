"""Frozen DistilBERT contextual word embeddings for the text and fusion pipelines.

Why the encoder is frozen (justified at length in the report, Section A/C):

* TESS transcripts are all the same carrier phrase, "Say the word <target>.", and the
  target word is *balanced across all seven emotions* by construction. The lexical
  content therefore carries essentially no emotion information, and fine-tuning
  66 M transformer parameters on 1 960 training sentences mostly buys overfitting,
  not signal.
* Freezing keeps the committed checkpoints small (a few MB of BiLSTM + MLP weights
  instead of ~265 MB of DistilBERT weights) so the repository stays cloneable and
  `test.py` can reproduce the saved results on a fresh clone.

`text_pipeline/train.py --finetune` runs the fine-tuned alternative as an ablation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from . import config


class TextEmbedder:
    """Wraps the DistilBERT tokenizer + encoder and returns token-level embeddings."""

    def __init__(self, model_name: str = config.TEXT_BACKBONE,
                 max_tokens: int = config.TEXT_MAX_TOKENS, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (embeddings (B, L, D), attention_mask (B, L)) for a list of texts."""
        enc = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self.max_tokens, return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        emb = out.last_hidden_state.cpu().numpy().astype(np.float32)
        mask = enc["attention_mask"].cpu().numpy().astype(bool)
        return emb, mask

    def encode_all(self, texts: list[str], batch_size: int = 64,
                   progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
        embs, masks = [], []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            e, m = self.encode(chunk)
            embs.append(e)
            masks.append(m)
            if progress:
                done = min(start + batch_size, len(texts))
                print(f"\r  text embeddings {done}/{len(texts)}", end="", flush=True)
        if progress:
            print()
        return np.concatenate(embs, axis=0), np.concatenate(masks, axis=0)


_EMBEDDER_CACHE: dict[tuple, "TextEmbedder"] = {}


def get_embedder(model_name: str = config.TEXT_BACKBONE,
                 max_tokens: int = config.TEXT_MAX_TOKENS,
                 device: str = "cpu") -> "TextEmbedder":
    """Return a process-wide shared TextEmbedder.

    Loading DistilBERT takes a couple of minutes on CPU, and a single training run
    needs embeddings for more than one split, so the instance is reused. This is a
    pure caching optimisation: the model is frozen and runs in eval mode, so it
    produces bit-identical embeddings either way.
    """
    key = (model_name, max_tokens, device)
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = TextEmbedder(model_name, max_tokens, device)
    return _EMBEDDER_CACHE[key]


def _content_tag(texts: list[str]) -> str:
    """Short stable hash identifying *which* texts are being embedded.

    The cache must be keyed on content, not just on the model and the sequence
    length: the train (1,960), validation (420) and test (420) splits all embed
    roughly the same number of items, so a length-based key lets one split silently
    overwrite another and, worse, lets a later run pick up the *wrong* split's
    embeddings whenever the row counts happen to coincide.
    """
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def cache_paths(cache_dir: Path, model_name: str, max_tokens: int,
                texts: list[str]) -> tuple[Path, Path]:
    tag = (f"{model_name.replace('/', '_')}_L{max_tokens}"
           f"_{len(texts)}_{_content_tag(texts)}")
    base = Path(cache_dir) / "text"
    return base / f"emb_{tag}.npy", base / f"mask_{tag}.npy"


def load_or_compute(texts: list[str], cache_dir: Path | None = None,
                    model_name: str = config.TEXT_BACKBONE,
                    max_tokens: int = config.TEXT_MAX_TOKENS,
                    device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return DistilBERT embeddings for `texts`, using an on-disk cache if present.

    The cache stores derived tensors only and is git-ignored.
    """
    if cache_dir is not None:
        emb_f, mask_f = cache_paths(cache_dir, model_name, max_tokens, texts)
        if emb_f.is_file() and mask_f.is_file():
            emb, mask = np.load(emb_f), np.load(mask_f)
            if emb.shape[0] == len(texts):
                return emb, mask

    embedder = get_embedder(model_name, max_tokens, device)
    emb, mask = embedder.encode_all(texts)

    if cache_dir is not None:
        emb_f, mask_f = cache_paths(cache_dir, model_name, max_tokens, texts)
        emb_f.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_f, emb)
        np.save(mask_f, mask)
    return emb, mask
