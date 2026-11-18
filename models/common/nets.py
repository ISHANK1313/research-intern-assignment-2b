"""Encoder and classifier blocks shared by the three pipelines.

Block mapping onto the assignment figure:

    Feature extraction      -> log-mel (speech, features.py) / DistilBERT embeddings
                               (text, text_features.py)
    Temporal modelling      -> SpeechTemporalEncoder  (BiLSTM over mel frames)
    Contextual modelling    -> TextContextualEncoder  (BiLSTM over word embeddings)
    Fusion                  -> torch.cat of the two pooled representations
    Classifier              -> EmotionMLP (softmax over 7 emotions)

Each encoder exposes a pooled ``representation`` so the same module can both
classify and hand its vector to the t-SNE plots.

The text encoder is mask-aware: padded positions are zeroed before the LSTM and
excluded from the mean pool, so a fixed-length padded batch yields exactly the
same representation as the unpadded sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `x` (B, T, D) over time, counting only positions where mask is True."""
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m).sum(dim=1) / denom


class SpeechTemporalEncoder(nn.Module):
    """BiLSTM over log-mel frames. Input (B, n_mels, T) -> pooled (B, 2*hidden)."""

    def __init__(self, n_mels: int, hidden: int, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, n_mels, T) -> (B, T, n_mels): the LSTM walks along time.
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        if mask is None:
            return self.norm(out.mean(dim=1))
        # Zero-padding is only there to make the batch rectangular; excluding it
        # from the pool keeps utterances of different true duration comparable.
        return self.norm(masked_mean(out, mask))


class TextContextualEncoder(nn.Module):
    """BiLSTM over DistilBERT word embeddings. Input (B, L, D) -> (B, 2*hidden)."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        out, _ = self.lstm(x * mask.unsqueeze(-1))
        return self.norm(masked_mean(out, mask))


class EmotionMLP(nn.Module):
    """MLP head producing logits over the seven emotions.

    Cross-entropy is applied to these logits during training, which is
    mathematically identical to softmax followed by NLL but numerically stabler;
    softmax is applied explicitly when probabilities are needed.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeechClassifier(nn.Module):
    """Speech-only model: SpeechTemporalEncoder -> EmotionMLP."""

    def __init__(self, n_mels: int, hidden: int, num_classes: int,
                 dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.encoder = SpeechTemporalEncoder(n_mels, hidden, num_layers, dropout)
        self.classifier = EmotionMLP(self.encoder.out_dim, hidden, num_classes, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                return_rep: bool = False):
        rep = self.encoder(x, mask)
        logits = self.classifier(rep)
        return (logits, rep) if return_rep else logits


class TextClassifier(nn.Module):
    """Text-only model: TextContextualEncoder -> EmotionMLP."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.encoder = TextContextualEncoder(in_dim, hidden, num_layers, dropout)
        self.classifier = EmotionMLP(self.encoder.out_dim, hidden, num_classes, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                return_rep: bool = False):
        rep = self.encoder(x, mask)
        logits = self.classifier(rep)
        return (logits, rep) if return_rep else logits


class FusionClassifier(nn.Module):
    """Multimodal model: concatenate the pooled speech and text representations.

    Both branches contribute a 2*hidden vector; their concatenation *is* the fused
    representation used for the fusion-block t-SNE plot.
    """

    def __init__(self, n_mels: int, text_in_dim: int, hidden: int, num_classes: int,
                 dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.speech_encoder = SpeechTemporalEncoder(n_mels, hidden, num_layers, dropout)
        self.text_encoder = TextContextualEncoder(text_in_dim, hidden, num_layers, dropout)
        self.rep_dim = self.speech_encoder.out_dim + self.text_encoder.out_dim
        self.classifier = EmotionMLP(self.rep_dim, hidden, num_classes, dropout)

    def forward(self, speech: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor | None = None,
                speech_mask: torch.Tensor | None = None,
                return_rep: bool = False):
        s_rep = self.speech_encoder(speech, speech_mask)
        t_rep = self.text_encoder(text, text_mask)
        fused = torch.cat([s_rep, t_rep], dim=-1)
        logits = self.classifier(fused)
        if return_rep:
            return logits, {"fused": fused, "speech": s_rep, "text": t_rep}
        return logits
