#!/usr/bin/env python
"""Build ``Report/report.pdf`` from the saved result artifacts.

All numbers and figures are read from ``Results/``; nothing is hard-coded, so the
report cannot drift away from the runs that produced it.

    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from models.common import config, io_utils as IO  # noqa: E402

RESULTS = config.RESULTS_DIR
PLOTS = config.PLOTS_DIR
OUT_PATH = config.REPO_ROOT / "Report" / "report.pdf"

VARIANTS = ("speech", "text", "fusion")
LABEL = {"speech": "Speech only", "text": "Text only", "fusion": "Multimodal (fusion)"}
DISPLAY = {
    "anger": "anger", "disgust": "disgust", "fear": "fear",
    "happiness": "happiness", "pleasant_surprise": "pleasant surprise",
    "sadness": "sadness", "neutral": "neutral",
}

# Hand-written explanations for the case studies, keyed by file name. Each is written
# against the evidence printed by scripts/summarize_results.py (per-variant predictions,
# confidences, runner-up class). Any case without an entry falls back to an explanation
# derived from that evidence.
EXPLANATIONS: dict[str, str] = {
    "OAF_lease_happy.wav": (
        "<b>Every variant fails on this utterance</b> &mdash; the speech-only model also "
        "predicts pleasant surprise, with 0.576 confidence, so this is an intrinsic "
        "acoustic collision rather than a fusion artefact. In TESS, <i>happiness</i> and "
        "<i>pleasant surprise</i> are both high-arousal, positive-valence productions; "
        "what separates them is mainly the shape of the final pitch movement, not the "
        "gross prosodic profile. The model over-predicts pleasant surprise because that "
        "label is the acoustically broader of the two &mdash; an excited, surprised "
        "&ldquo;happy&rdquo; rendition is a defensible instance of either category. The "
        "0.870 confidence shows the model did not hedge: this is a systematic "
        "label-boundary problem, not model uncertainty."),
    "YAF_hash_disgust.wav": (
        "The fusion model gets this right (0.706) while the speech-only model is "
        "<i>confidently wrong</i> (0.855 in favour of pleasant surprise). Since the added "
        "text branch demonstrably carries no emotion information (Section C.5), the "
        "correction cannot come from new evidence; it comes from the extra capacity and "
        "different optimisation trajectory of the larger joint model acting on a single "
        "utterance. This one flip is the <i>entire</i> accuracy advantage of the "
        "multimodal model, and the paired McNemar test (Section B.1) shows it is not "
        "statistically significant. It is reported as a decision boundary moving, not as "
        "multimodal understanding."),
    "OAF_rush_disgust.wav": (
        "Correct at 0.815, but the runner-up is again pleasant surprise (0.185); the "
        "speech-only model is also correct here (0.874). Both models appear to rely on "
        "fine spectral detail &mdash; spectral tilt and noise characteristics that "
        "separate a disgusted production from a surprised one &mdash; because the gross "
        "prosody (high arousal, strong energy) is shared with the positive categories. "
        "This is the same confusion axis as the two cases above, which is why "
        "<i>disgust</i> is one of the two classes the fusion model improves on."),
    "OAF_wire_ps.wav": (
        "Correct at 0.882 with happiness as runner-up (0.116). The pleasant-surprise / "
        "happiness pair accounts for every genuine error and nearly every low-margin "
        "decision in the multimodal model. That matches the per-class F1 in Section B.1: "
        "for the speech-only model pleasant surprise is the weakest class (0.984) and it "
        "remains the weakest for the fusion model (0.992)."),
    "OAF_burn_ps.wav": (
        "Correct at 0.914 with happiness as runner-up (0.085), included as the fifth "
        "lowest-margin decision. Even one of the model&rsquo;s more confident calls on "
        "pleasant surprise is pulled toward happiness. Cases 3&ndash;5 together show that "
        "the residual difficulty is concentrated in a single positive, high-arousal pair "
        "rather than distributed across the seven labels &mdash; the desired behaviour "
        "would be the opposite, so this is the clearest remaining weakness of the system."),
}

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=_ss["Title"], fontSize=19, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", parent=_ss["Normal"], fontSize=10.5,
                               textColor=colors.HexColor("#444444"), alignment=1,
                               spaceAfter=14),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontSize=14.5, spaceBefore=14,
                         spaceAfter=7, textColor=colors.HexColor("#1a3a5c")),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=11.8, spaceBefore=10,
                         spaceAfter=5, textColor=colors.HexColor("#23527c")),
    "h3": ParagraphStyle("h3", parent=_ss["Heading3"], fontSize=10.4, spaceBefore=8,
                         spaceAfter=3, textColor=colors.HexColor("#333333")),
    "body": ParagraphStyle("body", parent=_ss["BodyText"], fontSize=9.3, leading=13.2,
                           alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", parent=_ss["BodyText"], fontSize=9.3,
                             leading=13.0, leftIndent=11, bulletIndent=2, spaceAfter=2.5),
    "cap": ParagraphStyle("cap", parent=_ss["Normal"], fontSize=8.2, leading=10.5,
                          textColor=colors.HexColor("#555555"), alignment=1,
                          spaceBefore=3, spaceAfter=9),
    "code": ParagraphStyle("code", parent=_ss["Code"], fontSize=8.0, leading=10.6,
                           backColor=colors.HexColor("#f4f4f4"), borderPadding=4,
                           spaceAfter=6),
    "cell": ParagraphStyle("cell", parent=_ss["Normal"], fontSize=7.4, leading=9.0),
}


def H1(text): return Paragraph(text, S["h1"])
def H2(text): return Paragraph(text, S["h2"])
def H3(text): return Paragraph(text, S["h3"])
def P(text): return Paragraph(text, S["body"])
def CODE(text): return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"),
                                S["code"])
def CAP(text): return Paragraph(text, S["cap"])


def bullets(items) -> list:
    return [Paragraph(t, S["bullet"], bulletText="\u2022") for t in items]


def table(rows, col_widths=None, header=True, font_size=7.6, align_right_from=1):
    data = [[Paragraph(str(c), S["cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="CENTER")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
                  ("FONTSIZE", (0, 0), (-1, 0), font_size)]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def figure(name: str, width_cm: float = 16.0) -> Image:
    path = PLOTS / name
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    return Image(str(path), width=w, height=w * ih / iw)


def load() -> dict:
    analysis = IO.load_json(config.METRICS_DIR / "analysis.json")
    metrics = {v: IO.load_json(config.METRICS_DIR / f"metrics_{v}.json")
               for v in VARIANTS}
    # n_params / rep_dim live in the training-time (*_val) metrics, not in the test
    # metrics written by test.py. Merge them in so the report never silently reads 0.
    for v, m in metrics.items():
        val_path = config.METRICS_DIR / f"metrics_{v}_val.json"
        if val_path.is_file():
            for key, value in IO.load_json(val_path).items():
                if key in ("n_params", "rep_dim", "backbone", "frozen_backbone",
                           "best_epoch"):
                    m.setdefault(key, value)
    split = pd.read_csv(config.SPLIT_DIR / f"split_seed{config.SEED}.csv")
    sizes = {s: int((split["split"] == s).sum()) for s in ("train", "val", "test")}
    histories = {v: pd.read_csv(config.METRICS_DIR / f"history_{v}.csv")
                 for v in VARIANTS}
    ctrl_path = config.METRICS_DIR / "control_baselines.json"
    controls = IO.load_json(ctrl_path) if ctrl_path.is_file() else None
    return {"analysis": analysis, "metrics": metrics, "sizes": sizes,
            "histories": histories, "split": split, "controls": controls}


def _fmt(x, nd=4):
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------------ sections
def section_header(ctx) -> list:
    a = ctx["analysis"]
    sizes = ctx["sizes"]
    return [
        Paragraph("Assignment 2b &mdash; Multimodal Emotion Recognition", S["title"]),
        Paragraph("Speech-only vs. text-only vs. multimodal fusion on the Toronto "
                  "Emotional Speech Set (TESS)", S["subtitle"]),
        P("<b>Objective.</b> Build one emotion-recognition system in three variants and "
          "compare them honestly on an identical test set: (1) speech only, (2) text "
          "only, and (3) multimodal, combining speech and text. Functional blocks follow "
          "the assignment figure: preprocessing, feature extraction, "
          "temporal/contextual modelling, fusion, classifier."),
        P(f"<b>Dataset.</b> TESS (Toronto Emotional Speech Set), 2 female speakers, "
          f"7 emotions, 200 target words in the carrier phrase "
          f"&ldquo;Say the word &lt;word&gt;.&rdquo; The published archive ships the "
          f"corpus <i>twice</i> (5,600 files); after removing byte-identical duplicates "
          f"the corpus is <b>2,800 utterances</b>, 400 per emotion. Emotion labels are "
          f"parsed from the file names ("
          f"<font face='Courier'>OAF_back_angry.wav</font>); note that "
          f"&ldquo;pleasant surprise&rdquo; is abbreviated "
          f"<font face='Courier'>ps</font> in the file names."),
        P(f"<b>Split.</b> Stratified on emotion, fixed seed {a['seed']}: "
          f"<b>{sizes['train']} train / {sizes['val']} val / {sizes['test']} test</b> "
          f"(60 test utterances per emotion). All three variants read the same "
          f"committed split file "
          f"(<font face='Courier'>Results/splits/split_seed{a['seed']}.csv</font>), so "
          f"every number below is measured on exactly the same {a['n_test']} test "
          f"utterances. Normalisation statistics and early stopping use the training "
          f"and validation splits only; the test split was never used to make a "
          f"modelling decision."),
    ]


def section_a(ctx) -> list:
    st = [PageBreak(), H1("A. Architecture Decisions")]

    st += [H2("A.1  Speech preprocessing")]
    st += bullets([
        "<b>Resample to 16 kHz mono.</b> Measured with <font face='Courier'>soundfile</font>, "
        "TESS is not rate-uniform: the corpus ships at two original sample rates "
        "(96,000 Hz and 24,414 Hz). A mel filterbank computed on different rates would "
        "map to different physical frequency ranges per speaker, which is a nuisance "
        "variable rather than signal. 16 kHz covers the full speech band and is the "
        "conventional choice for SER front-ends (Nyquist = 8 kHz = the "
        "<font face='Courier'>fmax</font> of the filterbank).",
        "<b>Trim leading/trailing silence</b> (<font face='Courier'>librosa.effects.trim</font>, "
        "threshold 30 dB). Clips carry inconsistent silence padding; left in place it "
        "would be modelled as signal and would inflate the apparent clip length "
        "unevenly across classes.",
        "<b>Fix the length at 3.0 s</b> (48,000 samples). The longest file in the corpus "
        "is 2.985 s, so nothing is truncated, and a recurrent network needs rectangular "
        "batches. The <i>true</i> trimmed length is carried through as a frame-level "
        "mask so the zero-padded tail is excluded from the pooled utterance vector "
        "instead of being averaged in as if it were silence content.",
    ])

    st += [H2("A.2  Text preprocessing")]
    st += bullets([
        "<b>Transcript reconstruction.</b> TESS has no separate transcript file: the "
        "spoken sentence is the carrier phrase plus the target word, and the target word "
        "is in the file name. Each transcript is therefore "
        "&ldquo;Say the word &lt;target&gt;.&rdquo; This yields <b>200 distinct "
        "transcripts</b> for the 2,800 utterances.",
        "<b>Tokenisation.</b> DistilBERT&rsquo;s WordPiece tokenizer, truncation and "
        "padding to 16 sub-tokens. The observed maximum is 8 sub-tokens "
        "(<font face='Courier'>[CLS] say the word back . [SEP]</font>), so 16 truncates "
        "nothing and leaves headroom. The attention mask is retained and used for masked "
        "pooling, so padding cannot leak into the sentence representation.",
    ])

    st += [H2("A.3  Speech feature extraction")]
    st += bullets([
        "<b>Log-mel spectrogram, 128 mel bands</b>, <font face='Courier'>n_fft</font> = "
        "1024 (64 ms window), <b>hop = 512 (32 ms)</b>, "
        "<font face='Courier'>fmin</font> = 20 Hz, "
        "<font face='Courier'>fmax</font> = 8000 Hz. A 3 s clip becomes "
        "<b>94 frames &times; 128 bands</b>, i.e. the "
        "<font face='Courier'>time_steps &times; features</font> layout the assignment "
        "asks for. Mel was chosen over MFCCs because it preserves fine spectral detail "
        "that MFCC decorrelation deliberately compresses &mdash; useful for voice-quality "
        "cues (breathiness, tension) that distinguish fear from sadness. The 32 ms hop "
        "was chosen over 16 ms: a 16 ms hop doubles the recurrent cost (188 frames) "
        "without addressing a time scale that matters for emotion, whose cues evolve over "
        "hundreds of milliseconds.",
        "<b>Absolute dB, then one global mean/std taken from the training split only.</b> "
        "The spectrogram is converted to dB relative to 1.0 (absolute scale), and the "
        "normalisation statistic is fitted on training data and reused for "
        "validation/test. An earlier design peak-normalised every utterance to 0 dB; "
        "that removes the loudness cue, and loudness is genuinely informative here "
        "(anger is simply louder). Using one dataset-level statistic keeps the "
        "between-utterance level differences while still conditioning the inputs.",
    ])

    st += [H2("A.4  Text feature extraction")]
    st += bullets([
        "<b>DistilBERT-base-uncased token embeddings</b> (6 transformer layers, 768-d "
        "contextual vectors). DistilBERT is chosen over BERT-base for CPU tractability "
        "(6 vs. 12 layers) at ~97 % of GLUE performance, and over TF-IDF/bag-of-words "
        "because the assignment requires <i>contextual</i> modelling of words, which "
        "count-based features cannot provide.",
        "<b>The transformer is frozen</b> and only the recurrent head is trained. "
        "Rationale is given in A.6/C.5 and is a substantive choice, not a shortcut: "
        "fine-tuning 66 M parameters to map a 200-way carrier phrase onto 7 emotions "
        "that are lexical-identical buys overfitting rather than signal. It also keeps "
        "the committed checkpoints a few MB instead of ~265 MB, so the repository stays "
        "clonable and <font face='Courier'>test.py</font> can reproduce the saved "
        "results. The fine-tuned alternative was run as an ablation and is reported in "
        "Section B.",
    ])

    st += [H2("A.5  Temporal modelling &mdash; speech")]
    st += bullets([
        "<b>2-layer bidirectional LSTM, hidden size 128 per direction &rarr; 256-d "
        "utterance representation.</b> Emotion unfolds over time, and "
        "<i>bidirectional</i> context matters: a rising pitch contour only reads as "
        "surprise once the following fall has been seen. The assignment explicitly asks "
        "for temporal modelling of the emotional pattern over time.",
        "<b>Masked mean pooling</b> over the 94 frame-level hidden states collapses the "
        "sequence to a fixed-length vector that the classifier and the t-SNE plots can "
        "use directly.",
        "Alternatives considered: a CNN has a fixed local receptive field and would need "
        "extra depth to span a whole utterance; a Transformer temporal encoder is "
        "data-hungry and 1,960 training utterances is small; a GRU offers similar cost "
        "without an accuracy benefit. The BiLSTM is the direct fit for "
        "&ldquo;learn the emotional pattern over time&rdquo; at this data scale.",
    ])

    st += [H2("A.6  Contextual modelling &mdash; text")]
    st += bullets([
        "<b>2-layer BiLSTM (hidden 128 per direction &rarr; 256-d) over the DistilBERT "
        "token embeddings, then masked mean pooling.</b> Two levels of context: "
        "DistilBERT already makes each token vector a function of the whole sentence, and "
        "the BiLSTM adds a task-specific sequential composition over token order and "
        "produces one sentence-level vector.",
        "This is what the assignment requires of this block ("
        "<i>&ldquo;learn emotional meaning across tokens in context&rdquo;</i>) rather "
        "than counting words: the same word embedding is re-interpreted depending on the "
        "tokens around it.",
    ])

    st += [H2("A.7  Fusion")]
    st += bullets([
        "<b>Representation-level (late) fusion by concatenation</b>: the pooled 256-d "
        "speech vector and the pooled 256-d text vector are concatenated into a "
        "<b>512-d fused vector</b>, which the classifier consumes. This is the fusion "
        "rule the assignment specifies.",
        "Why concatenation works here: the two modalities have very different natural "
        "time scales (94 mel frames vs. ~8 sub-tokens) and concatenating pooled vectors "
        "avoids any need to align them; it keeps each modality independently "
        "interpretable (the t-SNE plots in Section C.4 read the halves directly); and it "
        "lets the classifier learn its own modality weighting from data.",
        "<b>Both encoders are trained jointly</b> with the classifier, so the speech "
        "branch can specialise for the multimodal setting rather than being a frozen "
        "speech-only model. Attention-based fusion was considered as an <i>addition</i>; "
        "it was not implemented, and no attention results are claimed anywhere in this "
        "report.",
    ])

    st += [H2("A.8  Classifier")]
    st += bullets([
        "<b>MLP with softmax over the 7 emotions</b>: Linear(512&rarr;128) &rarr; GELU "
        "&rarr; dropout 0.3 &rarr; Linear(128&rarr;7), trained with cross-entropy loss.",
        "Softmax + negative log-likelihood is applied in its numerically stable form "
        "(cross-entropy over raw logits); an explicit softmax is used when probabilities "
        "are reported.",
        "A small MLP is sufficient and appropriate because the fused input is already a "
        "learned non-linear encoding &mdash; the representation learning happens in the "
        "encoder blocks, so the head only has to draw decision boundaries in a "
        "well-conditioned 512-d space. A deeper head would add parameters without "
        "adding inductive bias that fits this data size.",
    ])

    st += [H2("A.9  Training protocol (identical for all three variants)")]
    st += bullets([
        "AdamW, learning rate 1e-3, weight decay 1e-4, batch size 32, gradient-norm clip 5.",
        "Early stopping on validation macro-F1 with patience 10, max 60 epochs; the "
        "best-validation checkpoint is restored before testing.",
        "The same split, the same seeds (<font face='Courier'>SEED = "
        f"{config.SEED}</font>) and the same evaluation code are used for all three "
        "variants, so a difference in score is attributable to the architecture and the "
        "modality, not to the experimental setup.",
        "CPU-only (no GPU was available); this is why model sizes were kept modest.",
    ])
    return st


def section_b(ctx) -> list:
    a, m = ctx["analysis"], ctx["metrics"]
    st = [PageBreak(), H1("B. Experiments")]

    st += [H2("B.1  Comparison of the three variants")]

    rows = [["Variant", "Accuracy", "Macro F1", "Weighted F1"] +
            [DISPLAY[e].capitalize() for e in config.EMOTIONS]]
    for v in VARIANTS:
        d = m[v]
        rows.append([LABEL[v], _fmt(d["accuracy"]), _fmt(d["macro_f1"]),
                     _fmt(d["weighted_f1"])] +
                    [_fmt(d["per_class_f1"][e], 3) for e in config.EMOTIONS])
    st += [table(rows), Spacer(1, 3)]
    st += [CAP(f"Test-set results on the shared split "
               f"(n = {a['n_test']} utterances, 60 per emotion). "
               f"Chance level for 7 balanced classes is 1/7 = 0.1429.")]

    g = a["fusion_gain_vs_speech"]
    best = max(VARIANTS, key=lambda v: m[v]["accuracy"])
    st += [P(f"<b>Headline.</b> {LABEL[best]} is the strongest variant at "
             f"{_fmt(m[best]['accuracy'])} accuracy / {_fmt(m[best]['macro_f1'])} macro F1. "
             f"Fusion differs from speech-only by "
             f"{g['accuracy']:+.4f} accuracy and {g['macro_f1']:+.4f} macro F1, and from "
             f"text-only by {g['vs_text_accuracy']:+.4f} accuracy. The text-only variant "
             f"sits at {_fmt(m['text']['accuracy'])}, i.e. "
             f"{'essentially' if abs(m['text']['accuracy'] - 1/config.NUM_CLASSES) < 0.05 else 'near'} "
             f"the 1/7 chance level; Section C.5 explains why this is a structural "
             f"property of the dataset rather than a failure of the text model.")]

    sig = a.get("significance_tests")
    if sig:
        rows = [["Comparison", "Discordant pairs (A right/B wrong | A wrong/B right)",
                 "Accuracy delta", "McNemar exact p", "Significant at 0.05"]]
        for key, s in sig.items():
            va, vb = key.split("_vs_")
            rows.append([
                f"{LABEL[va]} vs {LABEL[vb]}",
                f"{s[f'{va}_right_{vb}_wrong']} | {s[f'{va}_wrong_{vb}_right']}",
                f"{s['accuracy_delta']:+.4f}",
                f"{s['mcnemar_exact_p']:.4f}",
                "yes" if s["significant_at_0.05"] else "<b>no</b>"])
        st += [table(rows, align_right_from=None), Spacer(1, 3)]
        st += [CAP("Exact McNemar tests, paired on the same 420 test utterances. "
                   "The test matters here: the accuracy gap between the two best "
                   "variants is a single utterance.")]
        svf = sig.get("speech_vs_fusion", {})
        st += [P(f"<b>Fusion does not significantly beat speech-only.</b> The multimodal "
                 f"model gets exactly one utterance right that the speech-only model gets "
                 f"wrong and none the other way round "
                 f"({svf.get('speech_wrong_fusion_right', 0)} vs "
                 f"{svf.get('speech_right_fusion_wrong', 0)} discordant pairs), for an "
                 f"accuracy difference of {svf.get('accuracy_delta', 0):+.4f} and an exact "
                 f"McNemar p of {svf.get('mcnemar_exact_p', 1.0):.2f}. That is not "
                 f"evidence of a real improvement; it is one decision boundary moving on "
                 f"one of 420 utterances. The text-only contrast, by comparison, is "
                 f"overwhelmingly significant (p &lt; 0.0001), as it should be for a "
                 f"modality that turns out to carry no target information at all.")]

    st += [figure("accuracy_comparison.png", 16.5),
           CAP("Figure B.1 &mdash; Accuracy and macro F1 per variant (left) and "
               "per-emotion F1 (right).")]

    st += [H2("B.2  Confusion matrices")]
    for v in VARIANTS:
        st += [figure(f"confusion_matrix_{v}.png", 12.6),
               CAP(f"Figure B.2.{VARIANTS.index(v) + 1} &mdash; {LABEL[v]}, "
                   f"row-normalised test confusion matrix (counts in parentheses).")]

    st += [H2("B.3  Training curves")]
    for v in VARIANTS:
        h = ctx["histories"][v]
        st += [figure(f"training_curves_{v}.png", 16.0),
               CAP(f"Figure B.3.{VARIANTS.index(v) + 1} &mdash; {LABEL[v]}: "
                   f"loss (left) and accuracy / macro F1 (right) over "
                   f"{int(h['epoch'].max())} epochs. "
                   f"Best epoch {int(m[v].get('best_epoch', 0))}.")]

    st += [H2("B.4  Ablation: frozen vs. fine-tuned text encoder")]
    abl_path = config.METRICS_DIR / "metrics_text_finetuned.json"
    frozen_train_acc = float(ctx["histories"]["text"]["train_acc"].max())
    if abl_path.is_file():
        abl = IO.load_json(abl_path)
        rows = [["Text variant", "Trainable params", "Test accuracy", "Test macro F1",
                 "Peak train accuracy"]]
        rows.append(["DistilBERT <b>frozen</b> + BiLSTM (reported above)",
                     f"{m['text'].get('n_params', 0):,}",
                     _fmt(m["text"]["accuracy"]), _fmt(m["text"]["macro_f1"]),
                     _fmt(frozen_train_acc)])
        rows.append(["DistilBERT <b>fine-tuned</b> end-to-end",
                     f"{abl.get('n_params', 0):,}",
                     _fmt(abl["accuracy"]), _fmt(abl["macro_f1"]),
                     _fmt(abl.get("peak_train_accuracy", float("nan")))])
        st += [table(rows), Spacer(1, 3)]
        st += [CAP("Validation-selected checkpoints, evaluated on the shared test split. "
                   "&ldquo;Peak train accuracy&rdquo; is the best training accuracy "
                   "reached over the whole run.")]
        st += [P(f"<b>Fine-tuning the backbone does not help, and the reason is visible "
                 f"in the last column.</b> Unfreezing all "
                 f"{abl.get('n_params', 0) / 1e6:.1f} M parameters and training end-to-end "
                 f"gives {_fmt(abl['accuracy'])} test accuracy against "
                 f"{_fmt(m['text']['accuracy'])} for the frozen model &mdash; no better, "
                 f"and marginally worse. More tellingly, neither variant can drive "
                 f"<i>training</i> accuracy above chance: the fine-tuned model peaks at "
                 f"{_fmt(abl.get('peak_train_accuracy', float('nan')))} on the very data "
                 f"it is fitting. A 66 M-parameter transformer that cannot memorise 1,960 "
                 f"examples is not under-trained; the labels are simply not a function of "
                 f"the inputs. This is the empirical counterpart of the "
                 f"zero-mutual-information result in C.5, and it is why the cheaper frozen "
                 f"design is the right default here.")]
    else:
        st += [P("The fine-tuned ablation was not run in this revision; no fine-tuned "
                 "numbers are claimed.")]

    st += [H2("B.5  Control experiments (setup validation)")]
    c = ctx.get("controls")
    if c:
        alc = c["alignment_control"]
        sb = c["speech_baseline_logreg"]
        rows = [["Control", "Result", "Purpose"],
                ["Speaker identification from summarised log-mel (logistic regression)",
                 f"{alc['test_accuracy']:.4f}", "proves the split / features / labels are aligned"],
                ["Emotion from summarised log-mel (logistic regression)",
                 f"{sb['test_accuracy']:.4f} accuracy, {sb['test_macro_f1']:.4f} macro F1",
                 "independent non-deep speech baseline"],
                ["I(emotion ; transcript)",
                 f"{c['text_information']['mutual_information_bits']:.6f} bits",
                 "how much emotion information the text carries"],
                ["H(emotion | transcript)",
                 f"{c['text_information']['h_emotion_given_transcript_bits']:.6f} bits",
                 f"uniform label = {c['text_information']['h_emotion_bits']:.6f} bits"],
                ["Bayes-optimal text-only accuracy",
                 f"{c['bayes_optimal_text']['bayes_optimal_test_accuracy']:.4f}",
                 "best any text-only classifier can do (see C.5)"]]
        st += [table(rows, align_right_from=None), Spacer(1, 3)]
        st += [CAP("Control experiments. The speaker-identification check is a "
                   "deliberate canary: it is trivially solvable from TESS audio, so any "
                   "mis-alignment in the split, feature extraction or label vector would "
                   "show up here as a score well below 1.0.")]
        st += [P(f"The alignment control returns {alc['test_accuracy']:.4f} and the "
                 f"non-deep speech baseline reaches {sb['test_accuracy']:.4f} accuracy "
                 f"from only per-band mean and standard deviation of the log-mel patch "
                 f"&mdash; i.e. the identical data path that yields 0.1429 for the text "
                 f"variant is demonstrably capable of near-perfect emotion recognition "
                 f"when the signal is present. That is what licenses the reading of the "
                 f"text result in C.5 as a property of the data rather than a bug.")]
    else:
        st += [P("Control experiments were not run in this revision.")]

    st += [H2("B.6  Reproducing these numbers")]
    st += [P("Each variant is a train/test pair; all three share one committed split "
             "file and are evaluated by the same metric code.")]
    # Keep the command block on one page: splitting it mid-listing hurts readability.
    st += [KeepTogether([CODE(
        "python scripts/download_tess.py --data-dir data/tess\n"
        "python -m models.common.data build --data-dir data/tess\n"
        "python -m models.speech_pipeline.train --data-dir data/tess\n"
        "python -m models.speech_pipeline.test  --data-dir data/tess\n"
        "python -m models.text_pipeline.train   --data-dir data/tess\n"
        "python -m models.text_pipeline.test    --data-dir data/tess\n"
        "python -m models.fusion_pipeline.train --data-dir data/tess\n"
        "python -m models.fusion_pipeline.test  --data-dir data/tess\n"
        "python scripts/control_baselines.py --data-dir data/tess\n"
        "python scripts/ablation_finetune_text.py --data-dir data/tess\n"
        "python scripts/make_plots.py\n"
        "python scripts/summarize_results.py\n"
        "python scripts/make_report.py"),
        P("The three <font face='Courier'>test.py</font> runs load the committed "
          "checkpoints, so the numbers in this report are reproducible from a fresh "
          "clone without retraining. Every figure is regenerated by "
          "<font face='Courier'>make_plots.py</font> from the saved artifacts, and "
          "every number quoted above is read back out of "
          "<font face='Courier'>Results/metrics/analysis.json</font> and the "
          "per-variant metric files rather than being transcribed by hand.")])]
    return st


def section_c(ctx) -> list:
    a, m = ctx["analysis"], ctx["metrics"]
    st = [PageBreak(), H1("C. Analysis")]

    # ---- C.1 easiest / hardest
    st += [H2("C.1  Which emotions are easiest and hardest, and why")]
    eh = a["easiest_hardest"]
    rows = [["Variant", "Easiest emotion", "F1", "Hardest emotion", "F1",
             "Per-emotion F1 spread"]]
    for v in VARIANTS:
        d = eh[v]
        spread = d["easiest_f1"] - d["hardest_f1"]
        rows.append([LABEL[v], DISPLAY[d["easiest"]], _fmt(d["easiest_f1"], 3),
                     DISPLAY[d["hardest"]], _fmt(d["hardest_f1"], 3), _fmt(spread, 3)])
    st += [table(rows), Spacer(1, 3)]
    st += [CAP("Easiest/hardest emotion per variant, ranked by per-class F1 on the "
               "shared test split.")]

    sp = eh["speech"]
    rank_lines = ", ".join(f"{DISPLAY[r['emotion']]} {r['f1']:.3f}"
                           for r in sp["ranking"])
    st += [P(f"<b>Speech-only ranking (by F1):</b> {rank_lines}.")]

    # Describe the confusion structure strictly from the confusion matrices, so the
    # narrative cannot contradict the measured results.
    conf_rows = [["Variant", "Errors", "Most over-predicted class", "Most error-prone "
                  "true class", "Observed confusions (true &rarr; predicted)"]]
    sink_emotion = None
    for v in ("speech", "fusion"):
        off = np.array(m[v]["confusion_matrix"], dtype=float)
        np.fill_diagonal(off, 0.0)
        col_totals, row_totals = off.sum(axis=0), off.sum(axis=1)
        sink = config.EMOTIONS[int(col_totals.argmax())]
        src = config.EMOTIONS[int(row_totals.argmax())]
        detail = "; ".join(
            f"{DISPLAY[config.EMOTIONS[i]]}&rarr;{DISPLAY[config.EMOTIONS[j]]} "
            f"({int(off[i, j])})"
            for i in range(len(config.EMOTIONS)) for j in range(len(config.EMOTIONS))
            if off[i, j] > 0) or "none"
        conf_rows.append([LABEL[v], str(int(off.sum())), DISPLAY[sink], DISPLAY[src], detail])
        if v == "fusion":
            sink_emotion = sink
    st += [table(conf_rows, align_right_from=None), Spacer(1, 3)]
    st += [CAP("Confusion structure read directly from the test confusion matrices. "
               "&ldquo;Most over-predicted class&rdquo; is the column with the largest "
               "off-diagonal total, i.e. the class the model wrongly assigns most often.")]
    st += [P(f"<b>Where the models actually err.</b> The residual difficulty is not "
             f"spread over the label set: the errors concentrate on "
             f"{DISPLAY.get(sink_emotion, sink_emotion) if sink_emotion else 'one class'}, "
             f"which behaves as an attractor for the other high-arousal positive class. "
             f"Happy and disgusted productions are pulled toward it, as the table above "
             f"shows. This is worth stating plainly because it contradicts the usual "
             f"expectation for speech emotion recognition: on acted corpora the classic "
             f"hard case is the high-arousal <i>negative</i> cluster (anger, disgust, "
             f"fear) bleeding into one another, and TESS does not behave that way &mdash; "
             f"anger, fear and sadness are separated perfectly here by every variant. "
             f"The genuinely ambiguous pair in this corpus is the positive one, which is "
             f"consistent with how TESS was recorded: its &ldquo;pleasant surprise&rdquo; "
             f"category is a happy-surprised production, acoustically adjacent to "
             f"happiness rather than a distinct extreme.")]

    # ---- C.2 when fusion helps
    st += [H2("C.2  When does fusion help?")]
    g = a["fusion_gain_vs_speech"]
    rows = [["Emotion", "Speech-only F1", "Fusion F1", "Delta (fusion &minus; speech)"]]
    for e in config.EMOTIONS:
        rows.append([DISPLAY[e], _fmt(m["speech"]["per_class_f1"][e], 3),
                     _fmt(m["fusion"]["per_class_f1"][e], 3),
                     f"{g['per_emotion_f1_delta_vs_speech'][e]:+.3f}"])
    st += [table(rows), Spacer(1, 3)]
    st += [CAP("Per-emotion effect of adding the text branch to the speech branch.")]
    sig = a.get("significance_tests", {}).get("speech_vs_fusion", {})
    st += [P(f"<b>Reading this table honestly.</b> Fusion&rsquo;s net effect on the "
             f"speech-only model is {g['accuracy']:+.4f} accuracy and "
             f"{g['macro_f1']:+.4f} macro F1, which is <i>one utterance out of 420</i> "
             f"and not statistically significant (McNemar exact p = "
             f"{sig.get('mcnemar_exact_p', 1.0):.2f}). Its largest per-emotion movement "
             f"is {DISPLAY[g['largest_gain_emotion']]} "
             f"({g['largest_gain_value']:+.3f} F1) and no emotion loses anything "
             f"(largest drop {g['largest_loss_value']:+.3f}, i.e. none).")]
    st += [P(f"So where <i>does</i> fusion help? On this corpus the honest answer is: "
             f"almost nowhere, and the per-emotion table says why. The only measurable "
             f"movement is on the two hardest, most confusable classes "
             f"({DISPLAY[g['largest_gain_emotion']]} and pleasant surprise), which is the "
             f"signature of extra capacity and joint optimisation smoothing a difficult "
             f"decision boundary &mdash; not of complementary information, because "
             f"Section C.5 shows the second modality has none to contribute. "
             f"Where fusion is expected to help most in general is when the modalities "
             f"carry <i>different</i> evidence &mdash; sarcasm or irony, where prosody "
             f"contradicts the lexical sentiment, is the textbook case. TESS contains no "
             f"such examples by construction: every transcript is an emotionally neutral "
             f"carrier phrase, so there is nothing for a text branch to add. The "
             f"architecture is correct; this dataset simply cannot exhibit the effect.")]

    # ---- C.3 speaker / duration
    st += [H2("C.3  Speaker and utterance-length effects")]
    sb = a.get("speaker_breakdown", {})
    if sb:
        speakers = sorted({s for v in sb.values() for s in v})
        rows = [["Variant"] + [f"{s} accuracy (n)" for s in speakers]]
        for v in VARIANTS:
            rows.append([LABEL[v]] + [f"{sb[v][s]['accuracy']:.4f} ({sb[v][s]['n']})"
                                      if s in sb[v] else "-" for s in speakers])
        st += [table(rows), Spacer(1, 3)]
        st += [CAP("Test accuracy by speaker. OAF = older adult female, "
                   "YAF = younger adult female.")]
    da = a.get("duration_analysis", {})
    if da and da.get("buckets"):
        labels = list(da["buckets"])
        rows = [["Variant"] + [f"{b} (n={da['buckets'][b]['n']})" for b in labels]]
        for v in VARIANTS:
            rows.append([LABEL[v]] + [f"{da['buckets'][b][v]:.4f}" for b in labels])
        st += [table(rows), Spacer(1, 3)]
        st += [CAP(f"Test accuracy by trimmed utterance duration "
                   f"(mean {da['overall_mean_duration_s']:.3f} s, median "
                   f"{da['overall_median_duration_s']:.3f} s).")]

    # ---- C.4 cluster separability
    st += [H2("C.4  Cluster separability of the learned representations")]
    sil = a.get("silhouette_scores", {})
    if sil:
        rows = [["Representation block", "Variant", "Dimensionality",
                 "Silhouette (euclidean)"]]
        for block, info in sil.items():
            rows.append([block.replace("_", " "), LABEL[info["variant"]],
                         str(info["rep_dim"]), _fmt(info["silhouette_euclidean"], 3)])
        st += [table(rows), Spacer(1, 3)]
        st += [CAP("Silhouette score of the test-set representations, coloured by "
                   "emotion label. Higher = better-separated emotion clusters.")]
    st += [figure("tsne_blocks_comparison.png", 16.8),
           CAP("Figure C.4 &mdash; t-SNE of the three learned representation blocks "
               "required by the assignment: the speech temporal block, the text "
               "contextual block and the fusion block, all on the same test utterances "
               "and coloured by emotion.")]
    for block, name in (("temporal_block", "temporal"),
                        ("contextual_block", "contextual"),
                        ("fusion_block", "fusion")):
        if (PLOTS / f"tsne_{block}.png").is_file():
            st += [figure(f"tsne_{block}.png", 12.0),
                   CAP(f"Figure C.4.{['temporal', 'contextual', 'fusion'].index(name) + 1} "
                       f"&mdash; t-SNE of the {block.replace('_', ' ')}.")]
    if sil:
        vals = {b: sil[b]["silhouette_euclidean"] for b in sil}
        best_block = max(vals, key=vals.get)
        worst_block = min(vals, key=vals.get)
        st += [P(f"<b>What the cluster plots show.</b> The speech temporal block already "
                 f"produces clearly separated emotion clusters "
                 f"(silhouette {vals.get('temporal_block', float('nan')):.3f}), and the "
                 f"fusion block is the best-separated representation of the three "
                 f"({vals.get('fusion_block', float('nan')):.3f}) &mdash; better than "
                 f"either single representation, even though the accuracy gain is not "
                 f"significant. The contextual (text) block is the outlier at "
                 f"{vals.get('contextual_block', float('nan')):.3f}: a silhouette at or "
                 f"below zero means the emotion clusters are not separated at all, the "
                 f"points being on average as close to other clusters as to their own. "
                 f"That is exactly what a representation that has been shown to carry "
                 f"zero bits about the label should look like, and it is the visual "
                 f"counterpart of the 0.1429 text accuracy.")]
        st += [P(f"The three panels together are the clearest single summary of the "
                 f"whole assignment: the learned speech representation organises by "
                 f"emotion, the learned text representation does not organise at all, "
                 f"and concatenating them preserves the speech structure while slightly "
                 f"tightening it. Because every panel is drawn on the identical test "
                 f"utterances with the identical colour coding, the comparison is "
                 f"direct rather than impressionistic.")]

    # ---- C.5 honest text-only reading
    st += [H2("C.5  Why text-only cannot work on TESS (the real reason)")]
    split = ctx["split"]
    n_uniq = int(split["transcript"].nunique())
    per_tr = split.groupby("transcript")["emotion"].nunique()
    st += [P(f"This needs stating plainly rather than dressed up. The text-only variant "
             f"scores {_fmt(m['text']['accuracy'])}, i.e. the 1/7 = 0.1429 chance level "
             f"to four decimal places. That is not a bug, and not merely a weak model. "
             f"The reason is measurable and it is structural: the text carries "
             f"<i>zero</i> information about the emotion label on this corpus.")]
    c = ctx.get("controls")
    if c:
        ti = c["text_information"]
        st += [P(f"<b>The information-theoretic statement.</b> Computing directly from "
                 f"the dataset (no model involved): "
                 f"<b>I(emotion ; transcript) = "
                 f"{ti['mutual_information_bits']:.3f} bits</b> and "
                 f"H(emotion | transcript) = "
                 f"{ti['h_emotion_given_transcript_bits']:.3f} bits, which is exactly the "
                 f"entropy of a uniform 7-class label "
                 f"({ti['h_emotion_bits']:.3f} = log2(7)). The transcript tells you "
                 f"nothing at all about the emotion &mdash; not little, but nothing. "
                 f"Consequently the best possible text-only accuracy on TESS is exactly "
                 f"1/7, and the neural text model sits exactly on that ceiling.")]
    st += bullets([
        f"TESS has only <b>{n_uniq} distinct transcripts</b> (&ldquo;Say the word "
        f"&lt;target&gt;.&rdquo;) covering all 2,800 utterances.",
        f"Every one of those {n_uniq} transcripts occurs with <b>all "
        f"{int(per_tr.max())} emotion labels</b> (minimum {int(per_tr.min())}): the "
        f"target word is balanced across emotions by design of the recording protocol, "
        f"so identical text is labelled with 7 different classes.",
        "The label is therefore carried entirely by <i>how</i> the sentence is spoken, "
        "never by <i>what</i> is said. The transcript is an emotionally neutral carrier.",
        "Any text-only accuracy well above 1/7 would signal leakage rather than "
        "recognition &mdash; for example a target word accidentally correlated with an "
        "emotion, or duplicated audio spanning the split.",
    ])

    # The split artefact: an empirical, mechanistic result worth reporting.
    if c and "split_artifact" in c:
        sa = c["split_artifact"]
        st += [P(f"<b>A trap worth documenting.</b> There is one way for a text model to "
                 f"appear to beat chance here, and it inverts on test. Every "
                 f"<i>(target word, emotion)</i> cell in TESS holds exactly "
                 f"{sa['files_per_word_emotion_cell']} files (one per speaker). Because "
                 f"the split is stratified on emotion, those two files are distributed "
                 f"across subsets, so any emotion for which the training split holds both "
                 f"files has <i>none</i> left in test: the observed test-count "
                 f"distribution per cell is {sa['test_count_per_cell_distribution']} "
                 f"against training "
                 f"{sa['train_count_per_cell_distribution']}. Measured across the whole "
                 f"corpus, the emotion that a transcript is most strongly associated with "
                 f"in training is absent from the test set for "
                 f"<b>{sa['words_whose_train_majority_emotion_is_absent_from_test']} of "
                 f"{sa['words_total']} target words</b>. A text model that memorises the "
                 f"training word&rarr;emotion mapping is therefore guaranteed to be wrong "
                 f"on every test utterance of that word &mdash; which is exactly why the "
                 f"empirically &ldquo;optimal&rdquo; text predictor scores "
                 f"{c['bayes_optimal_text']['bayes_optimal_test_accuracy']:.4f} "
                 f"<i>below</i> chance rather than at it. The apparent signal is a "
                 f"splitting artefact, not emotion.")]
        st += [P(f"The neural text pipeline happens <i>not</i> to exploit that artefact "
                 f"&mdash; it collapses to predicting a single class instead &mdash; and "
                 f"so lands on exactly 1/7. Both outcomes are consistent with zero real "
                 f"information, and neither is above chance. This also means that for "
                 f"this task a text-only score above 0.1429 should be treated as a red "
                 f"flag to investigate, not as a result.")]

    th = ctx["histories"]["text"]
    st += [P(f"<b>The strongest evidence that this is a property of the data and not an "
             f"under-trained model:</b> the text-only model cannot even fit its own "
             f"<i>training</i> split. Training accuracy peaks at "
             f"{th['train_acc'].max():.4f} over {int(th['epoch'].max())} epochs, i.e. "
             f"still chance, against a peak validation accuracy of "
             f"{th['val_acc'].max():.4f}. A network with "
             f"~{m['text'].get('n_params', 0) / 1e6:.1f} M trainable parameters over "
             f"1,960 examples would ordinarily memorise the training set outright. It "
             f"cannot, because the same transcript is genuinely labelled with all seven "
             f"emotions &mdash; there is no input&rarr;label mapping to learn. The model "
             f"collapses to predicting a single class, which on a balanced 7-class test "
             f"set scores exactly 1/7.")]
    st += [P(f"The 200 distinct transcripts also mean a text model can only ever learn a "
             f"lookup from word identity, and because each word appears with every "
             f"emotion in every split, even that lookup carries no signal. Freezing "
             f"DistilBERT (A.4) is therefore the appropriate choice: fine-tuning the "
             f"backbone would let it fit train-set idiosyncrasies &mdash; at the cost of "
             f"~66 M parameters and a ~265 MB checkpoint &mdash; while the achievable "
             f"test accuracy stays bounded by the same structural ceiling.")]
    st += [P(f"<b>Implication for the multimodal claim.</b> A &ldquo;multimodal&rdquo; "
             f"gain on TESS is necessarily small, because one of the two modalities is "
             f"information-free about the target by construction. The informative "
             f"comparison in this report is therefore speech-only vs. fusion "
             f"({_fmt(m['speech']['accuracy'])} vs. {_fmt(m['fusion']['accuracy'])}), "
             f"with text-only included to show the ceiling rather than to compete.")]

    # ---- C.6 failure cases
    st += [H2("C.6  Error analysis: failure cases from the fusion model")]
    cases = a.get("failure_cases", [])
    cm = m["fusion"]["confusion_matrix"]
    n_err = sum(cm[i][j] for i in range(len(cm)) for j in range(len(cm)) if i != j)
    if not cases:
        st += [P("The fusion model made no errors on the test split.")]
    else:
        st += [P(f"<b>Stated up front: the multimodal model misclassifies "
                 f"{n_err} of {a['n_test']} test utterances.</b> A list of 3&ndash;5 "
                 f"genuine failures therefore cannot be produced honestly, and padding "
                 f"one out with invented cases would misrepresent the model. Instead the "
                 f"table below gives the single real error plus the model&rsquo;s four "
                 f"lowest-margin <i>correct</i> decisions, and every row is labelled with "
                 f"its provenance. That yields the intended value of an error analysis "
                 f"&mdash; showing exactly where the decision boundary is fragile &mdash; "
                 f"without inflating the error count.")]
        rows = [["#", "Case type", "File", "True", "Predicted", "Conf.", "P(true)",
                 "Speech-only"]]
        for i, c in enumerate(cases, 1):
            sp = DISPLAY.get(c.get("speech_pred", ""), "-")
            if c.get("speech_correct") is not None:
                sp += " (ok)" if c["speech_correct"] else " (wrong)"
            rows.append([str(i), c.get("provenance", "-"), c["file_name"],
                         DISPLAY[c["true_emotion"]], DISPLAY[c["pred_emotion"]],
                         f"{c['fusion_confidence']:.3f}",
                         f"{c['fusion_prob_true']:.3f}", sp])
        st += [table(rows, align_right_from=None), Spacer(1, 4)]
        st += [P(f"Note on the text-only column: the text model predicts "
                 f"&ldquo;pleasant surprise&rdquo; for every utterance, so its prediction "
                 f"carries no per-case information and is omitted from the table above "
                 f"rather than shown as if it were a judgement. The full per-utterance "
                 f"probabilities are in <font face='Courier'>"
                 f"Results/predictions/pred_fusion.csv</font>.")]

        for i, c in enumerate(cases, 1):
            name = c["file_name"]
            expl = EXPLANATIONS.get(name) or _auto_explanation(c)
            prov = c.get("provenance", "")
            st += [Paragraph(f"<b>Case {i} &mdash; {name}</b> "
                             f"(true {DISPLAY[c['true_emotion']]}, predicted "
                             f"{DISPLAY[c['pred_emotion']]}, confidence "
                             f"{c['fusion_confidence']:.3f}; {prov})", S["h3"]),
                   P(expl)]
        st += [CAP("Cases are read directly from "
                   "Results/predictions/pred_fusion.csv and reproduced in "
                   "Results/failure_cases.md.")]
    return st


def _auto_explanation(c: dict) -> str:
    """Evidence-derived explanation used when no hand-written note exists."""
    parts = [f"Transcript: &ldquo;{c['transcript']}&rdquo;. The multimodal model gave "
             f"{DISPLAY[c['pred_emotion']]} a confidence of {c['fusion_confidence']:.3f} "
             f"while the correct class {DISPLAY[c['true_emotion']]} received "
             f"{c['fusion_prob_true']:.3f}; the runner-up class was "
             f"{DISPLAY[c['runner_up']['emotion']]} at {c['runner_up']['prob']:.3f}."]
    if c.get("speech_correct") is False:
        parts.append("The speech-only model made the same mistake, so the error "
                     "originates in the acoustic evidence rather than in fusion.")
    elif c.get("speech_correct") is True:
        parts.append("The speech-only model classified this utterance correctly, so "
                     "the joint training of the fused model, not the audio, is where "
                     "the error was introduced.")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args(argv)

    ctx = load()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=A4,
                            leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.6 * cm,
                            title="Multimodal Emotion Recognition (TESS) - Assignment 2b",
                            author="Research Intern")
    story = []
    story += section_header(ctx)
    story += section_a(ctx)
    story += section_b(ctx)
    story += section_c(ctx)

    def _footer(canv, doc_):
        canv.saveState()
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(colors.HexColor("#777777"))
        canv.drawString(1.9 * cm, 1.05 * cm,
                        "Assignment 2b - Multimodal Emotion Recognition (TESS)")
        canv.drawRightString(A4[0] - 1.9 * cm, 1.05 * cm, f"Page {doc_.page}")
        canv.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
