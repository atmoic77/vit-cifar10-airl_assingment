# Vision Transformer on CIFAR-10 — AIRL/IISc Candidate Evaluation

**Candidate Name:** Shreyash Dwivedi <br>
**College:** The National Institute of Engineering, Mysuru <br>
**Semester:** 5th <br>
**Branch:** Information Science and Engineering <br>
**Email:** dwivedishreyash79@gmail.com <br>
**github:** https://github.com/atmoic77/

## A Vision Transformer implemented entirely from scratch in PyTorch — patch embedding, positional
embeddings, CLS token, multi-head self-attention, MLP, residual connections, and layer
normalization all hand-built, no pre-built ViT/Transformer modules, no pre-trained weights —
trained on CIFAR-10 for 10-class image classification, with two architectural ablations testing
targeted fixes for ViT's lack of locality inductive bias.

---

## Background

**CIFAR-10** is a benchmark image classification dataset of 60,000 tiny 32×32 color images
across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) —
50,000 for training, 10,000 for testing. Its small image size and modest dataset size make it a
useful, fast-to-iterate benchmark, but also a genuinely hard setting for models with no
built-in assumptions about images, as explained below.

**Vision Transformers (ViT)** adapt the Transformer architecture — originally built for text,
where a sentence is a sequence of word tokens — to images, by cutting an image into fixed-size
square patches (here, 4×4 pixels), flattening and linearly projecting each patch into an
embedding vector, and treating the resulting sequence of patch embeddings exactly like a
sequence of word tokens. A single extra learnable "CLS" token is prepended to this sequence and,
after passing through several self-attention layers, its final representation is used to
classify the whole image. Unlike a Convolutional Neural Network (CNN), which is built with a
built-in assumption that nearby pixels are related ("locality") and that a pattern means the
same thing regardless of where it appears in the image ("translation equivariance"),
self-attention makes no such assumption — every patch can directly attend to every other patch,
with no notion of "near" or "far" unless the model learns it from data. This makes ViT very
flexible, but also means it typically needs far more training data than a CNN to reach the same
accuracy — a gap that is especially pronounced on a small dataset like CIFAR-10, and is the
central problem this project investigates and attempts to partially close.

---

## 1. Results at a glance

| Model | Val Acc @ 100 epochs | Val Acc (final) | Test Acc | Epochs Trained |
|---|---|---|---|---|
| Baseline (standard patch embed) | 78.96% | 83.64% | 83.52% | 200 |
| **Ablation A — Overlapping patch embed (final model)** | 83.82% | **85.00%** | **84.82%** | 200 |
| Ablation B — Windowed attention | 83.22% | — | 83.20% | 100 |

**Seed:** 42 (fixed across `random`, `numpy`, `torch`, `torch.cuda`, with `cudnn.deterministic=True`)
**GPU:** Colab T4 (baseline epochs 1–157; Ablation A epochs 101–200) + Kaggle T4×2 (baseline
epochs 158–200; Ablation A epochs 1–100 and resumed portions; Ablation B full run)
**Approximate total runtime:** ~9–10 GPU-hours across baseline + both ablations combined
(see Section 5 for the per-run breakdown)

---

## 2. Architecture overview

```
                    ┌─────────────┐
                    │   Image     │  (B, 3, 32, 32)
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Patch Embedding      │  Conv2d → (B, 64, 384)
              │  (Baseline/A differ)   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  + CLS token            │
              │  + Positional Embedding │  (B, 65, 384)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Transformer Block ×7  │
              │  ┌───────────────────┐  │
              │  │ LayerNorm         │  │  (Baseline)
              │  │ Multi-Head Attn   │  │  (A: global)
              │  │      + residual   │  │  (B: 4×4 windows, shifted)
              │  ├───────────────────┤  │
              │  │ LayerNorm         │  │
              │  │ MLP (4× expand)   │  │
              │  │      + residual   │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Final LayerNorm       │
              │   Take CLS token        │
              │   Linear → 10 classes   │
              └────────────┬────────────┘
                           │
                           ▼
                    Logits (B, 10)
```

**Baseline:** `Conv2d(3, 384, kernel_size=4, stride=4)` — non-overlapping 4×4 patches, global
self-attention across all 65 tokens.

**Ablation A (final model):** patch embedding changed to `Conv2d(3, 384, kernel_size=8, stride=4,
padding=2)` — overlapping patches, everything else identical to baseline. See the green-highlighted
box in the diagram above.

**Ablation B:** attention restricted to local 4×4 token windows, shifted between blocks
(Swin-style), patch embedding unchanged from baseline. See the orange-highlighted box above.

---

## 3. Configuration

| Parameter | Value |
|---|---|
| Image size | 32×32 |
| Patch size | 4 (baseline/A patch embed), 4×4 windows (B attention) |
| Embedding dim | 384 |
| Depth | 7 transformer blocks |
| Attention heads | 12 |
| MLP expansion | 4× |
| Parameters | ~12.46M (baseline/A); comparable for B |
| Dropout | 0.0 |

---

## 4. Training recipe

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 → 1e-5, cosine decay |
| Warmup | 5 epochs, linear |
| Weight decay | 5e-5 |
| Batch size | 128 |
| Label smoothing | 0.1 |
| Augmentation | RandomCrop(pad=4), RandomHorizontalFlip, AutoAugment (CIFAR-10 policy) |
| Precision | Mixed precision (AMP) |
| Train/val split | 45,000 / 5,000 (from official 50,000 training images) |
| Test set | Official 10,000 images, evaluated once, at the end |
| Seed | 42 |

Recipe follows the reference configuration reported by
[omihub777/ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR) (see citations), adapted to this
from-scratch implementation.

---

## 5. Training runtime and platform notes

Training for both the baseline and Ablation A was interrupted partway through by Colab's
free-tier GPU usage limits, and resumed on Kaggle (and, for Ablation A, resumed again back on
Colab) using an epoch-level checkpoint/resume system (model, optimizer, scaler, and epoch number
saved every epoch to Google Drive / Kaggle Datasets). This is documented transparently in the
notebook at the point each interruption occurred, with the full log preserved for both segments
of each run.

| Run | Platform(s) | Approx. runtime |
|---|---|---|
| Baseline (200 epochs) | Colab T4 (ep. 1–157) → Kaggle T4×2 (ep. 158–200) | ~4h Colab + ~1.5h Kaggle |
| Ablation A (200 epochs) | Kaggle T4×2 (ep. 1–100) → Colab T4 (ep. 101–142) → Kaggle T4×2 (ep. 143–200) | ~2h Kaggle + ~1h Colab + ~1h Kaggle |
| Ablation B (100 epochs) | Kaggle T4×2 | ~1h |

All resumes used identical seed (42) and preserved optimizer/scaler state; only the LR scheduler
was rebuilt (not loaded from checkpoint) when a run's total epoch target changed mid-training
(e.g. Ablation A promoted from a 100-epoch ablation to a 200-epoch final model after outperforming
baseline at the 100-epoch checkpoint) — this is noted at the relevant cell in the notebook.

---

## 6. How to run this notebook in Colab
<br> <br>
**Note: the images in the notebook are loaded and disaplayed locally , it is mainly for representation and understanding purpose , while running the vit_cifar10.ipynb file ,the code structres will be full functional but images wont load again ,to refer visuals go through the (.ipynb) file once in starting then run it for further evaluation** 
<br> <br>
1. Open `vit_cifar10.ipynb` in Google Colab.
2. **Runtime → Change runtime type → GPU (T4)**.
3. Run all cells top to bottom (**Runtime → Run all**). The notebook will:
   - Download all pre-trained checkpoints (`best_model.pt` and `latest.pt` for baseline,
     Ablation A, and Ablation B) directly from publicly shared Google Drive links via `gdown`
     — no personal Google account authorization or Drive access required.
   - Build and unit-test each architecture component (attention, LayerNorm, patch embedding, etc.).
   - Load CIFAR-10 via `torchvision.datasets.CIFAR10(download=True)` from the official source
     and construct the 45k/5k/10k split.
   - Load the downloaded checkpoints for live verification of the reported results, rather than
     retraining from scratch — the `latest.pt` checkpoints ensure the training-loop cells
     correctly detect that training is already complete and skip straight to evaluation.
4. The final test-accuracy print statements for the baseline, Ablation A, and Ablation B are
   genuine, live executions against the downloaded best-model weights — not hardcoded values.
5. Ablation-specific cells (patch embedding variant, windowed attention variant) are clearly
   marked with markdown headers and can be run independently of the baseline sections.

**Note:** the notebook contains historical training logs (as text/markdown, not live executed
cells) for the portions of training that ran on Kaggle due to Colab GPU quota limits, clearly
labeled as such at each occurrence.

---

## 7. Analysis — why the architectural changes worked

See the notebook's Section 11 (Conclusion) for the full write-up. In short: plain ViT has no
built-in locality inductive bias — unlike a CNN, self-attention treats all patches as an
unordered set, so the model must learn spatial relationships entirely from data, and CIFAR-10's
50,000 low-resolution images provide too little signal to fully compensate from scratch.

- **Ablation A (overlapping patch embedding)** injects locality directly at the tokenization
  stage — adjacent patches share border pixels — without removing any of the model's existing
  attention capacity. This produced the largest gain tested (+4.86 points over baseline at the
  100-epoch comparison point) and was promoted to the final model.
- **Ablation B (windowed attention)** also clearly improved on baseline (+4.26 points),
  confirming locality is the limiting factor generally, not specific to one fix — but it
  slightly underperformed A, likely because restricting attention *trades away* global context
  to gain locality, whereas A's approach is purely additive. At only 64 tokens, the usual
  compute-efficiency motivation for windowing also doesn't apply at this scale.

---

## 8. Limitations and scope

- Ablation B was trained for 100 epochs only (not extended to 200), since Ablation A had already
  been established as the stronger candidate by that point — so B's *final*-epoch potential is
  not directly comparable to A or baseline on equal footing.
- A third ablation (relative positional bias, per Rethinking and Improving Relative Position
  Encoding for ViT) was considered but not run, given time constraints.
- Stronger augmentation (Mixup/CutMix) was not tested in combination with either architectural
  change — this is a natural next experiment, left for future work, and was kept out of scope
  to preserve a controlled comparison against a fixed baseline recipe.

---

## 9. Citations and resources used

**Papers**
- Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) — the original ViT
  paper; source of the patch embedding + CLS token + positional embedding architecture this
  implementation follows.
- Vaswani, A. et al. (2017). *Attention Is All You Need.*
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — source of the scaled dot-product /
  multi-head self-attention mechanism implemented from scratch here.
- Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted
  Windows.* [arXiv:2103.14030](https://arxiv.org/abs/2103.14030) — basis for Ablation B's
  windowed, shifted attention mechanism.
- Hassani, A. et al. (2021). *Escaping the Big Data Paradigm with Compact Transformers (CCT).*
  [arXiv:2104.05704](https://arxiv.org/abs/2104.05704) — motivated the overlapping/convolutional
  patch embedding tested in Ablation A.

**Code / reference implementations**
- [omihub777/ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR) — reference training recipe
  (Adam, cosine LR + warmup, label smoothing, AutoAugment) this implementation's hyperparameters
  follow.
- [priyammaz/PyTorch-Adventures — Vision Transformer notebook](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Computer%20Vision/Vision%20Transformer/VisionTransformer.ipynb) —
  reference for overall ViT structure and the CLS-vs-pooling / patch embedding walkthrough.

**Video resources**
- Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out* (YouTube) — attention
  and multi-head attention implementation walkthrough (causal masking adapted to non-causal for
  ViT).
- CampusX — Self-Attention / Multi-Head Attention playlist (YouTube) — conceptual introduction
  to attention prior to implementation.
- freeCodeCamp — *Building a Vision Transformer Model from Scratch with PyTorch* (Tunga Bayrak)
  — component-assembly reference.
- mildlyoverfitted — *Vision Transformer in PyTorch* (YouTube) — shape-handling reference for
  patch embedding and full model assembly.
- Priyam Mazumdar — *Let's Reproduce the ViT* (YouTube) — model architecture, CLS-vs-pooling
  discussion, and training/augmentation recipe reference.
- Yannic Kilcher — *ViT paper explained* (YouTube) — inductive bias framing that motivated the
  architectural ablations in this work.
- AI Coffee Break — *Swin Transformer animated* (YouTube) — windowed attention + relative
  position bias intuition, used for Ablation B.

**AI assistance**
- Claude (Anthropic) — used throughout development for code review, debugging assistance,
  explaining architectural and mathematical concepts, and structuring this documentation. All
  implementation code was written, tested, and verified by the author; Claude's role was
  explanatory and advisory, not autonomous code generation without review.
- Perplexity AI — used for supplementary research and fact-checking during development.

---

## 10. Repository contents

This repository contains exactly two files, per assignment requirements:
- `vit_cifar10.ipynb` — full implementation, training, evaluation, and ablation analysis.
- `README.md` — this file.

No datasets, checkpoints, or additional result files are included in the repository.
