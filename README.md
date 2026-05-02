# Neural Machine Translation (NMT)

> **F21NL Coursework 2 — 2025/2026**

Build and compare Neural Machine Translation systems for **English → French** translation using the [Multi30k](https://github.com/multi30k/multi30k) image caption dataset. The coursework progresses from a classical seq2seq LSTM model through a modern decoder-only transformer, with ablation studies, attention visualizations, and decoding experiments.

---

## Table of Contents

- [Overview](#overview)
- [Parts](#parts)
  - [Part 1: Seq2seq LSTM with Attention](#part-1-seq2seq-lstm-with-attention)
  - [Part 2: Decoder-Only Transformer (GPT-style)](#part-2-decoder-only-transformer-gpt-style)
  - [Part 3: Attention Visualizations](#part-3-attention-visualizations)
  - [Part 4: Decoding Experiments](#part-4-decoding-experiments)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Running on Google Colab](#running-on-google-colab)

---

## Overview

| Aspect | Detail |
|---|---|
| **Task** | Neural Machine Translation (English → French) |
| **Dataset** | [Multi30k](https://github.com/multi30k/multi30k) — ~30k English-French image captions |
| **Tokenization** | SentencePiece (Byte-Pair Encoding) subword tokenization |
| **Frameworks** | PyTorch, HuggingFace Transformers, sacrebleu, Weights & Biases |
| **Evaluation** | BLEU score (sacrebleu), cross-entropy loss |

---

## Parts

### Part 1: Seq2seq LSTM with Attention

A classical **encoder-decoder** architecture with dot-product attention.

**Architecture:**
- **Encoder:** Single-layer bidirectional LSTM (hidden size 512)
- **Decoder:** Single LSTMCell (hidden size 512), fed `[target_embed; prev_output]` at each step
- **Attention:** Dot-product attention over projected encoder hidden states, with padding mask
- **Vocabulary:** Separate SentencePiece models for source (7,000 subwords) and target (8,000 subwords)
- **Inference:** Greedy decoding (up to 70 time steps)

**Key files:**

| File | Description |
|---|---|
| `nmt_model.py` | Core `NMT` model: `encode()`, `decode()`, `step()`, `generate()` |
| `train.py` | Training loop with Adam optimizer, LR decay, and early stopping |
| `predict.py` | Inference + attention heatmap visualization (seaborn) |
| `vocab.py` | `VocabEntry` / `Vocab` classes for SentencePiece vocabulary management |
| `utils.py` | Tokenization, padding, batching utilities |
| `model_config.yaml` | Model hyperparameters (embed/hidden size, dropout) |
| `train_config.yaml` | Training hyperparameters (lr, batch size, patience, etc.) |

**Model config variants:** `small`, `medium`, `large`, `xlarge`, `xxlarge` — for ablation studies.

---

### Part 2: Decoder-Only Transformer (GPT-style)

Replaces the encoder-decoder with a **decoder-only transformer** inspired by GPT. Translation is framed as causal next-token prediction over a unified sequence.

**Input format:**
```
<en> source_tokens <fr> target_tokens </s>
```

The `<en>` and `<fr>` tokens act as language markers. The model learns to "translate" by predicting the next token given all preceding tokens (causal masking).

**Architecture:**
- **Model:** 6-layer transformer with 6 attention heads, 192-dim embeddings
- **Attention:** Causal (masked) multi-head self-attention only — no cross-attention
- **Positional encoding:** Learnable positional embeddings
- **Vocabulary:** Shared SentencePiece subword vocabulary (15,003 tokens) for both languages
- **Inference:** Greedy decoding, top-k sampling, top-p (nucleus) sampling, temperature scaling

**Key differences from Part 1:**

| Aspect | Part 1 | Part 2 |
|---|---|---|
| Architecture | Encoder-Decoder LSTM | Decoder-only Transformer |
| Attention | Cross-attention (encoder → decoder) | Causal self-attention only |
| Vocabulary | Separate src/tgt vocabs | Unified vocab |
| Positional encoding | None (LSTM is order-aware) | Learnable positional embeddings |
| LR schedule | Step decay on restart | Cosine decay with warmup |
| Weight decay | Uniform | Two-group (with/without decay) |

---

### Part 3: Attention Visualizations

Visual comparison of attention patterns between the BiLSTM (Part 1) and Transformer (Part 2) models.

- **`bilstm_example*.png`** — Sample translations from the BiLSTM encoder-decoder model
- **`trans_example*.png`** — Attention heatmaps from the Transformer model (4800×4800 resolution grids showing which source tokens each decoder step attended to)

These visualizations demonstrate how each architecture "aligns" source and target tokens during translation.

---

### Part 4: Decoding Experiments

Experiments with a trained 8.46M-parameter Transformer model, probing how decoding hyperparameters affect output quality.

**Test sentence:** *"Two boys are sitting on a couch that is red and black."*

**Greedy (default) output:** *"Deux garçons sont assis sur un canapé qui est vêtus de rouge et noir."*

The experiment sweeps over combinations of:
- **top-k:** 3, 7, 10, 25, 50
- **top-p:** 0.3, 0.5, 0.7, 0.8, 0.9, 1.0
- **temperature:** 0.3, 0.5, 0.7, 0.9, 1.0

Results (`part4_results.txt`) show how sampling parameters influence lexical choice — e.g., shifting from *"vêtus de rouge et noir"* (less fluent) to *"en rouge et noir"* (more natural) under higher temperature / top-p settings.

**Key files:**
| File | Description |
|---|---|
| `part4/nmt.model` | Trained model checkpoint (~33 MB, 8.46M parameters) |
| `part4_experiments.ipynb` | Notebook running the decoding parameter sweep |
| `part4_results.txt` | Full text output of all experiment combinations |

---

## Dataset

The [Multi30k](https://github.com/multi30k/multi30k) dataset consists of ~30,000 English-French parallel image captions from Flickr30k.

| Split | Files | Approx. lines |
|---|---|---|
| Train | `multi30k_data/train.en`, `multi30k_data/train.fr` | ~28,991 |
| Validation | `multi30k_data/val.en`, `multi30k_data/val.fr` | ~1,014 |
| Test | `multi30k_data/test_2016_flickr.en`, `multi30k_data/test_2016_flickr.fr` | ~1,000 |

**Sample:**

| English | French |
|---|---|
| `Two young, White males are outside near many bushes.` | `Deux jeunes hommes blancs sont dehors près de buissons.` |
| `Several men in hard hats are operating a giant pulley system.` | `Plusieurs hommes en casque font fonctionner un système de poulies géant.` |

---

## Project Structure

```
cw2_2025_2026-students/
├── multi30k_data/              # Multi30k EN↔FR parallel corpus
│   ├── train.en / train.fr     # Training data
│   ├── val.en / val.fr         # Validation data
│   └── test_2016_flickr.en / .fr  # Test data
├── part1/                      # Seq2seq LSTM with attention
│   ├── nmt_model.py            # Encoder-decoder model definition
│   ├── train.py                # Training loop + early stopping
│   ├── predict.py              # Inference + attention visualization
│   ├── vocab.py                # SentencePiece vocabulary management
│   ├── utils.py                # Tokenization, padding, batching
│   ├── model_config.yaml       # Model hyperparameters
│   ├── train_config.yaml       # Training hyperparameters
│   ├── test.py                 # BLEU evaluation
│   ├── sanity_check.py / .pt   # Sanity check for model correctness
│   ├── src.model / tgt.model   # SentencePiece subword models
│   ├── src.vocab / tgt.vocab   # Vocabulary files
│   ├── outputs/                # Training outputs & logs
│   └── vocab/                  # Serialized vocab JSON
├── part2/                      # Decoder-only Transformer (GPT-style)
│   ├── model.py                # GPT transformer: CausalSelfAttention, Block, GPT
│   ├── train.py                # Training with cosine LR decay + warmup
│   ├── predict.py              # Inference with sampling strategies
│   ├── vocab.py                # Unified vocabulary management
│   ├── utils.py                # Data utilities
│   ├── test.py                 # Corpus-level BLEU evaluation
│   ├── model_config.yaml       # Transformer hyperparameters
│   ├── train_config.yaml       # Training hyperparameters
│   ├── sanity_check.py / .pt   # Sanity check
│   ├── outputs/                # Training outputs & logs
│   ├── part4_experiments.ipynb # Decoding parameter sweep notebook
│   ├── part4_results.txt       # Experiment results
│   └── vocab/                  # Serialized vocab JSON
├── part3/                      # Attention visualizations
│   ├── bilstm_example[1-5].png # BiLSTM model translation samples
│   └── trans_example[1-5].png  # Transformer attention heatmaps
├── part4/                      # Trained model for Part 4 experiments
│   └── nmt.model               # Trained 8.46M-parameter checkpoint
├── colab_tips.ipynb            # Guide for running on Google Colab
├── pyproject.toml              # Project metadata & dependencies
├── requirements.txt            # pip dependencies
└── uv.lock                     # uv lockfile for reproducible installs
```

---

## Installation

### Prerequisites
- **Python 3.12+**
- **Git**

### Using uv (recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies from lockfile
uv sync
```

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU support:** CUDA packages are commented out in `requirements.txt`. Uncomment them if you have a CUDA-enabled GPU. Not needed on Google Colab (GPU is provided automatically).

---

## Usage

### Training

```bash
# Part 1: Seq2seq LSTM
cd part1
python train.py \
  --train_src ../multi30k_data/train.en \
  --train_tgt ../multi30k_data/train.fr \
  --dev_src   ../multi30k_data/val.en \
  --dev_tgt   ../multi30k_data/val.fr \
  --model_config model_config.yaml \
  --train_config train_config.yaml

# Part 2: Transformer
cd part2
python train.py \
  --train_src ../multi30k_data/train.en \
  --train_tgt ../multi30k_data/train.fr \
  --dev_src   ../multi30k_data/val.en \
  --dev_tgt   ../multi30k_data/val.fr \
  --model_config model_config.yaml \
  --train_config train_config.yaml
```

### Inference

```bash
# Part 1: Translate a single sentence
cd part1
python predict.py \
  --model outputs/model.pt \
  --vocab vocab/vocab.json \
  --src_model src.model \
  --tgt_model tgt.model \
  --src_text "Two boys are sitting on a red and black couch."

# Part 2: Translate with sampling
cd part2
python predict.py \
  --model outputs/model.pt \
  --vocab vocab/vocab.json \
  --src_text "Two boys are sitting on a red and black couch." \
  --do_sample --top_k 10 --top_p 0.9 --temperature 0.7
```

### Evaluation

```bash
# Evaluate on test set (BLEU score)
cd part2
python test.py \
  --model outputs/model.pt \
  --vocab vocab/vocab.json \
  --test_src ../multi30k_data/test_2016_flickr.en \
  --test_tgt ../multi30k_data/test_2016_flickr.fr
```

---

## Dependencies

| Package | Purpose |
|---|---|
| **torch** | Deep learning framework |
| **transformers** | HuggingFace utilities |
| **sentencepiece** | Subword tokenization (BPE) |
| **datasets** | HuggingFace dataset loading |
| **sacrebleu** | Standardized BLEU scoring |
| **wandb** | Experiment tracking & logging |
| **tensorboard** | Training visualization |
| **matplotlib / seaborn** | Attention heatmap plots |
| **numpy / pandas** | Numerical computing & data manipulation |
| **tqdm** | Progress bars |

---

## Running on Google Colab

See [`colab_tips.ipynb`](colab_tips.ipynb) for a full guide. Key steps:

1. **Enable GPU:** `Runtime → Change runtime type → T4 GPU`
2. **Clone your fork:** `!git clone <your-fork-url>`
3. **Install deps:** `!pip install -r requirements.txt`
4. **Train:** `!python part1/train.py --train_src ...`

> ⚠️ Colab sessions are ephemeral. Save models to [Hugging Face Hub](https://huggingface.co/) or download manually before disconnecting.

---

## Training Configuration Reference

### Part 1 — LSTM (`train_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `lr` | 0.001 | Initial learning rate |
| `lr_decay` | 0.5 | LR multiplier on restart |
| `train_batch_size` | 32 | Training batch size |
| `dev_batch_size` | 128 | Evaluation batch size |
| `max_epoch` | 30 | Maximum epochs |
| `valid_niter` | 2000 | Validation interval |
| `patience` | 1 | Restarts before LR decay |
| `max_num_trial` | 5 | Max restarts (early stop) |

### Part 2 — Transformer (`train_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `lr` | 0.0006 | Peak learning rate |
| `min_lr` | 0.00006 | Minimum learning rate |
| `warmup_percent_iters` | 0.1 | Warmup fraction (10%) |
| `decay_lr` | true | Cosine decay schedule |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `betas` | [0.9, 0.95] | AdamW betas |

---

*Coursework for F20NL/F21NL (2025–2026 academic year)*
