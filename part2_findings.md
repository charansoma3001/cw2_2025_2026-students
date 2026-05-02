# Part 2 — Research Findings

## What Part 2 Does

Part 2 replaces the Part 1 encoder-decoder RNN architecture with a **GPT-style decoder-only transformer** for Neural Machine Translation (NMT). It takes English sentences as input and generates French translations using autoregressive next-token prediction.

The translation is framed as a **single sequence-to-sequence task** with a unified vocabulary. The input format wraps the source and target together:

```
<en> source_tokens <fr> target_tokens </s>
```

During training, the model sees the full `<en> ... <fr> ...` sequence and learns to predict each token given all preceding tokens (causal masking). The `<fr>` token acts as a trigger for the model to begin generating in French.

---

## Architecture

### Overall Design: Decoder-Only Transformer (GPT-style)

| Component | Description |
|---|---|
| **Architecture type** | GPT (decoder-only transformer), no encoder |
| **Attention** | Causal (masked) multi-head self-attention only |
| **Training objective** | Cross-entropy loss over next-token prediction (teacher forcing) |
| **Inference** | Autoregressive token-by-token generation with greedy or sampled decoding |

### Key Classes

#### `GPTConfig` (dataclass)
Configuration container for the model, loaded from `model_config.yaml`. Default values match GPT-2:
- `block_size`: 1024 (context window, set to **128** in config)
- `vocab_size`: 50304 (set to **15003** for the SentencePiece subword vocabulary)
- `n_layer`: 12 (set to **6**)
- `n_head`: 12 (set to **6**)
- `n_embed`: 768 (set to **192**)
- `dropout`: 0.0 (set to **0.1**)

#### `CausalSelfAttention`
Implements vanilla multi-head masked self-attention:
1. Linear projections for Query, Key, Value
2. Scaled dot-product attention with a causal (lower-triangular) mask
3. Attention dropout + residual dropout
4. Output projection
5. Supports `output_attentions` for visualization

#### `Block`
A standard transformer block:
- **LayerNorm → CausalSelfAttention → Residual add**
- **LayerNorm → MLP (GELU, 4× expansion) → Residual add**

#### `GPT`
The full model:
1. **Token embeddings** (`nn.Embedding`)
2. **Learnable positional embeddings** (not rotary or sinusoidal — trainable parameters)
3. **N transformer blocks** (`Block`)
4. **Final LayerNorm**
5. **Linear head** projecting to vocabulary size

---

## Training Approach

### Key Parameters (`train_config.yaml`)

| Parameter | Value |
|---|---|
| `train_batch_size` | 32 |
| `dev_batch_size` | 128 |
| `max_epoch` | 30 |
| `valid_niter` | 2000 (validate every 2000 iterations) |
| `patience` | 5 (early stopping) |
| `lr` | 0.0006 |
| `min_lr` | 0.00006 |
| `decay_lr` | True (cosine schedule with warmup) |
| `warmup_percent_iters` | 0.1 (10% of total iterations) |
| `weight_decay` | 0.1 |
| `betas` | [0.9, 0.95] |
| `grad_clip` | 1.0 |

### Training Loop (`train.py`)

1. **Data loading**: Pairs of English-French sentences tokenized with SentencePiece subwords
2. **Batching**: `batch_iter` yields batches sorted by length (largest first), with optional shuffle
3. **Input formatting**: `vocab.src.to_input_tensor` wraps inputs as `<en> src_tokens <fr> tgt_tokens` and creates shifted targets
4. **Loss**: Cross-entropy with `ignore_index=0` to skip `<pad>` tokens
5. **Optimizer**: AdamW with two parameter groups — weight decay on ≥2D tensors (weights, embeddings), no decay on 1D tensors (biases, LayerNorm)
6. **Learning rate**: Cosine decay with linear warmup over 10% of total iterations
7. **Validation**: Every `valid_niter` iterations on dev set; saves best model by lowest validation loss
8. **Early stopping**: Stops if validation loss doesn't improve for `patience` consecutive checks
9. **Logging**: WandB tracking of train loss, dev loss, and learning rate

### Data Pipeline

- **SentencePiece** subword tokenization (shared vocab for EN+FR, size 15000)
- Special tokens: `<pad>`, `<s>`, `</s>`, `<unk>`, `<en>`, `<fr>`
- Targets are shifted right by 1 position; source portion of targets is masked with `<pad>`

---

## Inference / Prediction

### `predict.py` — Single Sentence Translation
- Tokenize input with SentencePiece
- Wrap with `<en>` source `<fr>` prefix
- Call `model.generate()` for autoregressive decoding
- Support for **greedy decoding** (default) or **sampling** with top-k, top-p, and temperature
- Visualizes attention heatmaps across all layers and heads

### `test.py` — Corpus-Level Evaluation
- Generates translations for all test sentences
- Computes **BLEU score** using `sacrebleu`
- Saves predictions, references, and BLEU score to JSON

### Decoding Options
| Parameter | Purpose |
|---|---|
| `do_sample` | Switch between greedy and sampled decoding |
| `top_k` | Keep only top-k highest-probability tokens |
| `top_p` | Nucleus sampling — smallest set with cumulative prob ≥ p |
| `temperature` | Scale logits before softmax (higher = more diverse) |
| `max_decoding_time_steps` | Max tokens to generate (default 70) |
| `eos_token_id` | `</s>` (id=2) — stop generating when reached |

---

## Supporting Modules

### `vocab.py`
- **`VocabEntry`**: Word↔index mapping with special tokens, padding, tensor conversion
- **`Vocab`**: Container for source/target vocabs (shared in part 2)
- **`to_input_tensor`**: Builds the `<en> src <fr> tgt` format and shifted targets
- **`get_vocab_list`**: Trains SentencePiece model and extracts subword vocabulary

### `utils.py`
- **`pad_sents`**: Pad variable-length sequences to equal length
- **`read_corpus`**: Load and tokenize EN-FR pairs with SentencePiece
- **`batch_iter`**: Generate batches sorted by length, with optional shuffle

---

## How Part 2 Differs from Part 1

| Aspect | Part 1 | Part 2 |
|---|---|---|
| **Architecture** | Encoder-Decoder LSTM (separate encoder + decoder with attention) | Decoder-only GPT transformer (no encoder) |
| **Attention** | Separate encoder-decoder cross-attention | Causal self-attention only |
| **Embeddings** | Separate source & target embeddings | Shared unified embeddings (one vocab) |
| **Positional encoding** | Not specified (LSTM is order-aware by nature) | Learnable positional embeddings |
| **Vocabulary** | Separate source and target word-level vocabs | Shared SentencePiece subword vocabulary |
| **Tokenization** | Word-level with separate SP models (`src.model`, `tgt.model`) | Single SentencePiece model (`spm.model`) for both languages |
| **Training format** | Encoder processes source, decoder generates target | Full sequence `<en> src <fr> tgt` with causal masking |
| **Model size** | embed_size=512, hidden_size=512 | n_layer=6, n_head=6, n_embed=192 (smaller per-layer) |
| **Optimization** | Standard AdamW | AdamW with cosine LR decay + warmup |
| **Decoding** | Beam search or greedy | Greedy, top-k, top-p (nucleus) sampling |
| **Visualization** | Basic | Attention heatmap visualization |
| **Config files** | Multiple configs (small/medium/large/xlarge/xxlarge) | Single model config + single train config |
| **Tracking** | WandB (basic) | WandB with detailed logging |

### Key Architectural Shift
Part 1 used a **classical encoder-decoder NMT** pattern: an LSTM encoder compresses the source sentence into a context vector, and an LSTM decoder generates the target sentence with attention over encoder states.

Part 2 uses a **decoder-only approach**: the source and target are concatenated into one sequence with language markers (`<en>`, `<fr>`), and the model learns translation through causal next-token prediction. This mirrors how GPT models work — the "encoder" functionality emerges naturally from self-attention over the prefix tokens.
