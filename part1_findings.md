# Part 1 — Neural Machine Translation (NMT) Findings

## What Part 1 Does

Part 1 implements a **seq2seq Neural Machine Translation system** with **Bahdanau-style additive dot-product attention**. It translates sentences from English (source) to French (target) using the Multi30k dataset. The pipeline covers vocabulary building, model training, evaluation, inference, and attention visualization.

---

## Architecture

### Overall Architecture: Encoder-Decoder with Attention (Seq2Seq)

```
Source Sentence → [Embedding] → [Bi-directional LSTM Encoder] → Encoder Hidden States
                                                                   │
Target <s> + prev output ──→ [Embedding] ──┐                      │
                                            ├─→ [LSTMCell Decoder] ──→ Decoder Hidden
                                            │                      │
                    Encoder Hiddens ──→ [Dot-Product Attention] ──→│
                                                                    │
                                                  Concat + Project ──→ [Linear] → Log-Softmax → Target Vocab
```

### Key Classes and Files

| File | Purpose | Key Classes / Functions |
|---|---|---|
| `nmt_model.py` | Core NMT model definition | `NMT` (nn.Module) with `encode`, `decode`, `step`, `generate`, `forward` |
| `train.py` | Training loop, validation, early stopping | `train()`, `evaluate()`, `parse_args()` |
| `predict.py` | Inference + attention heatmap visualization | `predict()`, `visualize_attention()` |
| `vocab.py` | Vocabulary construction & serialization | `VocabEntry`, `Vocab`, `get_vocab_list()` |
| `utils.py` | Tokenization, padding, batching | `read_corpus()`, `pad_sents()`, `batch_iter()` |
| `model_config.yaml` | Model hyperparameters | embed_size, hidden_size, dropout_rate |
| `train_config.yaml` | Training hyperparameters | lr, batch_size, patience, max_epoch, etc. |

---

## Model Details (`NMT` class in `nmt_model.py`)

### Encoder
- **Component:** Single-layer **bidirectional LSTM**
- **Input:** Embedded source tokens, shape `(src_len, batch, embed_size)`
- **Output:** Per-timestep hidden states `(batch, src_len, hidden_size * 2)` — the `*2` comes from concatenating forward and backward directions
- **Initialization:** The final concatenated bi-directional hidden state and cell state are projected via two linear layers (`h_projection`, `c_projection`) from `2×hidden_size` → `hidden_size` to produce the decoder's initial state `(h₀, c₀)`

### Decoder
- **Component:** Single **LSTMCell** (one step at a time)
- **Input at each step `t`:** Concatenation of `[target_embedding_t, prev_combined_output]` → shape `(batch, embed_size + hidden_size)`
- **Output at each step `t`:** Combined output vector `o_t` of shape `(batch, hidden_size)`

### Attention Mechanism (Dot-Product)
Implemented in `step()`:

1. **Encoder hiddens projection:** `enc_hiddens` projected from `2h` → `h` via `att_projection` (Linear layer)
2. **Attention scores:** `e_t = enc_hiddens_proj · dec_hiddenᵀ` → shape `(batch, src_len)`
3. **Masking:** Padding positions masked with `-∞` before softmax
4. **Attention weights:** `α_t = softmax(e_t, dim=1)` → shape `(batch, src_len)`
5. **Context vector:** `c_t = α_t · enc_hiddens` → shape `(batch, 2h)`
6. **Concatenation:** `u_t = [dec_hidden; c_t]` → shape `(batch, 3h)`
7. **Projection + nonlinearity:** `v_t = combined_output_projection(u_t)`, then `o_t = dropout(tanh(v_t))`

### Output Layer
- **Linear projection:** `target_vocab_projection` maps `o_t` from `hidden_size` → `tgt_vocab_size`
- **Softmax:** `log_softmax` over the target vocabulary

### Key Projection Layers

| Layer | Input → Output |
|---|---|
| `h_projection` | `2h` → `h` (init decoder hidden) |
| `c_projection` | `2h` → `h` (init decoder cell) |
| `att_projection` | `2h` → `h` (encoder states for attention) |
| `combined_output_projection` | `3h` → `h` (merge decoder + context) |
| `target_vocab_projection` | `h` → `tgt_vocab_size` (final output) |

---

## Model Hyperparameters (`model_config.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `embed_size` | **512** | Word embedding dimensionality |
| `hidden_size` | **512** | LSTM hidden state dimensionality |
| `dropout_rate` | **0.2** | Dropout applied to combined output |

---

## Vocabulary (`vocab.py`)

### Tokenization Strategy
- Uses **SentencePiece** for subword tokenization (Byte-Pair Encoding)
- Source vocabulary size: **7000** subwords
- Target vocabulary size: **8000** subwords

### Special Tokens
- `<pad>` (id=0): Padding token
- `<s>` (id=1): Start-of-sentence (prepended to target)
- `</s>` (id=2): End-of-sentence (appended to target, stopping condition)
- `<unk>` (id=3): Out-of-vocabulary fallback

### Key Classes
- **`VocabEntry`:** Maps words/subwords ↔ integer IDs; handles padding and tensor conversion via `to_input_tensor()`
- **`Vocab`:** Wraps `src` and `tgt` `VocabEntry` objects; serializable to/from JSON

---

## Training Approach (`train.py`)

### Data
- **Dataset:** Multi30k (English → French)
- **Source:** `train.en` / `val.en`
- **Target:** `train.fr` / `val.fr`

### Training Hyperparameters (`train_config.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `lr` | **0.001** | Initial learning rate |
| `lr_decay` | **0.5** | LR multiplied by this on each restart |
| `train_batch_size` | **32** | Training batch size |
| `dev_batch_size` | **128** | Evaluation batch size |
| `max_epoch` | **30** | Maximum training epochs |
| `valid_niter` | **2000** | Validation every N iterations |
| `patience` | **1** | Patience before triggering LR decay |
| `max_num_trial` | **5** | Max restarts before early stopping |
| `uniform_init` | **0.1** | Initialize all params uniformly in [-0.1, +0.1] |
| `log_every` | **10** | Logging frequency |

### Optimization
- **Optimizer:** Adam
- **Loss:** Average negative log-likelihood over non-padding target tokens
- **Initialization:** Uniform [-0.1, +0.1] for all parameters
- **Gradient computation:** `pack_padded_sequence` / `pad_packed_sequence` used in encoder for efficiency

### Early Stopping & LR Decay Schedule
1. Every `valid_niter` (2000) iterations, evaluate dev loss
2. If dev loss improves → save best model
3. If dev loss does not improve → increment `patience`
4. If `patience` is exhausted → **LR decay** (× 0.5), restore best model weights and optimizer state, reset patience
5. If `max_num_trial` (5) restarts exhausted → **early stop**
6. Also stops at `max_epoch` (30)

### Monitoring
- Uses **Weights & Biases (wandb)** for logging train loss, dev loss, and learning rate

---

## How It Works — Data Flow

### Forward Pass (`forward`)
1. **Encode:** Source tokens → embeddings → bi-LSTM → packed/padded encoder hiddens + initial decoder state
2. **Mask:** Generate padding masks from source lengths
3. **Decode:** Autoregressively decode target tokens using attention over encoder hiddens → collect combined outputs
4. **Project + Score:** Project to target vocab → log-softmax → gather gold token log-probs → sum (excluding padding)

### Training Loop
1. Read corpus → SentencePiece subword tokenize → build vocab
2. For each batch (sorted by source length):
   - Pad sequences → convert to tensors
   - Forward pass → compute loss → backward pass → optimizer step
3. Periodically validate on dev set → early stopping logic

### Inference (`generate`)
1. Encode single source sentence
2. Greedy decoding: start with `<s>`, at each step pick `argmax` predicted token
3. Stop on `</s>` or `max_decoding_time_step` (70)
4. Returns decoded token IDs (and optionally attention scores for visualization)

### Prediction (`predict.py`)
1. Load trained model + vocab
2. SentencePiece tokenize input sentence
3. Run `generate()` with `output_attentions=True`
4. Convert subword IDs → text (handling SentencePiece `▁` marker)
5. Render attention heatmap via `visualize_attention()` (seaborn)

---

## Summary

| Aspect | Detail |
|---|---|
| **Architecture** | Seq2seq encoder-decoder with dot-product attention |
| **Encoder** | 1-layer bidirectional LSTM (hidden=512) |
| **Decoder** | LSTMCell (hidden=512), input = [target_embed; prev_output] |
| **Attention** | Dot-product over projected encoder states, masked softmax |
| **Embedding** | 512-dim for both source and target |
| **Tokenization** | SentencePiece subword (7000 src, 8000 tgt) |
| **Loss** | Negative log-likelihood (excluding padding) |
| **Optimizer** | Adam, lr=0.001, decay=0.5 |
| **Training** | Max 30 epochs, early stopping after 5 trials, validation every 2000 iters |
| **Device** | GPU → MPS → CPU fallback |
| **Dataset** | Multi30k (English → French) |
