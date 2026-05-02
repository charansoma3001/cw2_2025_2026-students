# Part 3 & Part 4 — Research Findings

## Part 3: Neural Machine Translation Visualizations

Part 3 contains 10 visualization images divided into two groups:

### BiLSTM Examples (`bilstm_example1.png` – `bilstm_example5.png`)
- **File sizes:** ~26–31 KB each
- **What they represent:** Visualization outputs from a **Bidirectional LSTM (BiLSTM)** encoder-decoder machine translation model. These likely show sample translations produced by the BiLSTM model — i.e., source English sentences alongside their translated French outputs. The small file sizes suggest simple text-based output images (not attention heatmaps).

### Transformer Examples (`trans_example1.png` – `trans_example5.png`)
- **File sizes:** ~441–599 KB each
- **Image resolution:** 4800 × 4800 pixels
- **What they represent:** Visualization outputs from a **Transformer-based** machine translation model. The large resolution and file sizes strongly suggest these are **attention mechanism visualizations** (heatmaps) — grids showing encoder-decoder attention weights. Each cell in the grid represents how much the decoder attended to each source token at each decoding step. These are a hallmark way of interpreting Transformer NMT models.

---

## Part 4: Trained NMT Model & Inference Experiments

### `nmt.model`
- **Path:** `/part4/nmt.model`
- **Size:** 33 MB
- **Type:** Binary model checkpoint (identified as a Zip archive — consistent with PyTorch's `torch.save` format)
- **Parameter count:** **8,455,296 (8.46M parameters)**

### Experiment Notebook (`part2/part4_experiments.ipynb`)
The notebook runs inference on the trained NMT model using `predict.py`, testing various **decoding/sampling strategies**:

| Parameter | Values Tested |
|-----------|--------------|
| `top-k` | 3, 7, 10, 25, 50 |
| `top-p` (nucleus sampling) | 0.3, 0.5, 0.7, 0.8, 0.9, 1.0 |
| `temperature` | 0.3, 0.5, 0.7, 0.9, 1.0 |

- **Test sentence:** `"Two boys are sitting on a couch that is red and black."`
- **Greedy (no sampling) output:** `"Deux garçons sont assis sur un canapé qui est vêtus de rouge et noir."`
- **Observation:** With low `top-k` / `top-p` and low temperature, the model tends to produce `"vêtus de rouge et noir"`; with higher sampling parameters, it shifts to `"en rouge et noir"`. This demonstrates how decoding hyperparameters affect translation fluency and lexical choice.

---

## Dataset: Multi30k (English ↔ French)

### Location
`multi30k_data/`

### Language Pair
**English (source) → French (target)**

### Files
| File | Purpose |
|------|---------|
| `train.en` / `train.fr` | Training parallel corpus (~28,991 lines) |
| `val.en` / `val.fr` | Validation set |
| `test_2016_flickr.en` / `test_2016_flickr.fr` | Test set (Flickr30k 2016) |

### Format
- Plain text, **one sentence per line**
- Line-aligned between `.en` and `.fr` files (line *N* in `train.en` corresponds to line *N* in `train.fr`)
- Content consists of **image captions** (descriptive sentences about photographs)

#### Sample data (first 3 lines):
| English | French |
|---------|--------|
| `Two young, White males are outside near many bushes.` | `Deux jeunes hommes blancs sont dehors près de buissons.` |
| `Several men in hard hats are operating a giant pulley system.` | `Plusieurs hommes en casque font fonctionner un système de poulies géant.` |
| `A little girl climbing into a wooden playhouse.` | `Une petite fille grimpe dans une maisonnette en bois.` |

---

## Colab Setup Tips (from `colab_tips.ipynb`)

### GPU Usage
1. Go to **Runtime → Change runtime type → T4 GPU**
2. **Warning:** Colab GPU usage is limited — do **not** waste GPU time on idle sessions or debugging. Develop and debug locally or on CPU first.

### Saving Model Checkpoints
Colab VMs are ephemeral — files are deleted when the session ends. Two options:
1. **Upload to Hugging Face Hub** — use `huggingface_hub` to push models automatically after each run.
2. **Manual download** — click the file icon in Colab's left sidebar → hover over the file → three dots → **Download**.

### Uploading Code
- **Recommended:** Fork the GitHub repo and clone in Colab via `!git clone <your-fork-url>`.
- **Alternative:** Download the repo as a zip and upload/unzip in Colab.

### Useful Commands
- Prefix shell commands with `!` (e.g., `!ls`, `!cp`, `!rm`)
- Change directory with Python: `os.chdir("/content/your_path")`
- Run Python scripts: `!python my_script.py --arg1 value1`
- Install dependencies: `!pip install -r requirements.txt`

---

## .gitignore Notes
Key project-specific ignores (beyond standard Python patterns):
- `models*/` — trained model directories
- `wandb/` — Weights & Biases logging artifacts
- `.DS_Store` — macOS metadata
