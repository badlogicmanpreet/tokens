# gpt — build and train GPT-2 (124M) from scratch

Companion code for Token by Token, Chapter 4. Builds GPT-2 from first
principles and trains it on FineWeb-Edu; also includes a smaller char-level
transformer decoder used earlier in the chapter.

## Layout
- `src/gpt2_gpus/train_gpt_10BTokens_current.py` — **the canonical training
  script** (the run behind the book's numbers). The `_prod_v1/_prod_v2/_failed`
  variants are kept for history; use `_current`.
- `src/gpt2_gpus/fineweb.py` — downloads + tokenizes FineWeb-Edu sample-10BT
  into `.npy` shards (~20 GB; run this first, from `src/gpt2_gpus/`).
- `src/gpt2_gpus/hellaswag.py` — HellaSwag eval (downloads the dataset on demand).
- `src/gpt2_gpus/generate_tokens.py`, `watcher.py` — sampling and WeightWatcher
  analysis for a trained checkpoint (expects `checkpoints/model_57218.pt`).
- `src/decoder/` — the char-level transformer decoder (tinyshakespeare).
- `notebooks/` — exploratory copies; prefer `src/` for canonical code.
- `log124M_*/` — training logs from real runs: 30B tokens → val loss 2.9672,
  HellaSwag 0.3260; 20B → 2.9981 / 0.3200. (Note: `log124M_40B/` actually
  contains a 10B-token run.)

## Setup
```bash
pip install -r requirements.txt
cd src/gpt2_gpus
python fineweb.py                       # prepare data shards (once)
```

## Train
```bash
# single GPU
python train_gpt_10BTokens_current.py
# multi-GPU (adjust nproc to your machine)
torchrun --standalone --nproc_per_node=8 train_gpt_10BTokens_current.py
```
Defaults assume large GPUs: `B = 64, T = 1024` needs ~80 GB per device.
On a 40 GB A100 use `B = 32`; on a 24 GB card `B = 8` or `B = 16` — the
gradient-accumulation logic keeps the effective batch at 524,288 tokens.
bfloat16 autocast requires Ampere or newer.

## Model facts (as trained)
12 layers, 12 heads, 768 dim, 1024 context, vocab padded 50257 → 50304,
~124.5M parameters, weight-tied lm_head, top-k 50 sampling.
