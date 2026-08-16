"""Load and remap pre-trained weights into our model architecture.

Supports two checkpoint formats:
  - Qwen3 custom .pth files (from rasbt's repo)
  - HuggingFace safetensors (Llama, and other HF-hosted models)

Each format uses different parameter names. This module translates
them into our model's naming so the rest of the code stays clean.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from .config import QWEN3_06B, DEFAULT, MODELS
from .model import TransformerLM
from .generate import pick_device


# -----------------------------------------------------------------------
# Qwen3 checkpoint remapping  (.pth from rasbt repo)
# -----------------------------------------------------------------------

_QWEN_TOP = {
    "tok_emb.weight":    "embedding.weight",
    "final_norm.scale":  "norm.scale",
    "out_head.weight":   "head.weight",
}

_QWEN_LAYER = {
    "att.W_query.weight":   "attn.wq.weight",
    "att.W_key.weight":     "attn.wk.weight",
    "att.W_value.weight":   "attn.wv.weight",
    "att.out_proj.weight":  "attn.wo.weight",
    "att.q_norm.scale":     "attn.q_norm.scale",
    "att.k_norm.scale":     "attn.k_norm.scale",
    "norm1.scale":          "attn_norm.scale",
    "norm2.scale":          "ffn_norm.scale",
    "ff.fc1.weight":        "ffn.gate.weight",
    "ff.fc2.weight":        "ffn.up.weight",
    "ff.fc3.weight":        "ffn.down.weight",
}


def _remap_qwen(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in ckpt.items():
        if k in _QWEN_TOP:
            out[_QWEN_TOP[k]] = v
        elif k.startswith("trf_blocks."):
            _, idx, suffix = k.split(".", 2)
            out[f"layers.{idx}.{_QWEN_LAYER[suffix]}"] = v
        else:
            raise KeyError(f"Unknown Qwen checkpoint key: {k}")
    return out


# -----------------------------------------------------------------------
# HuggingFace remapping  (Llama, Qwen-HF, etc.)
# -----------------------------------------------------------------------

_HF_TOP = {
    "model.embed_tokens.weight":  "embedding.weight",
    "model.norm.weight":          "norm.scale",
    "lm_head.weight":             "head.weight",
}

_HF_LAYER = {
    "self_attn.q_proj.weight":              "attn.wq.weight",
    "self_attn.k_proj.weight":              "attn.wk.weight",
    "self_attn.v_proj.weight":              "attn.wv.weight",
    "self_attn.q_proj.bias":                "attn.wq.bias",
    "self_attn.k_proj.bias":                "attn.wk.bias",
    "self_attn.v_proj.bias":                "attn.wv.bias",
    "self_attn.o_proj.weight":              "attn.wo.weight",
    "self_attn.q_norm.weight":              "attn.q_norm.scale",
    "self_attn.k_norm.weight":              "attn.k_norm.scale",
    "input_layernorm.weight":               "attn_norm.scale",
    "post_attention_layernorm.weight":       "ffn_norm.scale",
    "mlp.gate_proj.weight":                 "ffn.gate.weight",
    "mlp.up_proj.weight":                   "ffn.up.weight",
    "mlp.down_proj.weight":                 "ffn.down.weight",
}


def _remap_hf(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in ckpt.items():
        if k in _HF_TOP:
            out[_HF_TOP[k]] = v
        elif k.startswith("model.layers."):
            parts = k.split(".", 3)
            idx, suffix = parts[2], parts[3]
            if suffix in _HF_LAYER:
                out[f"layers.{idx}.{_HF_LAYER[suffix]}"] = v
            else:
                print(f"  skipping: {k}")
        else:
            print(f"  skipping: {k}")

    # Handle tied embeddings: if lm_head is missing the checkpoint ties
    # embeddings and head. TransformerLM also ties the modules themselves
    # (head.weight IS embedding.weight), so loading either key updates both
    # and the tie survives fine-tuning.
    if "head.weight" not in out and "embedding.weight" in out:
        out["head.weight"] = out["embedding.weight"]

    return out


# -----------------------------------------------------------------------
# Download from HuggingFace Hub
# -----------------------------------------------------------------------

def download_hf_model(repo_id: str, local_dir: str | Path = "models") -> Path:
    """Download a model from HuggingFace Hub and return the local path.

    Downloads only the files needed: safetensors weights, tokenizer,
    and config. Requires `huggingface-cli login` for gated models.
    """
    from huggingface_hub import snapshot_download

    local = Path(local_dir) / repo_id.replace("/", "--")
    path = snapshot_download(
        repo_id,
        local_dir=str(local),
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )
    return Path(path)


def _load_safetensors_dir(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all .safetensors files from a directory into one state dict."""
    tensors = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        tensors.update(load_safetensors(str(f)))
    return tensors


# -----------------------------------------------------------------------
# Unified model loading
# -----------------------------------------------------------------------

def load_model(weights_path: str | Path, cfg: dict = QWEN3_06B,
               device: torch.device | None = None) -> TransformerLM:
    """Load a Qwen3 model from a .pth checkpoint file."""
    if device is None:
        device = pick_device()

    model = TransformerLM(cfg)
    ckpt = torch.load(weights_path, weights_only=True)
    model.load_state_dict(_remap_qwen(ckpt))
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {cfg['name']} ({n_params / 1e6:.0f}M params) on {device}")
    return model


def load_hf_model(model_name: str, device: torch.device | None = None,
                  local_dir: str | Path = "models") -> tuple[TransformerLM, Path]:
    """Download and load a HuggingFace model by config name.

    Args:
        model_name: key from MODELS registry (e.g. 'llama-3.2-1b')
        device: target device
        local_dir: where to cache downloads

    Returns:
        (model, model_dir) — the loaded model and path to local files
                             (model_dir contains the tokenizer too)
    """
    cfg = MODELS[model_name]
    if device is None:
        device = pick_device()

    print(f"Downloading {cfg['repo_id']}...")
    model_dir = download_hf_model(cfg["repo_id"], local_dir=local_dir)

    print("Loading weights...")
    raw = _load_safetensors_dir(model_dir)
    remapped = _remap_hf(raw)

    model = TransformerLM(cfg)
    model.load_state_dict(remapped)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {cfg['name']} ({n_params / 1e6:.0f}M params) on {device}")
    return model, model_dir
