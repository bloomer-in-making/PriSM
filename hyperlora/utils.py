import hashlib
import json
import os
import pickle
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Logging / IO
# ============================================================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Dtype / Device
# ============================================================
def get_torch_dtype(name: str):
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def get_main_device(cfg) -> torch.device:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg.main_device)
    return torch.device(cfg.device)


# ============================================================
# Misc helpers
# ============================================================
def deterministic_seed(user_id, base_seed: int) -> int:
    h = int(hashlib.sha1(str(user_id).encode()).hexdigest(), 16)
    return (h + base_seed) % (2 ** 32)


def format_metrics(d: dict) -> str:
    return " ".join([f"{k}={v:.4f}" for k, v in d.items()])


# ============================================================
# BGE encoder singleton (sentence embedding for the user posterior summary)
# ============================================================
_bge_model = None
_bge_tok = None
_bge_device = None
_bge_model_name = "BAAI/bge-small-en-v1.5"


def set_bge_model_name(name: str):
    """Set the global BGE encoder model name. Call once after parsing the
    config. If a different model was already loaded, the cache is reset and
    the new model is loaded on the next get_bge_encoder() call."""
    global _bge_model, _bge_tok, _bge_device, _bge_model_name
    if name == _bge_model_name:
        return
    _bge_model_name = str(name)
    _bge_model = None
    _bge_tok = None
    _bge_device = None


def get_bge_encoder(device: torch.device):
    """Lazy-load and cache the BGE encoder module + tokenizer."""
    global _bge_model, _bge_tok, _bge_device
    if _bge_model is None:
        from transformers import AutoModel, AutoTokenizer
        log(f"[bge] loading {_bge_model_name} ...")
        _bge_tok = AutoTokenizer.from_pretrained(_bge_model_name, use_fast=True)
        _bge_model = AutoModel.from_pretrained(
            _bge_model_name,
            torch_dtype=torch.float32,
        )
        _bge_model.eval()
        _bge_device = device
        _bge_model.to(device)
        log(f"[bge] loaded on {device}")
    return _bge_model, _bge_tok


@torch.no_grad()
def embed_text_mean(
    model,      # unused (kept for backwards-compat signature)
    tok,        # unused
    text: str,
    max_len: int = 512,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Encode a single string with the cached BGE encoder. Returns the L2
    normalised CLS embedding as a numpy float32 vector (dim 384 for
    bge-small)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bge_model, bge_tok = get_bge_encoder(device)
    text = "" if text is None else str(text)
    x = bge_tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
        add_special_tokens=True,
    )
    x = {k: v.to(_bge_device) for k, v in x.items()}
    out = bge_model(**x, return_dict=True)
    emb = out.last_hidden_state[:, 0, :]
    emb = F.normalize(emb.float(), dim=-1)
    return emb.detach().cpu().numpy()[0].astype(np.float32)


