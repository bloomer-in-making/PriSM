from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import log


# ============================================================
# HyperLoRA Linear
# ============================================================
class HyperLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, module_index: int, r: int, basis_m: int,
                 alpha: float):
        """Basis LoRA with per-basis mixing.

        dW = sum_m c_m * (B_m @ A_m): each basis is applied separately and the
        outputs are combined with the mixing coefficients, so no cross-terms
        between bases appear. This costs about M times the FLOPs of a single
        LoRA and is the formulation used in the paper.
        """
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.module_index = int(module_index)
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.r = int(r)
        self.basis_m = int(basis_m)
        self.scale = float(alpha) / float(r)

        basis_device = base_linear.weight.device
        self.A_basis = nn.ParameterList([
            nn.Parameter((0.01 * torch.randn(r, self.in_features, dtype=torch.float32)).to(basis_device))
            for _ in range(basis_m)
        ])
        self.B_basis = nn.ParameterList([
            nn.Parameter((0.01 * torch.randn(self.out_features, r, dtype=torch.float32)).to(basis_device))
            for _ in range(basis_m)
        ])
        self._coeffs: Optional[torch.Tensor] = None
        self._use_warmup_forward: bool = False
        self._warmup_basis_idx: int = 0
        self._warmup_dropout_p: float = 0.0
        self._compute_func_div: bool = False
        self._func_div_loss: Optional[torch.Tensor] = None

    def set_coeffs(self, coeffs_m: Optional[torch.Tensor]):
        self._coeffs = coeffs_m

    def forward_warmup(self, x: torch.Tensor, basis_idx: int, dropout_p: float) -> torch.Tensor:
        y = self.base(x)
        A = self.A_basis[basis_idx].to(dtype=x.dtype, device=x.device)
        B = self.B_basis[basis_idx].to(dtype=x.dtype, device=x.device)
        if self.training and dropout_p > 0.0:
            A = F.dropout(A, p=dropout_p, training=True)
            B = F.dropout(B, p=dropout_p, training=True)
        return y + self.scale * F.linear(F.linear(x, A), B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_warmup_forward:
            return self.forward_warmup(x, self._warmup_basis_idx, self._warmup_dropout_p)
        y = self.base(x)
        if self._coeffs is None:
            return y
        A_stack = torch.stack(list(self.A_basis))
        B_stack = torch.stack(list(self.B_basis))
        c = self._coeffs.to(device=A_stack.device, dtype=A_stack.dtype)

        # dW = sum_m c_m * (B_m @ A_m): per-basis forward then weighted sum.
        y_lora = None
        for m in range(self.basis_m):
            A_m = A_stack[m].to(dtype=x.dtype, device=x.device)
            B_m = B_stack[m].to(dtype=x.dtype, device=x.device)
            y_m = F.linear(F.linear(x, A_m), B_m)
            contrib = c[m].to(dtype=y_m.dtype, device=y_m.device) * y_m
            y_lora = contrib if y_lora is None else y_lora + contrib

        # functional diversity: minimise pairwise cosine similarity of basis outputs.
        if self._compute_func_div and self.training:
            with torch.no_grad():
                x_sample = x.detach().reshape(-1, x.shape[-1])[:8]  # at most 8 tokens
            # (basis_m, n_tokens, out_features)
            lora_outs = torch.stack([
                F.linear(F.linear(x_sample.float(), A_stack[m].to(x_sample.device)),
                         B_stack[m].to(x_sample.device))
                for m in range(self.basis_m)
            ])  # (M, T, O)
            # flatten the token dim -> (M, T*O)
            flat = lora_outs.reshape(self.basis_m, -1)
            flat = F.normalize(flat, p=2, dim=1)
            # mean of squared pairwise cosine similarity (off-diagonal)
            G = flat @ flat.T  # (M, M)
            mask = ~torch.eye(self.basis_m, dtype=torch.bool, device=G.device)
            self._func_div_loss = G[mask].pow(2).mean()
        else:
            self._func_div_loss = None

        return y + self.scale * y_lora


# ============================================================
# Prior / HyperNet
# ============================================================
class PriorTable(nn.Module):
    def __init__(self, K: int, dim: int, init_logvar: float = -2.0):
        super().__init__()
        self.mu = nn.Embedding(K, dim)
        self.logvar = nn.Embedding(K, dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.constant_(self.logvar.weight, init_logvar)

    def forward(self, cluster_ids: torch.Tensor):
        return self.mu(cluster_ids), self.logvar(cluster_ids)


# ============================================================
# Loss helpers
# ============================================================
# ============================================================
# Patch / policy helpers
# ============================================================
def find_target_linears(model: nn.Module, target_suffixes: Tuple[str, ...]):
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in target_suffixes)]


def get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def patch_with_hyperlora(model: nn.Module, cfg) -> Tuple[List[HyperLoRALinear], List[str]]:
    targets = find_target_linears(model, cfg.target_modules)
    target_names = [name for name, _ in targets]

    # Every requested suffix must match at least one nn.Linear. A name that
    # matches nothing (a typo, or a module this backbone does not have) would
    # otherwise shrink the adapter silently and still train to completion.
    unmatched = [s for s in cfg.target_modules
                 if not any(n.endswith(s) for n in target_names)]
    if unmatched or not targets:
        available = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                            if isinstance(m, nn.Linear)})
        raise ValueError(
            f"--target_modules did not match any linear layer of {type(model).__name__}: "
            f"{unmatched or list(cfg.target_modules)}.\n"
            f"  matched {len(targets)} linears from {list(cfg.target_modules)}\n"
            f"  linear suffixes this backbone has: {available}\n"
            "The paper uses gate_proj,up_proj,down_proj (the MLP projections)."
        )

    log(f"[hyperlora] patching {len(targets)} linears "
        f"({len(cfg.target_modules)} module types): "
        f"head={target_names[:4]} ... tail={target_names[-4:]}")
    wrapped = []
    for idx, (name, lin) in enumerate(targets):
        parent, child = get_parent_module(model, name)
        hl = HyperLoRALinear(
            lin, module_index=idx, r=cfg.lora_r, basis_m=cfg.basis_m, alpha=cfg.lora_alpha,
        )
        setattr(parent, child, hl)
        wrapped.append(hl)
    return wrapped, target_names


def set_all_coeffs(wrapped: List[HyperLoRALinear], coeffs: torch.Tensor):
    for i, m in enumerate(wrapped):
        m.set_coeffs(coeffs[i])


def collect_functional_div_loss(wrapped: List[HyperLoRALinear], device: torch.device = None) -> torch.Tensor:
    """Mean functional diversity loss across all wrapped modules."""
    losses = [m._func_div_loss for m in wrapped if m._func_div_loss is not None]
    if not losses:
        return torch.tensor(0.0, device=device)
    if device is None:
        device = losses[0].device
    total = sum(l.to(device) for l in losses)
    return total / len(losses)


def set_functional_div(wrapped: List[HyperLoRALinear], enabled: bool):
    for m in wrapped:
        m._compute_func_div = enabled


def set_basis_train_policy(wrapped: List[HyperLoRALinear]):
    """Make every LoRA basis trainable. The bases stay trainable in both
    phases; only their learning rate is scaled down (basis_{A,B}_lr_mult)."""
    for m in wrapped:
        for p in m.A_basis:
            p.requires_grad_(True)
        for p in m.B_basis:
            p.requires_grad_(True)


def freeze_base_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


# ============================================================
# Cluster helpers
# ============================================================
def infer_cluster_from_emb(x: np.ndarray, centroids: np.ndarray) -> int:
    sims = centroids @ x
    return int(np.argmax(sims))


def resolve_cluster_id(
    user_id: str,
    userid_to_cluster: Dict[str, int],
    profile_emb: Optional[np.ndarray],
    centroids: np.ndarray,
) -> int:
    cid = userid_to_cluster.get(str(user_id), None)
    if cid is not None:
        return int(cid)
    if profile_emb is not None:
        return infer_cluster_from_emb(profile_emb, centroids)
    raise ValueError(f"Cannot resolve cluster for user_id={user_id}")


# ============================================================
# Tokenization / LM loss / Generation
# ============================================================
def tokenize_batch(tok, examples, max_length, device, cfg):
    from .prompt import build_prompt
    full_input_ids, full_attention_masks, full_labels = [], [], []

    for task_name, profile_text, input_text, target_text, profile_items in examples:
        # train/eval parity: the tokenizer is passed here too, so the training prompt is
        # truncated exactly like the evaluation prompt.
        user_prompt = build_prompt(cfg, task_name, profile_text, input_text, profile_items, tok=tok)
        chat_prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        target_text = "" if target_text is None else str(target_text)

        prompt_ids = tok(chat_prompt, add_special_tokens=False, truncation=False)["input_ids"]
        # Up to max_target_tokens-1 of body, then EOS. This trains the model to predict EOS at the end of the target.
        target_ids = tok(target_text, add_special_tokens=False, truncation=True,
                         max_length=max(1, cfg.max_target_tokens - 1))["input_ids"]
        if len(target_ids) == 0:
            target_ids = tok(" ", add_special_tokens=False, truncation=True, max_length=1)["input_ids"]
        target_ids.append(tok.eos_token_id)  # train EOS (Qwen3: <|im_end|>, Llama: <|end_of_text|>)

        max_prompt_len = max(1, max_length - len(target_ids))
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + target_ids
        full_input_ids.append(input_ids)
        full_attention_masks.append([1] * len(input_ids))
        full_labels.append([-100] * len(prompt_ids) + target_ids)

    max_seq = max(len(x) for x in full_input_ids)
    pad_id = tok.pad_token_id
    batch_input_ids, batch_attention_masks, batch_labels = [], [], []
    for input_ids, attn, labels in zip(full_input_ids, full_attention_masks, full_labels):
        pad_len = max_seq - len(input_ids)
        batch_input_ids.append([pad_id] * pad_len + input_ids)
        batch_attention_masks.append([0] * pad_len + attn)
        batch_labels.append([-100] * pad_len + labels)

    return {
        "input_ids":      torch.tensor(batch_input_ids,       dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_attention_masks,  dtype=torch.long, device=device),
        "labels":         torch.tensor(batch_labels,           dtype=torch.long, device=device),
    }


def per_example_lm_loss(model, batch):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    B, Tm1, V = shift_logits.shape
    loss_flat = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_labels.reshape(-1),
        ignore_index=-100, reduction="none",
    )
    valid = (shift_labels != -100).float()
    return (loss_flat.reshape(B, Tm1) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def generate_answer(cfg, model, tok, task_name, profile_text, input_text, profile_items, device,
                    max_input_len=1024, max_new_tokens=None):
    from .prompt import build_prompt
    if max_new_tokens is None:
        max_new_tokens = cfg.max_target_tokens

    prompt = build_prompt(cfg, task_name, profile_text, input_text, profile_items, tok=tok)
    chat_prompt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(chat_prompt, return_tensors="pt", truncation=True,
              max_length=max_input_len, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        max_length=None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")],
        repetition_penalty=cfg.eval_repetition_penalty,
        no_repeat_ngram_size=cfg.eval_no_repeat_ngram_size,
    )
    gen = model.generate(**gen_kwargs)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_answer_qwen(cfg, model, tok, task_name, profile_text, input_text, profile_items, device,
                    max_input_len=1024, max_new_tokens=None):
    from .prompt import build_prompt
    if max_new_tokens is None:
        max_new_tokens = cfg.max_target_tokens

    prompt = build_prompt(cfg, task_name, profile_text, input_text, profile_items, tok=tok)
    chat_prompt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    enc = tok(chat_prompt, return_tensors="pt", truncation=True,
              max_length=max_input_len, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        max_length=None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=[
            tok.eos_token_id,
            tok.convert_tokens_to_ids("<|im_end|>"),  # Qwen3
        ],
        repetition_penalty=cfg.eval_repetition_penalty,
        no_repeat_ngram_size=cfg.eval_no_repeat_ngram_size,
    )
    gen = model.generate(**gen_kwargs)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


