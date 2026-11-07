"""
hyperlora/train.py
==================
Training pipeline for PriSM.

■ Two phases, run in order
    Phase 1: Basis warmup   - train the M LoRA bases one at a time (``run_basis_warmup``).
    Phase 2: Joint training - train the HyperNet and the bases together with the
                              language-modelling loss (``run_dirichlet_training``).

The cluster prior alpha is the fixed asymmetric table (cluster k -> 'on' on
basis k % M, 'off' elsewhere) and is never trained.

Bases stay trainable across both phases (never frozen).
"""

import math
import os
from typing import Dict, List

import numpy as np
import torch

from .buffer import (
    buffer_to_posterior, sample_user_buffer_indices, sample_prompt_buffer_indices,
    select_prompt_items,
)
from .dirichlet import (
    ConjugateHyperNet,
    DirichletPriorTable,
    kl_dirichlet,
    mixing_entropy,
)
from .model import (
    HyperLoRALinear,
    PriorTable,
    collect_functional_div_loss,
    per_example_lm_loss,
    set_all_coeffs,
    set_functional_div,
    tokenize_batch,
)
from .utils import log, deterministic_seed, write_json, embed_text_mean


def _get_train_bank(cache, record_id, user_id, row, cfg, model, tok, main_device):
    """Pick the user's profile items and their embeddings from the cache.

    Falls back to the query text itself when the user has no profile items."""
    from .utils import embed_text_mean
    all_bank_embs = cache["train_user_profile_item_embs"].get(record_id, None)
    all_bank_items = cache["train_user_bank_items"].get(record_id, [])
    if all_bank_embs is None or all_bank_embs.shape[0] == 0 or len(all_bank_items) == 0:
        fb = row.get("input_text", row.get("input", ""))
        all_bank_items = [fb]
        all_bank_embs = embed_text_mean(model, tok, fb, cfg.max_profile_len, device=main_device)[None, :]
    return all_bank_items, all_bank_embs


def _sample_buffer(all_bank_embs, all_bank_items, row, record_id, ep, ridx, cfg):
    """Buffer sampling helper.

    bank_embs (the HyperNet input) keeps buffer_size items (32 in the paper).
    bank_items (the prompt body) is rep-div sampled separately when
    max_history_items < buffer_size; otherwise it uses the same selection.
    """
    buf_idx = sample_user_buffer_indices(
        bank_embs=all_bank_embs, buffer_size=cfg.buffer_size,
        rep_ratio=cfg.buffer_rep_ratio,
        seed=deterministic_seed(f"{record_id}_{ep}_{ridx}", cfg.seed),
        pool_mult=cfg.buffer_pool_mult,
        div_q_low=cfg.buffer_div_q_low, div_q_high=cfg.buffer_div_q_high,
    )
    bank_embs = all_bank_embs[buf_idx]  # HyperNet input (32)
    prompt_idx = sample_prompt_buffer_indices(
        all_bank_embs, cfg,
        base_seed_key=f"{record_id}_{ep}_{ridx}",
    )
    prompt_src_idx = prompt_idx if prompt_idx is not None else buf_idx
    input_text = row.get("input_text", row.get("input", ""))
    bank_items = select_prompt_items(cfg, input_text, all_bank_items, prompt_src_idx)
    if len(bank_items) == 0:
        bank_items = [input_text]
    return bank_embs, bank_items


# ============================================================
# Joint training: fixed cluster prior + HyperNet + LoRA bases
# ============================================================
def run_dirichlet_training(
    cfg,
    model,
    tok,
    wrapped: List[HyperLoRALinear],
    hypernet: ConjugateHyperNet,
    prior: DirichletPriorTable,
    cache: Dict,
    train_query_rows: List[Dict],
    train_query_row_record_ids: List[str],
    train_userid_to_cluster: Dict[str, int],
    main_device: torch.device,
    basis_A_params: List,
    basis_B_params: List,
    opt: torch.optim.Optimizer,
    out_dir: str,
):
    """Train the HyperNet and the LoRA bases with the language-modelling loss.

    The cluster prior alpha is the fixed asymmetric table (cluster k -> 'on' on
    basis k % M, 'off' elsewhere) and is never trained. Basis warmup runs before
    this; see ``run_basis_warmup``.
    """
    num_modules = len(wrapped)
    total_steps = cfg.epochs * len(train_query_rows)

    H_max = math.log(cfg.basis_m)

    # ── Set the fixed cluster prior alpha (never trained) ───
    K, M = cfg.num_clusters, cfg.basis_m
    on_val = float(cfg.conjugate_alpha_on)
    off_val = float(cfg.conjugate_alpha_off)
    if not (on_val > 0 and off_val > 0):
        raise ValueError(
            f"--alpha_on and --alpha_off must both be > 0 (got on={on_val}, off={off_val}). "
            "The paper uses the asymmetric table on=2.0, off=1.0."
        )
    # Asymmetric per-cluster: cluster k -> 'on' on basis (k % M), 'off' on the rest.
    alphas_np = np.full((K, num_modules, M), off_val, dtype=np.float32)
    for k in range(K):
        alphas_np[k, :, k % M] = on_val
    log(f"[dirichlet] prior: asymmetric, on={on_val} off={off_val} "
        f"(cluster k -> basis (k%M)=on, others=off, shape=(K={K}, num_modules={num_modules}, M={M}))")
    prior.set_alphas(torch.from_numpy(alphas_np).float())

    torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    # ── Phase 2: joint training (CE + regularisers) ──
    log("[dirichlet] === Phase 2: joint training ===")
    log(f"[dirichlet] {cfg.epochs} epochs, {total_steps} total steps")

    # ── Phase 2 debug: report prior alpha and HyperNet beta state ──
    with torch.no_grad():
        _alpha = prior.alphas.cpu().numpy()
        log(f"[diag] prior alpha: shape={_alpha.shape} mean={_alpha.mean():.4f} "
            f"std={_alpha.std():.4f} min={_alpha.min():.4f} max={_alpha.max():.4f}")
        for k in range(prior.K):
            _a_k = _alpha[k].mean(axis=0) if _alpha.ndim == 3 else _alpha[k]
            log(f"[diag]   cluster {k} alpha(module-avg): {np.round(_a_k, 3)}")

        # Just dump a summary of the HyperNet state.
        log("[diag] hypernet params:")
        for name, p in hypernet.named_parameters():
            std_val = p.data.std().item() if p.data.numel() > 1 else 0.0
            log(f"[diag]   {name}: mean={p.data.mean():.4f} std={std_val:.4f} "
                f"range=[{p.data.min():.4f}, {p.data.max():.4f}]")

    # Reset the optimizer before the joint phase.
    opt.zero_grad(set_to_none=True)

    global_step = 0
    entropy_ema = H_max
    entropy_ema_alpha = 0.05

    # Enable functional diversity.
    _use_fdiv = cfg.use_functional_div
    _fdiv_w = float(cfg.functional_div_weight)
    if _use_fdiv:
        set_functional_div(wrapped, True)
        log(f"[dirichlet] functional_div enabled, weight={_fdiv_w}")

    for ep in range(cfg.epochs):
        log(f"[epoch {ep+1}/{cfg.epochs}] start")
        indices = np.random.permutation(len(train_query_rows))

        for ridx in indices:
            row = train_query_rows[ridx]
            record_id = str(train_query_row_record_ids[ridx])
            user_id = str(row["user_id"])

            if user_id not in train_userid_to_cluster:
                continue

            # ── buffer sampling ──
            all_bank_items, all_bank_embs = _get_train_bank(
                cache, record_id, user_id, row, cfg, model, tok, main_device
            )
            bank_embs, bank_items = _sample_buffer(
                all_bank_embs, all_bank_items, row, record_id, ep, ridx, cfg
            )

            # ── user embedding ──
            h = buffer_to_posterior(bank_embs).to(main_device)
            cid = int(train_userid_to_cluster[user_id])

            # ── HyperNet forward: β = α + M2·M1 ──
            pseudo_count = hypernet(h)  # (num_modules, basis_m)
            alpha_k = prior(torch.tensor(cid, device=main_device))  # (num_modules, basis_m)
            beta = alpha_k + pseudo_count

            # ── tokenize ──
            input_text = row.get("input_text", row.get("input", ""))
            target_text = row.get("target_text", row.get("output", ""))
            batch = tokenize_batch(
                tok, [(cfg.task_name, "", input_text, target_text, bank_items)],
                max_length=cfg.max_input_len, device=main_device, cfg=cfg,
            )

            try:
                # ── CE loss (mean w = β/Σβ) ──
                w = beta / beta.sum(dim=-1, keepdim=True)
                set_all_coeffs(wrapped, w)
                ce = per_example_lm_loss(model, batch).mean()

                # ── entropy tracking (per-module quantiles) ──
                w_mean = beta.detach() / beta.detach().sum(dim=-1, keepdim=True)
                H_per_module = mixing_entropy(w_mean)  # (num_modules,)
                H_current = H_per_module.mean().item()
                H_q10 = H_per_module.quantile(0.1).item()
                H_q50 = H_per_module.quantile(0.5).item()
                H_q90 = H_per_module.quantile(0.9).item()
                entropy_ema = (1 - entropy_ema_alpha) * entropy_ema + entropy_ema_alpha * H_current

                # ── KL(Dir(beta) || Dir(alpha)): logged as a diagnostic, not optimised ──
                alpha = prior(torch.tensor(cid, device=main_device))  # (num_modules, basis_m)
                kl_raw = kl_dirichlet(beta, alpha)  # (num_modules,)
                kl_loss = kl_raw.mean()

                # ── L_reg: auxiliary regularization ──
                loss_reg = torch.tensor(0.0, device=main_device)
                fdiv_val_t = torch.tensor(0.0, device=main_device)
                if _use_fdiv:
                    fdiv_val_t = _fdiv_w * collect_functional_div_loss(wrapped, device=main_device)
                    loss_reg = loss_reg + fdiv_val_t

                loss = ce + loss_reg

                # ── backward ──
                (loss / cfg.grad_accum).backward()
                global_step += 1

                if global_step % cfg.grad_accum == 0:
                    clip_params = (
                        list(hypernet.parameters())
                        + basis_A_params + basis_B_params
                    )
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # ── logging ──
                if global_step % 100 == 0:
                    log(f"[joint] step={global_step}/{total_steps} "
                        f"ce={ce.item():.4f} kl={kl_loss.item():.4f} "
                        f"H_q={H_q10/H_max:.2f}/{H_q50/H_max:.2f}/{H_q90/H_max:.2f} "
                        f"loss={loss.item():.4f}")

                    # ── Conjugate monitoring (M1, M2, pseudo_count) ──
                    if hasattr(hypernet, "forward_decomposed"):
                        with torch.no_grad():
                            m1_d, m2_d, pc_d = hypernet.forward_decomposed(h)
                            m1_ent = (-(m1_d * (m1_d + 1e-12).log()).sum(dim=-1)).mean()
                            # L2 norm of this user's module-mean M1.
                            # 1.0 -> one-hot collapse; 1/sqrt(M) (= 0.5 for M=4) -> uniform.
                            m1_l2 = m1_d.float().mean(dim=0).norm().item()
                            log(f"[conjugate] step={global_step} "
                                f"m2(mean/min/max)={m2_d.mean().item():.3f}/{m2_d.min().item():.3f}/{m2_d.max().item():.3f} "
                                f"m1_ent={m1_ent.item():.3f}/{H_max:.3f} "
                                f"m1_l2={m1_l2:.3f} "
                                f"pc_norm={pc_d.norm().item():.3f} "
                                f"β_sum(mean/max)={beta.sum(-1).mean().item():.3f}/{beta.sum(-1).max().item():.3f}")
                            m1_argmax = m1_d.argmax(dim=-1)
                            basis_hist = torch.bincount(m1_argmax, minlength=cfg.basis_m).tolist()
                            # Module-mean M1 (the user's basis preference direction).
                            m1_mean = m1_d.float().mean(dim=0).tolist()
                            # Module mean of w = beta / sum(beta) (the actual LoRA mixing coefficients).
                            w_mean = (beta.float() / beta.float().sum(-1, keepdim=True)).mean(dim=0).tolist()
                            log(f"[conjugate] step={global_step} basis_pick(this_user)={basis_hist} "
                                f"m1_mean=[{','.join(f'{v:.3f}' for v in m1_mean)}] "
                                f"w_mean=[{','.join(f'{v:.3f}' for v in w_mean)}]")

                if global_step % 500 == 0:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                log(f"[dirichlet] OOM at step={global_step} ridx={ridx} — skipping")
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

        # ── End-of-epoch residual grad ──
        if global_step % cfg.grad_accum != 0:
            clip_params = (
                list(hypernet.parameters())
                + basis_A_params + basis_B_params
            )
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # ── epoch checkpoint ──
        if cfg.epochs > 1 and (ep + 1) < cfg.epochs:
            _save_dirichlet_checkpoint(
                cfg, os.path.join(out_dir, f"epoch_{ep+1}"),
                hypernet, wrapped, prior, cache,
            )
            log(f"[ckpt] epoch={ep+1} saved")

    # Disable functional diversity.
    if _use_fdiv:
        set_functional_div(wrapped, False)

    return global_step
# ============================================================
# Checkpoint helpers
# ============================================================
def _save_dirichlet_checkpoint(cfg, out_dir, hypernet, wrapped, prior, cache):
    from .utils import ensure_dir
    ensure_dir(out_dir)

    # basis + hypernet
    torch.save(hypernet.state_dict(), os.path.join(out_dir, "hypernet_dirichlet.pt"))
    torch.save(prior.state_dict(), os.path.join(out_dir, "prior_dirichlet.pt"))

    basis = [{"A_basis": torch.stack([p.detach().cpu() for p in m.A_basis]),
              "B_basis": torch.stack([p.detach().cpu() for p in m.B_basis])} for m in wrapped]
    torch.save(basis, os.path.join(out_dir, "basis.pt"))
    np.save(os.path.join(out_dir, "train_user_centroids.npy"), cache["train_user_centroids"])
    np.save(os.path.join(out_dir, "train_user_cluster_labels.npy"), cache["train_user_cluster_labels"])
    write_json(os.path.join(out_dir, "config.json"), {
        k: v for k, v in cfg.__dict__.items()
        if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, tuple))
    })


def save_dirichlet_checkpoint(cfg, out_dir, hypernet, wrapped, prior, cache):
    """External-facing checkpoint save."""
    _save_dirichlet_checkpoint(cfg, out_dir, hypernet, wrapped, prior, cache)


def load_dirichlet_checkpoint(cfg, hypernet, wrapped, prior):
    """Load the Dirichlet checkpoint."""
    ckpt_dir = cfg.ckpt_dir if cfg.ckpt_dir else cfg.out_dir

    hypernet.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "hypernet_dirichlet.pt"), map_location="cpu")
    )
    prior.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "prior_dirichlet.pt"), map_location="cpu")
    )

    basis = torch.load(os.path.join(ckpt_dir, "basis.pt"), map_location="cpu")
    for m, b in zip(wrapped, basis):
        for i, p in enumerate(m.A_basis):
            p.data.copy_(b["A_basis"][i].to(p.device))
        for i, p in enumerate(m.B_basis):
            p.data.copy_(b["B_basis"][i].to(p.device))

    centroids = np.load(os.path.join(ckpt_dir, "train_user_centroids.npy"))
    log(f"[ckpt] dirichlet checkpoint loaded from {ckpt_dir}")

    return centroids


# ============================================================
# Basis warmup (Phase 1) and helpers
# ============================================================
def get_train_bank(cache, record_id, user_id, row, cfg, model, tok, main_device):
    """User-based cache: bank is keyed by user_id (== record_id).
    Fallback: when no bank is available, build a 1-item bank from the input text."""
    all_bank_embs  = cache["train_user_profile_item_embs"].get(record_id, None)
    all_bank_items = cache["train_user_bank_items"].get(record_id, [])

    if all_bank_embs is None or all_bank_embs.shape[0] == 0 or len(all_bank_items) == 0:
        fb = row.get("input_text", row.get("input", ""))
        all_bank_items = [fb]
        all_bank_embs = embed_text_mean(model, tok, fb, cfg.max_profile_len, device=main_device)[None, :]

    return all_bank_items, all_bank_embs


def run_basis_warmup(
    cfg,
    model,
    tok,
    wrapped: List[HyperLoRALinear],
    hypernet,
    prior: PriorTable,
    cache: Dict,
    train_query_rows: List[Dict],
    train_query_row_record_ids: List[str],
    train_userid_to_cluster: Dict[str, int],
    main_device: torch.device,
):
    log("[basis_warmup] phase 1 start: HyperNet/Prior frozen, basis CE training")
    hypernet.requires_grad_(False)
    prior.requires_grad_(False)

    dropout_rates = list(cfg.basis_warmup_dropout)
    if len(dropout_rates) < cfg.basis_m:
        dropout_rates += [dropout_rates[-1]] * (cfg.basis_m - len(dropout_rates))

    clusters_per_basis = max(1, cfg.num_clusters // cfg.basis_m)
    basis_subsets = []
    rng_warmup = np.random.default_rng(cfg.seed)

    for m_idx in range(cfg.basis_m):
        cid_start = m_idx * clusters_per_basis
        cid_end   = (m_idx + 1) * clusters_per_basis if m_idx < cfg.basis_m - 1 else cfg.num_clusters
        assigned_clusters = set(range(cid_start, cid_end))

        cluster_indices = [
            i for i, row in enumerate(train_query_rows)
            if train_userid_to_cluster.get(str(row["user_id"]), -1) in assigned_clusters
        ]
        chosen = list(cluster_indices)
        rng_warmup.shuffle(chosen)
        basis_subsets.append(chosen)
        log(f"[basis_warmup] basis={m_idx} clusters={list(assigned_clusters)} "
            f"pool={len(cluster_indices)} train={len(chosen)}")

    warmup_step = [0]  # mutable counter (updated inside the closure)

    warmup_epochs_default = cfg.basis_warmup_epochs

    def _train_single_basis(m_idx: int, epochs: int) -> None:
        """Train a single basis for the given number of epochs.
        warmup_step[0] accumulates across calls."""
        # Clear stale grads from previous basis training and release CUDA cache.
        for m in wrapped:
            for p in list(m.A_basis) + list(m.B_basis):
                p.grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dp = dropout_rates[m_idx]
        warmup_params_m = (
            [m.A_basis[m_idx] for m in wrapped] +
            [m.B_basis[m_idx] for m in wrapped]
        )
        warmup_opt = torch.optim.AdamW(
            warmup_params_m,
            lr=cfg.lr * cfg.basis_warmup_lr_mult,
            weight_decay=0.0,
        )
        warmup_opt.zero_grad(set_to_none=True)

        log(f"[basis_warmup] basis={m_idx} dropout={dp} "
            f"train={len(basis_subsets[m_idx])} epochs={epochs}")

        for ep in range(epochs):
            for ridx in basis_subsets[m_idx]:
                row       = train_query_rows[int(ridx)]
                record_id = str(train_query_row_record_ids[int(ridx)])
                user_id   = str(row["user_id"])

                if user_id not in train_userid_to_cluster:
                    continue

                all_bank_items, all_bank_embs = get_train_bank(
                    cache, record_id, user_id, row, cfg, model, tok, main_device
                )

                buf_idx = sample_user_buffer_indices(
                    bank_embs=all_bank_embs, buffer_size=cfg.buffer_size,
                    rep_ratio=cfg.buffer_rep_ratio,
                    seed=deterministic_seed(f"{record_id}_warmup_{ridx}_{ep}", cfg.seed),
                    pool_mult=cfg.buffer_pool_mult,
                    div_q_low=cfg.buffer_div_q_low, div_q_high=cfg.buffer_div_q_high,
                )
                prompt_idx = sample_prompt_buffer_indices(
                    all_bank_embs, cfg,
                    base_seed_key=f"{record_id}_warmup_{ridx}_{ep}",
                )
                prompt_src_idx = prompt_idx if prompt_idx is not None else buf_idx
                input_text  = row.get("input_text",  row.get("input",  ""))
                target_text = row.get("target_text", row.get("output", ""))
                bank_items = select_prompt_items(cfg, input_text, all_bank_items, prompt_src_idx)
                if len(bank_items) == 0:
                    bank_items = [input_text]

                batch = tokenize_batch(
                    tok, [(cfg.task_name, "", input_text, target_text, bank_items)],
                    max_length=cfg.max_input_len, device=main_device, cfg=cfg,
                )

                for w in wrapped:
                    w._warmup_basis_idx = m_idx
                    w._warmup_dropout_p = dp
                    w._use_warmup_forward = True
                ce_m = per_example_lm_loss(model, batch).mean()
                # NOTE: keep _use_warmup_forward=True until backward finishes
                # (gradient_checkpointing recomputes the forward, so the same path must be used).

                warmup_loss = ce_m

                (warmup_loss / cfg.grad_accum).backward()
                # Reset only after backward (incl. recompute) finishes.
                for w in wrapped:
                    w._use_warmup_forward = False
                warmup_step[0] += 1

                if warmup_step[0] % cfg.grad_accum == 0:
                    if cfg.basis_warmup_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(warmup_params_m, cfg.basis_warmup_grad_clip)
                    warmup_opt.step()
                    warmup_opt.zero_grad(set_to_none=True)

                if warmup_step[0] % 50 == 0:
                    log(f"[basis_warmup] step={warmup_step[0]} basis={m_idx} "
                        f"ep={ep+1}/{epochs} ce={ce_m.item():.6f}")

        # End of basis training: flush any leftover gradients.
        if warmup_step[0] % cfg.grad_accum != 0:
            if cfg.basis_warmup_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(warmup_params_m, cfg.basis_warmup_grad_clip)
            warmup_opt.step()
            warmup_opt.zero_grad(set_to_none=True)

        # Release optimizer state (Adam exp_avg etc.) from GPU memory.
        del warmup_opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for m_idx in range(cfg.basis_m):
        _train_single_basis(m_idx, warmup_epochs_default)

    hypernet.requires_grad_(True)
    prior.requires_grad_(True)
    log("[basis_warmup] phase 1 done: HyperNet/Prior unfrozen, starting phase 2 training")
