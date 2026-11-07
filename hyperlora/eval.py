from typing import Dict, List, Optional, Tuple
import json, os, time

import numpy as np
import torch

from .buffer import buffer_to_posterior
from .metrics import evaluate_predictions
from .model import (
    HyperLoRALinear,
    generate_answer, generate_answer_qwen, set_all_coeffs,
)
from .utils import log, embed_text_mean, deterministic_seed
from .prompt import build_prompt

from collections import defaultdict


def _derangement(n: int, rng: np.random.Generator, max_tries: int = 20) -> np.ndarray:
    """No-fixed-point random permutation. Returns identity if n<2 (cannot derange)."""
    if n < 2:
        return np.arange(n)
    arange_n = np.arange(n)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not (perm == arange_n).any():
            return perm
    # Fallback: swap any remaining fixed points with their neighbour.
    perm = rng.permutation(n)
    fixed = np.where(perm == arange_n)[0]
    for f in fixed:
        j = (int(f) + 1) % n
        perm[f], perm[j] = perm[j], perm[f]
    return perm


def _get_generate_fn(cfg):
    """Pick the generate function based on cfg.model_type."""
    if cfg.model_type == "qwen":
        return generate_answer_qwen
    return generate_answer
# ============================================================
# BGE encoder lazy-loaded once on first use.
# ============================================================
# ============================================================
# Rep-Div bank selection (same logic as in train).
# ============================================================

def buffer_rng(cfg, key) -> np.random.Generator:
    """RNG for the evaluation-time history buffers.

    cfg.eval_buffer_seeding:
      "deterministic" (default) -- seeded from (key, cfg.seed) with a stable SHA-1 hash, so the
          same --seed always yields the same buffers, independent of PYTHONHASHSEED.
      "random" -- drawn from OS entropy; buffers differ on every run. Use this only to measure
          how much the buffer draw itself moves a metric.
    """
    if str(cfg.eval_buffer_seeding) == "random":
        return np.random.default_rng()
    return np.random.default_rng(deterministic_seed(key, int(cfg.seed)))


def build_bank_items_rep_div(
    items: List[str],
    item_embs: np.ndarray,
    max_bank_items: int,
    rep_ratio: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[str], np.ndarray]:
    """Same rep_div selection used at training time."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(items)
    if n <= max_bank_items:
        return items, item_embs

    all_idx = np.arange(n, dtype=np.int64)
    centroid = item_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = item_embs @ centroid

    n_rep = max(0, min(int(round(max_bank_items * rep_ratio)), max_bank_items))
    n_div = max_bank_items - n_rep

    rep_pick = np.array([], dtype=np.int64)
    if n_rep > 0:
        rep_order = np.argsort(-sims)
        rep_pool_sz = min(n, max(n_rep, n_rep * 2))
        rep_pool = all_idx[rep_order[:rep_pool_sz]]
        rep_pick = rng.choice(rep_pool, size=min(n_rep, len(rep_pool)), replace=False)

    rep_set = set(rep_pick.tolist())
    remain_idx = np.array([i for i in all_idx if i not in rep_set], dtype=np.int64)

    div_pick = np.array([], dtype=np.int64)
    if n_div > 0 and len(remain_idx) > 0:
        sims_rem = sims[remain_idx]
        lo = np.quantile(sims_rem, 0.20)
        hi = np.quantile(sims_rem, 0.90)
        band_mask = (sims_rem >= lo) & (sims_rem <= hi)
        band = remain_idx[band_mask]
        if len(band) == 0:
            band = remain_idx

        cand_emb = item_embs[band]
        anchor_emb = item_embs[rep_pick] if len(rep_pick) > 0 else None

        chosen = []
        chosen_mask = np.zeros(len(band), dtype=bool)
        if anchor_emb is not None and anchor_emb.shape[0] > 0:
            max_sim_to_set = np.max(cand_emb @ anchor_emb.T, axis=1)
        else:
            max_sim_to_set = np.full(len(band), -1.0, dtype=np.float32)

        seed_i = int(np.argmin(max_sim_to_set))
        chosen.append(int(band[seed_i]))
        chosen_mask[seed_i] = True
        max_sim_to_set = np.maximum(max_sim_to_set, cand_emb @ cand_emb[seed_i])

        for _ in range(1, n_div):
            if len(chosen) >= len(band):
                break
            scores = np.where(chosen_mask, 1e9, max_sim_to_set)
            j = int(np.argmin(scores))
            if chosen_mask[j]:
                break
            chosen_mask[j] = True
            chosen.append(int(band[j]))
            max_sim_to_set = np.maximum(max_sim_to_set, cand_emb @ cand_emb[j])

        div_pick = np.array(chosen, dtype=np.int64)

    selected_idx = np.concatenate([rep_pick, div_pick]).astype(np.int64)

    if len(selected_idx) < max_bank_items:
        remain = np.setdiff1d(all_idx, selected_idx)
        if len(remain) > 0:
            extra = rng.choice(
                remain,
                size=min(max_bank_items - len(selected_idx), len(remain)),
                replace=False,
            )
            selected_idx = np.concatenate([selected_idx, extra])

    selected_idx = selected_idx[:max_bank_items]
    return [items[i] for i in selected_idx], item_embs[selected_idx]


def _select_profile_items(all_bank_items: List[str], rep_items: List[str],
                          input_text: str, cfg) -> List[str]:
    """Pick the profile_items shown in the prompt.

    The top-k items are retrieved from the user's full bank with BM25, matching
    the training path. When the user has no bank items the rep-div buffer is
    used as it is.
    """
    if all_bank_items:
        from .buffer import sample_prompt_bm25
        top_k = int(cfg.prompt_bm25_top_k)
        idx = sample_prompt_bm25(input_text, all_bank_items, top_k,
                                 task_name=cfg.task_name)
        return [all_bank_items[j] for j in idx]
    return rep_items


# ============================================================
# evaluate_base_dev
# ============================================================
@torch.no_grad()
def evaluate_base_dev(
    cfg,
    model,
    tok,
    dev_rows: List[Dict],
    main_device: torch.device,
    dev_row_user_ids: List[str],
    dev_user_bank_items: Dict[str, List[str]],
    dev_user_profile_item_embs: Dict[str, np.ndarray],
    max_samples: int = -1,
) -> dict:
    preds, refs = [], []
    model.eval()
    n = len(dev_rows) if max_samples < 0 else min(len(dev_rows), max_samples)

    _eval_start = time.time()
    log(f"[eval_base] starting: {n} samples")
    _diag = bool(cfg.log_diagnostics)

    for i in range(n):
        row = dev_rows[i]
        user_id = str(row.get("user_id") or dev_row_user_ids[i])
        input_text = row.get("input_text", row.get("input", ""))
        ref = row.get("target_text", row.get("output", ""))

        all_bank_items = dev_user_bank_items.get(user_id, [])
        all_bank_embs = dev_user_profile_item_embs.get(
            user_id, np.empty((0, 0), dtype=np.float32)
        )

        if len(all_bank_items) == 0 or all_bank_embs.shape[0] == 0:
            profile_items = [input_text]
        else:
            # base_dev runs without the HyperNet: build a rep_div buffer, then
            # pick the prompt items.
            rep_items, buffer_embs = build_bank_items_rep_div(
                all_bank_items, all_bank_embs,
                max_bank_items=cfg.buffer_size,
                rep_ratio=cfg.buffer_rep_ratio,
                rng=buffer_rng(cfg, user_id),
            )
            # Independent prompt-side sampling (only when max_history_items < buffer_size).
            if 0 < cfg.max_history_items < cfg.buffer_size:
                prompt_rep_items, _ = build_bank_items_rep_div(
                    all_bank_items, all_bank_embs,
                    max_bank_items=cfg.max_history_items,
                    rep_ratio=cfg.buffer_rep_ratio,
                    rng=buffer_rng(cfg, f"{user_id}_prompt"),
                )
            else:
                prompt_rep_items = rep_items
            profile_items = _select_profile_items(
                all_bank_items, prompt_rep_items, input_text, cfg)

        if _diag and i == 0:
            prompt = build_prompt(cfg, cfg.task_name, "", input_text, profile_items, tok=tok)
            log(f"[diag] prompt=\n{prompt}")

        _generate = _get_generate_fn(cfg)
        pred = _generate(
            cfg, model, tok, cfg.task_name, "", input_text, profile_items,
            device=main_device, max_input_len=cfg.max_input_len,
            max_new_tokens=cfg.max_target_tokens,
        )
        pred = pred.strip('"').strip("'").strip()
        preds.append(pred)
        refs.append(ref)

        if _diag and i < 10:
            log(f"[diag] pred={pred!r} ref={ref!r}")
            log(f"[diag] prompt(bm25)={len(profile_items)} items")

        # Progress log every 50 samples (and at the end).
        done = i + 1
        if done % 50 == 0 or done == n:
            elapsed = time.time() - _eval_start
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            log(f"[progress] {done}/{n} ({100*done/n:.1f}%) "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

        # Partial dump every 500 samples.
        if done % 500 == 0 and done < n:
            try:
                with open(os.path.join(cfg.out_dir, "dev_preds_refs_base_partial.json"), "w") as f:
                    json.dump({"preds": preds, "refs": refs, "n_done": done, "n_total": n}, f)
            except Exception as e:
                log(f"[warn] partial dump failed: {e}")
            try:
                from .metrics import rouge_scores
                r1s, rLs = zip(*[rouge_scores(p, r) for p, r in zip(preds, refs)])
                log(f"[partial@{done}/{n}] R1={sum(r1s)/len(r1s):.4f}  RL={sum(rLs)/len(rLs):.4f}")
            except Exception as e:
                log(f"[warn] partial rouge failed: {e}")

    # Persist raw pred/ref for post-hoc analysis (dedup, stop_strings comparison, ...).
    try:
        with open(os.path.join(cfg.out_dir, "dev_preds_refs_base.json"), "w") as f:
            json.dump({"preds": preds, "refs": refs}, f)
    except Exception as e:
        log(f"[warn] failed to save preds/refs: {e}")

    return evaluate_predictions(cfg.task_name, preds, refs)


# ============================================================
# evaluate_dev
# ============================================================
@torch.no_grad()
def evaluate_dev(
    cfg,
    model,
    tok,
    wrapped: List[HyperLoRALinear],
    hypernet: torch.nn.Module,
    prior: torch.nn.Module,
    dev_rows: List[Dict],
    train_user_centroids: np.ndarray,
    main_device: torch.device,
    dev_user_profile_embs: np.ndarray,
    dev_userid_to_index: Dict[str, int],
    dev_userid_to_cluster: Dict[str, int],
    dev_row_user_ids: List[str],
    dev_user_bank_items: Dict[str, List[str]],
    dev_user_profile_item_embs: Dict[str, np.ndarray],
    train_cluster_sizes: np.ndarray,
    max_samples: int = -1,
    train_user_profile_embs: Optional[np.ndarray] = None,
) -> dict:

    torch.cuda.empty_cache()

    preds, refs = [], []
    model.eval()
    hypernet.eval()
    prior.eval()

    _diag = bool(cfg.log_diagnostics)
    prev_buffer_embs = None

    cluster_repr_coeffs = {}

    _eval_start = time.time()
    _eval_n = len(dev_rows) if max_samples < 0 else min(len(dev_rows), max_samples)
    log(f"[eval] starting: {_eval_n} samples")

    # ── eval swap mapping (computed once) ──
    eval_swap_mode = cfg.eval_swap_mode
    swap_mapping: Dict[str, str] = {}
    if eval_swap_mode != "none":
        seed = int(cfg.eval_swap_seed)
        rng = np.random.default_rng(seed)
        dev_users = sorted({str(u) for u in dev_row_user_ids})
        if eval_swap_mode == "random":
            users_arr = np.array(dev_users)
            perm = _derangement(len(users_arr), rng)
            swap_mapping = dict(zip(users_arr.tolist(), users_arr[perm].tolist()))
        elif eval_swap_mode == "random_in_cluster":
            by_cluster: Dict[int, List[str]] = defaultdict(list)
            for u in dev_users:
                c = int(dev_userid_to_cluster.get(u, -1))
                if c >= 0:
                    by_cluster[c].append(u)
            for c, us in by_cluster.items():
                us_arr = np.array(us)
                if len(us_arr) < 2:
                    for u in us:
                        swap_mapping[u] = u  # singleton cluster: cannot derange, map to self
                    continue
                perm = _derangement(len(us_arr), rng)
                for orig, sw in zip(us_arr.tolist(), us_arr[perm].tolist()):
                    swap_mapping[orig] = sw
        else:
            raise ValueError(f"unknown eval_swap_mode={eval_swap_mode}")
        self_count = sum(1 for k, v in swap_mapping.items() if k == v)
        log(f"[eval-swap] mode={eval_swap_mode} seed={seed} "
            f"mapped={len(swap_mapping)} self={self_count}")

    for i, row in enumerate(dev_rows):
        if max_samples > 0 and i >= max_samples:
            break

        orig_user_id = str(row.get("user_id") or dev_row_user_ids[i])
        # The prompt path always uses the original user's history.
        user_id = orig_user_id
        # In swap mode the HyperNet path overrides buffer_embs with another user's items (handled below).
        swap_user_id = (
            swap_mapping.get(orig_user_id, orig_user_id)
            if eval_swap_mode != "none" else orig_user_id
        )
        input_text = row.get("input_text", row.get("input", ""))
        ref = row.get("target_text", row.get("output", ""))

        all_bank_items = dev_user_bank_items.get(user_id, [])
        all_bank_embs = dev_user_profile_item_embs.get(
            user_id, np.empty((0, 0), dtype=np.float32)
        )

        if len(all_bank_items) == 0 or all_bank_embs.shape[0] == 0:
            # fallback
            all_bank_items = [input_text]
            all_bank_embs = embed_text_mean(
                model, tok, input_text,
                cfg.max_profile_len, device=main_device,
            )[None, :]
            buffer_embs = all_bank_embs
            profile_items = all_bank_items

        else:
            # HyperNet input: rep_div (matches training; always).
            rep_items, buffer_embs = build_bank_items_rep_div(
                all_bank_items, all_bank_embs,
                max_bank_items=cfg.buffer_size,
                rep_ratio=cfg.buffer_rep_ratio,
                rng=buffer_rng(cfg, user_id),
            )

            # Independent prompt-side sampling (only when max_history_items < buffer_size).
            if 0 < cfg.max_history_items < cfg.buffer_size:
                prompt_rep_items, _ = build_bank_items_rep_div(
                    all_bank_items, all_bank_embs,
                    max_bank_items=cfg.max_history_items,
                    rep_ratio=cfg.buffer_rep_ratio,
                    rng=buffer_rng(cfg, f"{user_id}_prompt"),
                )
            else:
                prompt_rep_items = rep_items

            profile_items = _select_profile_items(
                all_bank_items, prompt_rep_items, input_text, cfg)

        # Right after buffer_embs is computed inside the loop.
        if _diag and prev_buffer_embs is not None and i < 10:
            inter_sims = buffer_embs @ prev_buffer_embs.T  # (32, 32)
            log(f"[diag] inter_user_sim user_{i-1}_vs_{i}: "
                f"mean={inter_sims.mean():.3f} max={inter_sims.max():.3f} min={inter_sims.min():.3f}")
        prev_buffer_embs = buffer_embs.copy()

        # ── eval-swap: replace HyperNet input with swap user's buffer_embs (prompt stays original) ──
        buffer_embs_for_h = buffer_embs
        if eval_swap_mode != "none" and swap_user_id != orig_user_id:
            swap_items = dev_user_bank_items.get(swap_user_id, [])
            swap_embs  = dev_user_profile_item_embs.get(
                swap_user_id, np.empty((0, 0), dtype=np.float32)
            )
            if len(swap_items) > 0 and swap_embs.shape[0] > 0:
                _, buffer_embs_for_h = build_bank_items_rep_div(
                    swap_items, swap_embs,
                    max_bank_items=cfg.buffer_size,
                    rep_ratio=cfg.buffer_rep_ratio,
                    rng=buffer_rng(cfg, swap_user_id),
                )
            if i < 3:
                log(f"[eval-swap] sample {i}: orig={orig_user_id} swap={swap_user_id} "
                    f"(prompt=orig, h=swap)")

        # HyperNet input: rep_div buffer (matches training).
        h = buffer_to_posterior(buffer_embs_for_h).to(main_device)

        # HyperNet forward
        # At eval time we infer the user's cluster from cached centroids and look up alpha.
        # (In swap mode buffer_embs_for_h belongs to the swap user, so cid is the swap user's cluster.)
        profile_emb = buffer_embs_for_h.mean(axis=0)
        profile_emb = profile_emb / (np.linalg.norm(profile_emb) + 1e-12)
        cid = int(np.argmax(train_user_centroids @ profile_emb))
        pseudo_count = hypernet(h)  # (num_modules, basis_m)
        alpha_k = prior(torch.tensor(cid, device=main_device))  # (num_modules, basis_m)
        beta = alpha_k + pseudo_count
        if _diag and i < 3:
            log(f"[diag] conjugate β = α+M2·M1 "
                f"pseudo_count_mean={pseudo_count.mean().item():.4f} "
                f"β_mean={beta.mean().item():.4f}")

        # Mixing coefficients: the Dirichlet mean E[w] = beta / beta.sum().
        coeffs = beta / beta.sum(dim=-1, keepdim=True)
        if cid not in cluster_repr_coeffs:
            cluster_repr_coeffs[cid] = coeffs.cpu().detach().numpy()
        # Print once all clusters have been seen.
        if _diag and len(cluster_repr_coeffs) == cfg.num_clusters \
                and "done" not in cluster_repr_coeffs:
            log("[diag] ===== per-cluster coefficients =====")
            for c_id, c in sorted(cluster_repr_coeffs.items()):
                log(f"[diag] cluster={c_id} coeffs[0]={c[0].tolist()}")
            cluster_repr_coeffs["done"] = True

        if _diag and i == 0:
            prompt = build_prompt(cfg, cfg.task_name, "", input_text, profile_items, tok=tok)
            log(f"[diag] prompt=\n{prompt}")

        _generate = _get_generate_fn(cfg)
        set_all_coeffs(wrapped, coeffs)
        pred = _generate(
            cfg, model, tok, cfg.task_name, "", input_text, profile_items,
            device=main_device, max_input_len=cfg.max_input_len,
            max_new_tokens=cfg.max_target_tokens,
        )
        pred = pred.strip('"').strip("'").strip()
        preds.append(pred)
        refs.append(ref)

        if _diag and i < 10:
            log(f"[diag] coeffs[0]={coeffs[:3].cpu().detach().numpy()}")
            log(f"[diag] pred={pred!r} ref={ref!r}")
            log(f"[diag] buffer={len(buffer_embs)} "
                f"prompt(bm25)={len(profile_items)}/{len(all_bank_embs)} items")
        
        # Progress log every 50 samples (and at the end).
        done = i + 1
        if done % 50 == 0 or done == _eval_n:
            elapsed = time.time() - _eval_start
            rate = done / max(elapsed, 1e-6)
            eta = (_eval_n - done) / max(rate, 1e-6)
            log(f"[progress] {done}/{_eval_n} ({100*done/_eval_n:.1f}%) "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

        # Partial dump every 500 samples (recoverable if killed/crashed).
        if done % 500 == 0 and done < _eval_n:
            try:
                with open(os.path.join(cfg.out_dir, "dev_preds_refs_partial.json"), "w") as f:
                    json.dump({"preds": preds, "refs": refs, "n_done": done, "n_total": _eval_n}, f)
            except Exception as e:
                log(f"[warn] partial dump failed: {e}")
            try:
                from .metrics import rouge_scores
                r1s, rLs = zip(*[rouge_scores(p, r) for p, r in zip(preds, refs)])
                log(f"[partial@{done}/{_eval_n}] R1={sum(r1s)/len(r1s):.4f}  RL={sum(rLs)/len(rLs):.4f}")
            except Exception as e:
                log(f"[warn] partial rouge failed: {e}")

    # Persist raw pred/ref for post-hoc analysis.
    # The partial file is kept as a safety net in case the final write is interrupted.
    try:
        with open(os.path.join(cfg.out_dir, "dev_preds_refs.json"), "w") as f:
            json.dump({"preds": preds, "refs": refs}, f)
    except Exception as e:
        log(f"[warn] failed to save preds/refs: {e}")

    return evaluate_predictions(cfg.task_name, preds, refs)
