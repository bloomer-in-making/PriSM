"""
hyperlora/buffer.py
===================
User-profile buffer sampling utilities.

Public API
    buffer_to_posterior      : bank item embeddings -> HyperNet input vector h
                               h = [mu, log(var)] (2D concat).
    sample_user_buffer_indices : pick buffer_size item indices out of the bank.
                                 In the cache produced by precompute.py the bank
                                 is already rep-div selected, so re-sampling at
                                 train time is only useful as a mild data
                                 augmentation. Policies: random / head / tail /
                                 rep_div.

rep_div policy (Representative + Diverse)
    Two complementary picking rules to cover the user's taste:
    - Representative: items close to the user-level centroid (high similarity).
    - Diverse       : items far apart from each other (farthest-first).
    - Default mix   : rep 50%, div 50%.

Dependencies: numpy and torch only.
"""
from typing import Optional

import numpy as np
import torch


def buffer_to_posterior(embs: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    """h = [mu, log(var)] pooled over the buffer items."""
    mu = embs.mean(axis=0)
    var = embs.var(axis=0) + eps
    h = np.concatenate([mu, np.log(var)], axis=0)
    return torch.tensor(h, dtype=torch.float32)


def _quantile_band_indices(values: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    '''Narrow down the candidate pool used for diverse selection.

    Low similarity to centroid <----------> High similarity
    [excluded outliers][--- diverse pool ---][rep candidates]
                  q_low=0.2          q_high=0.9
    '''

    if values.size == 0:
        return np.array([], dtype=np.int64)
    q_low = float(np.clip(q_low, 0.0, 1.0))
    q_high = float(np.clip(q_high, 0.0, 1.0))
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    lo, hi = np.quantile(values, q_low), np.quantile(values, q_high)
    return np.where((values >= lo) & (values <= hi))[0].astype(np.int64)


def _farthest_first_select(
    cand_emb: np.ndarray,
    cand_ids: np.ndarray,
    k: int,
    rng: np.random.Generator,
    anchor_embs: Optional[np.ndarray] = None,
) -> np.ndarray:
    Nc = cand_emb.shape[0]
    if Nc == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    if Nc <= k:
        return cand_ids.copy()

    chosen_mask = np.zeros(Nc, dtype=bool)
    if anchor_embs is not None and anchor_embs.shape[0] > 0:
        max_sim = np.max(cand_emb @ anchor_embs.T, axis=1)
        seed_i = int(np.argmin(max_sim))
    else:
        seed_i = int(rng.integers(0, Nc))

    chosen = [int(cand_ids[seed_i])]
    chosen_mask[seed_i] = True

    if anchor_embs is not None and anchor_embs.shape[0] > 0:
        max_sim_to_set = np.max(cand_emb @ anchor_embs.T, axis=1)
    else:
        max_sim_to_set = np.full(Nc, -1.0, dtype=np.float32)
    max_sim_to_set = np.maximum(max_sim_to_set, cand_emb @ cand_emb[seed_i])

    for _ in range(1, k):
        scores = np.where(chosen_mask, 1e9, max_sim_to_set)
        j = int(np.argmin(scores))
        if chosen_mask[j]:
            break
        chosen_mask[j] = True
        chosen.append(int(cand_ids[j]))
        max_sim_to_set = np.maximum(max_sim_to_set, cand_emb @ cand_emb[j])

    return np.array(chosen, dtype=np.int64)


def sample_user_buffer_indices(
    bank_embs: np.ndarray,
    buffer_size: int,
    rep_ratio: float = 0.5,
    seed: Optional[int] = None,
    pool_mult: int = 8,
    div_q_low: float = 0.20,
    div_q_high: float = 0.90,
) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    n = len(bank_embs)
    if n == 0:
        return np.array([], dtype=np.int64)
    if buffer_size <= 0 or buffer_size >= n:
        return np.arange(n, dtype=np.int64)

    all_idx = np.arange(n, dtype=np.int64)

    centroid = bank_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = bank_embs @ centroid

    n_rep = max(0, min(int(round(buffer_size * rep_ratio)), buffer_size))
    n_div = buffer_size - n_rep

    rep_pick = np.array([], dtype=np.int64)
    if n_rep > 0:
        rep_order = np.argsort(-sims)
        rep_pool_sz = min(n, max(n_rep, int(np.ceil(n_rep * 2.0))))
        rep_candidates = all_idx[rep_order[:rep_pool_sz]]
        rep_pick = (rng.choice(rep_candidates, size=n_rep, replace=False)
                    if len(rep_candidates) > n_rep else rep_candidates.copy())

    rep_set = set(rep_pick.tolist())
    div_pick = np.array([], dtype=np.int64)
    if n_div > 0:
        remain_idx = np.array([i for i in all_idx if i not in rep_set], dtype=np.int64)
        if len(remain_idx) > 0:
            sims_rem = sims[remain_idx]
            band_local = _quantile_band_indices(sims_rem, div_q_low, div_q_high)
            if band_local.size == 0:
                band_local = np.arange(len(remain_idx), dtype=np.int64)
            pool_sz = min(len(band_local), max(n_div, n_div * pool_mult))
            pool_local = band_local[np.argsort(sims_rem[band_local])][:pool_sz]
            div_pick = _farthest_first_select(
                cand_emb=bank_embs[remain_idx][pool_local],
                cand_ids=remain_idx[pool_local],
                k=n_div, rng=rng,
                anchor_embs=bank_embs[rep_pick] if len(rep_pick) > 0 else None,
            )

    buf = np.concatenate([rep_pick, div_pick])
    if len(buf) < buffer_size:
        remain = np.setdiff1d(all_idx, buf)
        if len(remain) > 0:
            extra = rng.choice(remain, size=min(buffer_size - len(buf), len(remain)), replace=False)
            buf = np.concatenate([buf, extra])

    return np.sort(buf[:buffer_size])


def sample_prompt_buffer_indices(
    bank_embs: np.ndarray,
    cfg,
    base_seed_key: str,
) -> Optional[np.ndarray]:
    """Independent buffer sample for the prompt path (separate from HyperNet input).

    Returns:
        - rep_div sampled indices of size cfg.max_history_items, when
          cfg.max_history_items < cfg.buffer_size.
        - None otherwise, signalling the caller to reuse the existing buf_idx.

    The seed is base_seed_key + "_prompt" so the prompt sample is drawn
    independently from the HyperNet sample.
    """
    from .utils import deterministic_seed
    max_items = int(cfg.max_history_items)
    buffer_size = int(cfg.buffer_size)
    if not (0 < max_items < buffer_size):
        return None
    return sample_user_buffer_indices(
        bank_embs=bank_embs,
        buffer_size=max_items,
        rep_ratio=cfg.buffer_rep_ratio,
        seed=deterministic_seed(f"{base_seed_key}_prompt", cfg.seed),
        pool_mult=cfg.buffer_pool_mult,
        div_q_low=cfg.buffer_div_q_low,
        div_q_high=cfg.buffer_div_q_high,
    )


def sample_prompt_bm25(
    query_text: str,
    bank_texts,
    top_k: int = 4,
    task_name: Optional[str] = None,
) -> np.ndarray:
    """Return the indices of the top-k bank items by BM25 score (in rank order).

    If task_name is given, the task-specific BM25 query/bank text extractor
    in prompt.py is applied. Returned indices are positions in the original
    bank_texts list, so the caller can index back into the raw items.
    """
    from rank_bm25 import BM25Okapi
    if not bank_texts:
        return np.array([], dtype=np.int64)
    if task_name is not None:
        from .prompt import extract_bm25_query, extract_bm25_bank_text
        query_text = extract_bm25_query(task_name, query_text)
        bank_texts = [extract_bm25_bank_text(task_name, t) for t in bank_texts]
    tokenized = [str(t).lower().split() for t in bank_texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(str(query_text).lower().split())
    k = min(int(top_k), len(bank_texts))
    return np.argsort(-scores)[:k].astype(np.int64)


def select_prompt_items(cfg, input_text: str, all_bank_items, fallback_indices):
    """Pick profile_items for the prompt (shared by train and eval).

    BM25 top-k against the input query. When the user has no bank items the
    caller's fallback_indices are used instead.
    """
    if all_bank_items:
        top_k = int(cfg.prompt_bm25_top_k)
        idx = sample_prompt_bm25(input_text, all_bank_items, top_k,
                                 task_name=cfg.task_name)
    else:
        idx = fallback_indices
    return [all_bank_items[j] for j in idx]
