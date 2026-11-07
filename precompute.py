"""
precompute.py
=============

Precompute the user-based cache used by training and evaluation:

  - Encode every profile item with a frozen BGE-small encoder.
  - For each user, select 32 representative+diverse items (rep-div bank).
  - Cluster users into K groups (k-means on per-user mean embeddings).
  - Save bank items, bank embeddings, cluster centroids/labels, and the
    train/dev query rows to ``--out_dir``.

Supported tasks (``--task_name``) — the six generation tasks of the nine
evaluated in the paper:

  LaMP_4                              news headline generation
  LaMP_5                              paper title generation
  LaMP_7                              tweet paraphrasing
  LongLaMP_abstract_generation_user   paper abstract generation
  LongLaMP_topic_writing_user         reddit topic writing
  LongLaMP_product_review_user        product review generation

Run once per task before any training. The expensive part is the BGE encoding
pass; on a single GPU with bf16 it takes a few minutes for LaMP-4 and longer
for the LongLaMP tasks, whose profiles are much longer.

Example:
  python precompute.py \
      --task_name       LaMP_4 \
      --train_questions data/LaMP_4/train/train_questions.json \
      --train_outputs   data/LaMP_4/train/train_outputs.json \
      --dev_questions   data/LaMP_4/dev/dev_questions.json \
      --dev_outputs     data/LaMP_4/dev/dev_outputs.json \
      --out_dir         cache/LaMP_4 \
      --seed 42 --cluster_seed 9
"""

import argparse
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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


# ============================================================
# Config
# ============================================================
@dataclass
class CacheCfg:
    task_name: str = "LaMP_4"
    encoder_model_name: str = "BAAI/bge-small-en-v1.5"

    train_questions: str = ""
    train_outputs: str = ""
    dev_questions: str = ""
    dev_outputs: str = ""
    out_dir: str = "cache/LaMP_4"

    num_clusters: int = 4
    item_max_len: int = 512
    item_batch_size: int = 8
    profile_batch_size: int = 4
    max_profile_len: int = 512

    # rep-div bank: select buffer_size items from the user's full profile.
    buffer_size: int = 32
    bank_rep_ratio: float = 0.5

    seed: int = 42
    # k-means cluster seed. -1 means: fall back to ``seed`` (single-seed mode).
    # The per-task values used in the paper are listed in scripts/run_task.sh;
    # LaMP-4 uses ``--seed 42 --cluster_seed 9``.
    cluster_seed: int = -1

    # Optional k-means seed sweep. When both bounds are >= 0 the cluster seed is
    # chosen by silhouette (cosine) among the seeds whose largest/smallest
    # cluster-size ratio stays below ``balance_threshold``. This is how the
    # per-task cluster seeds in scripts/run_task.sh were obtained (range 0-30);
    # the sweep looks at train users only.
    seed_sweep_low: int = -1
    seed_sweep_high: int = -1
    balance_threshold: float = 2.0
    sil_sample_size: int = 0            # <= 0 means use all users
    device: str = "cuda"
    dtype: str = "bfloat16"
    main_device: str = "cuda:0"


# ============================================================
# LaMP data loading (questions + outputs merge by id)
# ============================================================
def load_lamp_data(questions_path: str, outputs_path: str) -> List[Dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        qs = json.load(f)
    if isinstance(qs, dict):
        qs = qs.get("data", qs.get("golds", list(qs.values())[0]))

    with open(outputs_path, "r", encoding="utf-8") as f:
        outs = json.load(f)
    golds = outs.get("golds", outs.get("data", [])) if isinstance(outs, dict) else outs
    id_to_output = {str(g["id"]): g["output"] for g in golds}

    result = []
    for record in qs:
        rec_id = str(record["id"])
        result.append({
            "id": rec_id,
            "user_id": str(record.get("user_id", rec_id)),
            "input": record.get("input", ""),
            "output": id_to_output.get(rec_id, ""),
            "profile": record.get("profile", []),
        })
    return result


# ============================================================
# Profile item formatting (per task)
#
# Two text views of the same profile item:
#
#   profile_item_to_text       -> BGE encoder input. Never change this for an
#                                 existing cache: the embeddings (and therefore
#                                 the clustering) depend on it byte for byte.
#   profile_item_to_save_text  -> text written to *_bank_items.json, i.e. what
#                                 the prompt shows and what BM25 indexes.
#
# They differ only for LongLaMP_product_review_user (see below). For the other
# five tasks the two are identical.
#
# In every format the gold target field is placed last so that prompt.py can
# keep it as a suffix while truncating the body.
# ============================================================
def profile_item_to_text(item: Dict, task_name: str) -> str:
    """Format a profile entry as a single string (BGE encoder input)."""
    if not isinstance(item, dict):
        return str(item).strip()

    if task_name == "LaMP_4":
        # "article: {text} headline: {title}"
        parts = []
        if item.get("text"):
            parts.append(f"article: {item['text'].strip()}")
        if item.get("title"):
            parts.append(f"headline: {item['title'].strip()}")
        return " ".join(parts)

    if task_name == "LaMP_5":
        # "abstract: {abstract} title: {title}"
        parts = []
        if item.get("abstract"):
            parts.append(f"abstract: {item['abstract'].strip()}")
        if item.get("title"):
            parts.append(f"title: {item['title'].strip()}")
        return " ".join(parts)

    if task_name == "LaMP_7":
        # A past tweet; no gold-field suffix.
        return str(item.get("text", "")).strip()

    if task_name == "LongLaMP_abstract_generation_user":
        # Target = abstract, so title is the preserved suffix.
        parts = []
        if item.get("abstract"):
            parts.append(f"abstract: {str(item['abstract']).strip()}")
        if item.get("title"):
            parts.append(f"title: {str(item['title']).strip()}")
        return " ".join(parts)

    if task_name == "LongLaMP_topic_writing_user":
        # Target = content; summary acts as the TL;DR suffix.
        parts = []
        if item.get("content"):
            parts.append(f"content: {str(item['content']).strip()}")
        if item.get("summary"):
            parts.append(f"summary: {str(item['summary']).strip()}")
        return " ".join(parts)

    if task_name == "LongLaMP_product_review_user":
        # Encoder input puts the user's own review text first: the BGE encoder
        # truncates at 512 tokens, and the product description (which is not
        # the user's writing) must be the part that gets cut.
        parts = []
        if item.get("reviewText"):
            parts.append(f"review: {str(item['reviewText']).strip()}")
        if item.get("description"):
            parts.append(f"description: {str(item['description']).strip()}")
        if item.get("overall") is not None:
            parts.append(f"rating: {item['overall']}")
        return " ".join(parts)

    parts = [str(v).strip() for v in item.values()
             if isinstance(v, str) and v.strip()]
    return " ".join(parts)


def profile_item_to_save_text(item: Dict, task_name: str) -> str:
    """Text stored in the cache (prompt display + BM25 index).

    Identical to ``profile_item_to_text`` except for
    ``LongLaMP_product_review_user``, where the field order matches the query
    ("description -> summary -> review -> rating") so that BM25 compares like
    with like, and ``summary`` is included.
    """
    if not isinstance(item, dict):
        return str(item).strip()

    if task_name == "LongLaMP_product_review_user":
        parts = []
        if item.get("description"):
            parts.append(f"description: {str(item['description']).strip()}")
        if item.get("summary"):
            parts.append(f"summary: {str(item['summary']).strip()}")
        if item.get("reviewText"):
            parts.append(f"review: {str(item['reviewText']).strip()}")
        if item.get("overall") is not None:
            parts.append(f"rating: {item['overall']}")
        return " ".join(parts)

    return profile_item_to_text(item, task_name)


def get_bank_items_pair(profile: List[Dict],
                        task_name: str) -> Tuple[List[str], List[str]]:
    """profile list -> (encoder-input items, stored items), index-aligned.

    Deduplication is driven by the encoder-input text so that the two lists
    stay the same length and position i refers to the same profile entry.
    """
    seen = set()
    embed_items: List[str] = []
    save_items: List[str] = []
    for p in profile:
        embed_text = profile_item_to_text(p, task_name).strip()
        if not embed_text or embed_text in seen:
            continue
        seen.add(embed_text)
        embed_items.append(embed_text)
        save_items.append(profile_item_to_save_text(p, task_name).strip())
    return embed_items, save_items


def build_bank_text(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items])


# ============================================================
# BGE encoder (CLS pooling, L2 normalised)
# ============================================================
@torch.no_grad()
def embed_text_mean_batch(
    model,
    tok,
    texts: List[str],
    max_len: int = 512,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t is not None else "" for t in texts[i:i + batch_size]]
        x = tok(batch, return_tensors="pt", padding=True, truncation=True,
                max_length=max_len)
        x = {k: v.to(device) for k, v in x.items()}
        out = model(**x, return_dict=True)
        emb = out.last_hidden_state[:, 0, :]  # CLS
        emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb.cpu())
        del x, out, emb
    return torch.cat(all_embs, dim=0).numpy().astype(np.float32)


# ============================================================
# Rep-div bank selection
#
# For each user we select ``max_bank_items`` items from their full profile
# such that (a) ``rep_ratio`` of them lie close to the user-level centroid
# (representative) and (b) the rest are spread out via farthest-first in a
# similarity band that excludes outliers (diverse).
# ============================================================
def build_bank_items_rep_div(
    items: List[str],
    item_embs: np.ndarray,
    max_bank_items: int,
    rep_ratio: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[str], np.ndarray]:
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
        rep_pick = rng.choice(rep_pool, size=min(n_rep, len(rep_pool)),
                              replace=False)

    rep_set = set(rep_pick.tolist())
    remain_idx = np.array([i for i in all_idx if i not in rep_set],
                          dtype=np.int64)

    div_pick = np.array([], dtype=np.int64)
    if n_div > 0 and len(remain_idx) > 0:
        sims_rem = sims[remain_idx]
        lo = np.quantile(sims_rem, 0.20)
        hi = np.quantile(sims_rem, 0.90)
        band_mask = (sims_rem >= lo) & (sims_rem <= hi)
        band = remain_idx[band_mask] if band_mask.any() else remain_idx

        cand_emb = item_embs[band]
        anchor_emb = item_embs[rep_pick] if len(rep_pick) > 0 else None

        chosen = []
        chosen_mask = np.zeros(len(band), dtype=bool)
        if anchor_emb is not None and anchor_emb.shape[0] > 0:
            max_sim = np.max(cand_emb @ anchor_emb.T, axis=1)
        else:
            max_sim = np.full(len(band), -1.0, dtype=np.float32)

        seed_i = int(np.argmin(max_sim))
        chosen.append(int(band[seed_i]))
        chosen_mask[seed_i] = True
        max_sim = np.maximum(max_sim, cand_emb @ cand_emb[seed_i])

        for _ in range(1, n_div):
            if len(chosen) >= len(band):
                break
            scores = np.where(chosen_mask, 1e9, max_sim)
            j = int(np.argmin(scores))
            if chosen_mask[j]:
                break
            chosen_mask[j] = True
            chosen.append(int(band[j]))
            max_sim = np.maximum(max_sim, cand_emb @ cand_emb[j])
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


# ============================================================
# K-means (cosine, kmeans++ init) on user-level embeddings
# ============================================================
def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    centroids = [x[rng.integers(0, n)]]
    for _ in range(1, k):
        c_mat = np.stack(centroids)
        sims = x @ c_mat.T
        nearest_sim = sims.max(axis=1)
        dist_sq = np.maximum(0.0, 1.0 - nearest_sim) ** 2
        total = dist_sq.sum()
        idx = rng.integers(0, n) if total < 1e-12 else rng.choice(n, p=dist_sq / total)
        centroids.append(x[idx])
    return np.stack(centroids).astype(np.float32)


def simple_kmeans(x: np.ndarray, k: int, seed: int = 42, iters: int = 25):
    # Guard for the degenerate case of fewer users than clusters: fit k_eff
    # centroids instead of k. With the benchmark splits (thousands of train
    # users, K=4) this never binds; it only matters when someone tries the
    # pipeline on a small subset.
    k_eff = min(k, len(x))
    if k_eff < k:
        log(f"[cluster] WARNING: only {len(x)} users for K={k}; fitting {k_eff} clusters. "
            f"Train with --num_clusters {k_eff} to match.")
    rng = np.random.default_rng(seed)
    centroids = _kmeans_pp_init(x, k_eff, rng)
    for _ in range(iters):
        sims = x @ centroids.T
        labels = np.argmax(sims, axis=1)
        new = np.zeros_like(centroids)
        for j in range(k_eff):
            cur = x[labels == j]
            if len(cur) == 0:
                new[j] = centroids[j]
            else:
                c = cur.mean(axis=0)
                new[j] = c / (np.linalg.norm(c) + 1e-12)
        centroids = new
    sims = x @ centroids.T
    labels = np.argmax(sims, axis=1).astype(np.int64)
    return labels, centroids.astype(np.float32)


def assign_to_nearest_cluster(embs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmax(embs @ centroids.T, axis=1).astype(np.int64)


# ============================================================
# K-means seed sweep (silhouette + balance)
#
# How the per-task cluster seeds in scripts/run_task.sh were chosen. The sweep
# runs on train users only; dev users are never seen here.
# ============================================================
def kmeans_seedsweep(
    embs: np.ndarray, k: int, seeds: List[int],
    balance_thresh: float = 2.0, sil_sample_size: int = 0,
):
    """Pick the cluster seed with the highest cosine silhouette among the seeds
    whose largest/smallest cluster-size ratio is at most ``balance_thresh``.

    If no seed meets the balance constraint, fall back to the best silhouette
    over all seeds.

    Returns ``(best_seed, sweep_summary, best_labels, best_centroids)``.
    """
    from collections import Counter

    from sklearn.metrics import silhouette_score

    rows = []
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sd in seeds:
        labels, centroids = simple_kmeans(embs, k=k, seed=int(sd))
        cache[int(sd)] = (labels, centroids)
        sizes = Counter(labels.tolist())
        size_list = [sizes.get(i, 0) for i in range(k)]
        ratio = (float(max(size_list) / min(size_list)) if min(size_list) > 0
                 else float("inf"))
        sample = min(sil_sample_size, len(embs)) if sil_sample_size > 0 else None
        try:
            sil = float(silhouette_score(
                embs, labels, metric="cosine",
                sample_size=sample, random_state=int(sd),
            ))
        except ValueError:
            sil = float("nan")
        rows.append({"seed": int(sd), "silhouette_cos": sil,
                     "sizes": size_list, "max_min_ratio": ratio})
        log(f"[sweep] seed={int(sd):>3d}  sil={sil:.4f}  sizes={size_list}  "
            f"max/min={ratio:.2f}")

    valid = [r for r in rows if r["silhouette_cos"] == r["silhouette_cos"]]
    survivors = [r for r in valid if r["max_min_ratio"] <= balance_thresh]
    fallback = False
    if survivors:
        best = max(survivors, key=lambda r: r["silhouette_cos"])
    elif valid:
        log(f"[sweep] no seed met balance <= {balance_thresh}; "
            f"falling back to the best silhouette")
        fallback = True
        best = max(valid, key=lambda r: r["silhouette_cos"])
    else:
        finite = [r for r in rows if r["max_min_ratio"] != float("inf")]
        if not finite:
            raise RuntimeError("seed sweep: every seed produced an empty cluster")
        fallback = True
        best = min(finite, key=lambda r: r["max_min_ratio"])

    best_seed = int(best["seed"])
    best_labels, best_centroids = cache[best_seed]
    summary = {
        "k": k,
        "balance_thresh": balance_thresh,
        "sil_sample_size": sil_sample_size,
        "best_seed": best_seed,
        "fallback": fallback,
        "all_seeds": sorted(
            rows,
            key=lambda r: (-r["silhouette_cos"]
                           if r["silhouette_cos"] == r["silhouette_cos"]
                           else float("inf")),
        ),
    }
    log(f"[sweep] best seed={best_seed}  sil={best['silhouette_cos']:.4f}  "
        f"sizes={best['sizes']}  fallback={fallback}")
    return best_seed, summary, best_labels, best_centroids


# ============================================================
# Per-user bank construction (train side: encode + rep-div + cluster mean)
# ============================================================
def build_train_bank_and_queries(
    cfg: CacheCfg,
    train_data: List[Dict],
    enc_model,
    enc_tok,
    device: torch.device,
    np_rng: np.random.Generator,
):
    user_bank_items: Dict[str, List[str]] = {}         # stored text
    user_bank_items_embed: Dict[str, List[str]] = {}   # encoder-input text
    user_bank_text: Dict[str, str] = {}
    query_rows = []
    query_row_user_ids = []
    user_profile_emb: Dict[str, np.ndarray] = {}
    unique_user_ids = []

    log(f"[train] processing {len(train_data)} users (BGE encode + rep-div) ...")
    for i, record in enumerate(train_data):
        user_id = str(record["user_id"])
        unique_user_ids.append(user_id)

        all_embed_items, all_save_items = get_bank_items_pair(
            record["profile"], cfg.task_name,
        )
        if len(all_embed_items) == 0:
            fallback = record.get("input", "")
            all_embed_items = [fallback]
            all_save_items = [fallback]

        pool_embs = embed_text_mean_batch(
            enc_model, enc_tok, all_embed_items,
            max_len=cfg.item_max_len, device=device,
            batch_size=cfg.item_batch_size,
        )

        if len(all_embed_items) > cfg.buffer_size:
            bank_embed_items, bank_embs = build_bank_items_rep_div(
                all_embed_items, pool_embs,
                max_bank_items=cfg.buffer_size,
                rep_ratio=cfg.bank_rep_ratio, rng=np_rng,
            )
            # rep-div returns a subset of all_embed_items; take the stored text
            # at the same positions so the two views stay aligned.
            pos = {t: j for j, t in enumerate(all_embed_items)}
            bank_items = [all_save_items[pos[t]] for t in bank_embed_items]
        else:
            bank_embed_items, bank_embs = all_embed_items, pool_embs
            bank_items = all_save_items

        user_bank_items[user_id] = bank_items
        user_bank_items_embed[user_id] = bank_embed_items
        user_bank_text[user_id] = build_bank_text(bank_items)

        mean_emb = bank_embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-12:
            mean_emb = mean_emb / norm
        user_profile_emb[user_id] = mean_emb

        query_rows.append({
            "user_id": user_id,
            "input_text": record["input"],
            "target_text": record["output"],
        })
        query_row_user_ids.append(user_id)

        if (i + 1) % 100 == 0 or (i + 1) == len(train_data):
            log(f"[train bank] {i+1}/{len(train_data)} users")

    return (unique_user_ids, user_bank_items, user_bank_items_embed,
            user_bank_text, query_rows, query_row_user_ids, user_profile_emb)


def embed_bank_items_bge(
    user_ids: List[str],
    user_bank_items: Dict[str, List[str]],
    enc_model, enc_tok, device: torch.device,
    item_max_len: int, item_batch_size: int, split_name: str,
) -> Dict[str, np.ndarray]:
    log(f"[{split_name}] embedding {len(user_ids)} users' bank items ...")
    out: Dict[str, np.ndarray] = {}
    for i, uid in enumerate(user_ids):
        items = user_bank_items.get(uid, [])
        if not items:
            items = [""]
        embs = embed_text_mean_batch(
            enc_model, enc_tok, items,
            max_len=item_max_len, device=device,
            batch_size=item_batch_size,
        )
        out[uid] = embs
        if (i + 1) % 100 == 0 or (i + 1) == len(user_ids):
            log(f"[{split_name} encode] {i+1}/{len(user_ids)}")
    log(f"[{split_name}] done. emb_dim={next(iter(out.values())).shape[-1]}")
    return out


def build_test_bank_items(cfg: CacheCfg, test_data: List[Dict]):
    """Dev side: keep the full profile (no rep-div) so retrieval has the
    largest possible candidate set per user.

    Returns ``(bank_items, bank_items_embed, bank_text, bank_text_embed)``:
    the first and third are the stored text (prompt + BM25), the second and
    fourth the encoder-input text (BGE).
    """
    test_bank_items: Dict[str, List[str]] = {}
    test_bank_items_embed: Dict[str, List[str]] = {}
    test_bank_text: Dict[str, str] = {}
    test_bank_text_embed: Dict[str, str] = {}
    for i, record in enumerate(test_data):
        uid = str(record["user_id"])
        embed_items, save_items = get_bank_items_pair(
            record["profile"], cfg.task_name,
        )
        if not embed_items:
            fallback = record.get("input", "")
            embed_items = [fallback]
            save_items = [fallback]
        test_bank_items[uid] = save_items
        test_bank_items_embed[uid] = embed_items
        test_bank_text[uid] = build_bank_text(save_items)
        test_bank_text_embed[uid] = build_bank_text(embed_items)
        if (i + 1) % 200 == 0 or (i + 1) == len(test_data):
            log(f"[test bank] {i+1}/{len(test_data)}")
    return (test_bank_items, test_bank_items_embed,
            test_bank_text, test_bank_text_embed)


def build_user_profile_embs_from_text(
    cfg: CacheCfg, enc_model, enc_tok,
    user_ids: List[str], user_bank_text: Dict[str, str],
    device: torch.device, split_name: str,
) -> np.ndarray:
    """User-level BGE embedding for cluster assignment. Encodes the joined
    bank text per user."""
    log(f"[{split_name}] embedding user bank_text (BGE, clustering) ...")
    texts = [user_bank_text[uid] for uid in user_ids]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]
    sorted_embs = embed_text_mean_batch(
        enc_model, enc_tok, sorted_texts,
        max_len=cfg.max_profile_len, device=device,
        batch_size=cfg.profile_batch_size,
    )
    inv = [0] * len(order)
    for new_i, orig_i in enumerate(order):
        inv[orig_i] = new_i
    return sorted_embs[inv].astype(np.float32)


# ============================================================
# Main
# ============================================================
def main(cfg: CacheCfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    ensure_dir(cfg.out_dir)
    log(f"[cfg] {json.dumps(asdict(cfg), ensure_ascii=False)}")

    device = (torch.device(cfg.main_device)
              if cfg.device.startswith("cuda") and torch.cuda.is_available()
              else torch.device(cfg.device))
    log(f"[device] {device}")

    log(f"[model] loading BGE encoder: {cfg.encoder_model_name}")
    enc_tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name, use_fast=True)
    enc_model = AutoModel.from_pretrained(cfg.encoder_model_name)
    enc_model.eval().to(device)

    log("[data] loading train + dev splits")
    train_data = load_lamp_data(cfg.train_questions, cfg.train_outputs)
    dev_data = load_lamp_data(cfg.dev_questions, cfg.dev_outputs)
    log(f"[data] train rows={len(train_data)}  dev rows={len(dev_data)}")

    train_uids = {str(d["user_id"]) for d in train_data}
    dev_uids = {str(d["user_id"]) for d in dev_data}
    overlap = train_uids & dev_uids
    if overlap:
        log(f"[data] WARNING: train/dev user overlap = {len(overlap)} (expected 0 for user-based split)")

    # ---- Train: encode + rep-div + cluster ----
    (unique_train_user_ids, train_user_bank_items, train_user_bank_items_embed,
     train_user_bank_text, train_query_rows, train_query_row_user_ids,
     train_user_profile_emb) = build_train_bank_and_queries(
        cfg, train_data, enc_model, enc_tok, device, np_rng,
    )
    write_json(os.path.join(cfg.out_dir, "train_user_ids.json"), unique_train_user_ids)
    write_json(os.path.join(cfg.out_dir, "train_user_bank_items.json"), train_user_bank_items)
    write_json(os.path.join(cfg.out_dir, "train_user_bank_text.json"), train_user_bank_text)
    write_json(os.path.join(cfg.out_dir, "train_query_rows.json"), train_query_rows)
    write_json(os.path.join(cfg.out_dir, "train_query_row_user_ids.json"), train_query_row_user_ids)

    train_user_profile_embs = np.stack(
        [train_user_profile_emb[uid] for uid in unique_train_user_ids]
    ).astype(np.float32)
    np.save(os.path.join(cfg.out_dir, "train_user_profile_embs.npy"), train_user_profile_embs)

    do_sweep = cfg.seed_sweep_low >= 0 and cfg.seed_sweep_high >= 0
    if do_sweep:
        sweep_seeds = list(range(cfg.seed_sweep_low, cfg.seed_sweep_high + 1))
        log(f"[cluster] K={cfg.num_clusters} seed sweep over "
            f"[{cfg.seed_sweep_low}..{cfg.seed_sweep_high}] "
            f"(balance <= {cfg.balance_threshold})")
        cluster_seed, sweep_summary, train_labels, centroids = kmeans_seedsweep(
            train_user_profile_embs, k=cfg.num_clusters, seeds=sweep_seeds,
            balance_thresh=cfg.balance_threshold,
            sil_sample_size=cfg.sil_sample_size,
        )
        write_json(os.path.join(cfg.out_dir, "sweep_log.json"), sweep_summary)
    else:
        cluster_seed = cfg.cluster_seed if cfg.cluster_seed >= 0 else cfg.seed
        log(f"[cluster] running kmeans++ (K={cfg.num_clusters}, "
            f"cluster_seed={cluster_seed}) ...")
        train_labels, centroids = simple_kmeans(
            train_user_profile_embs, k=cfg.num_clusters, seed=cluster_seed,
        )
    np.save(os.path.join(cfg.out_dir, "train_user_cluster_labels.npy"), train_labels)
    np.save(os.path.join(cfg.out_dir, "train_user_centroids.npy"), centroids)
    write_json(
        os.path.join(cfg.out_dir, "train_userid_to_cluster.json"),
        {uid: int(train_labels[i]) for i, uid in enumerate(unique_train_user_ids)},
    )
    from collections import Counter
    log(f"[cluster] train cluster sizes = {sorted(Counter(train_labels.tolist()).items())}")

    # ---- Dev: bank items (full profile), cluster assignment ----
    dev_user_ids = [str(d["user_id"]) for d in dev_data]
    write_json(os.path.join(cfg.out_dir, "dev_user_ids.json"), dev_user_ids)
    write_json(os.path.join(cfg.out_dir, "dev_row_user_ids.json"), dev_user_ids)

    (dev_bank_items, dev_bank_items_embed,
     dev_bank_text, dev_bank_text_embed) = build_test_bank_items(cfg, dev_data)
    write_json(os.path.join(cfg.out_dir, "dev_user_bank_items.json"), dev_bank_items)
    write_json(os.path.join(cfg.out_dir, "dev_user_bank_text.json"), dev_bank_text)

    # Cluster assignment runs on the encoder-input view, matching the train side.
    dev_user_profile_embs = build_user_profile_embs_from_text(
        cfg, enc_model, enc_tok, dev_user_ids, dev_bank_text_embed, device, "dev",
    )
    np.save(os.path.join(cfg.out_dir, "dev_user_profile_embs.npy"), dev_user_profile_embs)
    dev_labels = assign_to_nearest_cluster(dev_user_profile_embs, centroids)
    write_json(
        os.path.join(cfg.out_dir, "dev_userid_to_cluster.json"),
        {uid: int(dev_labels[i]) for i, uid in enumerate(dev_user_ids)},
    )

    # ---- Per-item BGE embeddings (used by HyperNet input) ----
    train_user_bank_embs = embed_bank_items_bge(
        unique_train_user_ids, train_user_bank_items_embed,
        enc_model, enc_tok, device,
        cfg.item_max_len, cfg.item_batch_size, "train",
    )
    with open(os.path.join(cfg.out_dir, "train_user_profile_item_embs.pkl"), "wb") as f:
        pickle.dump(train_user_bank_embs, f)

    dev_user_bank_embs = embed_bank_items_bge(
        dev_user_ids, dev_bank_items_embed,
        enc_model, enc_tok, device,
        cfg.item_max_len, cfg.item_batch_size, "dev",
    )
    with open(os.path.join(cfg.out_dir, "dev_user_profile_item_embs.pkl"), "wb") as f:
        pickle.dump(dev_user_bank_embs, f)

    test_rows = [{
        "user_id": str(d["user_id"]),
        "input_text": d["input"],
        "target_text": d["output"],
    } for d in dev_data]
    write_json(os.path.join(cfg.out_dir, "test_rows.json"), test_rows)

    write_json(os.path.join(cfg.out_dir, "cache_meta.json"), {
        "task_name": cfg.task_name,
        "encoder_model_name": cfg.encoder_model_name,
        "num_clusters": cfg.num_clusters,
        "item_max_len": cfg.item_max_len,
        "buffer_size": cfg.buffer_size,
        "bank_rep_ratio": cfg.bank_rep_ratio,
        "seed": cfg.seed,
        "cluster_seed": cluster_seed,
        "n_train_users": len(train_data),
        "n_dev_users": len(dev_data),
        "train_dev_overlap": len(overlap),
    })
    log("DONE")


# ============================================================
# CLI
# ============================================================
SUPPORTED_TASKS = (
    "LaMP_4",                             # news headline generation
    "LaMP_5",                             # paper title generation
    "LaMP_7",                             # tweet paraphrasing
    "LongLaMP_abstract_generation_user",  # paper abstract generation
    "LongLaMP_topic_writing_user",        # reddit topic writing
    "LongLaMP_product_review_user",       # product review generation
)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_name",       type=str, required=True,
                   choices=list(SUPPORTED_TASKS),
                   help="Task to build the cache for.")
    p.add_argument("--encoder_model_name", type=str, default=CacheCfg.encoder_model_name)
    p.add_argument("--train_questions", type=str, required=True,
                   help="train_questions.json of the task")
    p.add_argument("--train_outputs",   type=str, required=True,
                   help="train_outputs.json of the task")
    p.add_argument("--dev_questions",   type=str, required=True)
    p.add_argument("--dev_outputs",     type=str, required=True)
    p.add_argument("--out_dir",         type=str, required=True)
    p.add_argument("--num_clusters",    type=int, default=CacheCfg.num_clusters)
    p.add_argument("--item_max_len",    type=int, default=CacheCfg.item_max_len)
    p.add_argument("--item_batch_size", type=int, default=CacheCfg.item_batch_size)
    p.add_argument("--profile_batch_size", type=int, default=CacheCfg.profile_batch_size)
    p.add_argument("--max_profile_len", type=int, default=CacheCfg.max_profile_len)
    p.add_argument("--buffer_size",     type=int, default=CacheCfg.buffer_size)
    p.add_argument("--bank_rep_ratio",  type=float, default=CacheCfg.bank_rep_ratio)
    p.add_argument("--seed",            type=int, default=CacheCfg.seed,
                   help="Master seed for rep-div bank sampling and BGE encode order.")
    p.add_argument("--cluster_seed",    type=int, default=CacheCfg.cluster_seed,
                   help="k-means cluster seed (independent of --seed). "
                        "-1 (default) means: fall back to --seed. "
                        "The per-task values used in the paper are listed in "
                        "scripts/run_task.sh; LaMP-4 uses --seed 42 --cluster_seed 9.")
    p.add_argument("--seed_sweep_low",  type=int, default=CacheCfg.seed_sweep_low,
                   help="If this and --seed_sweep_high are both >= 0, sweep k-means "
                        "over that inclusive seed range and pick the best seed by "
                        "cosine silhouette subject to --balance_threshold, instead "
                        "of using --cluster_seed. The per-task seeds in "
                        "scripts/run_task.sh came from the range 0..30.")
    p.add_argument("--seed_sweep_high", type=int, default=CacheCfg.seed_sweep_high)
    p.add_argument("--balance_threshold", type=float, default=CacheCfg.balance_threshold,
                   help="Sweep filter: maximum largest/smallest cluster-size ratio.")
    p.add_argument("--sil_sample_size", type=int, default=CacheCfg.sil_sample_size,
                   help="Silhouette subsample size during the sweep; <= 0 uses all users.")
    p.add_argument("--device",          type=str, default=CacheCfg.device)
    p.add_argument("--dtype",           type=str, default=CacheCfg.dtype)
    p.add_argument("--main_device",     type=str, default=CacheCfg.main_device)
    a = p.parse_args()

    cfg = CacheCfg(**{k: v for k, v in vars(a).items() if k in CacheCfg.__dataclass_fields__})
    main(cfg)
