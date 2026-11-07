import os
import numpy as np

from .utils import log, load_json, load_pickle


def maybe_load_cache_arrays(cfg):
    """Load the per-user cache produced by precompute.py.

    User-based cache layout:
        - train_query_row_user_ids -> list of user_id (record_id == user_id)
        - bank lookup: keyed by user_id
        - cluster lookup: query_row["user_id"]
    """
    required = [
        "train_user_ids.json",
        "train_user_profile_embs.npy",
        "train_user_cluster_labels.npy",
        "train_user_centroids.npy",
        "train_userid_to_cluster.json",
        "train_user_bank_items.json",
        "train_user_bank_text.json",
        "train_user_profile_item_embs.pkl",
        "train_query_rows.json",
        "train_query_row_user_ids.json",
        "dev_user_ids.json",
        "dev_user_profile_embs.npy",
        "dev_userid_to_cluster.json",
        "dev_user_bank_items.json",
        "dev_user_bank_text.json",
        "dev_user_profile_item_embs.pkl",
        "dev_row_user_ids.json",
        "test_rows.json",
    ]

    missing = [f for f in required if not os.path.exists(os.path.join(cfg.cache_dir, f))]
    if missing:
        log(f"[cache] missing files: {missing}")
        return None

    cache = {
        "train_user_ids":               load_json(os.path.join(cfg.cache_dir, "train_user_ids.json")),
        "train_user_profile_embs":      np.load(os.path.join(cfg.cache_dir, "train_user_profile_embs.npy")),
        "train_user_cluster_labels":    np.load(os.path.join(cfg.cache_dir, "train_user_cluster_labels.npy")),
        "train_user_centroids":         np.load(os.path.join(cfg.cache_dir, "train_user_centroids.npy")),
        "train_userid_to_cluster":      load_json(os.path.join(cfg.cache_dir, "train_userid_to_cluster.json")),
        "train_user_bank_items":        load_json(os.path.join(cfg.cache_dir, "train_user_bank_items.json")),
        "train_user_bank_text":         load_json(os.path.join(cfg.cache_dir, "train_user_bank_text.json")),
        "train_user_profile_item_embs": load_pickle(os.path.join(cfg.cache_dir, "train_user_profile_item_embs.pkl")),
        "train_query_rows":             load_json(os.path.join(cfg.cache_dir, "train_query_rows.json")),
        "train_query_row_user_ids":     load_json(os.path.join(cfg.cache_dir, "train_query_row_user_ids.json")),
        "dev_user_ids":                 load_json(os.path.join(cfg.cache_dir, "dev_user_ids.json")),
        "dev_user_profile_embs":        np.load(os.path.join(cfg.cache_dir, "dev_user_profile_embs.npy")),
        "dev_userid_to_cluster":        load_json(os.path.join(cfg.cache_dir, "dev_userid_to_cluster.json")),
        "dev_user_bank_items":          load_json(os.path.join(cfg.cache_dir, "dev_user_bank_items.json")),
        "dev_user_bank_text":           load_json(os.path.join(cfg.cache_dir, "dev_user_bank_text.json")),
        "dev_user_profile_item_embs":   load_pickle(os.path.join(cfg.cache_dir, "dev_user_profile_item_embs.pkl")),
        "dev_row_user_ids":             load_json(os.path.join(cfg.cache_dir, "dev_row_user_ids.json")),
        "test_rows":                    load_json(os.path.join(cfg.cache_dir, "test_rows.json")),
    }

    cache["train_userid_to_index"] = {str(uid): i for i, uid in enumerate(cache["train_user_ids"])}
    cache["dev_userid_to_index"]   = {str(uid): i for i, uid in enumerate(cache["dev_user_ids"])}

    cache["dev_userid_to_cluster"]   = {str(k): int(v) for k, v in cache["dev_userid_to_cluster"].items()}
    cache["train_userid_to_cluster"] = {str(k): int(v) for k, v in cache["train_userid_to_cluster"].items()}

    log(f"[cache] loaded from {cfg.cache_dir}")
    return cache
