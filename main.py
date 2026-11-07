"""
Conjugate Dirichlet HyperLoRA — train + eval entry point.

Pipeline:
    precompute.py  -> cache/<task>/  (BGE embeddings + bank items + clusters)
    main.py        -> train -> save -> (separate) --eval_only -> dev metrics

Supported tasks (``--task_name``) — the six generation tasks of the nine
evaluated in the paper:

    LaMP_4                              news headline generation
    LaMP_5                              paper title generation
    LaMP_7                              tweet paraphrasing
    LongLaMP_abstract_generation_user   paper abstract generation
    LongLaMP_topic_writing_user         reddit topic writing
    LongLaMP_product_review_user        product review generation

``scripts/run_task.sh <task>`` runs precompute + train + eval for one task with
the per-task settings used in the paper.

Package layout:
    hyperlora/
        config.py     TrainCfg dataclass (all hyperparameters)
        utils.py      logging / IO helpers
        cache.py      cache loader
        prompt.py     per-task prompt formatting
        metrics.py    ROUGE-1/L
        buffer.py     rep-div sampling, BM25 retrieval
        model.py      HyperLoRALinear (basis LoRA), patch/loss/generate
        dirichlet.py  ConjugateHyperNet (M1, M2 -> beta = alpha + M2*M1),
                      DirichletPriorTable
        train.py      basis warmup + joint training
        eval.py       dev evaluation
"""

import argparse
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperlora import (
    TrainCfg,
    log, write_json, get_torch_dtype, get_main_device, format_metrics,
    maybe_load_cache_arrays,
    patch_with_hyperlora, freeze_base_model, set_basis_train_policy,
    ConjugateHyperNet, DirichletPriorTable,
    save_dirichlet_checkpoint, load_dirichlet_checkpoint,
    evaluate_base_dev, evaluate_dev,
    run_basis_warmup,
    run_dirichlet_training,
)


def _check_cache_matches_cfg(cfg):
    """Fail fast when --cache_dir was built for a different task or K.

    The cache fixes the profile-item text format (task-specific) and the number
    of cluster centroids. Using it with a different --task_name silently feeds
    the wrong field layout into the prompt; a different --num_clusters makes the
    cluster ids run past the prior table. Both are caught here instead of
    surfacing as bad numbers or a mid-training IndexError.
    """
    meta_path = os.path.join(cfg.cache_dir, "cache_meta.json")
    if not os.path.exists(meta_path):
        log(f"[cache] WARNING: {meta_path} not found, so the task/K consistency "
            f"check was skipped. Rebuild the cache with the bundled precompute.py "
            f"if the results look off.")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    problems = []
    cached_task = meta.get("task_name")
    if cached_task is not None and str(cached_task) != str(cfg.task_name):
        problems.append(f"task_name: the cache was built for {cached_task!r}, "
                        f"but --task_name is {cfg.task_name!r}")
    cached_k = meta.get("num_clusters")
    if cached_k is not None and int(cached_k) != int(cfg.num_clusters):
        problems.append(f"num_clusters: the cache has {cached_k}, "
                        f"but --num_clusters is {cfg.num_clusters}")
    if problems:
        raise SystemExit(
            f"[cache] {cfg.cache_dir} does not match this run:\n  - "
            + "\n  - ".join(problems)
            + "\nRebuild it with:\n"
            + f"  python precompute.py --task_name {cfg.task_name} "
              f"--num_clusters {cfg.num_clusters} --out_dir {cfg.cache_dir} ...\n"
              "Refusing to run: the cache fixes the profile-item format and the "
              "cluster count, so continuing would produce wrong results silently."
        )
    log(f"[cache] ok: task={cached_task} K={cached_k} seed={meta.get('seed')} "
        f"cluster_seed={meta.get('cluster_seed')}")


def main(cfg: TrainCfg):
    # Apply cfg.encoder_model_name to the BGE singleton (default: bge-small-en-v1.5).
    from hyperlora.utils import set_bge_model_name
    set_bge_model_name(cfg.encoder_model_name)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if str(cfg.eval_buffer_seeding) == "random":
        log("[seed] eval_buffer_seeding=random - evaluation history buffers are redrawn every run, "
            "so results are NOT reproducible.")

    _check_cache_matches_cfg(cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log(f"[cfg] {json.dumps(asdict(cfg), ensure_ascii=False)}")

    main_device = get_main_device(cfg)
    log(f"[device] main_device={main_device}")

    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log("[device] TF32 enabled (matmul + cudnn)")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tok.truncation_side = "left"
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _max_memory = None
    _device_map_arg = None
    _use_cuda_dm = cfg.device.startswith("cuda")
    if _use_cuda_dm and cfg.gpu0_layers >= 0:
        # Manual device_map: embed -> GPU 0; layers 0..k-1 -> GPU 0; rest + norm + lm_head -> GPU 1.
        from transformers import AutoConfig
        _hf_cfg = AutoConfig.from_pretrained(cfg.model_name)
        _n_layers = int(_hf_cfg.num_hidden_layers)
        _k = max(0, min(int(cfg.gpu0_layers), _n_layers))
        _device_map_arg = {
            "model.embed_tokens": 0,
            **{f"model.layers.{i}": (0 if i < _k else 1) for i in range(_n_layers)},
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 1,
        }
        log(f"[device] manual device_map: GPU0 layers 0..{_k - 1}, GPU1 layers {_k}..{_n_layers - 1}")
    elif _use_cuda_dm:
        _device_map_arg = "auto"
        if cfg.max_memory_per_gpu:
            _max_memory = {}
            for tok_ in cfg.max_memory_per_gpu.split(","):
                if not tok_.strip():
                    continue
                k, v = tok_.split(":", 1)
                _max_memory[int(k.strip())] = v.strip()
            log(f"[device] max_memory={_max_memory}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=get_torch_dtype(cfg.dtype),
        device_map=_device_map_arg,
        max_memory=_max_memory,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log("[mem] gradient_checkpointing enabled (use_reentrant=False)")
    if not cfg.device.startswith("cuda"):
        model.to(main_device)
    if hasattr(model, "hf_device_map"):
        log(f"[hf_device_map] {model.hf_device_map}")

    # ── Load cache ──────────────────────────────────────────
    cache = maybe_load_cache_arrays(cfg)
    if cache is None:
        raise ValueError(f"Run precompute.py first to populate the cache: {cfg.cache_dir}")

    train_query_rows          = cache["train_query_rows"]
    train_query_row_record_ids = cache["train_query_row_user_ids"]  # user-based: record_id == user_id
    train_userid_to_cluster   = cache["train_userid_to_cluster"]
    train_user_profile_embs   = cache["train_user_profile_embs"]
    train_user_cluster_labels = cache["train_user_cluster_labels"]
    train_user_centroids      = cache["train_user_centroids"]
    train_cluster_sizes = np.bincount(train_user_cluster_labels, minlength=cfg.num_clusters)

    dev_rows = cache["test_rows"]
    log(f"[data] train_query_rows={len(train_query_rows)} dev_rows={len(dev_rows)}")

    # Label distribution sanity check.
    c = {}
    for row in train_query_rows:
        key = str(row.get("target_text", row.get("output", ""))).strip()
        c[key] = c.get(key, 0) + 1
    print("[label dist]", sorted(c.items(), key=lambda x: -x[1])[:20])

    # ── base eval ───────────────────────────────────────────
    if cfg.run_base_eval:
        log("[base] dev evaluation (pretrained + history prompt)")
        base_out = evaluate_base_dev(
            cfg=cfg, model=model, tok=tok, dev_rows=dev_rows,
            main_device=main_device,
            dev_row_user_ids=cache["dev_row_user_ids"],
            dev_user_bank_items=cache["dev_user_bank_items"],
            dev_user_profile_item_embs=cache["dev_user_profile_item_embs"],
            max_samples=cfg.base_eval_max_samples,
        )
        log(f"[base] {format_metrics(base_out)}")
        write_json(os.path.join(cfg.out_dir, "dev_base.json"), base_out)

    # ── HyperLoRA patch ─────────────────────────────────────
    wrapped, target_names = patch_with_hyperlora(model, cfg)
    write_json(os.path.join(cfg.out_dir, "target_names.json"), target_names)

    freeze_base_model(model)
    set_basis_train_policy(wrapped)

    num_modules = len(wrapped)
    # Read embedding dim from the first user's item embeddings (BGE-small: 384).
    sample_emb = next(iter(cache["train_user_profile_item_embs"].values()))
    D = sample_emb.shape[1]
    log(f"[cache] train_user_profile_embs shape={train_user_profile_embs.shape}")

    # buffer_to_posterior returns [mu, log(var)] -> in_dim = 2*D.
    hyper_in_dim = 2 * D
    log(f"[cache] HyperNet in_dim={hyper_in_dim} "
        f"(mu/logvar standardised separately, one (M1, M2) broadcast to all modules)")
    log(f"[hypernet] ConjugateHyperNet (β = α + M2·M1), "
        f"m2_bias_init={cfg.conjugate_m2_bias_init}")
    hypernet = ConjugateHyperNet(
        in_dim=hyper_in_dim, num_modules=num_modules, basis_m=cfg.basis_m,
        hidden=cfg.hidden_dim,
        head_init_std=cfg.dirichlet_head_init_std,
        m2_bias_init=cfg.conjugate_m2_bias_init,
        m2_min=float(cfg.conjugate_m2_min),
    ).to(main_device)
    # Compute train mean/std for mu and logvar separately and inject them.
    # The stats come from the same distribution the runtime h is pooled over (the rep-div bank).
    item_embs_dict = cache["train_user_profile_item_embs"]
    mu_u, lv_u = [], []
    for uid in cache["train_user_ids"]:
        arr = np.asarray(item_embs_dict[uid])
        mu_u.append(arr.mean(axis=0))
        lv_u.append(np.log(arr.var(axis=0) + 1e-6))
    mu_u = np.stack(mu_u).astype(np.float32)
    lv_u = np.stack(lv_u).astype(np.float32)
    mu_mean = mu_u.mean(axis=0); mu_std = mu_u.std(axis=0) + 1e-6
    lv_mean = lv_u.mean(axis=0); lv_std = lv_u.std(axis=0) + 1e-6
    hypernet.set_input_stats_dual(
        torch.from_numpy(mu_mean), torch.from_numpy(mu_std),
        torch.from_numpy(lv_mean), torch.from_numpy(lv_std),
    )
    log(f"[hypernet] dual_standardize stats set: "
        f"mu σ p50={float(np.quantile(mu_std, 0.5)):.4f}, "
        f"lv σ p50={float(np.quantile(lv_std, 0.5)):.4f}")
    prior = DirichletPriorTable(
        K=cfg.num_clusters, M=cfg.basis_m, num_modules=num_modules,
    ).to(main_device)

    # ── prior-only ablation (HyperNet disabled in both train and eval) ──
    if cfg.disable_hypernet:
        if hasattr(hypernet, "_force_m2_zero"):
            hypernet._force_m2_zero = True
        for p in hypernet.parameters():
            p.requires_grad_(False)
        log("[ablation] disable_hypernet=True → β=α (cluster prior only). hypernet frozen + m2=0 forced.")

    # ── eval-only mode ───────────────────────────────────────
    if cfg.eval_only:
        train_user_centroids = load_dirichlet_checkpoint(cfg, hypernet, wrapped, prior)
        if cfg.eval_disable_hypernet and hasattr(hypernet, "_force_m2_zero"):
            hypernet._force_m2_zero = True
            log("[eval-only] HyperNet ablation: forcing m2=0 -> beta=alpha (prior-only LoRA)")
        final_dev = evaluate_dev(
            cfg=cfg, model=model, tok=tok, wrapped=wrapped, hypernet=hypernet, prior=prior,
            dev_rows=dev_rows, train_user_centroids=train_user_centroids, main_device=main_device,
            dev_user_profile_embs=cache["dev_user_profile_embs"],
            dev_userid_to_index=cache["dev_userid_to_index"],
            dev_userid_to_cluster=cache["dev_userid_to_cluster"],
            dev_row_user_ids=cache["dev_row_user_ids"],
            dev_user_bank_items=cache["dev_user_bank_items"],
            dev_user_profile_item_embs=cache["dev_user_profile_item_embs"],
            train_cluster_sizes=train_cluster_sizes,
            train_user_profile_embs=train_user_profile_embs,
        )
        log(f"[eval-only] {format_metrics(final_dev)}")
        write_json(os.path.join(cfg.out_dir, "dev_eval_only.json"), final_dev)
        return

    # ── Optimizer ────────────────────────────────────────────
    # DirichletPriorTable holds a non-learnable buffer; not added to the optimizer.
    opt_groups = [
        {"params": hypernet.parameters(), "lr": cfg.lr, "weight_decay": cfg.wd},
    ]
    basis_B_params = [p for m in wrapped for p in m.B_basis]
    basis_A_params = [p for m in wrapped for p in m.A_basis]
    opt_groups.append({"params": basis_B_params, "lr": cfg.lr * cfg.basis_B_lr_mult, "weight_decay": 0.0})
    opt_groups.append({"params": basis_A_params, "lr": cfg.lr * cfg.basis_A_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.AdamW(opt_groups)

    model.train()
    hypernet.train()
    prior.train()

    # ── Phase 1: Basis warmup (or load) ─────────────────────────
    basis_ckpt_path = os.path.join(cfg.out_dir, "basis_warmup.pt")
    if os.path.exists(basis_ckpt_path):
        basis_ckpt = torch.load(basis_ckpt_path, map_location="cpu")
        for m, b in zip(wrapped, basis_ckpt):
            for i, p in enumerate(m.A_basis):
                p.data.copy_(b["A_basis"][i].to(p.device))
            for i, p in enumerate(m.B_basis):
                p.data.copy_(b["B_basis"][i].to(p.device))
        log(f"[basis_warmup] loaded from {basis_ckpt_path}, skipping warmup")
    elif cfg.use_basis_warmup:
        run_basis_warmup(
            cfg=cfg, model=model, tok=tok, wrapped=wrapped,
            hypernet=hypernet, prior=prior, cache=cache,
            train_query_rows=train_query_rows,
            train_query_row_record_ids=train_query_row_record_ids,
            train_userid_to_cluster=train_userid_to_cluster,
            main_device=main_device,
        )
        # Save the basis snapshot after warmup completes.
        basis_state = [
            {"A_basis": [p.data.cpu().clone() for p in m.A_basis],
             "B_basis": [p.data.cpu().clone() for p in m.B_basis]}
            for m in wrapped
        ]
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save(basis_state, basis_ckpt_path)
        log(f"[basis_warmup] saved to {basis_ckpt_path}")

    # ── Phase 2: joint training of the HyperNet and the bases ──
    def dev_eval_fn():
        return evaluate_dev(
            cfg=cfg, model=model, tok=tok, wrapped=wrapped,
            hypernet=hypernet, prior=prior,
            dev_rows=dev_rows, train_user_centroids=train_user_centroids,
            main_device=main_device,
            dev_user_profile_embs=cache["dev_user_profile_embs"],
            dev_userid_to_index=cache["dev_userid_to_index"],
            dev_userid_to_cluster=cache["dev_userid_to_cluster"],
            dev_row_user_ids=cache["dev_row_user_ids"],
            dev_user_bank_items=cache["dev_user_bank_items"],
            dev_user_profile_item_embs=cache["dev_user_profile_item_embs"],
            train_cluster_sizes=train_cluster_sizes,
            train_user_profile_embs=train_user_profile_embs,
        )


    def _hypernet_opt_groups(lr):
        # The HyperNet emits (1, ...) which the caller broadcasts to N=num_modules
        # modules. Backward accumulates gradients into every trainable parameter N
        # times along the same direction, so Adam's second moment rises with it and
        # the effective learning rate is ~N times the raw lr. Compensate by putting
        # every HyperNet parameter into one group at lr/num_modules.
        return [{"params": list(hypernet.parameters()), "lr": lr / num_modules}]

    # Joint training: basis params get a small lr (their existing lr_mult).
    set_basis_train_policy(wrapped)
    # The prior alpha is a fixed buffer, so it is not added to the optimizer.
    opt_groups = _hypernet_opt_groups(cfg.lr) + [
        {"params": basis_B_params,
         "lr": cfg.lr * cfg.basis_B_lr_mult, "weight_decay": 0.0},
        {"params": basis_A_params,
         "lr": cfg.lr * cfg.basis_A_lr_mult, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(opt_groups)
    log(f"[joint] bases trainable with a reduced lr: "
        f"A_lr={cfg.lr * cfg.basis_A_lr_mult:.6f}, "
        f"B_lr={cfg.lr * cfg.basis_B_lr_mult:.6f}")

    run_dirichlet_training(
        cfg=cfg, model=model, tok=tok, wrapped=wrapped,
        hypernet=hypernet, prior=prior, cache=cache,
        train_query_rows=train_query_rows,
        train_query_row_record_ids=train_query_row_record_ids,
        train_userid_to_cluster=train_userid_to_cluster,
        main_device=main_device,
        basis_A_params=basis_A_params,
        basis_B_params=basis_B_params,
        opt=opt,
        out_dir=cfg.out_dir,
    )

    # ── Save checkpoint (right after training, before final eval) ──
    save_dirichlet_checkpoint(cfg, cfg.out_dir, hypernet, wrapped, prior, cache)

    # ── Final eval ───────────────────────────────────────────
    if cfg.no_final_eval:
        log("[final dev] skipped (--no_final_eval). Run with --eval_only later to evaluate.")
    else:
        log("[final dev] start")
        model.eval(); hypernet.eval(); prior.eval()
        final_dev = dev_eval_fn()
        log(f"[final dev] {format_metrics(final_dev)}")
        write_json(os.path.join(cfg.out_dir, "dev_final.json"), final_dev)
    log("DONE")


# ============================================================
# CLI — only the essential knobs are exposed. Every other field uses the
# TrainCfg dataclass default; see the note at the end of this file.
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

    # Data + paths
    p.add_argument("--task_name",       type=str, default="LaMP_4",
                   choices=list(SUPPORTED_TASKS),
                   help="Task to train/evaluate on. Must match the --task_name "
                        "passed to precompute.py for --cache_dir.")
    p.add_argument("--cache_dir",       type=str, required=True)
    p.add_argument("--out_dir",         type=str, default=TrainCfg.out_dir)
    p.add_argument("--ckpt_dir",        type=str, default="",
                   help="Path to a trained checkpoint dir (used with --eval_only).")
    p.add_argument("--eval_only",       action="store_true")
    p.add_argument("--no_final_eval",   action="store_true",
                   help="Skip the dev evaluation that otherwise runs at the end of "
                        "training, so that the only evaluation is a later --eval_only "
                        "run. The paper used this for the three LongLaMP tasks, whose "
                        "evaluation generation length differs from the training one.")

    # Model
    p.add_argument("--base_model",      type=str, default=TrainCfg.model_name,
                   help="HuggingFace model id (e.g., Qwen/Qwen3-8B).")
    p.add_argument("--model_type",      type=str, default="qwen",
                   choices=["llama", "qwen"])
    p.add_argument("--encoder_model_name", type=str, default=TrainCfg.encoder_model_name,
                   help="Frozen sentence encoder for user posterior summary.")
    p.add_argument("--target_modules",  type=str,
                   default="gate_proj,up_proj,down_proj",
                   help="Comma-separated list of linear modules to wrap with basis LoRA.")
    p.add_argument("--lora_r",          type=int, default=16)
    p.add_argument("--lora_alpha",      type=float, default=32.0)
    p.add_argument("--basis_m",         type=int, default=4,
                   help="Number of LoRA bases per module (mixture size).")

    # Method
    p.add_argument("--num_clusters",    type=int, default=4,
                   help="K, number of user clusters (= number of Dirichlet priors).")
    p.add_argument("--hidden_dim",      type=int, default=512)
    p.add_argument("--alpha_on",        type=float, default=2.0,
                   help="Asymmetric prior diagonal entry (cluster k -> basis k%%M).")
    p.add_argument("--alpha_off",       type=float, default=1.0,
                   help="Asymmetric prior off-diagonal entries.")

    # Optimisation
    p.add_argument("--epochs",          type=int,   default=1)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--wd",              type=float, default=0.01)
    p.add_argument("--grad_accum",      type=int,   default=TrainCfg.grad_accum)
    p.add_argument("--grad_clip",       type=float, default=0.5)

    # Sequence budgets
    p.add_argument("--max_input_len",       type=int, default=4096)
    p.add_argument("--max_profile_len",     type=int, default=2048)
    p.add_argument("--max_history_items",   type=int, default=32)
    p.add_argument("--history_min_budget",  type=int, default=TrainCfg.history_min_budget,
                   help="tokens reserved for the user history (the query is capped to protect it)")
    p.add_argument("--max_target_tokens",   type=int, default=24)

    # Retrieval (rep-div bank + BM25)
    p.add_argument("--buffer_size",         type=int,   default=32)
    p.add_argument("--buffer_rep_ratio",    type=float, default=0.5)
    p.add_argument("--prompt_bm25_top_k",   type=int,   default=4)

    # Evaluation
    p.add_argument("--run_base_eval",   action="store_true",
                   help="Also evaluate the frozen base LM with the same prompts and "
                        "decoding, before training. This is the no-personalization "
                        "reference; it writes dev_base.json.")
    p.add_argument("--base_eval_max_samples", type=int, default=TrainCfg.base_eval_max_samples,
                   help="Cap the number of dev rows in the base-LM evaluation "
                        "(-1 = all of them).")
    p.add_argument("--eval_repetition_penalty",   type=float, default=1.1)
    p.add_argument("--eval_no_repeat_ngram_size", type=int,   default=0)

    # Runtime
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--log_diagnostics", action="store_true", default=TrainCfg.log_diagnostics,
                   help="Log per-sample evaluation diagnostics (first prompt, predictions, "
                        "beta, mixing coefficients, buffer sizes). Logging only.")
    p.add_argument("--eval_buffer_seeding", type=str, default=TrainCfg.eval_buffer_seeding,
                   choices=["deterministic", "random"],
                   help="deterministic: eval history buffers are reproducible from --seed alone; "
                        "random: redrawn every run (only for measuring buffer-draw variability)")
    p.add_argument("--device",          type=str, default="cuda")
    p.add_argument("--dtype",           type=str, default="bfloat16",
                   choices=["bfloat16", "float16"],
                   help="Weight dtype for the frozen base LM. The paper uses bfloat16.")
    p.add_argument("--main_device",     type=str, default="cuda:0")
    p.add_argument("--gpu0_layers",     type=int, default=TrainCfg.gpu0_layers,
                   help="-1 (default): single GPU / device_map=auto. >=0: split the "
                        "model over two GPUs with this many decoder layers on GPU 0. "
                        "The LongLaMP topic-writing run in the paper used 4.")
    p.add_argument("--max_memory_per_gpu", type=str, default=TrainCfg.max_memory_per_gpu,
                   help='Per-GPU cap for device_map="auto", e.g. "0:22GiB,1:18GiB". '
                        "Ignored when --gpu0_layers >= 0.")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false", default=True,
                   help="Turn off HF gradient checkpointing. It is on by default (as in "
                        "the paper) because the long-profile tasks otherwise run out of "
                        "memory; turning it off is about 20%% faster when memory allows.")

    a = p.parse_args()

    cfg = TrainCfg(
        # Paths
        task_name=a.task_name,
        cache_dir=a.cache_dir,
        out_dir=a.out_dir,
        ckpt_dir=a.ckpt_dir,
        eval_only=bool(a.eval_only),

        # Model
        model_name=a.base_model,
        model_type=a.model_type,
        encoder_model_name=a.encoder_model_name,
        target_modules=tuple(x.strip() for x in a.target_modules.split(",") if x.strip()),
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        basis_m=a.basis_m,

        # Method
        num_clusters=a.num_clusters,
        hidden_dim=a.hidden_dim,
        conjugate_alpha_on=a.alpha_on,
        conjugate_alpha_off=a.alpha_off,

        # Optimisation
        epochs=a.epochs,
        lr=a.lr,
        wd=a.wd,
        grad_accum=a.grad_accum,
        grad_clip=a.grad_clip,

        # Sequence budgets
        max_input_len=a.max_input_len,
        max_profile_len=a.max_profile_len,
        max_history_items=a.max_history_items,
        history_min_budget=a.history_min_budget,
        max_target_tokens=a.max_target_tokens,

        # Retrieval
        prompt_bm25_top_k=a.prompt_bm25_top_k,
        buffer_size=a.buffer_size,
        buffer_rep_ratio=a.buffer_rep_ratio,

        # Evaluation
        eval_repetition_penalty=a.eval_repetition_penalty,
        eval_no_repeat_ngram_size=a.eval_no_repeat_ngram_size,
        run_base_eval=bool(a.run_base_eval),
        base_eval_max_samples=a.base_eval_max_samples,
        no_final_eval=bool(a.no_final_eval),

        # Runtime
        seed=a.seed,
        eval_buffer_seeding=a.eval_buffer_seeding,
        log_diagnostics=bool(a.log_diagnostics),
        device=a.device,
        dtype=a.dtype,
        main_device=a.main_device,
        gpu0_layers=a.gpu0_layers,
        max_memory_per_gpu=a.max_memory_per_gpu,
        gradient_checkpointing=bool(a.gradient_checkpointing),
    )
    main(cfg)



# ----------------------------------------------------------------
# The CLI above exposes the settings needed to reproduce the reported numbers.
# The three ablations reported in the paper are defined on TrainCfg but not
# surfaced here; set them on TrainCfg directly to run them:
#   - eval_swap_mode              route the HyperNet input through another user
#                                 (the own / within-cluster / cross-user analysis)
#   - disable_hypernet            train with the cluster prior only
#   - eval_disable_hypernet       force M2=0 at evaluation time (beta = alpha)
