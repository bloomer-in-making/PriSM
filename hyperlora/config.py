from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainCfg:
    """Training configuration.

    The defaults below are the configuration used for the numbers reported in
    the paper, so reading this file tells you what the method was run with. The
    values that differ per task -- cluster seed, ``max_input_len``,
    ``max_target_tokens``, ``grad_accum`` and ``eval_no_repeat_ngram_size`` --
    are listed in ``scripts/run_task.sh`` and passed on the command line.

    Most fields are exposed via the CLI in main.py. The three ablations reported
    in the paper (``eval_swap_mode``, ``disable_hypernet``,
    ``eval_disable_hypernet``) are defined here and set on TrainCfg directly
    rather than via the CLI.
    """

    model_name: str = "Qwen/Qwen3-8B"
    model_type: str = "qwen"  # "llama" | "qwen"
    encoder_model_name: str = "BAAI/bge-small-en-v1.5"

    # ── Data paths ────────────────────────────────────────────
    cache_dir: str = ""
    out_dir: str = "runs/lamp_4"
    task_name: str = "LaMP_4"

    # ── Sequence-length budgets ───────────────────────────────
    max_profile_len: int = 2048
    # Per task: 8192 for the three LaMP tasks, 4096 for the three LongLaMP ones.
    max_input_len: int = 4096
    max_history_items: int = 32
    history_min_budget: int = 1024      # tokens reserved for the history; the query is capped to protect it
    # Per task: 24 (LaMP-4/5), 64 (LaMP-7), 512/1024 (LongLaMP training). It caps
    # the training target, the generation length at evaluation, and the prompt
    # budget reserved for the answer.
    max_target_tokens: int = 24

    # ── Optimisation ──────────────────────────────────────────
    epochs: int = 1
    grad_accum: int = 8                 # 8 for LaMP, 4 for LongLaMP
    lr: float = 5e-4
    wd: float = 0.01
    grad_clip: float = 0.5

    # ── HyperLoRA architecture ────────────────────────────────
    # dW = sum_m c_m * (B_m @ A_m): each basis is applied separately and the
    # outputs are mixed, so no cross-terms between bases appear.
    lora_r: int = 16
    basis_m: int = 4
    lora_alpha: float = 32.0
    target_modules: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    # ── Basis training policy ─────────────────────────────────
    # The bases stay trainable in both phases; only their lr is scaled down.
    basis_A_lr_mult: float = 0.1
    basis_B_lr_mult: float = 0.1

    # ── Basis warmup (Phase 1: sequential MoE-style activation) ──
    use_basis_warmup: bool = True
    basis_warmup_dropout: Tuple[float, ...] = (0.05, 0.05, 0.05, 0.05)
    basis_warmup_epochs: int = 1
    basis_warmup_lr_mult: float = 0.1
    basis_warmup_grad_clip: float = 0.3

    # ── HyperNet / Prior ─────────────────────────────────────
    hidden_dim: int = 512
    num_clusters: int = 4

    # ── Evaluation ────────────────────────────────────────────
    # Evaluate the frozen base LM too (no personalization reference).
    run_base_eval: bool = False
    base_eval_max_samples: int = -1
    eval_only: bool = False
    ckpt_dir: str = ""
    # Skip the dev evaluation at the end of training; used for the LongLaMP
    # tasks, which are evaluated by a separate --eval_only run.
    no_final_eval: bool = False
    # Force m2=0 at eval time (beta = alpha, prior-only LoRA). Ablation knob.
    eval_disable_hypernet: bool = False
    # Disable HyperNet during both training and eval (pure cluster-prior LoRA).
    disable_hypernet: bool = False
    # eval-only "swap" diagnostic: route the HyperNet input through a different
    # user (none / random / random_in_cluster).
    eval_swap_mode: str = "none"
    eval_swap_seed: int = 42
    # Decoding is greedy. The penalty matches the setting of the paper this
    # release compares against (the HF default is 1.0).
    eval_repetition_penalty: float = 1.1
    # 0 = off. The paper uses 4 for LongLaMP topic-writing and product-review,
    # where long-form generation otherwise falls into repetition loops.
    eval_no_repeat_ngram_size: int = 0
    # Per-sample evaluation diagnostics (first prompt, predictions, beta, mixing
    # coefficients, buffer sizes). Logging only; it changes nothing that is computed.
    log_diagnostics: bool = False

    # ── Runtime ──────────────────────────────────────────────
    seed: int = 42
    # Evaluation history buffers: "deterministic" reproduces the same buffers for a given --seed
    # (stable SHA-1 hash, independent of PYTHONHASHSEED); "random" redraws them on every run.
    eval_buffer_seeding: str = "deterministic"
    device: str = "cuda"
    dtype: str = "bfloat16"
    main_device: str = "cuda:0"
    # "0:22GiB,1:18GiB" -> {0:"22GiB", 1:"18GiB"}; empty leaves device_map=auto.
    max_memory_per_gpu: str = ""
    # -1: device_map="auto" (default). >=0: number of decoder layers placed on
    # GPU 0 (embed_tokens always on GPU 0; remaining layers + norm + lm_head on GPU 1).
    gpu0_layers: int = -1
    # HF gradient_checkpointing_enable(); on in the paper, since the
    # long-profile tasks otherwise run out of memory (~20% slower).
    gradient_checkpointing: bool = True

    # ── Buffer (rep-div) sampling ────────────────────────────
    buffer_size: int = 32
    buffer_rep_ratio: float = 0.5
    buffer_pool_mult: int = 8
    buffer_div_q_low: float = 0.20
    buffer_div_q_high: float = 0.90

    # ── Prompt history selection (independent of HyperNet input buffer) ──
    # BM25 top-k retrieval against the query, over the user's full bank.
    prompt_bm25_top_k: int = 4
    # Per-item token cap in BM25 mode (0 = no cap). The gold suffix is always preserved.
    prompt_max_tokens_per_item: int = 1000

    # ── HyperNet ─────────────────────────────────────────────
    dirichlet_head_init_std: float = 0.1        # m1_head / m2_head weight init std

    # ── Basis diversity regularisation ───────────────────────
    use_functional_div: bool = True            # functional diversity loss (basis differentiation)
    functional_div_weight: float = 0.1

    # ── Conjugate update (beta = alpha + M2 * M1) ────────────
    conjugate_alpha_on: float = 2.0             # asymmetric prior: diagonal entry (cluster k -> basis k%M)
    conjugate_alpha_off: float = 1.0            # asymmetric prior: off-diagonal entries
    conjugate_m2_bias_init: float = 0.0         # m2_head bias init
    conjugate_m2_min: float = 0.0               # M2 floor (m2 = softplus(raw) + m2_min)
