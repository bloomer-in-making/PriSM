from .config import TrainCfg
from .utils import log, write_json, load_json, load_pickle, get_torch_dtype, get_main_device, format_metrics
from .cache import maybe_load_cache_arrays
from .prompt import build_prompt
from .metrics import evaluate_predictions
from .buffer import (
    buffer_to_posterior, sample_user_buffer_indices, sample_prompt_buffer_indices,
    sample_prompt_bm25, select_prompt_items,
)
from .model import (
    HyperLoRALinear, PriorTable,
    patch_with_hyperlora, freeze_base_model, set_basis_train_policy,
    set_all_coeffs,
    tokenize_batch, per_example_lm_loss, generate_answer,
    resolve_cluster_id,
)
from .eval import evaluate_base_dev, evaluate_dev
from .dirichlet import (
    DirichletPriorTable, ConjugateHyperNet,
    kl_dirichlet, sample_dirichlet, mixing_entropy,
)
from .train import (
    get_train_bank, run_basis_warmup,
    run_dirichlet_training,
    save_dirichlet_checkpoint, load_dirichlet_checkpoint,
)

__all__ = [
    "TrainCfg",
    "log", "write_json", "load_json", "load_pickle",
    "get_torch_dtype", "get_main_device", "format_metrics",
    "maybe_load_cache_arrays",
    "build_prompt",
    "evaluate_predictions",
    "buffer_to_posterior", "sample_user_buffer_indices", "sample_prompt_buffer_indices",
    "sample_prompt_bm25", "select_prompt_items",
    "HyperLoRALinear", "PriorTable",
    "patch_with_hyperlora", "freeze_base_model", "set_basis_train_policy",
    "set_all_coeffs",
    "tokenize_batch", "per_example_lm_loss", "generate_answer",
    "resolve_cluster_id",
    "evaluate_base_dev", "evaluate_dev",
    "get_train_bank", "run_basis_warmup",
    "DirichletPriorTable", "ConjugateHyperNet",
    "kl_dirichlet", "sample_dirichlet", "mixing_entropy",
    "run_dirichlet_training",
    "save_dirichlet_checkpoint", "load_dirichlet_checkpoint",
]
