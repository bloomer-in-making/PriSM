# PriSM

Reference implementation of PriSM: a prior-guided mixture of shared LoRA bases
for personalized text generation. One frozen backbone, `M` shared LoRA bases,
and a HyperNet that turns a user's history into the coefficients that mix them.

| `--task_name` | domain | metric |
|---|---|---|
| `LaMP_4` | news headline generation | ROUGE-1 / ROUGE-L |
| `LaMP_5` | paper title generation | ROUGE-1 / ROUGE-L |
| `LaMP_7` | tweet paraphrasing | ROUGE-1 / ROUGE-L |
| `LongLaMP_abstract_generation_user` | paper abstract generation | ROUGE-1 / ROUGE-L |
| `LongLaMP_topic_writing_user` | reddit topic writing | ROUGE-1 / ROUGE-L |
| `LongLaMP_product_review_user` | product review generation | ROUGE-1 / ROUGE-L |

All six share one implementation and one set of method hyperparameters. What
differs per task is the data format, the sequence budget and the generation
length; `scripts/run_task.sh` applies those automatically.

## Method

**Stage I — prior basis formation.** Train users are clustered by their profile
mean, and each cluster is paired with one of the `M` shared LoRA bases. Each
basis is warmed up on the training rows of its own cluster. A fixed asymmetric
Dirichlet prior `α_c` is defined per cluster, putting higher mass on that
cluster's basis. The priors are never trained.

**Stage II — prior-guided personalization.** For a user `u` in cluster `c(u)`:

1. A frozen encoder (BGE-small) embeds the user's history items. Their
   element-wise mean and log-variance form the profile representation
   `h = [μ_u, log v_u]`, which the HyperNet standardizes half by half with
   train statistics before using it.
2. The HyperNet emits `M1(h)`, a softmax over the `M` bases, and `M2(h)`, a
   positive scalar setting how much user-specific evidence to add.
3. The posterior-like Dirichlet parameters are

       β = α_{c(u)} + M2(h) · M1(h)

   This is learned pseudo-evidence, not a Bayesian posterior over observed
   counts.
4. The mixing coefficients are the Dirichlet mean `w = β / Σβ`, and the
   completed basis updates are combined:

       ΔW_u = Σ_m w_m · B_m A_m

   The bases are mixed as finished products; `A` and `B` are not averaged
   separately.
5. The HyperNet and the Stage-I bases are trained jointly with the
   language-modelling cross-entropy and a small `functional_div` term that
   discourages basis collapse. The priors and the backbone stay fixed.

## Requirements

Python 3.10 or newer, and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Quick start

```
export TRAIN_QUESTIONS=/path/to/LaMP_4/train/train_questions.json
export TRAIN_OUTPUTS=/path/to/LaMP_4/train/train_outputs.json
export DEV_QUESTIONS=/path/to/LaMP_4/dev/dev_questions.json
export DEV_OUTPUTS=/path/to/LaMP_4/dev/dev_outputs.json

bash scripts/run_task.sh LaMP_4
```

The driver precomputes the cache if `cache/<task>` is missing, trains, then
evaluates, and writes `runs/<task>_eval/dev_eval_only.json` (ROUGE-1 /
ROUGE-L) and `dev_preds_refs.json`. The environment variables it accepts are
documented at the top of the script.

A single 24 GB GPU is enough for the three LaMP tasks with Qwen3-8B in
`bfloat16`; the LongLaMP tasks need more, and `GPU0_LAYERS=4` splits the model
across two GPUs without changing the computation.

## Pipeline

`precompute.py` builds a task's cache, `main.py` trains and evaluates, and
`scripts/run_task.sh` runs both with the per-task flags. The two Python entry
points take `--help`. The method lives in `hyperlora/`. The user-level k-means
in the cache is fit on **train** users only.

Per-task values are in the `case` block at the top of `scripts/run_task.sh`;
everything else is identical across the six tasks and is the set of defaults in
`hyperlora/config.py`. `--eval_repetition_penalty` is 1.1 everywhere, matching
the work this release compares against (the HuggingFace default is 1.0).

## Reproducibility

Two seeds fix everything that is drawn at random. `--seed`, passed to both
steps, fixes the per-user history buffers, the parameter init and the data
order; `--cluster_seed`, passed to `precompute.py`, fixes the k-means and was
chosen on the **train split only**.

A user's 32-item buffer is a random subsample of their profile, so it has to be
seeded. `--eval_buffer_seeding deterministic` (the default) seeds it from a
SHA-1 hash of `(user_id, --seed)`, so the same `--seed` yields the same buffers
across runs and machines, independently of `PYTHONHASHSEED`. Passing `random`
redraws on every run; use it to measure how much the draw alone moves a metric.

**Reported numbers.** Evaluation is deterministic by default in this code, but
the reported numbers were produced under a different buffer draw, so metrics
can differ slightly. On LaMP-4 the same trained checkpoint evaluated under four
seeds gives ROUGE-1 of 0.2093–0.2106 (SD 0.0006), which contains the reported
0.2101.

## Data

No data is bundled. LaMP follows the standard user-based split published at
<https://lamp-benchmark.github.io/>, and LongLaMP is at
<https://longlamp-benchmark.github.io/>. `precompute.py` expects each split as
a `*_questions.json` / `*_outputs.json` pair, which is the layout both
benchmarks distribute. LongLaMP's Hugging Face release calls the validation
split `val` where the LaMP layout calls it `dev`; pass whichever files hold it
to `--dev_questions` / `--dev_outputs`.

## Assets and credits

This release bundles no data and no model weights. It builds on the following,
each obtained from its own distributor:

| Asset | Used as | License | Reference |
|---|---|---|---|
| [LaMP](https://lamp-benchmark.github.io/) — tasks 4, 5, 7 | benchmark data | none published by the benchmark | Salemi, Mysore, Bendersky, Zamani, *LaMP: When Large Language Models Meet Personalization*, [arXiv:2304.11406](https://arxiv.org/abs/2304.11406) |
| [LongLaMP](https://longlamp-benchmark.github.io/) — abstract, topic, product | benchmark data | none published by the benchmark | Kumar et al., *LongLaMP: A Benchmark for Personalized Long-form Text Generation*, [arXiv:2407.11016](https://arxiv.org/abs/2407.11016) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | frozen backbone LM | Apache-2.0, per its model card | Qwen Team, *Qwen3 Technical Report*, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | frozen sentence encoder | MIT, per its model card | Xiao, Liu, Zhang, Muennighoff, *C-Pack*, [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) |

Both benchmarks also contain tasks built from the Avocado Research Email
Collection, which is not publicly available. **None of the six tasks in this
release uses it**, so these numbers can be reproduced without that corpus.

## License

MIT; see [LICENSE](LICENSE). It covers this repository only, not the benchmark
data or the model weights listed above.
