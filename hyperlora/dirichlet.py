"""
hyperlora/dirichlet.py
======================
Dirichlet-based HyperLoRA. The per-cluster prior alpha is a fixed, non-learnable
table; only the HyperNet and the LoRA bases are trained.

Key components:
    DirichletPriorTable   : per-cluster alpha (fixed, non-learnable buffer)
    ConjugateHyperNet     : h -> (M1, M2) so that beta = alpha + M2 * M1
    kl_dirichlet          : KL(Dir(beta) || Dir(alpha))
    sample_dirichlet      : w ~ Dir(beta) with the reparameterisation trick
    mixing_entropy        : entropy of the mixing coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dirichlet Prior Table (non-learnable, fitted)
# ============================================================
class DirichletPriorTable(nn.Module):
    """Per-cluster Dirichlet prior alpha. Stored as a non-learnable buffer
    and set once via set_alphas().

    The prior is per-module: the buffer has shape (K, num_modules, M).
    """
    def __init__(self, K: int, M: int, num_modules: int, init_alpha: float = 1.0):
        """
        Args:
            K: number of clusters.
            M: Dirichlet dimension (= number of basis components).
            num_modules: number of target layers wrapped with basis LoRA.
            init_alpha: initial alpha value (uniform prior).
        """
        super().__init__()
        if num_modules <= 0:
            raise ValueError(f"num_modules must be > 0 (got {num_modules})")
        self.K = K
        self.M = M
        self.num_modules = num_modules
        self.register_buffer("alphas", torch.full((K, num_modules, M), init_alpha))

    def set_alphas(self, alphas: torch.Tensor):
        """Set the prior alpha. Shape (K, num_modules, M)."""
        assert alphas.shape == self.alphas.shape, \
            f"Expected {self.alphas.shape}, got {alphas.shape}"
        self.alphas.copy_(alphas.clamp(min=1e-3))

    def forward(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        """cluster_ids: (B,) or scalar -> alpha: (B, num_modules, M) or (num_modules, M)."""
        return self.alphas[cluster_ids]
# ============================================================
# ConjugateHyperNet: β = α + M2(r) · M1(r)
# ============================================================
class ConjugateHyperNet(nn.Module):
    """User embedding h → posterior Dirichlet concentration via conjugate update.

    The classical conjugate update beta_i = alpha_i + n_i is emulated by a
    HyperNet:
        beta = alpha + M2(r) * M1(r)
    where
        M1(r) in the simplex (softmax)   : pseudo-count direction
        M2(r) in R+ (softplus)           : pseudo-count strength (per-module scalar)

    Architecture (M1 and M2 use separate trunks):
        h (in_dim), with mu and logvar standardised separately by train stats
        -> trunk_m1 -> hidden -> m1_head -> softmax  (direction)
        -> trunk_m2 -> hidden -> m2_head -> softplus (strength)

    One (M1, M2) pair is produced per user and applied to every wrapped module.

    The two trunks share the same shape but no parameters; this avoids the
    gradient interference that happens when one head's gradient distorts the
    features used by the other.

    forward returns the pseudo_count. The caller composes beta = alpha + pseudo_count.
    """
    def __init__(
        self,
        in_dim: int,
        num_modules: int,
        basis_m: int,
        hidden: int = 512,
        head_init_std: float = 0.1,
        m2_bias_init: float = 0.0,    # m2_head bias init
        m2_min: float = 0.0,          # M2 floor: m2 = softplus(raw) + m2_min; >0 prevents dead zones
    ):
        super().__init__()
        self.num_modules = num_modules
        self.basis_m = basis_m
        self.m2_min = float(m2_min)
        # Eval-time HyperNet ablation: when True, force m2=0 in forward -> beta=alpha (prior-only LoRA).
        # Always False during training. Toggled at eval time via main.py's eval_only + eval_disable_hypernet.
        self._force_m2_zero = False

        # h is [mu, log var]: standardise each half separately using train mean/std.
        half_dim = in_dim // 2
        self.register_buffer("in_mu_mean", torch.zeros(half_dim))
        self.register_buffer("in_mu_std", torch.ones(half_dim))
        self.register_buffer("in_lv_mean", torch.zeros(half_dim))
        self.register_buffer("in_lv_std", torch.ones(half_dim))

        trunk_in_dim = in_dim

        def _make_trunk() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(trunk_in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )

        # Separate trunks for M1 (direction) and M2 (magnitude).
        self.trunk_m1 = _make_trunk()
        self.trunk_m2 = _make_trunk()

        self.m1_head = nn.Linear(hidden, basis_m)   # softmax -> direction
        self.m2_head = nn.Linear(hidden, 1)         # softplus → scalar

        # Head init.
        self._head_init_std = head_init_std
        nn.init.zeros_(self.m1_head.bias)
        nn.init.normal_(self.m1_head.weight, std=head_init_std)
        nn.init.normal_(self.m2_head.weight, std=head_init_std)
        nn.init.constant_(self.m2_head.bias, float(m2_bias_init))

    def _trunk_input(self, h: torch.Tensor) -> torch.Tensor:
        """Standardise mu and logvar separately with the train stats, and emit a
        single row that the two trunks share across all modules."""
        half = h.shape[-1] // 2
        mu_part = (h[:half] - self.in_mu_mean) / (self.in_mu_std + 1e-6)
        lv_part = (h[half:2 * half] - self.in_lv_mean) / (self.in_lv_std + 1e-6)
        h = torch.cat([mu_part, lv_part], dim=-1)
        return h.unsqueeze(0)  # (1, in_dim)

    @torch.no_grad()
    def set_input_stats_dual(
        self,
        mu_mean: torch.Tensor, mu_std: torch.Tensor,
        lv_mean: torch.Tensor, lv_std: torch.Tensor,
    ):
        """Inject the train mean/std for mu and logvar."""
        self.in_mu_mean.copy_(mu_mean.to(self.in_mu_mean.device, self.in_mu_mean.dtype))
        self.in_mu_std.copy_(mu_std.to(self.in_mu_std.device, self.in_mu_std.dtype))
        self.in_lv_mean.copy_(lv_mean.to(self.in_lv_mean.device, self.in_lv_mean.dtype))
        self.in_lv_std.copy_(lv_std.to(self.in_lv_std.device, self.in_lv_std.dtype))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            pseudo_count : (num_modules, basis_m) — M2·M1. β = α + pseudo_count.
            (M1, M2) is computed once and broadcast to every module.
        """
        # Ablation: HyperNet off -> skip the trunk and return pseudo_count=0.
        if self._force_m2_zero:
            shape = (self.num_modules, self.basis_m)
            return torch.zeros(shape, device=h.device, dtype=h.dtype)
        x = self._trunk_input(h)
        m1 = F.softmax(self.m1_head(self.trunk_m1(x)), dim=-1)         # (1, M)
        m2 = F.softplus(self.m2_head(self.trunk_m2(x))) + self.m2_min  # (1, 1)
        # Apply the same (M1, M2) to every module.
        return (m2 * m1).expand(self.num_modules, -1)

    def forward_decomposed(self, h: torch.Tensor):
        """Return m1, m2, and pseudo_count separately (used by the warmup loss),
        each expanded to (num_modules, *)."""
        x = self._trunk_input(h)
        m1 = F.softmax(self.m1_head(self.trunk_m1(x)), dim=-1)
        if self._force_m2_zero:
            m2 = torch.zeros_like(m1[..., :1])
            pseudo_count = torch.zeros_like(m1)
        else:
            m2 = F.softplus(self.m2_head(self.trunk_m2(x))) + self.m2_min
            pseudo_count = m2 * m1
        return (m1.expand(self.num_modules, -1),
                m2.expand(self.num_modules, -1),
                pseudo_count.expand(self.num_modules, -1))


# ============================================================
# Dirichlet Sampling (Reparameterization Trick)
# ============================================================
def sample_dirichlet(
    beta: torch.Tensor,
    sample: bool = True,
    n_samples: int = 1,
) -> torch.Tensor:
    """Sample from Dir(beta) with the Gamma reparameterisation trick.

    w_m = g_m / Σg, where g_m ~ Gamma(β_m, 1)

    PyTorch's torch.distributions.Dirichlet supports rsample(), so gradients
    flow through the sample.

    Args:
        beta: (..., M) Dirichlet concentration params (positive)
        sample: if False, return the mean (= beta / sum(beta))
        n_samples: number of samples (squeezed when 1)
    Returns:
        w: (..., M) mixing coefficients on the simplex
    """
    if not sample:
        return beta / beta.sum(dim=-1, keepdim=True)

    dist = torch.distributions.Dirichlet(beta)
    if n_samples == 1:
        return dist.rsample()
    else:
        return dist.rsample((n_samples,))


# ============================================================
# KL Divergence: Dir(β) || Dir(α)
# ============================================================
def kl_dirichlet(
    beta: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL(Dir(β) || Dir(α)).

    KL = log Γ(Σβ) - Σ log Γ(β_m) - log Γ(Σα) + Σ log Γ(α_m)
         + Σ (β_m - α_m)(ψ(β_m) - ψ(Σβ))

    Args:
        beta:  (..., M) posterior Dirichlet params
        alpha: (..., M) prior Dirichlet params (broadcastable)
    Returns:
        kl: (...) scalar per batch element
    """
    beta_sum = beta.sum(dim=-1)
    alpha_sum = alpha.sum(dim=-1)

    kl = (
        torch.lgamma(beta_sum) - torch.lgamma(alpha_sum)
        - (torch.lgamma(beta) - torch.lgamma(alpha)).sum(dim=-1)
        + ((beta - alpha) * (torch.digamma(beta) - torch.digamma(beta_sum).unsqueeze(-1))).sum(dim=-1)
    )
    return kl
# ============================================================
# Entropy monitoring (collapse prevention)
# ============================================================
def mixing_entropy(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """H(w) = -sum_m w_m log w_m. Higher = more diverse.

    Args:
        w: (..., M) simplex vectors
    Returns:
        entropy: (...) per-element
    """
    return -(w * torch.log(w + eps)).sum(dim=-1)
