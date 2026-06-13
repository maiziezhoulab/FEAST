from __future__ import annotations

import torch


def log_sinkhorn(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.1,
    n_iter: int = 100,
    tol: float = 1e-5,
    unbalanced: bool = False,
    reg_m: float = 5.0,
) -> torch.Tensor:
    """Log-domain Sinkhorn solver for local de novo conditional transport."""
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    log_a = torch.log(a + 1e-10)
    log_b = torch.log(b + 1e-10)
    fi = reg_m / (reg_m + epsilon) if unbalanced else 1.0

    for _ in range(n_iter):
        f_prev = f.clone()
        tmp = (g.unsqueeze(0) - C) / epsilon
        f = fi * (epsilon * log_a - epsilon * torch.logsumexp(tmp, dim=1))

        tmp = (f.unsqueeze(1) - C) / epsilon
        g = fi * (epsilon * log_b - epsilon * torch.logsumexp(tmp, dim=0))

        if torch.max(torch.abs(f - f_prev)) < tol:
            break

    log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon
    return torch.exp(log_pi)
