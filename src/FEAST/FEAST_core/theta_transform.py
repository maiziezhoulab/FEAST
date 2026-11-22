"""Bijective transforms between per-gene statistics and unconstrained coordinates.

  theta = [log(mu), log(omega), logit(pi0)]

where omega = variance / mean (Fano factor).
"""

import numpy as np
import pandas as pd

EPS = 1e-10


def stats_to_theta(stats_df: pd.DataFrame) -> np.ndarray:
    """Convert per-gene (mean, variance, zero_prop) to (G, 3) theta matrix.

    Columns: log_mu, log_omega, logit_pi0.
    """
    mu = np.asarray(stats_df["mean"].values, dtype=np.float64)
    var = np.asarray(stats_df["variance"].values, dtype=np.float64)
    pi0 = np.asarray(stats_df["zero_prop"].values, dtype=np.float64)

    mu_clipped = np.clip(mu, EPS, None)
    omega = var / mu_clipped
    omega_clipped = np.clip(omega, EPS, None)
    pi0_clipped = np.clip(pi0, EPS, 1.0 - EPS)

    log_mu = np.log(mu_clipped)
    log_omega = np.log(omega_clipped)
    logit_pi0 = np.log(pi0_clipped / (1.0 - pi0_clipped))

    return np.column_stack([log_mu, log_omega, logit_pi0])


def theta_to_stats(theta: np.ndarray) -> pd.DataFrame:
    """Inverse of stats_to_theta: (G, 3) theta -> DataFrame with [mean, variance, zero_prop]."""
    theta = np.asarray(theta, dtype=np.float64)
    log_mu = theta[:, 0]
    log_omega = theta[:, 1]
    logit_pi0 = theta[:, 2]

    mu = np.exp(log_mu)
    omega = np.exp(log_omega)
    var = omega * mu
    pi0 = 1.0 / (1.0 + np.exp(-logit_pi0))

    return pd.DataFrame({"mean": mu, "variance": var, "zero_prop": pi0})
