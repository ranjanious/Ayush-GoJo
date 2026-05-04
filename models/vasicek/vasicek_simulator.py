"""
Vasicek short-rate simulator (Step 13, Phase C).

SDE:    dr_t = a*(b - r_t)*dt + sigma*dW_t
Update: r_{k+1} = r_k + a*(b - r_k)*dt + sigma*sqrt(dt)*Z_k,  Z_k ~ N(0,1) i.i.d.

Placeholder calibration (per project plan):
    a = 0.5     (mean-reversion speed)
    b = 0.04    (long-run mean)
    sigma = 0.01
    r0 = 0.0372 (3-month CMT, March 17 2026)
    dt = 1/12   (monthly grid)
    horizon T = 10 years
    n_paths = 10,000

Provides:
    simulate_vasicek_paths(...)        -> ndarray[n_paths, n_steps+1]
    vasicek_theoretical_mean(...)      -> ndarray[n_steps+1]
    vasicek_theoretical_variance(...)  -> ndarray[n_steps+1]
    vasicek_stationary_moments(...)    -> tuple[float, float]
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Default model and grid constants (placeholder; Phase F replaces with OLS).
# -----------------------------------------------------------------------------
A_DEFAULT: float = 0.5
B_DEFAULT: float = 0.04
SIGMA_DEFAULT: float = 0.01
R0_DEFAULT: float = 0.0372
DT_DEFAULT: float = 1.0 / 12.0
T_DEFAULT: float = 10.0
N_PATHS_DEFAULT: int = 10_000
SEED_DEFAULT: int = 20260417  # reproducibility anchor


def simulate_vasicek_paths(
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    r0: float = R0_DEFAULT,
    dt: float = DT_DEFAULT,
    T: float = T_DEFAULT,
    n_paths: int = N_PATHS_DEFAULT,
    seed: int | None = SEED_DEFAULT,
) -> np.ndarray:
    """
    Simulate Vasicek short-rate paths under Euler-Maruyama.

    Returns an ndarray of shape (n_paths, n_steps + 1) where column 0 is r0
    and column k is r at time t_k = k*dt.
    """
    n_steps = int(round(T / dt))
    rng = np.random.default_rng(seed)
    # Pre-draw the full shock matrix for vectorized stepping.
    Z = rng.standard_normal(size=(n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = r0
    for k in range(n_steps):
        r_k = paths[:, k]
        paths[:, k + 1] = r_k + a * (b - r_k) * dt + sigma * sqrt_dt * Z[:, k]
    return paths


def vasicek_theoretical_mean(
    t_grid: np.ndarray,
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    r0: float = R0_DEFAULT,
) -> np.ndarray:
    """E[r_t] = b + (r0 - b)*exp(-a*t)."""
    return b + (r0 - b) * np.exp(-a * t_grid)


def vasicek_theoretical_variance(
    t_grid: np.ndarray,
    a: float = A_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
) -> np.ndarray:
    """Var[r_t] = sigma^2 * (1 - exp(-2*a*t)) / (2*a)."""
    # Guard a -> 0 limit: variance -> sigma^2 * t.
    if a <= 0:
        return sigma ** 2 * t_grid
    return (sigma ** 2) * (1.0 - np.exp(-2.0 * a * t_grid)) / (2.0 * a)


def vasicek_stationary_moments(
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
) -> tuple[float, float]:
    """Stationary mean and variance: (b, sigma^2 / (2a))."""
    return b, (sigma ** 2) / (2.0 * a)
