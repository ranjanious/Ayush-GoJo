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
    horizon_years: float | None = None,
) -> np.ndarray:
    """
    Simulate Vasicek short-rate paths under Euler-Maruyama.

    Returns an ndarray of shape (n_paths, n_steps + 1) where column 0 is r0
    and column k is r at time t_k = k*dt.

    `horizon_years` is an explicit-name alias for `T`, accepted for Step 16+
    callers; if provided, it overrides the positional `T` argument.
    """
    if horizon_years is not None:
        T = horizon_years
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


def simulate_vasicek(
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    r0: float = R0_DEFAULT,
    dt: float = DT_DEFAULT,
    n_steps: int | None = None,
    n_paths: int = N_PATHS_DEFAULT,
    seed: int | None = SEED_DEFAULT,
) -> np.ndarray:
    """
    Step 14 wrapper around `simulate_vasicek_paths` that takes `n_steps`
    instead of `T`.  Provided so downstream Step 14+ code can import a
    function with the explicit-grid signature used throughout Phase C.
    """
    if n_steps is None:
        T_local = T_DEFAULT
    else:
        T_local = n_steps * dt
    return simulate_vasicek_paths(
        a=a, b=b, sigma=sigma, r0=r0, dt=dt, T=T_local,
        n_paths=n_paths, seed=seed,
    )


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


# -----------------------------------------------------------------------------
# Step 15 additions: discount factor helper + MC zero-coupon bond pricer
# -----------------------------------------------------------------------------
def riemann_discount_factor(rate_paths: np.ndarray, dt: float) -> np.ndarray:
    """
    Left-endpoint Riemann approximation of D_T = exp(- int_0^T r_s ds) along
    each path.  Uses columns 0 .. n_steps - 1 so the integral excludes r_T,
    matching the project's bond_pricing.py convention.

    Args
    ----
    rate_paths : ndarray (n_paths, n_steps + 1)
    dt         : float, grid spacing in years

    Returns
    -------
    ndarray (n_paths,)
    """
    integ = rate_paths[:, :-1].sum(axis=1) * dt
    return np.exp(-integ)


def riemann_left_integral(rate_paths: np.ndarray, dt: float) -> np.ndarray:
    """
    Cumulative left-endpoint integral of the short rate along each path.

        I[p, k] = sum_{j < k} r[p, j] * dt   for k = 0, 1, ..., n_steps
        I[p, 0] = 0

    Shape (n_paths, n_steps + 1).  Reused by Step 16 for both the
    money-market account M(t_k) = exp(I[k]) and the reinvested-coupon
    growth factor exp(I[k] - I[T_j]).
    """
    n_paths, n_grid = rate_paths.shape
    I = np.zeros((n_paths, n_grid), dtype=np.float64)
    I[:, 1:] = np.cumsum(rate_paths[:, :-1] * dt, axis=1)
    return I


def zcb_price_mc(
    a: float,
    b: float,
    sigma: float,
    r0: float,
    T: float,
    dt: float = DT_DEFAULT,
    n_paths: int = N_PATHS_DEFAULT,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Monte Carlo zero-coupon bond price P_MC(0, T) under Vasicek.

    Returns
    -------
    (price, standard_error, discount_factor_array)
        price    -- sample mean of the path-specific discount factors
        se       -- sample std / sqrt(n_paths)
        D        -- ndarray (n_paths,) of path-specific discount factors
    """
    paths = simulate_vasicek_paths(
        a=a, b=b, sigma=sigma, r0=r0, dt=dt, T=T,
        n_paths=n_paths, seed=seed,
    )
    D = riemann_discount_factor(paths, dt)
    price = float(D.mean())
    se = float(D.std(ddof=1) / np.sqrt(n_paths))
    return price, se, D
