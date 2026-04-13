"""
dbm_simulator.py
================
Drifted Brownian Motion (DBM) short-rate simulator.

Model:  dr_t = mu * dt + sigma * dW_t
Update: r_{k+1} = r_k + mu*dt + sigma*sqrt(dt)*Z_k,  Z_k ~ N(0,1) i.i.d.

Parameters (baseline):
    r0    = 0.0372   # 3-month CMT par yield, March 17 2026
    mu    = 0.0      # zero drift (benchmark case)
    sigma = 0.01     # annualised short-rate volatility
    dt    = 1/12     # monthly grid
    T     = 2.0      # horizon = 2 years (Bond A maturity)
    N     = 10_000   # number of independent paths
"""

import numpy as np


def simulate_dbm(
    r0: float = 0.0372,
    mu: float = 0.0,
    sigma: float = 0.01,
    dt: float = 1 / 12,
    T: float = 2.0,
    N: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate N independent paths of the DBM short-rate model.

    Returns
    -------
    paths : ndarray, shape (N, steps + 1)
        Full rate paths; paths[:, 0] == r0 for every path.
    """
    rng = np.random.default_rng(seed)
    steps = round(T / dt)

    # Draw all innovations upfront: shape (N, steps)
    Z = rng.standard_normal((N, steps))

    paths = np.empty((N, steps + 1))
    paths[:, 0] = r0

    drift = mu * dt
    diffusion = sigma * np.sqrt(dt)

    for k in range(steps):
        paths[:, k + 1] = paths[:, k] + drift + diffusion * Z[:, k]

    return paths


def bond_price_along_path(
    paths: np.ndarray,
    coupon: float,
    face: float,
    payment_times: np.ndarray,
    dt: float = 1 / 12,
) -> np.ndarray:
    """
    Price a fixed-coupon bond along each simulated rate path.

    Discount factor at T_j:
        D(0, T_j) = exp( -sum_{i=0}^{j-1} r_{t_i} * dt )   [left-endpoint Riemann sum]

    Returns
    -------
    prices : ndarray, shape (N,)   -- one present value per path
    """
    N = paths.shape[0]
    prices = np.zeros(N)

    for j, t_j in enumerate(payment_times):
        n_steps = round(t_j / dt)
        log_df = -np.sum(paths[:, :n_steps], axis=1) * dt
        df = np.exp(log_df)
        cf = (face if j == len(payment_times) - 1 else 0.0) + coupon
        prices += cf * df

    return prices


if __name__ == "__main__":
    paths = simulate_dbm(
        r0=0.0372, mu=0.0, sigma=0.01,
        dt=1 / 12, T=2.0, N=10_000, seed=42,
    )

    r_T = paths[:, -1]
    print(f"Paths shape : {paths.shape}")
    print(f"E[r_T]      : {r_T.mean():.4f}  (expected 0.0372)")
    print(f"Std[r_T]    : {r_T.std():.4f}  (expected {0.01 * (2.0 ** 0.5):.4f})")

    # Bond A: 2-yr, coupon $1.84, face $100, semiannual
    payment_times_A = np.arange(0.5, 2.5, 0.5)  # [0.5, 1.0, 1.5, 2.0]
    prices_A = bond_price_along_path(
        paths, coupon=1.84, face=100.0,
        payment_times=payment_times_A, dt=1 / 12,
    )

    print(f"\nBond A prices:")
    print(f"Mean   : {prices_A.mean():.4f}")
    print(f"Std    : {prices_A.std():.4f}")

    for alpha, label in [(0.05, "95%"), (0.01, "99%")]:
        var = np.percentile(prices_A, alpha * 100)
        es = prices_A[prices_A <= var].mean()
        print(f"VaR({label}): {var:.4f}   ES({label}): {es:.4f}")
