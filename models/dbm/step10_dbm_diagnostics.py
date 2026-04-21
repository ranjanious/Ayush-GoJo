"""
models/dbm/step10_dbm_diagnostics.py
------------------------------------
Phase B, Step 10: negative-rate diagnostics for the DBM simulator.

DBM is Gaussian by construction, so negative rates are not merely possible,
they are guaranteed at some frequency for any non-degenerate parameter set.
This script quantifies that frequency across the three dimensions called for
in the Step 10 spec:

    (1) fraction of paths with ANY negative rate over the horizon
    (2) fraction of paths with a negative rate at the terminal date
    (3) minimum simulated rate across all paths and all horizons

Analytical benchmarks are computed in parallel for validation:
    - P(r_T < 0)                    = Phi(-r_0 / (sigma * sqrt(T)))
    - P(min_{0<=t<=T} r_t < 0)     = 2 * Phi(-r_0 / (sigma * sqrt(T)))
      (reflection principle, mu = 0)
    - E[min_{0<=t<=T} r_t]         = r_0 - sigma * sqrt(2 T / pi)
      (expected min of Brownian motion, mu = 0)

Seeds, grid, and path count are locked to Step 8 / Step 9 (seed = 42,
10,000 paths, Delta t = 1/12) so the diagnostics reconcile exactly with
the terminal-value distributions already filed.

Run:
    python models/dbm/step10_dbm_diagnostics.py
"""

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------
# 1. Configuration -- identical to Phase B, Step 8 / Step 9
# ----------------------------------------------------------------------
SEED     = 42
N_PATHS  = 10_000
DT       = 1.0 / 12.0
T_MAX    = 10.0
N_STEPS  = int(round(T_MAX / DT))          # 120 monthly steps

R0       = 0.0372                          # 3-month CMT, 17 March 2026
MU       = 0.0                             # zero-drift benchmark
SIGMA    = 0.01                            # placeholder volatility

HORIZONS = [1.0, 2.0, 5.0, 10.0]           # years
H_STEPS  = [int(round(h / DT)) for h in HORIZONS]


# ----------------------------------------------------------------------
# 2. DBM simulator (vectorised Euler-Maruyama, same as Steps 7-9)
# ----------------------------------------------------------------------
def simulate_dbm(n_paths, n_steps, r0, mu, sigma, dt, seed):
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        r[:, k + 1] = r[:, k] + mu * dt + sigma * np.sqrt(dt) * Z[:, k]
    return r


# ----------------------------------------------------------------------
# 3. Analytical benchmarks (closed form for mu = 0)
# ----------------------------------------------------------------------
def p_terminal_negative(r0, sigma, T):
    """P(r_T < 0) for DBM with zero drift."""
    return norm.cdf(-r0 / (sigma * np.sqrt(T)))


def p_any_negative(r0, sigma, T):
    """P(min_{0<=t<=T} r_t < 0) via the reflection principle (mu = 0)."""
    return 2.0 * norm.cdf(-r0 / (sigma * np.sqrt(T)))


def expected_min_rate(r0, sigma, T):
    """E[min_{0<=t<=T} r_t] for BM with zero drift, started at r0."""
    return r0 - sigma * np.sqrt(2.0 * T / np.pi)


# ----------------------------------------------------------------------
# 4. Diagnostics
# ----------------------------------------------------------------------
def diagnostics(r_paths, horizons, h_steps, dt, r0, sigma):
    """Return the three Step 10 diagnostic tables as dicts of lists."""
    frac_any    = []
    frac_any_a  = []
    frac_term   = []
    frac_term_a = []
    min_rate    = []
    min_rate_a  = []

    for T, k in zip(horizons, h_steps):
        window = r_paths[:, : k + 1]
        mins   = window.min(axis=1)

        frac_any.append(float((mins < 0.0).mean()))
        frac_any_a.append(p_any_negative(r0, sigma, T))

        frac_term.append(float((r_paths[:, k] < 0.0).mean()))
        frac_term_a.append(p_terminal_negative(r0, sigma, T))

        min_rate.append(float(mins.min()))
        min_rate_a.append(expected_min_rate(r0, sigma, T))

    return {
        "horizons":    horizons,
        "frac_any":    frac_any,
        "frac_any_a":  frac_any_a,
        "frac_term":   frac_term,
        "frac_term_a": frac_term_a,
        "min_rate":    min_rate,
        "min_rate_a":  min_rate_a,
    }


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    print("Phase B, Step 10: DBM negative-rate diagnostics")
    print("-" * 60)
    print(f"seed = {SEED}, N = {N_PATHS}, dt = {DT:.6f}, T_max = {T_MAX}")
    print(f"r0   = {R0}, mu = {MU}, sigma = {SIGMA}")
    print()

    r_paths = simulate_dbm(N_PATHS, N_STEPS, R0, MU, SIGMA, DT, SEED)
    d = diagnostics(r_paths, HORIZONS, H_STEPS, DT, R0, SIGMA)

    global_min      = float(r_paths.min())
    global_any      = float((r_paths.min(axis=1) < 0.0).mean())
    global_analytic = expected_min_rate(R0, SIGMA, T_MAX)

    print(f"{'T (yr)':>7}  {'any<0 sim':>11}  {'any<0 ana':>11}  "
          f"{'r_T<0 sim':>11}  {'r_T<0 ana':>11}  "
          f"{'min sim':>10}  {'E[min] ana':>12}")
    for i, T in enumerate(HORIZONS):
        print(f"{T:>7.1f}  "
              f"{d['frac_any'][i]:>11.5f}  {d['frac_any_a'][i]:>11.5f}  "
              f"{d['frac_term'][i]:>11.5f}  {d['frac_term_a'][i]:>11.5f}  "
              f"{d['min_rate'][i]:>10.5f}  {d['min_rate_a'][i]:>12.5f}")

    print()
    print(f"Across all paths and all horizons up to T = {T_MAX}:")
    print(f"  fraction with any negative rate = {global_any:.5f}")
    print(f"  minimum simulated rate          = {global_min:.5f}")
    print(f"  analytical E[min_{{0..T}}]       = {global_analytic:.5f}")

    return d


if __name__ == "__main__":
    main()
