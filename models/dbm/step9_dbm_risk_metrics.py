"""
models/dbm/step9_dbm_risk_metrics.py
------------------------------------
Phase B, Step 9: feed Step 8's DBM terminal-value distributions into the
shared risk engine and record the full metric set (mean, std, skewness,
kurtosis, VaR95, VaR99, ES95, ES99, P(loss)) with simulation standard
errors (delta method + bootstrap for VaR; i.i.d. + bootstrap for ES) at
horizons T = 1, 2, 5, 10 years, separately for Bond A, Bond B, and the
50/50 portfolio.

Run:
    python models/dbm/step9_dbm_risk_metrics.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..")))
from shared.risk_engine import risk_report


# ----------------------------------------------------------------------
# 1. Configuration (placeholder parameters; replaced in Phase F)
# ----------------------------------------------------------------------
SEED      = 42
N_PATHS   = 10_000
DT        = 1.0 / 12.0
T_MAX     = 10.0
N_STEPS   = int(round(T_MAX / DT))   # 120 monthly steps

R0        = 0.0372   # 3-month CMT, 17 March 2026
MU        = 0.0      # zero-drift benchmark for DBM
SIGMA     = 0.01     # placeholder volatility

FACE      = 100.0
COUPON_A  = 1.84                               # 3.68% par, semiannual
COUPON_B  = 2.10                               # 4.20% par, semiannual
DATES_A   = [0.5 * i for i in range(1, 5)]     # 0.5, 1.0, 1.5, 2.0
DATES_B   = [0.5 * i for i in range(1, 21)]    # 0.5, 1.0, ..., 10.0

HORIZONS  = [1.0, 2.0, 5.0, 10.0]


# ----------------------------------------------------------------------
# 2. DBM simulator (vectorised Euler-Maruyama)
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
# 3. Time-T bond valuation (left-endpoint Riemann discount)
# ----------------------------------------------------------------------
def cumulative_integrated_rate(rates, dt):
    """cumR[p, k] = integral of r_u du from 0 to k*dt, left-endpoint rule."""
    cumR = np.zeros_like(rates)
    cumR[:, 1:] = np.cumsum(rates[:, :-1] * dt, axis=1)
    return cumR


def bond_time_T_values(cumR, coupon_dates, coupon, face, T, dt):
    """
    Time-T value of a fixed-coupon bond along each path.
    Past coupons (t <= T) are accrued forward to T at path rates;
    future cash flows (t > T) are discounted back to T at path rates.
    """
    T_step  = int(round(T / dt))
    logAF_T = cumR[:, T_step]
    values  = np.zeros(cumR.shape[0])
    last    = coupon_dates[-1]
    for t in coupon_dates:
        t_step  = int(round(t / dt))
        logAF_t = cumR[:, t_step]
        cf      = coupon + face if np.isclose(t, last) else coupon
        values += cf * np.exp(logAF_T - logAF_t)
    return values


# ----------------------------------------------------------------------
# 4. Driver
# ----------------------------------------------------------------------
def main():
    print(f"Simulating DBM: N = {N_PATHS:,}, steps = {N_STEPS}, seed = {SEED}")
    rates = simulate_dbm(N_PATHS, N_STEPS, R0, MU, SIGMA, DT, SEED)
    cumR  = cumulative_integrated_rate(rates, DT)

    rows = []
    for T in HORIZONS:
        V_A    = bond_time_T_values(cumR, DATES_A, COUPON_A, FACE, T, DT)
        V_B    = bond_time_T_values(cumR, DATES_B, COUPON_B, FACE, T, DT)
        V_port = 0.5 * V_A + 0.5 * V_B

        for name, dist, threshold in [
            ("Bond A",    V_A,    FACE),
            ("Bond B",    V_B,    FACE),
            ("Portfolio", V_port, FACE),
        ]:
            report = risk_report(dist, threshold=threshold,
                                 alphas=(0.05, 0.01),
                                 n_boot=1000, seed=SEED)
            report.update({"Horizon": T, "Instrument": name})
            rows.append(report)

    df = pd.DataFrame(rows)
    cols = ["Horizon", "Instrument",
            "mean", "std", "skew", "kurt", "min", "max", "p_loss",
            "VaR95", "VaR95_SE_delta", "VaR95_SE_boot",
            "ES95",  "ES95_SE_iid",    "ES95_SE_boot",
            "VaR99", "VaR99_SE_delta", "VaR99_SE_boot",
            "ES99",  "ES99_SE_iid",    "ES99_SE_boot"]
    df = df[cols]
    return df


if __name__ == "__main__":
    df = main()
    pd.set_option("display.float_format", lambda x: f"{x:9.4f}")
    print(df.to_string(index=False))
    out_path = os.path.join(os.path.dirname(__file__), "step9_risk_metrics.csv")
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nWrote results to {out_path}")
