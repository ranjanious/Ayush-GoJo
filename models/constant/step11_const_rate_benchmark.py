"""
models/constant/step11_const_rate_benchmark.py
----------------------------------------------
Phase B, Step 11: constant-rate benchmark comparison.

Anchors the risk-measurement pipeline to its degenerate limit.  Holds the
short rate at r_t = r0 for all t in [0, 10] and routes the constant rate
through the same Step 8 / Step 9 pricing and risk-metric pipeline used for
DBM.  Under r_t = r0 the terminal-value distribution collapses to a point
mass at exp(r0 * T_h) * P0(r0): standard deviation is zero, VaR and ES
coincide with the deterministic terminal value, and P(loss) is identically
zero.

For direct side-by-side comparison the script also re-runs the Step 8 DBM
pipeline with seed = 42 and writes the paired results to
models/constant/step11_results.json.

Run:
    python models/constant/step11_const_rate_benchmark.py
    # or, with the package layout, from project root:
    python -m models.constant.step11_const_rate_benchmark
"""

import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

from shared.risk_engine import (
    expected_shortfall,
    prob_loss,
    var_empirical,
)


# ----------------------------------------------------------------------
# 1. Configuration -- identical to Phase B, Steps 8 / 9 / 10
# ----------------------------------------------------------------------
SEED = 42
N_PATHS = 10_000
DT = 1.0 / 12.0
T_MAX = 10.0
N_STEPS = int(round(T_MAX / DT))      # 120 monthly steps

R0 = 0.0372                            # 3-month CMT, 17 March 2026
DBM_MU = 0.0                           # zero-drift benchmark
DBM_SIGMA = 0.01                       # placeholder volatility

FACE = 100.0
COUPON_A = 1.84                        # 3.68% par, semiannual
COUPON_B = 2.10                        # 4.20% par, semiannual
DATES_A = [0.5 * i for i in range(1, 5)]   # 0.5, 1.0, 1.5, 2.0
DATES_B = [0.5 * i for i in range(1, 21)]  # 0.5, 1.0, ..., 10.0

HORIZONS = [1.0, 2.0, 5.0, 10.0]
V0_REF = 100.0                         # paper normalization for P(loss)


# ----------------------------------------------------------------------
# 2. Rate-path generators
# ----------------------------------------------------------------------
def simulate_constant(n_paths, n_steps, r0):
    """Rate-path matrix with r_k = r0 for all k.  Shape (n_paths, n_steps+1)."""
    return np.full((n_paths, n_steps + 1), r0, dtype=float)


def simulate_dbm(n_paths, n_steps, r0, mu, sigma, dt, seed):
    """Vectorised Euler-Maruyama; identical to Step 9 for bit-exact replication."""
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        r[:, k + 1] = r[:, k] + mu * dt + sigma * np.sqrt(dt) * Z[:, k]
    return r


# ----------------------------------------------------------------------
# 3. Pricing infrastructure (reuses Step 8 conventions without modification)
# ----------------------------------------------------------------------
def path_integrals(rate_paths, dt):
    """I[p, k] = integral of r_u du from 0 to k*dt, left-endpoint Riemann."""
    I = np.zeros_like(rate_paths)
    I[:, 1:] = np.cumsum(rate_paths[:, :-1] * dt, axis=1)
    return I


def bond_value_paths(I, coupon_dates, coupon, face, T, dt):
    """
    Time-T value of a fixed-coupon bond along each path.
    Past coupons (t <= T) are accrued forward to T at path rates;
    future cash flows (t > T) are discounted back to T at path rates.
    At T = 0 this collapses to the initial bond price.
    """
    T_step = int(round(T / dt))
    I_T = I[:, T_step]
    values = np.zeros(I.shape[0])
    last = coupon_dates[-1]
    for t in coupon_dates:
        t_step = int(round(t / dt))
        I_t = I[:, t_step]
        cf = coupon + face if np.isclose(t, last) else coupon
        values += cf * np.exp(I_T - I_t)
    return values


# ----------------------------------------------------------------------
# 4. Risk metrics -- light wrapper; spec requires only core statistics
# ----------------------------------------------------------------------
def risk_metrics(dist, threshold):
    """Mean, std, min, max, VaR95/99, ES95/99, P(loss) for one distribution."""
    dist = np.asarray(dist, dtype=float)
    return {
        "mean":   float(np.mean(dist)),
        "std":    float(np.std(dist, ddof=1)) if dist.size > 1 else 0.0,
        "min":    float(np.min(dist)),
        "max":    float(np.max(dist)),
        "VaR95":  var_empirical(dist, 0.05),
        "VaR99":  var_empirical(dist, 0.01),
        "ES95":   expected_shortfall(dist, 0.05),
        "ES99":   expected_shortfall(dist, 0.01),
        "p_loss": prob_loss(dist, threshold),
    }


# ----------------------------------------------------------------------
# 5. Pipeline driver
# ----------------------------------------------------------------------
def run_pipeline(rate_paths, label):
    """Route rate_paths through path_integrals -> bond_value_paths -> risk_metrics."""
    I = path_integrals(rate_paths, DT)

    P0_A_paths = bond_value_paths(I, DATES_A, COUPON_A, FACE, 0.0, DT)
    P0_B_paths = bond_value_paths(I, DATES_B, COUPON_B, FACE, 0.0, DT)
    P0_A = float(np.mean(P0_A_paths))
    P0_B = float(np.mean(P0_B_paths))

    out = {
        "label": label,
        "P0_A": P0_A,
        "P0_B": P0_B,
        "V0_model": 0.5 * P0_A + 0.5 * P0_B,
        "V0_ref": V0_REF,
        "horizons": {},
    }

    for t_h in HORIZONS:
        V_A = bond_value_paths(I, DATES_A, COUPON_A, FACE, t_h, DT)
        V_B = bond_value_paths(I, DATES_B, COUPON_B, FACE, t_h, DT)
        V_pf = 0.5 * V_A + 0.5 * V_B

        out["horizons"][str(t_h)] = {
            "bond_a":    risk_metrics(V_A,  FACE),
            "bond_b":    risk_metrics(V_B,  FACE),
            "portfolio": risk_metrics(V_pf, V0_REF),
        }

    return out


# ----------------------------------------------------------------------
# 6. Sanity checks
# ----------------------------------------------------------------------
def sanity_checks(const_res, dbm_res):
    """Five checks promised in the completion log, Section 6."""
    checks = {}

    # (1) point-mass distribution for constant rate: range = 0 everywhere
    point_mass_ok = True
    rate_paths_const = simulate_constant(N_PATHS, N_STEPS, R0)
    I_const = path_integrals(rate_paths_const, DT)
    for t_h in HORIZONS:
        V_A = bond_value_paths(I_const, DATES_A, COUPON_A, FACE, t_h, DT)
        V_B = bond_value_paths(I_const, DATES_B, COUPON_B, FACE, t_h, DT)
        if (V_A.max() - V_A.min() > 1e-9) or (V_B.max() - V_B.min() > 1e-9):
            point_mass_ok = False
            break
    checks["(1) constant-rate distribution is a point mass"] = point_mass_ok

    # (2) closed-form cross-check: V(T_h) = exp(r0 * T_h) * P0(r0)
    P0_A = const_res["P0_A"]
    P0_B = const_res["P0_B"]
    cf_ok = True
    for t_h in HORIZONS:
        expected_A = np.exp(R0 * t_h) * P0_A
        expected_B = np.exp(R0 * t_h) * P0_B
        h = const_res["horizons"][str(t_h)]
        if (abs(h["bond_a"]["mean"] - expected_A) > 1e-8
                or abs(h["bond_b"]["mean"] - expected_B) > 1e-8):
            cf_ok = False
    checks["(2) closed-form identity V(T) = exp(r0 T) * P0"] = cf_ok

    # (3) P(loss) = 0 under constant rate for every horizon
    ploss_ok = all(
        const_res["horizons"][str(t_h)][inst]["p_loss"] == 0.0
        for t_h in HORIZONS
        for inst in ("bond_a", "bond_b", "portfolio")
    )
    checks["(3) P(loss) identically zero under constant rate"] = ploss_ok

    # (4) DBM re-run matches Step 9 Portfolio VaR95 at T=1Y (94.341)
    vaR95_1y_port = dbm_res["horizons"]["1.0"]["portfolio"]["VaR95"]
    step9_match_ok = abs(vaR95_1y_port - 94.341) < 5e-3
    checks["(4) DBM reproduces Step 9 Portfolio VaR95(1Y) = 94.341"] = step9_match_ok

    # (5) no-arbitrage: simulated bond return under constant rate = exp(r0*T)-1
    noarb_ok = True
    for t_h in HORIZONS:
        expected_return = np.exp(R0 * t_h) - 1.0
        h = const_res["horizons"][str(t_h)]
        # Return on Bond A from t=0 to T:  V_A(T) / P0_A - 1
        realised = h["bond_a"]["mean"] / P0_A - 1.0
        if abs(realised - expected_return) > 1e-9:
            noarb_ok = False
    checks["(5) no-arbitrage: bond return = exp(r0 T) - 1"] = noarb_ok

    return checks


# ----------------------------------------------------------------------
# 7. Presentation helpers
# ----------------------------------------------------------------------
def _format_metric_row(t_h, inst_label, m):
    return (
        f"{t_h:>5.1f}Y  {inst_label:<10s}  "
        f"{m['mean']:>9.4f}  {m['std']:>7.4f}  "
        f"{m['VaR95']:>9.4f}  {m['VaR99']:>9.4f}  "
        f"{m['p_loss']:>7.4f}"
    )


def print_result_block(res):
    print(f"\n=== {res['label']} ===")
    print(f"  P0(A)      = {res['P0_A']:.4f}")
    print(f"  P0(B)      = {res['P0_B']:.4f}")
    print(f"  V0_model   = {res['V0_model']:.4f}")
    print(f"  V0_ref     = {res['V0_ref']:.4f}")
    print()
    print(f"  {'Horizon':<6}  {'Inst':<10}  {'Mean':>9}  {'Std':>7}  "
          f"{'VaR95':>9}  {'VaR99':>9}  {'P(loss)':>7}")
    for t_h in HORIZONS:
        h = res["horizons"][str(t_h)]
        print(_format_metric_row(t_h, "Bond A",    h["bond_a"]))
        print(_format_metric_row(t_h, "Bond B",    h["bond_b"]))
        print(_format_metric_row(t_h, "Portfolio", h["portfolio"]))


def print_comparison_block(const_res, dbm_res):
    print("\n=== Portfolio comparison: Constant vs DBM ===")
    print(f"  {'Horizon':<8}  {'V_T Const':>10}  {'V_T DBM':>10}  "
          f"{'sigma DBM':>10}  {'VaR95 C':>9}  {'VaR95 D':>9}  "
          f"{'DeltaVaR95':>10}  {'P(loss) D':>10}")
    for t_h in HORIZONS:
        c = const_res["horizons"][str(t_h)]["portfolio"]
        d = dbm_res["horizons"][str(t_h)]["portfolio"]
        print(f"  {t_h:>7.1f}Y  {c['mean']:>10.4f}  {d['mean']:>10.4f}  "
              f"{d['std']:>10.4f}  {c['VaR95']:>9.4f}  {d['VaR95']:>9.4f}  "
              f"{c['VaR95'] - d['VaR95']:>10.4f}  {d['p_loss']:>10.4f}")


# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------
def main():
    print("Phase B, Step 11: constant-rate benchmark comparison")
    print("-" * 60)
    print(f"seed       = {SEED}")
    print(f"N_paths    = {N_PATHS}")
    print(f"dt         = {DT:.6f}")
    print(f"T_max      = {T_MAX}")
    print(f"r0         = {R0}")
    print(f"DBM mu     = {DBM_MU}")
    print(f"DBM sigma  = {DBM_SIGMA}")

    # Constant-rate benchmark
    const_paths = simulate_constant(N_PATHS, N_STEPS, R0)
    const_res = run_pipeline(const_paths, "Constant-rate (r = r0)")

    # DBM re-run of Step 8 for side-by-side comparison
    dbm_paths = simulate_dbm(
        N_PATHS, N_STEPS, R0, DBM_MU, DBM_SIGMA, DT, SEED
    )
    dbm_res = run_pipeline(dbm_paths, "DBM (mu=0, sigma=0.01)")

    print_result_block(const_res)
    print_result_block(dbm_res)
    print_comparison_block(const_res, dbm_res)

    # Sanity checks
    checks = sanity_checks(const_res, dbm_res)
    print("\n=== Sanity checks ===")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")

    # Write combined results JSON
    payload = {
        "config": {
            "seed": SEED,
            "n_paths": N_PATHS,
            "dt": DT,
            "t_max": T_MAX,
            "n_steps": N_STEPS,
            "r0": R0,
            "dbm_mu": DBM_MU,
            "dbm_sigma": DBM_SIGMA,
            "face": FACE,
            "coupon_a": COUPON_A,
            "coupon_b": COUPON_B,
            "horizons": HORIZONS,
            "v0_ref": V0_REF,
        },
        "constant": const_res,
        "dbm": dbm_res,
        "sanity_checks": checks,
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "step11_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"\nWrote results to {out_path}")

    return payload


if __name__ == "__main__":
    main()
