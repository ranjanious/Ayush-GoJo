"""
models/vasicek/step14_vasicek_pricing.py
----------------------------------------
Phase C, Step 14:
    Run Vasicek bond pricing and risk metrics.
    Build the first side-by-side comparison: DBM vs. Vasicek vs. constant
    rate at horizons T = 1, 2, 5, 10 years.

Reuses Phase A infrastructure (shared/bond_pricing.py, shared/risk_engine.py)
and Phase B / Phase C simulators (Step 8 DBM, Step 13 Vasicek).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.bond_pricing import (  # noqa: E402
    constant_rate_bond_price,
    portfolio_value,
    price_bond_at_horizon,
    price_bond,
)
from shared.risk_engine import full_metric_set  # noqa: E402
from models.vasicek.vasicek_simulator import simulate_vasicek  # noqa: E402


# ----------------------------------------------------------------------
# 1. Configuration: bonds, grid, models
# ----------------------------------------------------------------------
DT = 1.0 / 12.0
HORIZONS_YEARS = [1, 2, 5, 10]
N_STEPS = int(round(10 / DT))      # 120 monthly steps to T = 10
N_PATHS = 10_000
SEED = 42                          # mulberry-style; matches Step 8/9/11

R0 = 0.0372                         # 3-month CMT, March 17 2026

BOND_A = {                          # 2-year, 3.68% par coupon
    "label": "Bond A (2y, 3.68%)",
    "coupon": 1.84,
    "face": 100.0,
    "payment_idx": np.array([6, 12, 18, 24]),
    "payment_times": np.array([0.5, 1.0, 1.5, 2.0]),
    "maturity": 2.0,
}
BOND_B = {                          # 10-year, 4.20% par coupon
    "label": "Bond B (10y, 4.20%)",
    "coupon": 2.10,
    "face": 100.0,
    "payment_idx": np.arange(6, 121, 6),
    "payment_times": np.arange(0.5, 10.0 + 1e-9, 0.5),
    "maturity": 10.0,
}

VASICEK_PARAMS = {"a": 0.5, "b": 0.04, "sigma": 0.01, "r0": R0}
DBM_PARAMS = {"mu": 0.0, "sigma": 0.01, "r0": R0}


# ----------------------------------------------------------------------
# 2. DBM simulator (PCG64 with explicit seed; matches Step 9/11 conventions)
# ----------------------------------------------------------------------
def simulate_dbm(
    mu: float,
    sigma: float,
    r0: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """dr_t = mu*dt + sigma*dW_t.  Vectorised Euler-Maruyama with default_rng."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for k in range(n_steps):
        paths[:, k + 1] = paths[:, k] + mu * dt + sigma * sqrt_dt * Z[:, k]
    return paths


# ----------------------------------------------------------------------
# 3. Generate short-rate paths under each model
# ----------------------------------------------------------------------
def main() -> Dict[str, Any]:
    print("Phase C, Step 14: Vasicek pricing + three-model comparison")
    print("-" * 60)
    print(f"seed = {SEED} | n_paths = {N_PATHS} | n_steps = {N_STEPS} "
          f"| dt = {DT:.6f}")
    print()
    print("Simulating short-rate paths ...")
    vasicek_rates = simulate_vasicek(
        **VASICEK_PARAMS, dt=DT, n_steps=N_STEPS, n_paths=N_PATHS, seed=SEED
    )
    dbm_rates = simulate_dbm(
        **DBM_PARAMS, dt=DT, n_steps=N_STEPS, n_paths=N_PATHS, seed=SEED
    )
    const_rates = np.full((N_PATHS, N_STEPS + 1), R0)
    print(f"  Vasicek paths shape : {vasicek_rates.shape}")
    print(f"  DBM     paths shape : {dbm_rates.shape}")
    print(f"  Const   paths shape : {const_rates.shape}")

    # ------------------------------------------------------------------
    # 4. Price each bond at each horizon, along every path, under each model
    # ------------------------------------------------------------------
    def price_at_all_horizons(rate_paths, bond):
        out = {}
        for h in HORIZONS_YEARS:
            h_idx = int(round(h / DT))
            prices = price_bond_at_horizon(
                rate_paths,
                coupon=bond["coupon"],
                face=bond["face"],
                payment_idx=bond["payment_idx"],
                dt=DT,
                horizon_idx=h_idx,
            )
            out[h] = prices
        return out

    print("\nPricing bonds along every path ...")
    prices = {
        "Vasicek": {
            "A": price_at_all_horizons(vasicek_rates, BOND_A),
            "B": price_at_all_horizons(vasicek_rates, BOND_B),
        },
        "DBM": {
            "A": price_at_all_horizons(dbm_rates, BOND_A),
            "B": price_at_all_horizons(dbm_rates, BOND_B),
        },
        "Constant": {
            "A": price_at_all_horizons(const_rates, BOND_A),
            "B": price_at_all_horizons(const_rates, BOND_B),
        },
    }

    # ------------------------------------------------------------------
    # 5. 50/50 portfolio at each horizon
    # ------------------------------------------------------------------
    def portfolio_at_horizons(model_prices):
        out = {}
        for h in HORIZONS_YEARS:
            out[h] = portfolio_value(
                model_prices["A"][h], model_prices["B"][h],
                w_a=0.5, w_b=0.5,
            )
        return out

    portfolios = {model: portfolio_at_horizons(prices[model]) for model in prices}

    # ------------------------------------------------------------------
    # 6. Full metric set per model, per bond, per horizon
    # ------------------------------------------------------------------
    print("Computing risk metrics ...")
    metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for model in ("Vasicek", "DBM", "Constant"):
        metrics[model] = {"A": {}, "B": {}, "Portfolio": {}}
        for h in HORIZONS_YEARS:
            metrics[model]["A"][h] = full_metric_set(prices[model]["A"][h])
            metrics[model]["B"][h] = full_metric_set(prices[model]["B"][h])
            metrics[model]["Portfolio"][h] = full_metric_set(portfolios[model][h])

    # ------------------------------------------------------------------
    # 7. Sanity checks
    # ------------------------------------------------------------------
    print("\nSanity check 1: time-zero par prices under continuous compounding")
    p0_close = {}
    for bond in (BOND_A, BOND_B):
        p0 = constant_rate_bond_price(
            bond["coupon"], bond["face"], bond["payment_times"], R0
        )
        p0_close[bond["label"]] = p0
        print(f"  {bond['label']:<22s}  P0 = {p0:7.4f}")

    print("\nSanity check 2: Monte Carlo Bond price at t = 0 vs. closed form")
    mc_p0: Dict[str, Dict[str, float]] = {}
    for model_name, rates in [("Vasicek", vasicek_rates), ("DBM", dbm_rates)]:
        cfs_a = np.full(len(BOND_A["payment_times"]), BOND_A["coupon"])
        cfs_a[-1] += BOND_A["face"]
        cfs_b = np.full(len(BOND_B["payment_times"]), BOND_B["coupon"])
        cfs_b[-1] += BOND_B["face"]
        mc_A = price_bond(rates, cfs_a, BOND_A["payment_times"], 0.0, DT)
        mc_B = price_bond(rates, cfs_b, BOND_B["payment_times"], 0.0, DT)
        mc_p0[model_name] = {"A": float(mc_A.mean()), "B": float(mc_B.mean())}
        print(f"  {model_name:<8s}  MC mean(P0_A) = {mc_A.mean():7.4f}   "
              f"MC mean(P0_B) = {mc_B.mean():7.4f}")

    sanity_pass = True
    for model in ("Vasicek", "DBM"):
        for asset in ("A", "B"):
            v = mc_p0[model][asset]
            if not (90.0 <= v <= 110.0):
                sanity_pass = False
                print(f"  [FAIL] MC P0 out of [90,110] band: "
                      f"{model} Bond {asset} = {v:.4f}")
    if sanity_pass:
        print("  [PASS] All MC P0 means inside the [90,110] acceptance band")

    # ------------------------------------------------------------------
    # 8. Console summary tables (Sections 6 and 7 of the completion log)
    # ------------------------------------------------------------------
    def fmt_row(m):
        return (f"{m['mean']:>7.2f}  {m['std']:>6.3f}  "
                f"{m['VaR95']:>7.2f} ({m['VaR95_SE']:>5.3f})  {m['ES95']:>7.2f}  "
                f"{m['VaR99']:>7.2f} ({m['VaR99_SE']:>5.3f})  {m['ES99']:>7.2f}  "
                f"{m['p_loss']*100:>5.2f}%")

    header = (f"  {'T':>3}  {'Mean':>7}  {'Std':>6}  "
              f"{'VaR95':>7} {'(SE)':>7}  {'ES95':>7}  "
              f"{'VaR99':>7} {'(SE)':>7}  {'ES99':>7}  P(loss)")

    for model in ("Vasicek", "DBM", "Constant"):
        for asset in ("A", "B", "Portfolio"):
            print(f"\n=== {model} | {asset} ===")
            print(header)
            for h in HORIZONS_YEARS:
                m = metrics[model][asset][h]
                print(f"  {h:>3d}  {fmt_row(m)}")

    # Synthesis (Section 7.3)
    print("\n=== Synthesis: Portfolio tail risk across models ===")
    print(f"  {'T':>3}  {'Const':>7}  {'Vas mean':>9}  {'DBM mean':>9}  "
          f"{'Vas VaR99':>10}  {'DBM VaR99':>10}  "
          f"{'Vas ES99':>9}  {'DBM ES99':>9}")
    for h in HORIZONS_YEARS:
        c = metrics["Constant"]["Portfolio"][h]
        v = metrics["Vasicek"]["Portfolio"][h]
        d = metrics["DBM"]["Portfolio"][h]
        print(f"  {h:>3d}  {c['mean']:>7.2f}  "
              f"{v['mean']:>9.2f}  {d['mean']:>9.2f}  "
              f"{v['VaR99']:>10.2f}  {d['VaR99']:>10.2f}  "
              f"{v['ES99']:>9.2f}  {d['ES99']:>9.2f}")

    # ------------------------------------------------------------------
    # 9. Persist outputs to disk (alongside Step 13 artefacts)
    # ------------------------------------------------------------------
    here = Path(__file__).resolve().parent
    fig_dir = ROOT / "figures" / "step14"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    metrics_json: Dict[str, Any] = {
        "config": {
            "seed": SEED, "n_paths": N_PATHS, "n_steps": N_STEPS,
            "dt": DT, "r0": R0,
            "horizons_years": HORIZONS_YEARS,
            "vasicek_params": VASICEK_PARAMS,
            "dbm_params": DBM_PARAMS,
        },
        "sanity_checks": {
            "closed_form_P0": p0_close,
            "mc_P0_mean": mc_p0,
            "p0_in_acceptance_band": bool(sanity_pass),
        },
        "metrics": {
            m: {
                asset: {str(h): {k: (None if isinstance(val, float)
                                     and (np.isnan(val) or np.isinf(val))
                                     else (float(val) if isinstance(val, (int, float, np.floating))
                                           else val))
                                 for k, val in d.items()}
                        for h, d in inner.items()}
                for asset, inner in m_dict.items()
            }
            for m, m_dict in metrics.items()
        },
    }
    metrics_path = here / "step14_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2, allow_nan=False)
    print(f"\nWrote metrics to {metrics_path}")

    # NPZ archive of distributions
    npz_path = here / "step14_terminal_distributions.npz"
    np.savez_compressed(
        npz_path,
        vasicek_rates_first200=vasicek_rates[:200],
        dbm_rates_first200=dbm_rates[:200],
        portfolio_vasicek_10y=portfolios["Vasicek"][10],
        portfolio_dbm_10y=portfolios["DBM"][10],
        portfolio_const_10y=portfolios["Constant"][10],
        bondA_vasicek_1y=prices["Vasicek"]["A"][1],
        bondB_vasicek_1y=prices["Vasicek"]["B"][1],
        bondA_dbm_1y=prices["DBM"]["A"][1],
        bondB_dbm_1y=prices["DBM"]["B"][1],
    )
    print(f"Wrote distributions to {npz_path}")

    # ------------------------------------------------------------------
    # 10. Diagnostic figures (Section 8)
    # ------------------------------------------------------------------
    t_grid = np.arange(N_STEPS + 1) * DT

    # Figure 8.1: 150 representative rate paths from each model
    fig, ax = plt.subplots(figsize=(11, 5))
    rng_plot = np.random.default_rng(0)
    idx = rng_plot.choice(N_PATHS, size=150, replace=False)
    for i in idx:
        ax.plot(t_grid, dbm_rates[i] * 100.0, color="#ED7D31",
                alpha=0.07, linewidth=0.5)
    for i in idx:
        ax.plot(t_grid, vasicek_rates[i] * 100.0, color="#4472C4",
                alpha=0.10, linewidth=0.5)
    ax.axhline(VASICEK_PARAMS["b"] * 100.0, color="black",
               linestyle="--", linewidth=1.0,
               label=f"Vasicek long-run mean b = {VASICEK_PARAMS['b']*100:.2f}%")
    ax.axhline(0.0, color="red", linestyle=":", linewidth=0.8,
               label="Zero rate")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short rate (%)")
    ax.set_title("Figure 8.1. Vasicek (blue) vs DBM (orange) rate paths "
                 "(150 of 10,000 each)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "fig1_rate_paths.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 8.2: portfolio terminal distributions at T = 10
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(portfolios["DBM"][10].min(), portfolios["Vasicek"][10].min()) - 1.0,
        max(portfolios["DBM"][10].max(), portfolios["Vasicek"][10].max()) + 1.0,
        80,
    )
    ax.hist(portfolios["DBM"][10], bins=bins, density=True, alpha=0.55,
            color="#ED7D31",
            label=f"DBM (sd = {metrics['DBM']['Portfolio'][10]['std']:.3f})")
    ax.hist(portfolios["Vasicek"][10], bins=bins, density=True, alpha=0.55,
            color="#4472C4",
            label=f"Vasicek (sd = {metrics['Vasicek']['Portfolio'][10]['std']:.3f})")
    ax.axvline(metrics["Constant"]["Portfolio"][10]["mean"],
               color="black", linestyle="--", linewidth=1.2,
               label=f"Constant rate = "
                     f"{metrics['Constant']['Portfolio'][10]['mean']:.2f}")
    ax.set_xlabel("Portfolio terminal value at T = 10 ($)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 8.2. Portfolio terminal distributions at T = 10")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "fig2_portfolio_T10.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 8.3: VaR95 bars + ES99 markers across horizons (Bond B + Portfolio)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.35
    x = np.arange(len(HORIZONS_YEARS))
    for ax, asset, title in zip(
        axes, ("B", "Portfolio"),
        ("Bond B (10y)", "50/50 Portfolio"),
    ):
        var_v = [metrics["Vasicek"][asset][h]["VaR95"] for h in HORIZONS_YEARS]
        var_d = [metrics["DBM"][asset][h]["VaR95"] for h in HORIZONS_YEARS]
        es_v = [metrics["Vasicek"][asset][h]["ES99"] for h in HORIZONS_YEARS]
        es_d = [metrics["DBM"][asset][h]["ES99"] for h in HORIZONS_YEARS]
        ax.bar(x - width/2, var_v, width, color="#4472C4",
               label="Vasicek VaR95", alpha=0.85)
        ax.bar(x + width/2, var_d, width, color="#ED7D31",
               label="DBM VaR95", alpha=0.85)
        ax.scatter(x - width/2, es_v, color="#1F3864", marker="v", s=70,
                   label="Vasicek ES99", zorder=5)
        ax.scatter(x + width/2, es_d, color="#843C0C", marker="v", s=70,
                   label="DBM ES99", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T={h}" for h in HORIZONS_YEARS])
        ax.set_ylabel("Value ($)")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 8.3. VaR95 (bars) and ES99 (markers) across horizons",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_var_es_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 8.4: Vasicek Bond A vs Bond B at T = 1
    fig, ax = plt.subplots(figsize=(10, 5))
    pa = prices["Vasicek"]["A"][1]
    pb = prices["Vasicek"]["B"][1]
    bins = np.linspace(min(pa.min(), pb.min()) - 1.0,
                       max(pa.max(), pb.max()) + 1.0, 80)
    ax.hist(pa, bins=bins, density=True, alpha=0.55, color="#4472C4",
            label=f"Bond A  (sd = {metrics['Vasicek']['A'][1]['std']:.3f})")
    ax.hist(pb, bins=bins, density=True, alpha=0.55, color="#ED7D31",
            label=f"Bond B  (sd = {metrics['Vasicek']['B'][1]['std']:.3f})")
    ax.axvline(100.0, color="black", linestyle="--", linewidth=1.0,
               label="Par ($100)")
    ratio = (metrics["Vasicek"]["B"][1]["std"]
             / metrics["Vasicek"]["A"][1]["std"])
    ax.set_xlabel("Bond price at T = 1 ($)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Figure 8.4. Vasicek Bond A vs Bond B at T = 1 "
        f"(sd ratio = {ratio:.2f})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "fig4_bondA_vs_bondB.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to {fig_dir}")
    return metrics_json


if __name__ == "__main__":
    main()
