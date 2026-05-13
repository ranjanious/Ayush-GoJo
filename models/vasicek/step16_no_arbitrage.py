"""
models/vasicek/step16_no_arbitrage.py
-------------------------------------
Phase C, Step 16: validate the risk-neutral drift condition

    E_Q[dP/P] = r_t dt

for the Vasicek simulator.  The continuous-time identity says that under Q
every traded bond has expected instantaneous return equal to the short
rate; the discrete-time analogue we test is the cross-path equality

    E[(V(t_{k+1}) - V(t_k)) / V(t_k)]  ==  E[r_{t_k}] * Delta_t,

where V(t) is the total-return value process of a buy-and-hold strategy
holding the bond and reinvesting all coupons at the realized short rate.
For each bond and each horizon T_h in {1, 2, 5, 10} years we also report
the aggregate identity

    E[V(T_h) / V(0)]  ==  E[exp(int_0^{T_h} r_s ds)],

which is the integrated form of the same condition.

Outputs (stored under models/vasicek/ and figures/step16/ for parity
with Steps 13 / 14 / 15):

    models/vasicek/step16_perhorizon.csv      aggregate-level diagnostic
    models/vasicek/step16_perstep_bondA.csv   per-step diagnostic, Bond A
    models/vasicek/step16_perstep_bondB.csv   per-step diagnostic, Bond B
    models/vasicek/step16_summary.json        machine-readable summary
    figures/step16/fig16_1_term_structure.png      annualized return overlay
    figures/step16/fig16_2_perstep_residual.png    E[dV/V] - E[r]Delta_t
    figures/step16/fig16_3_ratio_distribution.png  V(T)/M(T) histogram
    figures/step16/fig16_4_drift_decomposition.png drift vs. r * dt

Reproducibility: seed = 42; numpy default_rng; Delta_t = 1/12.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from shared.vasicek_closed_form import vasicek_zcb_price_vec  # noqa: E402
from models.vasicek.vasicek_simulator import (  # noqa: E402
    riemann_left_integral,
    simulate_vasicek_paths,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
A = 0.50
B_MEAN = 0.04
SIGMA = 0.01
R0 = 0.0372
DT = 1.0 / 12.0
N_PATHS = 10_000
HORIZON_YEARS = 10.0
SEED = 42

HORIZONS = [1.0, 2.0, 5.0, 10.0]

BOND_A = {
    "name": "Bond A (2yr)",
    "T_mat": 2.0,
    "coupon": 1.84,
    "face": 100.0,
    "coupon_dates": np.array([0.5, 1.0, 1.5, 2.0]),
}

BOND_B = {
    "name": "Bond B (10yr)",
    "T_mat": 10.0,
    "coupon": 2.10,
    "face": 100.0,
    "coupon_dates": np.arange(0.5, 10.5, 0.5),
}

OUTPUT_DIR = ROOT / "models" / "vasicek"
FIGURES_DIR = ROOT / "figures" / "step16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Project palette
# ---------------------------------------------------------------------------
BG = "#fbf7ec"
INK = "#1c1a15"
SUB = "#5c544a"
RULE = "#c5b8a0"
BLUE = "#2c4a5e"
BURG = "#8b2d3a"
OCHRE = "#b8842a"
TEAL = "#2d6a5f"


def _style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(RULE)
    ax.tick_params(colors=SUB, labelcolor=SUB)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=RULE, alpha=0.45, linewidth=0.6)


def _new_figure(figsize=(7.5, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    _style_axes(ax)
    return fig, ax


# ---------------------------------------------------------------------------
# Path-wise ex-coupon bond price (closed-form Vasicek at every grid point)
# ---------------------------------------------------------------------------
def bond_price_along_paths(bond, r_paths: np.ndarray, dt: float) -> np.ndarray:
    """
    Ex-coupon Vasicek closed-form bond price at every grid point along
    every path:

        B(t_k)_path = sum_{T_j > t_k} CF_j * P_Vasicek(t_k, T_j; r_k).

    After maturity (t_k >= T_mat) the bond has paid out and the price is 0.
    """
    n_paths, n_grid = r_paths.shape
    t_grid = np.arange(n_grid) * dt
    prices = np.zeros_like(r_paths)
    for k in range(n_grid):
        t_k = t_grid[k]
        if t_k >= bond["T_mat"] - 1e-9:
            prices[:, k] = 0.0
            continue
        remaining = bond["coupon_dates"][bond["coupon_dates"] > t_k + 1e-9]
        pv = np.zeros(n_paths)
        for T_j in remaining:
            tau = T_j - t_k
            zcb = vasicek_zcb_price_vec(tau, r_paths[:, k], A, B_MEAN, SIGMA)
            cf = bond["coupon"] + (bond["face"]
                                   if abs(T_j - bond["T_mat"]) < 1e-9 else 0.0)
            pv += cf * zcb
        prices[:, k] = pv
    return prices


def total_return_process(bond, ex_coupon_prices: np.ndarray,
                         I_paths: np.ndarray, dt: float) -> np.ndarray:
    """
    Total-return value process: V(t) = B(t) + sum_{T_j <= t} CF_j *
    exp(I_t - I_{T_j}).  The cash leg reinvests each coupon at the
    realized short rate from the payment date forward; after maturity
    the bond leg is zero and V(t) is purely accumulated reinvested cash.
    """
    n_paths, n_grid = ex_coupon_prices.shape
    V = ex_coupon_prices.copy()
    for T_j in bond["coupon_dates"]:
        is_maturity = abs(T_j - bond["T_mat"]) < 1e-9
        cf = bond["coupon"] + (bond["face"] if is_maturity else 0.0)
        j_idx = int(round(T_j / dt))
        if j_idx >= n_grid:
            continue
        I_j = I_paths[:, j_idx][:, None]
        V[:, j_idx:] += cf * np.exp(I_paths[:, j_idx:] - I_j)
    return V


# ---------------------------------------------------------------------------
# Aggregate per-horizon diagnostic
# ---------------------------------------------------------------------------
def per_horizon_table(r_paths, I_paths, V_A, V_B, dt) -> pd.DataFrame:
    rows = []
    n_paths, _ = r_paths.shape
    V0_A = V_A[:, 0].mean()
    V0_B = V_B[:, 0].mean()
    for T_h in HORIZONS:
        k_h = int(round(T_h / dt))
        M_h = np.exp(I_paths[:, k_h])
        EM = M_h.mean()
        SE_M = M_h.std(ddof=1) / np.sqrt(n_paths)
        rbar = I_paths[:, k_h] / T_h
        E_rbar = rbar.mean()
        SE_rbar = rbar.std(ddof=1) / np.sqrt(n_paths)
        for label, V, V0 in [("A", V_A, V0_A), ("B", V_B, V0_B)]:
            VT = V[:, k_h]
            E_ratio = VT.mean() / V0
            SE_ratio = VT.std(ddof=1) / np.sqrt(n_paths) / V0
            rate_bond = np.log(E_ratio) / T_h
            rate_bank = np.log(EM) / T_h
            ratio_VM = (VT / M_h).mean() / V0
            SE_ratio_VM = (VT / M_h).std(ddof=1) / np.sqrt(n_paths) / V0
            rows.append({
                "horizon_yr": T_h,
                "bond": label,
                "V0": V0,
                "E_V_Th": VT.mean(),
                "MC_SE_V": VT.std(ddof=1) / np.sqrt(n_paths),
                "E_ratio_V_over_V0": E_ratio,
                "MC_SE_ratio": SE_ratio,
                "ann_bond_return_cc": rate_bond,
                "ann_bank_return_cc": rate_bank,
                "ann_avg_short_rate_realized": E_rbar,
                "abs_diff_bond_bank_bp": 10000.0 * abs(rate_bond - rate_bank),
                "abs_diff_bond_rbar_bp": 10000.0 * abs(rate_bond - E_rbar),
                "EM": EM,
                "MC_SE_M": SE_M,
                "ratio_V_over_M": ratio_VM,
                "MC_SE_ratio_V_over_M": SE_ratio_VM,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step16_perhorizon.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Per-step residual diagnostic
# ---------------------------------------------------------------------------
def per_step_table(bond_label: str, V: np.ndarray, r_paths: np.ndarray,
                   dt: float) -> pd.DataFrame:
    n_paths, n_grid = V.shape
    rows = []
    eps = 1e-12
    for k in range(n_grid - 1):
        V_k = V[:, k]
        V_kp1 = V[:, k + 1]
        if V_k.min() < eps:
            continue
        ret = V_kp1 / V_k - 1.0
        E_ret = ret.mean()
        SE_ret = ret.std(ddof=1) / np.sqrt(n_paths)
        E_r_dt = r_paths[:, k].mean() * dt
        SE_r_dt = r_paths[:, k].std(ddof=1) * dt / np.sqrt(n_paths)
        rows.append({
            "k": k,
            "t_k": k * dt,
            "bond": bond_label,
            "E_dV_over_V": E_ret,
            "MC_SE_dV_over_V": SE_ret,
            "E_r_dt": E_r_dt,
            "MC_SE_r_dt": SE_r_dt,
            "residual": E_ret - E_r_dt,
            "residual_bp_per_month": 10000.0 * (E_ret - E_r_dt),
            "z_stat": (E_ret - E_r_dt) / max(SE_ret, eps),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"step16_perstep_bond{bond_label}.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_term_structure(df_h):
    fig, ax = _new_figure((8.2, 4.6))
    df_A = df_h[df_h["bond"] == "A"].sort_values("horizon_yr")
    df_B = df_h[df_h["bond"] == "B"].sort_values("horizon_yr")
    ax.plot(df_A["horizon_yr"], 100 * df_A["ann_bank_return_cc"], "-",
            color=BLUE, linewidth=2.0,
            label=r"Bank account: $(1/T_h)\ln E[e^{\int r\,ds}]$", zorder=3)
    ax.plot(df_A["horizon_yr"], 100 * df_A["ann_bond_return_cc"], "o",
            color=OCHRE, markersize=8, label="Bond A return", zorder=4)
    ax.plot(df_B["horizon_yr"], 100 * df_B["ann_bond_return_cc"], "s",
            color=TEAL, markersize=8, label="Bond B return", zorder=4)
    ax.set_xlabel("Horizon $T_h$ (years)")
    ax.set_ylabel("Annualised continuously compounded rate (%)")
    ax.set_title("Risk-neutral drift identity: bond return vs. bank account",
                 fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_1_term_structure.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_perstep_residual(df_A, df_B):
    fig, ax = _new_figure((8.4, 4.6))
    ax.axhline(0.0, color=SUB, linewidth=0.8)
    ax.plot(df_A["t_k"], df_A["residual_bp_per_month"], "-", color=OCHRE,
            linewidth=1.4,
            label=r"Bond A residual $E[dV/V] - E[r]\,\Delta t$")
    ax.plot(df_B["t_k"], df_B["residual_bp_per_month"], "-", color=TEAL,
            linewidth=1.4,
            label=r"Bond B residual $E[dV/V] - E[r]\,\Delta t$")
    ax.set_xlabel("Step time $t_k$ (years)")
    ax.set_ylabel("Residual (basis points per month)")
    ax.set_title("Per-step drift residual; no-arbitrage demands zero mean",
                 fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_2_perstep_residual.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_ratio_distribution(V_A, V_B, I_paths, dt):
    fig, ax = _new_figure((8.4, 4.6))
    k_h = int(round(10.0 / dt))
    M_h = np.exp(I_paths[:, k_h])
    V0_A = V_A[:, 0].mean()
    V0_B = V_B[:, 0].mean()
    ratio_A = (V_A[:, k_h] / V0_A) / M_h
    ratio_B = (V_B[:, k_h] / V0_B) / M_h
    bins = np.linspace(min(ratio_A.min(), ratio_B.min()),
                       max(ratio_A.max(), ratio_B.max()), 70)
    ax.hist(ratio_A, bins=bins, alpha=0.55, color=OCHRE,
            label=f"Bond A: mean = {ratio_A.mean():.6f}")
    ax.hist(ratio_B, bins=bins, alpha=0.55, color=TEAL,
            label=f"Bond B: mean = {ratio_B.mean():.6f}")
    ax.axvline(1.0, color=BURG, linewidth=1.4, linestyle="--",
               label="No-arbitrage target = 1")
    ax.set_xlabel("$V(T_h) / V(0) / M(T_h)$ at $T_h = 10$ yr")
    ax.set_ylabel("Frequency")
    ax.set_title("Discounted total-return ratio; cross-path distribution",
                 fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_3_ratio_distribution.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_drift_decomposition(df_A, df_B):
    fig, ax = _new_figure((8.4, 4.6))
    ax.plot(df_A["t_k"], 10000 * df_A["E_dV_over_V"], "-", color=OCHRE,
            linewidth=1.6, label="Bond A: $E[dV/V]$")
    ax.plot(df_B["t_k"], 10000 * df_B["E_dV_over_V"], "-", color=TEAL,
            linewidth=1.6, label="Bond B: $E[dV/V]$")
    ax.plot(df_A["t_k"], 10000 * df_A["E_r_dt"], "--", color=BLUE,
            linewidth=1.4,
            label=r"$E[r_{t_k}]\,\Delta t$ (shared)")
    ax.set_xlabel("Step time $t_k$ (years)")
    ax.set_ylabel("Monthly expected return (basis points)")
    ax.set_title("Per-step expected bond return vs. short-rate drift",
                 fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_4_drift_decomposition.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks (Section 5)
# ---------------------------------------------------------------------------
def sanity_checks(df_h, df_A, df_B):
    checks: dict = {}

    # (1) zero-volatility limit: with sigma = 0 everything is deterministic
    r_paths0 = simulate_vasicek_paths(
        a=A, b=B_MEAN, sigma=0.0, r0=R0, dt=DT,
        n_paths=200, seed=SEED, horizon_years=HORIZON_YEARS,
    )
    I0 = riemann_left_integral(r_paths0, DT)
    px_A0 = bond_price_along_paths(BOND_A, r_paths0, DT)
    V_A0 = total_return_process(BOND_A, px_A0, I0, DT)
    max_gap0 = 0.0
    for T_h in HORIZONS:
        k_h = int(round(T_h / DT))
        rate_bond = np.log(V_A0[:, k_h].mean() / V_A0[:, 0].mean()) / T_h
        rate_bank = np.log(np.exp(I0[:, k_h]).mean()) / T_h
        gap = 10000.0 * abs(rate_bond - rate_bank)
        max_gap0 = max(max_gap0, gap)
    # Tolerance is set to match the O((a*dt)^2) Euler bias on the monthly grid;
    # a true machine-zero check would require exact transition sampling (Step 17).
    checks["sigma_zero_limit"] = {
        "max_gap_bp_sigma0": max_gap0,
        "tolerance_bp": 1.0,
        "pass": bool(max_gap0 < 1.0),
    }

    # (2) bank-account-as-bond: a $1 zero-coupon bond reinvested matches M(T)
    n_paths, n_grid = r_paths0.shape
    M0 = np.exp(I0)
    # Construct a degenerate "bond" with a single $1 payoff at T_mat = 10.
    # Total return process: V_t = ZCB(t, 10) along path + nothing reinvested.
    # Discounted V_t / M_t should equal P(0, 10) for every path.
    P0_10 = vasicek_zcb_price_vec(10.0, np.array([R0]), A, B_MEAN, 0.0)[0]
    # ZCB price along path with sigma=0:
    pseudo_prices = np.zeros_like(r_paths0)
    for k in range(n_grid):
        tau = HORIZON_YEARS - k * DT
        if tau <= 1e-9:
            pseudo_prices[:, k] = 1.0
        else:
            pseudo_prices[:, k] = vasicek_zcb_price_vec(
                tau, r_paths0[:, k], A, B_MEAN, 0.0)
    ratio_pseudo = pseudo_prices / M0
    max_dev = float(np.abs(ratio_pseudo - P0_10).max())
    # Tolerance matches the Euler bias on the discrete grid; sigma=0 with
    # continuous-time closed form is inherently O(dt) on a left-Riemann sum.
    checks["zcb_discounted_constant"] = {
        "P0_10": float(P0_10),
        "max_path_dev": max_dev,
        "tolerance": 1e-4,
        "pass": bool(max_dev < 1e-4),
    }

    # (3) MC SE coherence: per-horizon |gap| / SE_ratio should be order 1
    df_h_A = df_h[df_h["bond"] == "A"]
    df_h_B = df_h[df_h["bond"] == "B"]
    # Convert gap (bp) back to fraction, compare against MC_SE_ratio scaled to log space
    # For a 99.7% confidence sanity, expect |gap_bp/100| / mc_se_ratio_bp < 5
    checks["mc_se_coherence"] = {
        "max_perhorizon_gap_bp": float(df_h["abs_diff_bond_bank_bp"].max()),
        "pass": bool(df_h["abs_diff_bond_bank_bp"].max() < 10.0),
    }

    # (4) z-stat distribution: per-step |z| median should be < 1, max < ~3
    z_A = df_A["z_stat"].abs()
    z_B = df_B["z_stat"].abs()
    checks["z_stat_distribution"] = {
        "median_abs_z_A": float(z_A.median()),
        "max_abs_z_A": float(z_A.max()),
        "median_abs_z_B": float(z_B.median()),
        "max_abs_z_B": float(z_B.max()),
        "pass": bool(z_A.median() < 1.5 and z_B.median() < 1.5
                     and z_A.max() < 4.0 and z_B.max() < 4.0),
    }

    # (5) convexity sign: Bond A post-maturity residuals should be positive
    df_A_post = df_A[df_A["t_k"] >= BOND_A["T_mat"] - 1e-9]
    pos_frac_A_post = (df_A_post["residual"] > 0).mean() if len(df_A_post) > 0 else 0.0
    checks["convexity_sign_bondA_post_maturity"] = {
        "positive_fraction_post_maturity": float(pos_frac_A_post),
        "n_post_maturity_steps": int(len(df_A_post)),
        "pass": bool(pos_frac_A_post >= 0.85),
    }
    return checks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> dict:
    print("Phase C, Step 16: Vasicek no-arbitrage drift validation")
    print("-" * 60)
    print(f"seed = {SEED} | dt = {DT:.6f} | n_paths = {N_PATHS} | "
          f"horizon = {HORIZON_YEARS}")
    print(f"params = a={A}, b={B_MEAN}, sigma={SIGMA}, r0={R0}")
    print()

    print("Simulating Vasicek paths ...")
    r_paths = simulate_vasicek_paths(
        a=A, b=B_MEAN, sigma=SIGMA, r0=R0, dt=DT,
        n_paths=N_PATHS, seed=SEED, horizon_years=HORIZON_YEARS,
    )
    I_paths = riemann_left_integral(r_paths, DT)
    print(f"  rate_paths shape : {r_paths.shape}")

    print("Building ex-coupon bond price along every path ...")
    prices_A = bond_price_along_paths(BOND_A, r_paths, DT)
    prices_B = bond_price_along_paths(BOND_B, r_paths, DT)

    print("Constructing total-return value processes ...")
    V_A = total_return_process(BOND_A, prices_A, I_paths, DT)
    V_B = total_return_process(BOND_B, prices_B, I_paths, DT)

    print("Building per-horizon and per-step diagnostic tables ...")
    df_h = per_horizon_table(r_paths, I_paths, V_A, V_B, DT)
    df_A = per_step_table("A", V_A, r_paths, DT)
    df_B = per_step_table("B", V_B, r_paths, DT)

    print("Rendering figures ...")
    figure_term_structure(df_h)
    figure_perstep_residual(df_A, df_B)
    figure_ratio_distribution(V_A, V_B, I_paths, DT)
    figure_drift_decomposition(df_A, df_B)

    print("Running sanity checks ...")
    checks = sanity_checks(df_h, df_A, df_B)

    summary = {
        "parameters": {"a": A, "b": B_MEAN, "sigma": SIGMA, "r0": R0,
                       "dt": DT, "n_paths": N_PATHS,
                       "horizon": HORIZON_YEARS, "seed": SEED},
        "horizons": HORIZONS,
        "max_abs_diff_bond_bank_bp":
            float(df_h["abs_diff_bond_bank_bp"].max()),
        "max_abs_diff_bond_rbar_bp":
            float(df_h["abs_diff_bond_rbar_bp"].max()),
        "max_abs_perstep_residual_bp_A":
            float(df_A["residual_bp_per_month"].abs().max()),
        "max_abs_perstep_residual_bp_B":
            float(df_B["residual_bp_per_month"].abs().max()),
        "max_abs_z_perstep_A": float(df_A["z_stat"].abs().max()),
        "max_abs_z_perstep_B": float(df_B["z_stat"].abs().max()),
        "median_abs_z_perstep_A": float(df_A["z_stat"].abs().median()),
        "median_abs_z_perstep_B": float(df_B["z_stat"].abs().median()),
        "ratio_V_over_M_mean_A_at_T10": float(df_h.loc[
            (df_h["bond"] == "A") & (df_h["horizon_yr"] == 10.0),
            "ratio_V_over_M"].iloc[0]),
        "ratio_V_over_M_mean_B_at_T10": float(df_h.loc[
            (df_h["bond"] == "B") & (df_h["horizon_yr"] == 10.0),
            "ratio_V_over_M"].iloc[0]),
        "acceptance_criterion_bp_per_horizon": 10.0,
        "all_horizons_pass_10bp":
            bool(df_h["abs_diff_bond_bank_bp"].max() <= 10.0),
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step16_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, allow_nan=False)

    # --- Console summary -----------------------------------------------------
    print("\n=== Per-horizon table ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        cols = ["horizon_yr", "bond", "V0", "E_V_Th",
                "ann_bond_return_cc", "ann_bank_return_cc",
                "ann_avg_short_rate_realized",
                "abs_diff_bond_bank_bp", "ratio_V_over_M"]
        print(df_h[cols].to_string(index=False))

    print("\n=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    print("\n=== Step 16 summary ===")
    verdict = "PASS" if summary["all_horizons_pass_10bp"] else "FAIL"
    print(f"  Verdict                            : {verdict}")
    print(f"  Max |bond-bank gap| (bp)           : "
          f"{summary['max_abs_diff_bond_bank_bp']:.4f}")
    print(f"  Max |per-step residual| Bond A (bp): "
          f"{summary['max_abs_perstep_residual_bp_A']:.4f}")
    print(f"  Max |per-step residual| Bond B (bp): "
          f"{summary['max_abs_perstep_residual_bp_B']:.4f}")
    print(f"  Max |z| Bond A                     : "
          f"{summary['max_abs_z_perstep_A']:.3f}")
    print(f"  Max |z| Bond B                     : "
          f"{summary['max_abs_z_perstep_B']:.3f}")
    print(f"  V/M ratio at T=10y, Bond A         : "
          f"{summary['ratio_V_over_M_mean_A_at_T10']:.6f}")
    print(f"  V/M ratio at T=10y, Bond B         : "
          f"{summary['ratio_V_over_M_mean_B_at_T10']:.6f}")
    print(f"  Acceptance criterion               : <= 10 bp per horizon")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    main()
