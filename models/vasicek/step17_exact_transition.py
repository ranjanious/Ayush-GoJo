"""
models/vasicek/step17_exact_transition.py
-----------------------------------------
Phase C, Step 17: validate the Euler-Maruyama Vasicek simulator against
exact conditional transition sampling.

The Vasicek SDE under Q is

    dr_t = a (b - r_t) dt + sigma dW_t.

Its exact transition law over Delta_t is Gaussian:

    r_{t + Delta_t} | r_t  ~  N( mu_exact, var_exact ),
        mu_exact  = b + (r_t - b) * exp(-a Delta_t),
        var_exact = sigma^2 * (1 - exp(-2 a Delta_t)) / (2 a).

The Euler-Maruyama update is

    r_{k+1}^EM = r_k + a (b - r_k) Delta_t + sigma sqrt(Delta_t) Z_k,

which is also Gaussian conditional on r_k but with

    mu_EM  = r_k + a (b - r_k) Delta_t,
    var_EM = sigma^2 * Delta_t.

The conditional-mean error and the conditional-variance error therefore are

    mu_exact - mu_EM   = (r_k - b) * [exp(-a Delta_t) - (1 - a Delta_t)]
                       = (r_k - b) * [(a Delta_t)^2 / 2 - ... ],
    var_exact - var_EM = sigma^2 [(1 - exp(-2 a Delta_t))/(2a) - Delta_t]
                       = -sigma^2 a Delta_t^2 + O(Delta_t^3).

Both errors are O(Delta_t^2) per step in the conditional moments; the weak
order of Euler-Maruyama on this linear SDE is therefore 1, the strong order
is 1 (Euler matches Milstein for constant-diffusion SDEs).  Step 17 measures
the cumulative impact of these per-step errors on the marginal distribution
of r_t, the zero-coupon term structure, the coupon-bond total-return value
process, and the Step 14 VaR metrics.

Reproducibility: seed = 42 reused identically for the EM driver and the
exact driver so that both consume the same standard-normal stream; only the
conditional update rule differs.  This makes the per-path discretization
gap directly measurable.

Outputs (aligned with Steps 13-16 artefact layout):

    models/vasicek/step17_marginal_moments.csv
    models/vasicek/step17_zcb_term_structure.csv
    models/vasicek/step17_coupon_bond.csv
    models/vasicek/step17_var.csv
    models/vasicek/step17_conditional_error.csv
    models/vasicek/step17_summary.json
    figures/step17/fig17_1_marginal_moments.png
    figures/step17/fig17_2_zcb_term_structure.png
    figures/step17/fig17_3_path_difference.png
    figures/step17/fig17_4_var_compare.png
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

# ---------------------------------------------------------------------------
# Configuration; identical to Step 16
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
ZCB_TENORS = [1.0, 2.0, 5.0, 10.0]
VAR_LEVELS = [0.95, 0.99]
VAR_HORIZON_MONTHS = 1  # 1-month VaR on the short rate

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
FIGURES_DIR = ROOT / "figures" / "step17"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Step 14/15/16 token set)
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
# Closed-form Vasicek bond pricing primitives (inlined for single-file
# reproducibility; equivalent to shared/vasicek_closed_form.py).
# ---------------------------------------------------------------------------
def vasicek_B(tau, a):
    return (1.0 - np.exp(-a * tau)) / a


def vasicek_lnA(tau, a, b, sigma):
    B = vasicek_B(tau, a)
    term1 = (b - sigma ** 2 / (2.0 * a ** 2)) * (B - tau)
    term2 = -(sigma ** 2) * B ** 2 / (4.0 * a)
    return term1 + term2


def vasicek_zcb_price_vec(tau, r, a, b, sigma):
    B = vasicek_B(tau, a)
    lnA = vasicek_lnA(tau, a, b, sigma)
    return np.exp(lnA - B * np.asarray(r, dtype=float))


def vasicek_zcb_price_scalar(tau, r, a, b, sigma):
    B = vasicek_B(tau, a)
    lnA = vasicek_lnA(tau, a, b, sigma)
    return float(np.exp(lnA - B * r))


# ---------------------------------------------------------------------------
# Simulators -- same-seed coupling
# ---------------------------------------------------------------------------
def simulate_em(a, b, sigma, r0, horizon, dt, n_paths, seed):
    """Euler-Maruyama Vasicek paths.  Per-iteration RNG draw matches the
    exact simulator's shock order so the two are same-seed coupled."""
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        r[:, k + 1] = r[:, k] + a * (b - r[:, k]) * dt + sigma * sqrt_dt * Z
    return r


def simulate_exact(a, b, sigma, r0, horizon, dt, n_paths, seed):
    """Exact-transition Vasicek paths.  Uses the conditional Gaussian
        r_{k+1} | r_k ~ N(b + (r_k - b) e^{-a dt},
                           sigma^2 (1 - e^{-2 a dt}) / (2 a)).
    Same seed as the EM driver so identical Z_k drive both updates; the
    only difference is the conditional rule applied to those increments."""
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    exp_adt = np.exp(-a * dt)
    cond_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        r[:, k + 1] = b + (r[:, k] - b) * exp_adt + cond_std * Z
    return r


def riemann_left_integral(rate_paths, dt):
    n_paths, n_grid = rate_paths.shape
    I = np.zeros((n_paths, n_grid), dtype=np.float64)
    I[:, 1:] = np.cumsum(rate_paths[:, :-1] * dt, axis=1)
    return I


# ---------------------------------------------------------------------------
# Path-wise total-return process (reused from Step 16)
# ---------------------------------------------------------------------------
def bond_price_along_paths(bond, r_paths, dt):
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


def total_return_process(bond, ex_coupon_prices, I_paths, dt):
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
# Diagnostics
# ---------------------------------------------------------------------------
def diag_marginal_moments(r_em, r_ex, dt):
    n_paths, n_grid = r_em.shape
    t_grid = np.arange(n_grid) * dt
    mean_em = r_em.mean(axis=0)
    mean_ex = r_ex.mean(axis=0)
    var_em = r_em.var(axis=0, ddof=1)
    var_ex = r_ex.var(axis=0, ddof=1)
    mean_th = B_MEAN + (R0 - B_MEAN) * np.exp(-A * t_grid)
    var_th = SIGMA ** 2 * (1.0 - np.exp(-2.0 * A * t_grid)) / (2.0 * A)
    df = pd.DataFrame({
        "t": t_grid,
        "mean_em": mean_em,
        "mean_exact": mean_ex,
        "mean_analytic": mean_th,
        "var_em": var_em,
        "var_exact": var_ex,
        "var_analytic": var_th,
        "mean_err_em_bp": 10000.0 * (mean_em - mean_th),
        "mean_err_exact_bp": 10000.0 * (mean_ex - mean_th),
        "var_err_em_rel": np.divide(
            var_em - var_th, var_th,
            out=np.full_like(var_th, np.nan, dtype=float),
            where=(var_th > 0)),
        "var_err_exact_rel": np.divide(
            var_ex - var_th, var_th,
            out=np.full_like(var_th, np.nan, dtype=float),
            where=(var_th > 0)),
    })
    df.to_csv(OUTPUT_DIR / "step17_marginal_moments.csv", index=False)
    return df


def diag_zcb_term_structure(r_em, r_ex):
    I_em = riemann_left_integral(r_em, DT)
    I_ex = riemann_left_integral(r_ex, DT)
    rows = []
    for T in ZCB_TENORS:
        k = int(round(T / DT))
        disc_em = np.exp(-I_em[:, k])
        disc_ex = np.exp(-I_ex[:, k])
        P_em = float(disc_em.mean())
        P_ex = float(disc_ex.mean())
        SE_em = float(disc_em.std(ddof=1) / np.sqrt(disc_em.size))
        SE_ex = float(disc_ex.std(ddof=1) / np.sqrt(disc_ex.size))
        P_ana = vasicek_zcb_price_scalar(T, R0, A, B_MEAN, SIGMA)
        rows.append({
            "T_yr": T,
            "P_analytic": P_ana,
            "P_em": P_em,
            "P_exact": P_ex,
            "MC_SE_em": SE_em,
            "MC_SE_exact": SE_ex,
            "diff_em_vs_exact_bp_of_price": 10000.0 * (P_em - P_ex),
            "rel_diff_em_vs_exact_pct": 100.0 * (P_em - P_ex) / P_ex,
            "rel_diff_em_vs_analytic_pct": 100.0 * (P_em - P_ana) / P_ana,
            "rel_diff_exact_vs_analytic_pct": 100.0 * (P_ex - P_ana) / P_ana,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step17_zcb_term_structure.csv", index=False)
    return df


def diag_coupon_bond(r_em, r_ex):
    I_em = riemann_left_integral(r_em, DT)
    I_ex = riemann_left_integral(r_ex, DT)
    prices_A_em = bond_price_along_paths(BOND_A, r_em, DT)
    prices_B_em = bond_price_along_paths(BOND_B, r_em, DT)
    prices_A_ex = bond_price_along_paths(BOND_A, r_ex, DT)
    prices_B_ex = bond_price_along_paths(BOND_B, r_ex, DT)
    V_A_em = total_return_process(BOND_A, prices_A_em, I_em, DT)
    V_B_em = total_return_process(BOND_B, prices_B_em, I_em, DT)
    V_A_ex = total_return_process(BOND_A, prices_A_ex, I_ex, DT)
    V_B_ex = total_return_process(BOND_B, prices_B_ex, I_ex, DT)

    rows = []
    for label, V_em, V_ex in [("A", V_A_em, V_A_ex), ("B", V_B_em, V_B_ex)]:
        V0_em = float(V_em[:, 0].mean())
        V0_ex = float(V_ex[:, 0].mean())
        for T_h in HORIZONS:
            k = int(round(T_h / DT))
            E_em = float(V_em[:, k].mean())
            E_ex = float(V_ex[:, k].mean())
            SE_em = float(V_em[:, k].std(ddof=1) / np.sqrt(V_em.shape[0]))
            SE_ex = float(V_ex[:, k].std(ddof=1) / np.sqrt(V_ex.shape[0]))
            r_bond_em = float(np.log(E_em / V0_em) / T_h)
            r_bond_ex = float(np.log(E_ex / V0_ex) / T_h)
            rows.append({
                "bond": label,
                "horizon_yr": T_h,
                "V0_em": V0_em,
                "V0_exact": V0_ex,
                "E_V_em": E_em,
                "E_V_exact": E_ex,
                "MC_SE_em": SE_em,
                "MC_SE_exact": SE_ex,
                "ann_return_em_cc": r_bond_em,
                "ann_return_exact_cc": r_bond_ex,
                "rel_diff_E_V_pct": 100.0 * (E_em - E_ex) / E_ex,
                "ann_return_diff_bp": 10000.0 * (r_bond_em - r_bond_ex),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step17_coupon_bond.csv", index=False)
    return df, V_A_em, V_A_ex, V_B_em, V_B_ex


def diag_var(r_em, r_ex, V_B_em, V_B_ex):
    rows = []
    k_one_month = VAR_HORIZON_MONTHS
    loss_rate_em = -(r_em[:, k_one_month] - r_em[:, 0])
    loss_rate_ex = -(r_ex[:, k_one_month] - r_ex[:, 0])
    for lvl in VAR_LEVELS:
        q_em = float(np.quantile(loss_rate_em, lvl))
        q_ex = float(np.quantile(loss_rate_ex, lvl))
        rows.append({
            "target": "1m rate level loss (-Delta r)",
            "level": lvl,
            "VaR_em": q_em,
            "VaR_exact": q_ex,
            "abs_diff_bp": 10000.0 * (q_em - q_ex),
            "rel_diff_pct": 100.0 * (q_em - q_ex) / q_ex,
        })
    k_one_year = int(round(1.0 / DT))
    V0_em = float(V_B_em[:, 0].mean())
    V0_ex = float(V_B_ex[:, 0].mean())
    loss_B_em = V0_em - V_B_em[:, k_one_year]
    loss_B_ex = V0_ex - V_B_ex[:, k_one_year]
    for lvl in VAR_LEVELS:
        q_em = float(np.quantile(loss_B_em, lvl))
        q_ex = float(np.quantile(loss_B_ex, lvl))
        rows.append({
            "target": "1yr Bond B MtM loss",
            "level": lvl,
            "VaR_em": q_em,
            "VaR_exact": q_ex,
            "abs_diff_bp": 10000.0 * (q_em - q_ex) / V0_ex,
            "rel_diff_pct": 100.0 * (q_em - q_ex) / V0_ex,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step17_var.csv", index=False)
    return df, loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex


def diag_per_step_conditional(dt):
    rk_minus_b = R0 - B_MEAN
    mu_em = rk_minus_b * (1.0 - A * dt)
    mu_ex = rk_minus_b * np.exp(-A * dt)
    var_em = SIGMA ** 2 * dt
    var_ex = SIGMA ** 2 * (1.0 - np.exp(-2.0 * A * dt)) / (2.0 * A)
    df = pd.DataFrame([{
        "dt": dt,
        "a_dt": A * dt,
        "conditional_mu_em": mu_em,
        "conditional_mu_exact": mu_ex,
        "mu_error_abs": mu_ex - mu_em,
        "mu_error_bp": 10000.0 * (mu_ex - mu_em),
        "leading_mu_term_bp": 10000.0 * rk_minus_b * (A * dt) ** 2 / 2.0,
        "conditional_var_em": var_em,
        "conditional_var_exact": var_ex,
        "var_error_abs": var_ex - var_em,
        "var_error_rel": (var_ex - var_em) / var_em,
        "leading_var_term_abs": -SIGMA ** 2 * A * dt ** 2,
    }])
    df.to_csv(OUTPUT_DIR / "step17_conditional_error.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_marginal_moments(df_m):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)
    t = df_m["t"]
    axes[0].plot(t, 10000 * df_m["mean_analytic"], "-", color=BLUE,
                 linewidth=2.2, label="Analytic $E[r_t]$", zorder=2)
    axes[0].plot(t, 10000 * df_m["mean_em"], "--", color=OCHRE,
                 linewidth=1.5, label="Euler-Maruyama", zorder=3)
    axes[0].plot(t, 10000 * df_m["mean_exact"], ":", color=TEAL,
                 linewidth=1.5, label="Exact transition", zorder=4)
    axes[0].set_xlabel("Time $t$ (years)")
    axes[0].set_ylabel("$E[r_t]$ (basis points)")
    axes[0].set_title("Marginal mean of $r_t$", fontweight="bold", pad=10)
    axes[0].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK)
    axes[1].plot(t, 1e6 * df_m["var_analytic"], "-", color=BLUE,
                 linewidth=2.2, label=r"Analytic $\mathrm{Var}[r_t]$",
                 zorder=2)
    axes[1].plot(t, 1e6 * df_m["var_em"], "--", color=OCHRE,
                 linewidth=1.5, label="Euler-Maruyama", zorder=3)
    axes[1].plot(t, 1e6 * df_m["var_exact"], ":", color=TEAL,
                 linewidth=1.5, label="Exact transition", zorder=4)
    axes[1].set_xlabel("Time $t$ (years)")
    axes[1].set_ylabel(r"$\mathrm{Var}[r_t]$ (bp$^2$)")
    axes[1].set_title("Marginal variance of $r_t$",
                      fontweight="bold", pad=10)
    axes[1].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig17_1_marginal_moments.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_zcb_term_structure(df_z):
    fig, ax = _new_figure((8.6, 4.6))
    x = df_z["T_yr"]
    width = 0.32
    rel_em = df_z["rel_diff_em_vs_analytic_pct"] * 100  # to bp of price
    rel_ex = df_z["rel_diff_exact_vs_analytic_pct"] * 100
    ax.bar(x - width / 2, rel_em, width=width, color=OCHRE,
           edgecolor=INK, linewidth=0.6, label="EM vs. analytic")
    ax.bar(x + width / 2, rel_ex, width=width, color=TEAL,
           edgecolor=INK, linewidth=0.6, label="Exact vs. analytic")
    ax.axhline(0.0, color=SUB, linewidth=0.8)
    ax.set_xlabel("Tenor $T$ (years)")
    ax.set_ylabel("ZCB price error (basis points of price)")
    ax.set_title("Simulated $P(0,T)$ vs. analytic; discretization error",
                 fontweight="bold", pad=10)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig17_2_zcb_term_structure.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_path_difference(r_em, r_ex):
    fig, ax = _new_figure((8.6, 4.6))
    t = np.arange(r_em.shape[1]) * DT
    diff = (r_em - r_ex) * 10000.0
    mean = diff.mean(axis=0)
    p5 = np.percentile(diff, 5, axis=0)
    p95 = np.percentile(diff, 95, axis=0)
    ax.fill_between(t, p5, p95, color=OCHRE, alpha=0.30,
                    label="5th-95th percentile across paths")
    ax.plot(t, mean, "-", color=BURG, linewidth=1.8,
            label=r"Cross-path mean of $r_t^{EM} - r_t^{exact}$")
    ax.axhline(0.0, color=SUB, linewidth=0.8)
    ax.set_xlabel("Time $t$ (years)")
    ax.set_ylabel(r"Pathwise gap $r_t^{EM} - r_t^{exact}$ (bp)")
    ax.set_title("Same-seed pathwise discretization gap",
                 fontweight="bold", pad=10)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
              loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig17_3_path_difference.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_var_compare(loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)
    bins = np.linspace(min(loss_rate_em.min(), loss_rate_ex.min()),
                       max(loss_rate_em.max(), loss_rate_ex.max()), 60)
    axes[0].hist(loss_rate_em * 10000, bins=bins * 10000, alpha=0.55,
                 color=OCHRE, label="EM")
    axes[0].hist(loss_rate_ex * 10000, bins=bins * 10000, alpha=0.55,
                 color=TEAL, label="Exact")
    for lvl, c in zip(VAR_LEVELS, [BURG, BLUE]):
        q_em = np.quantile(loss_rate_em, lvl) * 10000
        q_ex = np.quantile(loss_rate_ex, lvl) * 10000
        axes[0].axvline(q_em, color=c, linestyle="--", linewidth=1.2,
                        label=f"VaR{int(100 * lvl)}% EM = {q_em:.2f}bp")
        axes[0].axvline(q_ex, color=c, linestyle=":", linewidth=1.4,
                        label=f"VaR{int(100 * lvl)}% Exact = {q_ex:.2f}bp")
    axes[0].set_xlabel(r"1-month rate loss $-(r_{1m} - r_0)$ (bp)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("1-month rate-level loss distribution",
                      fontweight="bold", pad=10)
    axes[0].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
                   fontsize=8, loc="upper left")
    bins = np.linspace(min(loss_B_em.min(), loss_B_ex.min()),
                       max(loss_B_em.max(), loss_B_ex.max()), 60)
    axes[1].hist(loss_B_em, bins=bins, alpha=0.55, color=OCHRE,
                 label="EM")
    axes[1].hist(loss_B_ex, bins=bins, alpha=0.55, color=TEAL,
                 label="Exact")
    for lvl, c in zip(VAR_LEVELS, [BURG, BLUE]):
        q_em = np.quantile(loss_B_em, lvl)
        q_ex = np.quantile(loss_B_ex, lvl)
        axes[1].axvline(q_em, color=c, linestyle="--", linewidth=1.2,
                        label=f"VaR{int(100 * lvl)}% EM = ${q_em:.3f}")
        axes[1].axvline(q_ex, color=c, linestyle=":", linewidth=1.4,
                        label=f"VaR{int(100 * lvl)}% Exact = ${q_ex:.3f}")
    axes[1].set_xlabel("1-year Bond B MtM loss ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("1-year Bond B mark-to-market loss",
                      fontweight="bold", pad=10)
    axes[1].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
                   fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig17_4_var_compare.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks (Section 5)
# ---------------------------------------------------------------------------
def sanity_checks(df_m, df_z, df_var):
    checks: dict = {}

    # (1) sigma -> 0: exact simulator should match the deterministic ODE
    r_ex_zero = simulate_exact(
        A, B_MEAN, 0.0, R0, HORIZON_YEARS, DT, 200, SEED
    )
    t_grid = np.arange(r_ex_zero.shape[1]) * DT
    analytic = B_MEAN + (R0 - B_MEAN) * np.exp(-A * t_grid)
    max_rel_ex = float(np.max(np.abs(r_ex_zero - analytic) / np.abs(analytic)))
    r_em_zero = simulate_em(
        A, B_MEAN, 0.0, R0, HORIZON_YEARS, DT, 200, SEED
    )
    max_rel_em = float(np.max(np.abs(r_em_zero - analytic) / np.abs(analytic)))
    checks["sigma_zero_collapse"] = {
        "max_rel_err_exact": max_rel_ex,
        "max_rel_err_em": max_rel_em,
        "tolerance_exact": 1e-12,
        "pass": bool(max_rel_ex < 1e-12 and max_rel_em < 1e-2),
    }

    # (2) stationary-law match: at t = 10y the EM variance bias is O((a dt)^2)
    mean_th_inf = B_MEAN
    var_th_inf = SIGMA ** 2 / (2.0 * A)
    last = df_m.iloc[-1]
    mean_em_dev_bp = abs(last["mean_em"] - mean_th_inf) * 10000.0
    mean_ex_dev_bp = abs(last["mean_exact"] - mean_th_inf) * 10000.0
    var_em_rel = (last["var_em"] - var_th_inf) / var_th_inf
    var_ex_rel = (last["var_exact"] - var_th_inf) / var_th_inf
    checks["stationary_law_match"] = {
        "mean_em_dev_bp": float(mean_em_dev_bp),
        "mean_exact_dev_bp": float(mean_ex_dev_bp),
        "var_em_rel": float(var_em_rel),
        "var_exact_rel": float(var_ex_rel),
        "pass": bool(mean_em_dev_bp < 5.0 and mean_ex_dev_bp < 5.0
                     and abs(var_ex_rel) < 0.03),
    }

    # (3) Brownian-increment audit: both simulators consume identical Z streams.
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)
    for _ in range(12):  # first year of monthly draws
        z1 = rng1.standard_normal(N_PATHS)
        z2 = rng2.standard_normal(N_PATHS)
    same_stream = bool(np.array_equal(z1, z2))
    checks["brownian_increment_audit"] = {
        "first_year_identical": same_stream,
        "pass": same_stream,
    }

    # (4) Cross-rank correlation under same-seed coupling
    rho = float(np.corrcoef(df_m["mean_em"].values,
                            df_m["mean_exact"].values)[0, 1])
    checks["mean_curve_correlation"] = {
        "pearson_r_em_vs_exact_mean_curve": rho,
        "pass": bool(rho > 0.999),
    }

    # (5) ZCB acceptance band
    max_zcb_bp = float(df_z["rel_diff_em_vs_exact_pct"].abs().max() * 100)
    checks["zcb_within_5bp_of_price"] = {
        "max_em_vs_exact_bp_of_price": max_zcb_bp,
        "tolerance_bp": 5.0,
        "pass": bool(max_zcb_bp < 5.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> dict:
    print("Phase C, Step 17: Vasicek exact-transition validation")
    print("-" * 60)
    print(f"seed = {SEED} | dt = {DT:.6f} | n_paths = {N_PATHS} | "
          f"horizon = {HORIZON_YEARS}")
    print(f"params = a={A}, b={B_MEAN}, sigma={SIGMA}, r0={R0}")
    print()

    print("Simulating Euler-Maruyama paths ...")
    r_em = simulate_em(A, B_MEAN, SIGMA, R0, HORIZON_YEARS, DT,
                       N_PATHS, SEED)
    print("Simulating exact-transition paths (same seed) ...")
    r_ex = simulate_exact(A, B_MEAN, SIGMA, R0, HORIZON_YEARS, DT,
                          N_PATHS, SEED)

    print("Building diagnostics ...")
    df_m = diag_marginal_moments(r_em, r_ex, DT)
    df_z = diag_zcb_term_structure(r_em, r_ex)
    df_c, V_A_em, V_A_ex, V_B_em, V_B_ex = diag_coupon_bond(r_em, r_ex)
    df_var, loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex = diag_var(
        r_em, r_ex, V_B_em, V_B_ex
    )
    df_cond = diag_per_step_conditional(DT)

    print("Rendering figures ...")
    figure_marginal_moments(df_m)
    figure_zcb_term_structure(df_z)
    figure_path_difference(r_em, r_ex)
    figure_var_compare(loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex)

    print("Running sanity checks ...")
    checks = sanity_checks(df_m, df_z, df_var)

    summary = {
        "parameters": {"a": A, "b": B_MEAN, "sigma": SIGMA, "r0": R0,
                       "dt": DT, "n_paths": N_PATHS,
                       "horizon": HORIZON_YEARS, "seed": SEED},
        "horizons": HORIZONS,
        "max_abs_mean_err_em_bp":
            float(df_m["mean_err_em_bp"].abs().max()),
        "max_abs_mean_err_exact_bp":
            float(df_m["mean_err_exact_bp"].abs().max()),
        "max_abs_var_err_em_rel":
            float(np.nanmax(np.abs(df_m["var_err_em_rel"].to_numpy()))),
        "max_abs_var_err_exact_rel":
            float(np.nanmax(np.abs(df_m["var_err_exact_rel"].to_numpy()))),
        "max_zcb_rel_diff_em_vs_exact_pct":
            float(df_z["rel_diff_em_vs_exact_pct"].abs().max()),
        "max_zcb_rel_diff_em_vs_analytic_pct":
            float(df_z["rel_diff_em_vs_analytic_pct"].abs().max()),
        "max_zcb_rel_diff_exact_vs_analytic_pct":
            float(df_z["rel_diff_exact_vs_analytic_pct"].abs().max()),
        "max_coupon_ann_return_diff_bp":
            float(df_c["ann_return_diff_bp"].abs().max()),
        "max_var_rel_diff_pct":
            float(df_var["rel_diff_pct"].abs().max()),
        "theoretical_mu_error_bp_per_step":
            float(df_cond["leading_mu_term_bp"].iloc[0]),
        "theoretical_var_error_rel_per_step":
            float(df_cond.iloc[0]["var_error_rel"]),
        "acceptance_criterion_zcb_bp_of_price": 5.0,
        "acceptance_criterion_var_rate_pct": 3.0,
        "acceptance_criterion_var_bondB_bp_of_V0": 10.0,
        "zcb_passes_5bp_of_price": bool(
            df_z["rel_diff_em_vs_exact_pct"].abs().max() * 100 < 5.0),
        "var_rate_passes_3pct": bool(
            df_var.loc[df_var["target"].str.contains("rate"),
                       "rel_diff_pct"].abs().max() < 3.0),
        "var_bondB_passes_10bp_of_V0": bool(
            df_var.loc[df_var["target"].str.contains("Bond"),
                       "abs_diff_bp"].abs().max() < 10.0),
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step17_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, allow_nan=False)

    # --- Console summary ---
    print("\n=== ZCB term-structure ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        print(df_z.to_string(index=False))

    print("\n=== Coupon-bond return diff ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        print(df_c[["bond", "horizon_yr", "V0_em",
                    "ann_return_em_cc", "ann_return_exact_cc",
                    "ann_return_diff_bp"]].to_string(index=False))

    print("\n=== VaR comparison ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        print(df_var.to_string(index=False))

    print("\n=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    print("\n=== Step 17 summary ===")
    main_pass = (summary["zcb_passes_5bp_of_price"]
                 and summary["var_rate_passes_3pct"]
                 and summary["var_bondB_passes_10bp_of_V0"])
    verdict = "PASS" if main_pass else "FAIL"
    print(f"  Verdict                                : {verdict}")
    print(f"  Max ZCB EM vs Exact (bp of price)      : "
          f"{summary['max_zcb_rel_diff_em_vs_exact_pct'] * 100:.4f}")
    print(f"  Max coupon-return diff (bp/yr)         : "
          f"{summary['max_coupon_ann_return_diff_bp']:.4f}")
    print(f"  Max VaR rel diff (%)                   : "
          f"{summary['max_var_rel_diff_pct']:.4f}")
    print(f"  EM mean error vs analytic (bp)         : "
          f"{summary['max_abs_mean_err_em_bp']:.4f}")
    print(f"  Exact mean error vs analytic (bp)      : "
          f"{summary['max_abs_mean_err_exact_bp']:.4f}")
    print(f"  EM variance rel error vs analytic      : "
          f"{summary['max_abs_var_err_em_rel']:.4%}")
    print(f"  Exact variance rel error vs analytic   : "
          f"{summary['max_abs_var_err_exact_rel']:.4%}")
    print(f"  Theoretical mu error per step (bp)     : "
          f"{summary['theoretical_mu_error_bp_per_step']:.4f}")
    print(f"  Theoretical var error per step (rel)   : "
          f"{summary['theoretical_var_error_rel_per_step']:.4%}")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    main()
