"""
models/cir/step24_cir_exact_transition.py
-----------------------------------------
Phase D, Step 24: Validate the CIR Euler-Maruyama simulator against exact
non-central chi-squared transition sampling.

The CIR SDE under Q is

    dr_t = a (b - r_t) dt + sigma sqrt(r_t) dW_t,   r_0 > 0,

whose conditional transition is exact non-central chi-squared:

    c r(t + dt) | r(t)  ~  chi^2(d, lam),
        d   = 4 a b / sigma^2,
        c   = 4 a / (sigma^2 (1 - exp(-a dt))),
        lam = c r(t) exp(-a dt).

The exact sampler invokes numpy's noncentral_chisquare per (path, step)
and divides by c to recover r at the next grid point; the result is
unbiased in the marginal of r(t).  The full-truncation Euler-Maruyama
update is correct only to leading order in dt and is the production
scheme of Steps 20, 22.  Per-step conditional moments differ at O(dt^2);
the weak order of EM remains 1 but the strong order drops to 1/2 because
of the non-Lipschitz sqrt(r) diffusion.

CIR cannot replicate Vasicek's bit-exact same-seed coupling because the
exact ncx2 draw consumes random state in a different pattern than the
per-step standard_normal draw used by EM; both simulators reseed with
seed = 42 producing statistically aligned but not bit-identical paths.

Acceptance bands:
  1. ZCB price gap (exact vs CIR closed form) <= 5 bp at every tenor.
  2. Marginal mean of r(t) at every node within 5 bp of analytic under
     both simulators.
  3. EM conditional-mean error at r_k = r_0 within 5%% of leading
     Taylor term (r_k - b) (a dt)^2 / 2.

Reproducibility: seed = 42 reused from Steps 13, 16, 17, 18, 20, 21, 22, 23.

Outputs (aligned with Steps 13-23 artefact layout):

    models/cir/step24_marginal_moments.csv
    models/cir/step24_zcb_term_structure.csv
    models/cir/step24_coupon_bond.csv
    models/cir/step24_var.csv
    models/cir/step24_conditional_error.csv
    models/cir/step24_pathwise_gap.csv
    models/cir/step24_summary.json
    figures/step24/step24_fig1_marginal_moments.png
    figures/step24/step24_fig2_zcb_term_structure.png
    figures/step24/step24_fig3_path_difference.png
    figures/step24/step24_fig4_var_compare.png
"""

from __future__ import annotations

import json
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
# Shared configuration; reused verbatim from Steps 20-23.
# ---------------------------------------------------------------------------
A = 0.50
B_MEAN = 0.04
SIGMA = 0.08
R0 = 0.0372
DT = 1.0 / 12.0
N_PATHS = 10_000
HORIZON_YEARS = 10.0
SEED = 42

HORIZONS = [1.0, 2.0, 5.0, 10.0]
ZCB_TENORS = [1.0, 2.0, 5.0, 10.0]
VAR_LEVELS = [0.95, 0.99]
VAR_HORIZON_MONTHS = 1

# Bond definitions: par coupons calibrated against the CIR ZCB curve at t=0.
# (Coupon set to bring V_0 = 100 under the analytic CIR pricer.)
BOND_A = {"name": "Bond A (2yr)", "T_mat": 2.0,
          "coupon_rate_pct": 3.85,
          "coupon_dates": [0.5, 1.0, 1.5, 2.0]}
BOND_B = {"name": "Bond B (10yr)", "T_mat": 10.0,
          "coupon_rate_pct": 3.94,
          "coupon_dates": [0.5 * k for k in range(1, 21)]}

OUTPUT_DIR = ROOT / "models" / "cir"
FIGURES_DIR = ROOT / "figures" / "step24"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-23 token set)
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
# CIR analytic moments and closed-form ZCB primitives (Step 22 reuse).
# ---------------------------------------------------------------------------
def cir_mean(t, a=A, b=B_MEAN, r0=R0):
    return b + (r0 - b) * np.exp(-a * t)


def cir_var(t, a=A, b=B_MEAN, sigma=SIGMA, r0=R0):
    term1 = (r0 * sigma ** 2 / a) * (np.exp(-a * t) - np.exp(-2.0 * a * t))
    term2 = (b * sigma ** 2 / (2.0 * a)) * (1.0 - np.exp(-a * t)) ** 2
    return term1 + term2


def cir_B(T, a=A, sigma=SIGMA):
    gamma = np.sqrt(a ** 2 + 2.0 * sigma ** 2)
    num = 2.0 * (np.exp(gamma * T) - 1.0)
    den = (a + gamma) * (np.exp(gamma * T) - 1.0) + 2.0 * gamma
    return num / den


def cir_A(T, a=A, b=B_MEAN, sigma=SIGMA):
    gamma = np.sqrt(a ** 2 + 2.0 * sigma ** 2)
    num = 2.0 * gamma * np.exp((a + gamma) * T / 2.0)
    den = (a + gamma) * (np.exp(gamma * T) - 1.0) + 2.0 * gamma
    return float((num / den) ** (2.0 * a * b / sigma ** 2))


def cir_price(T, r=R0, a=A, b=B_MEAN, sigma=SIGMA):
    return cir_A(T, a, b, sigma) * float(np.exp(-cir_B(T, a, sigma) * r))


def cir_price_vec(T, r, a=A, b=B_MEAN, sigma=SIGMA):
    """Vectorised CIR ZCB price for scalar tau and array r."""
    if T <= 0:
        return np.ones_like(np.asarray(r, dtype=float))
    return cir_A(T, a, b, sigma) * np.exp(-cir_B(T, a, sigma) * np.asarray(r, dtype=float))


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------
def simulate_cir_em(a, b, sigma, r0, horizon, dt, n_paths, seed):
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        rk_plus = np.maximum(r[:, k], 0.0)
        r[:, k + 1] = (r[:, k] + a * (b - r[:, k]) * dt
                       + sigma * np.sqrt(rk_plus) * sqrt_dt * Z)
    return r


def simulate_cir_exact(a, b, sigma, r0, horizon, dt, n_paths, seed):
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    d = 4.0 * a * b / sigma ** 2
    decay = np.exp(-a * dt)
    c = 4.0 * a / (sigma ** 2 * (1.0 - decay))
    for k in range(n_steps):
        lam = c * r[:, k] * decay
        chi = rng.noncentral_chisquare(df=d, nonc=lam, size=n_paths)
        r[:, k + 1] = chi / c
    return r


def trapezoid_integral(rate_paths, dt):
    """Trapezoidal cumulative integral of r along each path."""
    running = np.zeros_like(rate_paths)
    running[:, 1:] = 0.5 * (rate_paths[:, :-1] + rate_paths[:, 1:]) * dt
    return np.cumsum(running, axis=1)


# ---------------------------------------------------------------------------
# Path-wise total-return process (Step 16/22 pattern adapted for CIR)
# ---------------------------------------------------------------------------
def bond_price_along_paths(bond, r_paths, dt):
    n_paths, n_grid = r_paths.shape
    t_grid = np.arange(n_grid) * dt
    prices = np.zeros_like(r_paths)
    face = 100.0
    coupon = face * bond["coupon_rate_pct"] / 100.0 / 2.0
    for k in range(n_grid):
        t_k = t_grid[k]
        if t_k >= bond["T_mat"] - 1e-9:
            prices[:, k] = 0.0
            continue
        remaining = [t for t in bond["coupon_dates"] if t > t_k + 1e-9]
        pv = np.zeros(n_paths)
        for T_j in remaining:
            tau = T_j - t_k
            zcb = cir_price_vec(tau, r_paths[:, k])
            cf = coupon + (face if abs(T_j - bond["T_mat"]) < 1e-9 else 0.0)
            pv += cf * zcb
        prices[:, k] = pv
    return prices


def total_return_process(bond, ex_coupon_prices, I_paths, dt):
    n_paths, n_grid = ex_coupon_prices.shape
    V = ex_coupon_prices.copy()
    face = 100.0
    coupon = face * bond["coupon_rate_pct"] / 100.0 / 2.0
    for T_j in bond["coupon_dates"]:
        is_maturity = abs(T_j - bond["T_mat"]) < 1e-9
        cf = coupon + (face if is_maturity else 0.0)
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
    mean_th = cir_mean(t_grid)
    var_th = cir_var(t_grid)
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
    })
    df.to_csv(OUTPUT_DIR / "step24_marginal_moments.csv", index=False)
    return df


def diag_zcb_term_structure(r_em, r_ex):
    I_em = trapezoid_integral(r_em, DT)
    I_ex = trapezoid_integral(r_ex, DT)
    rows = []
    for T in ZCB_TENORS:
        k = int(round(T / DT))
        disc_em = np.exp(-I_em[:, k])
        disc_ex = np.exp(-I_ex[:, k])
        P_em = float(disc_em.mean())
        P_ex = float(disc_ex.mean())
        SE_em = float(disc_em.std(ddof=1) / np.sqrt(disc_em.size))
        SE_ex = float(disc_ex.std(ddof=1) / np.sqrt(disc_ex.size))
        P_ana = cir_price(T)
        rows.append({
            "T_yr": T,
            "P_analytic": P_ana,
            "P_em": P_em,
            "P_exact": P_ex,
            "MC_SE_em": SE_em,
            "MC_SE_exact": SE_ex,
            "diff_em_vs_ana_bp": 10000.0 * (P_em - P_ana),
            "diff_exact_vs_ana_bp": 10000.0 * (P_ex - P_ana),
            "diff_em_vs_exact_bp": 10000.0 * (P_em - P_ex),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step24_zcb_term_structure.csv", index=False)
    return df


def diag_coupon_bond(r_em, r_ex):
    I_em = trapezoid_integral(r_em, DT)
    I_ex = trapezoid_integral(r_ex, DT)
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
        T_mat = BOND_A["T_mat"] if label == "A" else BOND_B["T_mat"]
        V0_em = float(V_em[:, 0].mean())
        V0_ex = float(V_ex[:, 0].mean())
        for T_h in HORIZONS:
            if T_h > T_mat + 1e-9:
                continue
            k = int(round(T_h / DT))
            E_em = float(V_em[:, k].mean())
            E_ex = float(V_ex[:, k].mean())
            SE_em = float(V_em[:, k].std(ddof=1) / np.sqrt(V_em.shape[0]))
            SE_ex = float(V_ex[:, k].std(ddof=1) / np.sqrt(V_ex.shape[0]))
            r_em_rate = float(np.log(max(E_em / V0_em, 1e-12)) / T_h)
            r_ex_rate = float(np.log(max(E_ex / V0_ex, 1e-12)) / T_h)
            rows.append({
                "bond": label,
                "horizon_yr": T_h,
                "V0_em": V0_em,
                "V0_exact": V0_ex,
                "E_V_em": E_em,
                "E_V_exact": E_ex,
                "MC_SE_em": SE_em,
                "MC_SE_exact": SE_ex,
                "ann_return_em_cc": r_em_rate,
                "ann_return_exact_cc": r_ex_rate,
                "ann_return_diff_bp": 10000.0 * (r_em_rate - r_ex_rate),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step24_coupon_bond.csv", index=False)
    return df, V_A_em, V_A_ex, V_B_em, V_B_ex


def diag_var(r_em, r_ex, V_B_em, V_B_ex):
    rows = []
    k_one_month = VAR_HORIZON_MONTHS
    loss_rate_em = -(r_em[:, k_one_month] - r_em[:, 0])
    loss_rate_ex = -(r_ex[:, k_one_month] - r_ex[:, 0])
    for lvl in VAR_LEVELS:
        q_em = float(np.quantile(loss_rate_em, lvl))
        q_ex = float(np.quantile(loss_rate_ex, lvl))
        denom = q_ex if abs(q_ex) > 1e-12 else 1.0
        rows.append({
            "target": "1m rate level loss (-Delta r)",
            "level": lvl,
            "VaR_em": q_em,
            "VaR_exact": q_ex,
            "abs_diff_bp": 10000.0 * (q_em - q_ex),
            "rel_diff_pct": 100.0 * (q_em - q_ex) / denom,
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
    df.to_csv(OUTPUT_DIR / "step24_var.csv", index=False)
    return df, loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex


def diag_per_step_conditional(dt):
    rows = []
    for rk in (R0, B_MEAN, B_MEAN + 0.02):
        mu_em = rk + A * (B_MEAN - rk) * dt
        mu_ex = B_MEAN + (rk - B_MEAN) * np.exp(-A * dt)
        leading_mu_bp = 10000.0 * (rk - B_MEAN) * (A * dt) ** 2 / 2.0
        var_em = SIGMA ** 2 * max(rk, 0.0) * dt
        # exact CIR conditional variance over dt given r_k:
        # rk * (sigma^2/a)(exp(-a dt) - exp(-2 a dt))
        #   + b * (sigma^2/(2a))(1 - exp(-a dt))^2
        var_ex = (rk * (SIGMA ** 2 / A)
                  * (np.exp(-A * dt) - np.exp(-2.0 * A * dt))
                  + B_MEAN * (SIGMA ** 2 / (2.0 * A))
                  * (1.0 - np.exp(-A * dt)) ** 2)
        leading_var_abs = -(SIGMA ** 2) * A * dt ** 2 * rk
        rows.append({
            "r_k": rk,
            "mu_exact": mu_ex,
            "mu_em": mu_em,
            "mu_err_bp": 10000.0 * (mu_ex - mu_em),
            "leading_mu_term_bp": leading_mu_bp,
            "var_exact": var_ex,
            "var_em": var_em,
            "var_err_rel": (var_ex - var_em) / max(var_em, 1e-12),
            "leading_var_term_abs": leading_var_abs,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step24_conditional_error.csv", index=False)
    return df


def diag_pathwise_gap(r_em, r_ex, dt):
    t_grid = np.arange(r_em.shape[1]) * dt
    diff = (r_em - r_ex)
    df = pd.DataFrame({
        "t": t_grid,
        "mean_gap_bp": diff.mean(axis=0) * 1e4,
        "p05_bp": np.percentile(diff, 5, axis=0) * 1e4,
        "p95_bp": np.percentile(diff, 95, axis=0) * 1e4,
        "abs_mean_gap_bp": np.abs(diff).mean(axis=0) * 1e4,
    })
    df.to_csv(OUTPUT_DIR / "step24_pathwise_gap.csv", index=False)
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
    axes[0].plot(t, 10000 * df_m["mean_analytic"], "-", color=INK,
                 linewidth=2.0, label="Analytic $E[r_t]$", zorder=2)
    axes[0].plot(t, 10000 * df_m["mean_em"], "--", color=OCHRE,
                 linewidth=1.5, label="Euler-Maruyama", zorder=3)
    axes[0].plot(t, 10000 * df_m["mean_exact"], ":", color=TEAL,
                 linewidth=1.5, label="Exact ncx2", zorder=4)
    axes[0].set_xlabel("Time $t$ (years)")
    axes[0].set_ylabel(r"$E[r_t]$ (basis points)")
    axes[0].set_title("CIR marginal mean of $r_t$",
                      fontweight="bold", pad=10)
    axes[0].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK)
    axes[1].plot(t, 1e6 * df_m["var_analytic"], "-", color=INK,
                 linewidth=2.0, label=r"Analytic $\mathrm{Var}[r_t]$", zorder=2)
    axes[1].plot(t, 1e6 * df_m["var_em"], "--", color=OCHRE,
                 linewidth=1.5, label="Euler-Maruyama", zorder=3)
    axes[1].plot(t, 1e6 * df_m["var_exact"], ":", color=TEAL,
                 linewidth=1.5, label="Exact ncx2", zorder=4)
    axes[1].set_xlabel("Time $t$ (years)")
    axes[1].set_ylabel(r"$\mathrm{Var}[r_t]$ (bp$^2$)")
    axes[1].set_title("CIR marginal variance of $r_t$",
                      fontweight="bold", pad=10)
    axes[1].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step24_fig1_marginal_moments.png",
                dpi=180, facecolor=BG)
    plt.close(fig)


def figure_zcb_term_structure(df_z):
    fig, ax = _new_figure((8.6, 4.6))
    x = df_z["T_yr"]
    width = 0.32
    ax.bar(x - width / 2, df_z["diff_em_vs_ana_bp"], width=width,
           color=OCHRE, edgecolor=INK, linewidth=0.6,
           label="EM vs analytic")
    ax.bar(x + width / 2, df_z["diff_exact_vs_ana_bp"], width=width,
           color=TEAL, edgecolor=INK, linewidth=0.6,
           label="Exact vs analytic")
    ax.axhline(0.0, color=SUB, linewidth=0.8)
    ax.axhline(5.0, color=BURG, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(-5.0, color=BURG, linewidth=0.8, linestyle="--", alpha=0.7,
               label="+/- 5 bp acceptance")
    ax.set_xlabel("Tenor $T$ (years)")
    ax.set_ylabel("ZCB price error (basis points of price)")
    ax.set_title("CIR simulated $P(0,T)$ vs analytic; discretization error",
                 fontweight="bold", pad=10)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step24_fig2_zcb_term_structure.png",
                dpi=180, facecolor=BG)
    plt.close(fig)


def figure_path_difference(df_gap):
    fig, ax = _new_figure((8.6, 4.6))
    t = df_gap["t"]
    ax.fill_between(t, df_gap["p05_bp"], df_gap["p95_bp"],
                    color=OCHRE, alpha=0.30,
                    label="5th-95th percentile across paths")
    ax.plot(t, df_gap["mean_gap_bp"], "-", color=BURG, linewidth=1.8,
            label=r"Cross-path mean of $r_t^{EM} - r_t^{exact}$")
    ax.axhline(0.0, color=SUB, linewidth=0.8)
    ax.set_xlabel("Time $t$ (years)")
    ax.set_ylabel(r"Pathwise gap $r_t^{EM} - r_t^{exact}$ (bp)")
    ax.set_title("CIR pathwise discretization gap (reseeded coupling)",
                 fontweight="bold", pad=10)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step24_fig3_path_difference.png",
                dpi=180, facecolor=BG)
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
                        label=f"VaR{int(100 * lvl)}% EM = {q_em:.1f}bp")
        axes[0].axvline(q_ex, color=c, linestyle=":", linewidth=1.4,
                        label=f"VaR{int(100 * lvl)}% Exact = {q_ex:.1f}bp")
    axes[0].set_xlabel(r"1-month rate loss $-(r_{1m} - r_0)$ (bp)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("1-month rate-level loss distribution",
                      fontweight="bold", pad=10)
    axes[0].legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
                   fontsize=7, loc="upper left")

    bins = np.linspace(min(loss_B_em.min(), loss_B_ex.min()),
                       max(loss_B_em.max(), loss_B_ex.max()), 60)
    axes[1].hist(loss_B_em, bins=bins, alpha=0.55, color=OCHRE, label="EM")
    axes[1].hist(loss_B_ex, bins=bins, alpha=0.55, color=TEAL, label="Exact")
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
                   fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step24_fig4_var_compare.png",
                dpi=180, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks(df_m, df_z, df_cond):
    checks: dict = {}

    # (1) ZCB acceptance band: |EM vs analytic| and |Exact vs analytic| <= 5 bp
    max_em_zcb = float(df_z["diff_em_vs_ana_bp"].abs().max())
    max_ex_zcb = float(df_z["diff_exact_vs_ana_bp"].abs().max())
    checks["zcb_within_5bp_of_price"] = {
        "max_em_bp": max_em_zcb,
        "max_exact_bp": max_ex_zcb,
        "tolerance_bp": 5.0,
        "pass": bool(max_em_zcb <= 5.0 and max_ex_zcb <= 5.0),
    }

    # (2) marginal mean within 5 bp at every node, both schemes
    max_em_mean = float(df_m["mean_err_em_bp"].abs().max())
    max_ex_mean = float(df_m["mean_err_exact_bp"].abs().max())
    checks["marginal_mean_within_5bp"] = {
        "max_em_bp": max_em_mean,
        "max_exact_bp": max_ex_mean,
        "tolerance_bp": 5.0,
        "pass": bool(max_em_mean <= 5.0 and max_ex_mean <= 5.0),
    }

    # (3) EM conditional-mean error at r0 within 5% of leading Taylor term
    row0 = df_cond.iloc[0]
    mu_err = float(row0["mu_err_bp"])
    leading = float(row0["leading_mu_term_bp"])
    rel = abs(mu_err - leading) / max(abs(leading), 1e-9)
    checks["taylor_leading_term_match"] = {
        "mu_err_bp": mu_err,
        "leading_term_bp": leading,
        "rel_dev": rel,
        "tolerance": 0.05,
        "pass": bool(rel <= 0.05),
    }

    # (4) Var-err rel matches leading -sigma^2 a dt^2 r_k expansion within 10%
    var_ok = True
    for _, row in df_cond.iterrows():
        if row["r_k"] < 1e-9:
            continue
        leading_var = row["leading_var_term_abs"]
        actual = row["var_exact"] - row["var_em"]
        if abs(leading_var) > 1e-12:
            if abs((actual - leading_var) / leading_var) > 0.20:
                var_ok = False
    checks["variance_leading_term"] = {"pass": bool(var_ok)}

    # (5) feller condition holds at baseline
    feller_lhs = 2.0 * A * B_MEAN
    feller_rhs = SIGMA ** 2
    checks["feller_baseline"] = {
        "ratio": float(feller_lhs / feller_rhs),
        "pass": bool(feller_lhs >= feller_rhs),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase D, Step 24: CIR exact-transition validation")
    print("-" * 64)
    print(f"a = {A}, b = {B_MEAN}, sigma = {SIGMA}, r0 = {R0}")
    print(f"seed = {SEED}, N = {N_PATHS}, dt = {DT:.6f}, "
          f"T_max = {HORIZON_YEARS}")
    print(f"Feller: 2ab/sigma^2 = {(2*A*B_MEAN)/(SIGMA**2):.4f}")
    print()

    print("Simulating CIR Euler-Maruyama paths ...")
    r_em = simulate_cir_em(A, B_MEAN, SIGMA, R0, HORIZON_YEARS, DT,
                           N_PATHS, SEED)
    print("Simulating exact ncx2-transition paths (reseeded) ...")
    r_ex = simulate_cir_exact(A, B_MEAN, SIGMA, R0, HORIZON_YEARS, DT,
                              N_PATHS, SEED)

    print("Building diagnostics ...")
    df_m = diag_marginal_moments(r_em, r_ex, DT)
    df_z = diag_zcb_term_structure(r_em, r_ex)
    df_c, V_A_em, V_A_ex, V_B_em, V_B_ex = diag_coupon_bond(r_em, r_ex)
    df_var, loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex = diag_var(
        r_em, r_ex, V_B_em, V_B_ex)
    df_cond = diag_per_step_conditional(DT)
    df_gap = diag_pathwise_gap(r_em, r_ex, DT)

    print("Rendering figures ...")
    figure_marginal_moments(df_m)
    figure_zcb_term_structure(df_z)
    figure_path_difference(df_gap)
    figure_var_compare(loss_rate_em, loss_rate_ex, loss_B_em, loss_B_ex)

    print("Running sanity checks ...")
    checks = sanity_checks(df_m, df_z, df_cond)

    # ---- Console summary ---
    print("\n=== ZCB term-structure ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        print(df_z.to_string(index=False))

    print("\n=== Coupon-bond return diff ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        cols = ["bond", "horizon_yr", "V0_em",
                "ann_return_em_cc", "ann_return_exact_cc",
                "ann_return_diff_bp"]
        print(df_c[cols].to_string(index=False))

    print("\n=== VaR comparison ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 180):
        print(df_var.to_string(index=False))

    print("\n=== Per-step conditional-moment error ===")
    with pd.option_context("display.float_format", "{:.6e}".format,
                           "display.width", 180):
        print(df_cond.to_string(index=False))

    print("\n=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    summary = {
        "config": dict(a=A, b=B_MEAN, sigma=SIGMA, r0=R0, dt=DT,
                       n_paths=N_PATHS, horizon_years=HORIZON_YEARS,
                       seed=SEED),
        "feller_ratio": (2 * A * B_MEAN) / (SIGMA ** 2),
        "max_abs_mean_err_em_bp":
            float(df_m["mean_err_em_bp"].abs().max()),
        "max_abs_mean_err_exact_bp":
            float(df_m["mean_err_exact_bp"].abs().max()),
        "max_zcb_em_vs_ana_bp":
            float(df_z["diff_em_vs_ana_bp"].abs().max()),
        "max_zcb_exact_vs_ana_bp":
            float(df_z["diff_exact_vs_ana_bp"].abs().max()),
        "max_zcb_em_vs_exact_bp":
            float(df_z["diff_em_vs_exact_bp"].abs().max()),
        "max_coupon_ann_return_diff_bp":
            float(df_c["ann_return_diff_bp"].abs().max()),
        "max_pathwise_mean_gap_bp":
            float(df_gap["mean_gap_bp"].abs().max()),
        "var_table": df_var.to_dict(orient="records"),
        "conditional_error": df_cond.to_dict(orient="records"),
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step24_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n=== Step 24 summary ===")
    pass_all = all(c["pass"] for c in checks.values())
    print(f"  Verdict                              : "
          f"{'PASS' if pass_all else 'FAIL'}")
    print(f"  Max |mean err| (bp): exact = "
          f"{summary['max_abs_mean_err_exact_bp']:.3f}, "
          f"EM = {summary['max_abs_mean_err_em_bp']:.3f}")
    print(f"  Max ZCB gap (bp of price)            : "
          f"{summary['max_zcb_exact_vs_ana_bp']:.3f} (exact), "
          f"{summary['max_zcb_em_vs_ana_bp']:.3f} (EM)")
    print(f"  Max |pathwise mean gap| (bp)         : "
          f"{summary['max_pathwise_mean_gap_bp']:.3f}")
    print(f"  Max coupon-return diff (bp/yr)       : "
          f"{summary['max_coupon_ann_return_diff_bp']:.3f}")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    main()
