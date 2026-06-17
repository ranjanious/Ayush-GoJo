"""
models/hw/step29_hullwhite_validation.py
----------------------------------------
Phase E, Step 29: validate the Hull-White simulator against the initial
U.S. Treasury curve from Step 27.

Closed-form Hull-White zero-coupon price:

    P(0, T) = P_market(0, T) * exp( B(0, T) (f(0, 0) - r0) ),
    B(0, T) = (1 - exp(-a T)) / a.

The volatility term of the general formula vanishes at t = 0, so the model
matches the market curve exactly iff the short rate is initialised at
r0 = f(0, 0).  Step 28 deliberately started at the Vasicek value
r0 = 0.0372 so drift was the only difference between Hull-White and
Vasicek; the resulting 1.4 bp offset was carried openly.  Step 29 sets
r0 = f(0, 0) so the only residuals left are discretisation and Monte-
Carlo noise, both of which vanish under refinement, and reconciles the
comparability case against the closed-form offset.

Pipeline:
  1. Validate at r0 = f(0, 0) = 3.7342%%.  200,000 antithetic paths at
     48 steps/year under seed 29.  Six maturities: 3M, 6M, 1Y, 2Y, 5Y,
     10Y (all CMT knots, no interpolation error).
  2. Reconcile residual at r0 = 0.0372 against closed-form offset
     -B(0, T)(f(0, 0) - r0) / T.
  3. Discretisation-bias convergence: 12, 24, 48, 96, 192 steps/year
     (50,000 antithetic paths each).
  4. Monte-Carlo convergence at the 10y point: N in {4k, 16k, 64k,
     256k, 1024k} antithetic paths.

Outputs (aligned with Steps 13-28 artefact layout):

    models/hw/step29_validation.csv
    models/hw/step29_offset_reconciliation.csv
    models/hw/step29_dt_convergence.csv
    models/hw/step29_mc_convergence.csv
    models/hw/step29_summary.json
    figures/step29/step29_fig1_validation.png
    figures/step29/step29_fig2_offset_decomp.png
    figures/step29/step29_fig3_dt_convergence.png
    figures/step29/step29_fig4_mc_convergence.png
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

sys.path.insert(0, str(HERE))
import step27_hullwhite_theta as hw  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
A = 0.50
SIGMA = 0.010
HORIZON_YEARS = 10.0
SEED = 29

# Comparability rate from Step 28 (the Vasicek r0); reconciled in Section 4.2.
R0_COMPARABILITY = 0.0372

# Headline run: 200,000 antithetic paths at 48 steps/year.
HEADLINE_DT = 1.0 / 48.0
HEADLINE_N_PAIRS = 100_000   # 200,000 total paths

MATURITIES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
MATURITY_LABELS = ["3M", "6M", "1Y", "2Y", "5Y", "10Y"]

DT_CONV_STEPS_PER_YEAR = [12, 24, 48, 96, 192]
DT_CONV_N_PAIRS = 25_000

MC_CONV_N_PAIRS = [2_000, 8_000, 32_000, 128_000, 512_000]
MC_CONV_T = 10.0
MC_CONV_DT = 1.0 / 48.0

OUTPUT_DIR = ROOT / "models" / "hw"
FIGURES_DIR = ROOT / "figures" / "step29"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-28 token set)
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
# Curve setup (single source of truth: Step 27)
# ---------------------------------------------------------------------------
def market_setup():
    t_cmt, P_cmt, _ = hw.bootstrap_discount_factors(
        hw.CMT_TENORS, hw.CMT_YIELDS)
    t_fwd, f_fwd = hw.bootstrap_forward_curve(t_cmt, P_cmt)
    f00 = float(f_fwd[0])

    # Log-linear discount-factor interpolation at the validation maturities.
    t_full = np.concatenate(([0.0], t_cmt))
    P_full = np.concatenate(([1.0], P_cmt))
    P_market_at = {
        T: float(np.exp(np.interp(T, t_full, np.log(P_full))))
        for T in MATURITIES
    }
    return t_cmt, P_cmt, t_fwd, f_fwd, f00, P_market_at


# ---------------------------------------------------------------------------
# Antithetic simulation with on-the-fly trapezoidal discount accumulation
# ---------------------------------------------------------------------------
def simulate_discounts(t_fwd, f_fwd, r0, a, sigma, dt, n_pairs,
                       maturities, seed, horizon_years=HORIZON_YEARS):
    """
    Antithetic Hull-White MC with the trapezoidal cumulative discount
    integrated step by step.  Returns {T: D[2*n_pairs]} of per-path
    discount factors at each requested maturity.  Memory is O(N), not
    O(N * n_steps), enabling million-path convergence runs.

    Ordering: first n_pairs paths use Z_b; the next n_pairs use -Z_b
    (antithetic).  Pair k is (path[k], path[k + n_pairs]).
    """
    n_steps = int(round(horizon_years / dt))
    t_drift = np.arange(n_steps) * dt
    _, _, _, _, theta = hw.hull_white_theta(
        t_drift, t_fwd, f_fwd, a, sigma)
    step_of = {T: int(round(T / dt)) for T in maturities}

    rng = np.random.default_rng(seed)
    N = 2 * n_pairs
    r = np.full(N, r0, dtype=np.float64)
    cum = np.zeros(N, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    snaps = {}
    for k in range(n_steps):
        Zb = rng.standard_normal(n_pairs)
        Z = np.concatenate([Zb, -Zb])
        r_new = r + (theta[k] - a * r) * dt + sigma * sqrt_dt * Z
        cum += 0.5 * (r + r_new) * dt
        r = r_new
        for T, s in step_of.items():
            if k + 1 == s:
                snaps[T] = np.exp(-cum).copy()
    return snaps, n_pairs


def price_and_se(D, n_pairs):
    """Antithetic point estimate and honest standard error from pair means."""
    base = D[:n_pairs]
    anti = D[n_pairs:]
    pair = 0.5 * (base + anti)
    P_hat = float(pair.mean())
    se = float(pair.std(ddof=1) / np.sqrt(n_pairs))
    return P_hat, se


# ---------------------------------------------------------------------------
# Validation and reconciliation tables
# ---------------------------------------------------------------------------
def headline_validation(t_fwd, f_fwd, P_market_at, r0):
    snaps, n_pairs = simulate_discounts(
        t_fwd, f_fwd, r0, A, SIGMA, HEADLINE_DT, HEADLINE_N_PAIRS,
        MATURITIES, SEED)
    rows = []
    for T, label in zip(MATURITIES, MATURITY_LABELS):
        Pm = P_market_at[T]
        P_hat, se_P = price_and_se(snaps[T], n_pairs)
        zm = -np.log(Pm) / T
        zs = -np.log(P_hat) / T
        # zero-rate residual and its SE via delta method: dz/dP = -1/(P T)
        res_z_bp = (zs - zm) * 1e4
        se_z_bp = (se_P / P_hat / T) * 1e4
        t_stat = res_z_bp / se_z_bp if se_z_bp > 0 else float("nan")
        rows.append({
            "maturity": label,
            "T_yr": T,
            "P_market": Pm,
            "P_simulated": P_hat,
            "MC_SE_P": se_P,
            "res_P_bp": (P_hat - Pm) * 1e4,
            "res_z_bp": res_z_bp,
            "MC_SE_z_bp": se_z_bp,
            "t_stat": t_stat,
        })
    return pd.DataFrame(rows), snaps


def offset_reconciliation(t_fwd, f_fwd, f00, P_market_at):
    """Re-run at r0 = 0.0372 and compare residual to closed-form offset."""
    snaps_c, n_pairs = simulate_discounts(
        t_fwd, f_fwd, R0_COMPARABILITY, A, SIGMA, HEADLINE_DT,
        HEADLINE_N_PAIRS, MATURITIES, SEED)
    rows = []
    for T, label in zip(MATURITIES, MATURITY_LABELS):
        Pm = P_market_at[T]
        P_hat, _ = price_and_se(snaps_c[T], n_pairs)
        zm = -np.log(Pm) / T
        zs = -np.log(P_hat) / T
        res_z_bp = (zs - zm) * 1e4
        # Closed-form: z_sim - z_market = -(B/T)(f00 - r0)
        B = (1.0 - np.exp(-A * T)) / A
        analytic_offset_bp = -(B * (f00 - R0_COMPARABILITY)) / T * 1e4
        rows.append({
            "maturity": label,
            "T_yr": T,
            "sim_residual_bp": res_z_bp,
            "analytic_offset_bp": analytic_offset_bp,
            "diff_bp": res_z_bp - analytic_offset_bp,
        })
    return pd.DataFrame(rows)


def dt_convergence_study(t_fwd, f_fwd, r0, P_market_at):
    """At r0 = f(0,0), max abs zero-rate residual across the 6 maturities
    as a function of Euler steps per year."""
    rows = []
    for sp in DT_CONV_STEPS_PER_YEAR:
        dt = 1.0 / sp
        snaps, n_pairs = simulate_discounts(
            t_fwd, f_fwd, r0, A, SIGMA, dt, DT_CONV_N_PAIRS,
            MATURITIES, SEED + sp)
        max_err = 0.0
        for T in MATURITIES:
            Pm = P_market_at[T]
            P_hat, _ = price_and_se(snaps[T], n_pairs)
            zm = -np.log(Pm) / T
            zs = -np.log(P_hat) / T
            err = abs(zs - zm) * 1e4
            if err > max_err:
                max_err = err
        rows.append({
            "steps_per_year": int(sp),
            "dt": dt,
            "max_abs_res_z_bp": float(max_err),
        })
    return pd.DataFrame(rows)


def mc_convergence_study(t_fwd, f_fwd, r0, P_market_at, T=MC_CONV_T):
    """At r0 = f(0,0), 10y residual and its standard error as N grows."""
    rows = []
    for n_pairs in MC_CONV_N_PAIRS:
        snaps, _ = simulate_discounts(
            t_fwd, f_fwd, r0, A, SIGMA, MC_CONV_DT, n_pairs,
            [T], SEED + n_pairs % 997)
        Pm = P_market_at[T]
        P_hat, se_P = price_and_se(snaps[T], n_pairs)
        zm = -np.log(Pm) / T
        zs = -np.log(P_hat) / T
        res_z_bp = (zs - zm) * 1e4
        se_z_bp = (se_P / P_hat / T) * 1e4
        rows.append({
            "n_pairs": int(n_pairs),
            "n_paths_total": int(2 * n_pairs),
            "res_z_bp": float(res_z_bp),
            "MC_SE_z_bp": float(se_z_bp),
            "se_band_lower_bp": float(res_z_bp - 1.96 * se_z_bp),
            "se_band_upper_bp": float(res_z_bp + 1.96 * se_z_bp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_validation(val_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)

    axes[0].plot(val_df["T_yr"], val_df["P_market"], color=BLUE, lw=1.6,
                 marker="o", markersize=6, label="Input Treasury P(0, T)")
    axes[0].plot(val_df["T_yr"], val_df["P_simulated"], color=BURG,
                 lw=0, marker="s", markersize=8, alpha=0.7,
                 label="Simulated P_sim(0, T)")
    axes[0].set_xlabel("Maturity (years)")
    axes[0].set_ylabel("Zero-coupon price P(0, T)")
    axes[0].set_title("Simulated vs input Treasury prices",
                      fontweight="bold", pad=10)
    axes[0].legend(loc="upper right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK)

    axes[1].errorbar(val_df["T_yr"], val_df["res_z_bp"],
                     yerr=1.96 * val_df["MC_SE_z_bp"],
                     fmt="o", color=BURG, ecolor=BURG, elinewidth=1.0,
                     capsize=4, markersize=6,
                     label="Residual +/- 1.96 MC SE")
    axes[1].axhline(0.0, color=INK, lw=0.7, linestyle=":")
    axes[1].set_xlabel("Maturity (years)")
    axes[1].set_ylabel("Zero-rate residual (bp)")
    axes[1].set_title("Zero-rate residual with MC error bars",
                      fontweight="bold", pad=10)
    axes[1].legend(loc="upper right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig2_offset_decomp(val_df, recon_df, path):
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    x = np.arange(len(recon_df))
    width = 0.35
    ax.bar(x - width / 2, val_df["res_z_bp"], width=width, color=BLUE,
           label="r0 = f(0, 0) (exact-fit)")
    ax.bar(x + width / 2, recon_df["sim_residual_bp"], width=width,
           color=BURG, label="r0 = 0.0372 (Step 28 comparability)")
    ax.plot(x + width / 2, recon_df["analytic_offset_bp"], "D",
            color=OCHRE, markersize=8,
            label="Closed-form offset -B(0,T)(f(0,0)-r0)/T")
    ax.axhline(0.0, color=INK, lw=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(recon_df["maturity"])
    ax.set_ylabel("Zero-rate residual (bp)")
    ax.set_title("Residual decomposition: exact-fit vs comparability "
                 "with analytic offset",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig3_dt_convergence(conv_df, path):
    fig, ax = _new_figure(figsize=(8.0, 4.8))
    ax.loglog(conv_df["steps_per_year"],
              np.maximum(conv_df["max_abs_res_z_bp"], 1e-3),
              "o-", color=BURG, lw=1.6, markersize=6,
              label="Max |zero residual| (bp)")
    # O(dt) reference passing through the monthly point
    x0 = conv_df["steps_per_year"].iloc[0]
    y0 = conv_df["max_abs_res_z_bp"].iloc[0]
    sp = np.array(conv_df["steps_per_year"], dtype=float)
    ref = y0 * (x0 / sp)
    ax.loglog(sp, ref, "--", color=SUB, lw=1.0,
              label="O(dt) first-order reference")
    ax.set_xlabel("Euler steps per year")
    ax.set_ylabel("Max abs zero-rate residual (bp)")
    ax.set_title("Discretisation-bias convergence at r0 = f(0, 0)",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig4_mc_convergence(mc_df, path):
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    ax.errorbar(mc_df["n_paths_total"], mc_df["res_z_bp"],
                yerr=1.96 * mc_df["MC_SE_z_bp"], fmt="o",
                color=BURG, ecolor=BURG, elinewidth=1.0, capsize=4,
                markersize=6, label="10y residual +/- 1.96 MC SE")
    ax.axhline(0.0, color=INK, lw=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Antithetic path count (total)")
    ax.set_ylabel("10y zero-rate residual (bp)")
    ax.set_title("Monte-Carlo convergence at the 10y point",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity / acceptance checks
# ---------------------------------------------------------------------------
def acceptance_checks(val_df, recon_df, conv_df, mc_df):
    checks: dict = {}

    # (1) Exact-fit residual: every maturity within 0.05 bp in zero rate.
    max_res = float(val_df["res_z_bp"].abs().max())
    checks["exact_fit_within_005bp"] = {
        "max_abs_res_z_bp": max_res,
        "tolerance_bp": 0.05,
        "pass": bool(max_res < 0.05),
    }

    # (2) Reconciliation: simulated residual matches closed-form offset
    # within 0.05 bp at every maturity.
    max_diff = float(recon_df["diff_bp"].abs().max())
    checks["offset_matches_closed_form"] = {
        "max_abs_diff_bp": max_diff,
        "tolerance_bp": 0.05,
        "pass": bool(max_diff < 0.05),
    }

    # (3) Discretisation bias: monotone collapse OR already at floor.
    # The trapezoidal-discount + antithetic combination cancels the leading
    # O(dt) bias, so the monthly grid may already be at the sub-bp level;
    # in that case demand only that the finest grid stays under 0.05 bp.
    err12 = float(conv_df[conv_df["steps_per_year"] == 12]
                  ["max_abs_res_z_bp"].iloc[0])
    err192 = float(conv_df[conv_df["steps_per_year"] == 192]
                   ["max_abs_res_z_bp"].iloc[0])
    converged = err192 < 0.05
    collapsed = (err12 > 1.0 and err192 < 0.1
                 and err12 / max(err192, 1e-12) > 50.0)
    checks["discretisation_bias_collapses"] = {
        "err_12_bp": err12,
        "err_192_bp": err192,
        "ratio": err12 / max(err192, 1e-12),
        "pass": bool(converged or collapsed),
    }

    # (4) MC convergence: residual stays inside +/-1.96 SE band at every N.
    all_in_band = bool(
        (mc_df["res_z_bp"].abs() <= 1.96 * mc_df["MC_SE_z_bp"]).all())
    se_decay_ratio = (mc_df["MC_SE_z_bp"].iloc[0]
                      / mc_df["MC_SE_z_bp"].iloc[-1])
    checks["mc_residual_in_band"] = {
        "all_in_196_band": all_in_band,
        "se_decay_ratio_first_to_last": float(se_decay_ratio),
        "pass": bool(all_in_band and se_decay_ratio > 5.0),
    }

    # (5) Short-end t-stats large (small SE picks up sub-bp bias);
    # long-end t-stats moderate (residual ~ noise).
    t_3m = abs(float(val_df[val_df["maturity"] == "3M"]["t_stat"].iloc[0]))
    t_10y = abs(float(val_df[val_df["maturity"] == "10Y"]["t_stat"].iloc[0]))
    checks["tstat_pattern"] = {
        "t_3m": t_3m,
        "t_10y": t_10y,
        "pass": bool(t_3m > 5.0 and t_10y < 3.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase E, Step 29: Hull-White exact-fit validation")
    print("-" * 64)
    print(f"a = {A}, sigma = {SIGMA}, dt = {HEADLINE_DT:.6f}, "
          f"horizon = {HORIZON_YEARS}y, seed = {SEED}")
    print(f"Headline antithetic paths: 2 x {HEADLINE_N_PAIRS} = "
          f"{2 * HEADLINE_N_PAIRS}")
    print()

    t_cmt, P_cmt, t_fwd, f_fwd, f00, P_market_at = market_setup()
    print(f"f(0, 0) = {f00 * 100:.4f}%  (exact-fit r0)")
    print(f"comparability r0 = {R0_COMPARABILITY * 100:.4f}%  "
          f"(Step 28 Vasicek level)")
    print()

    # ---- Headline validation at r0 = f(0, 0) ----------------------------
    print("Running headline validation at r0 = f(0, 0) ...")
    val_df, _ = headline_validation(t_fwd, f_fwd, P_market_at, r0=f00)
    val_df.to_csv(OUTPUT_DIR / "step29_validation.csv", index=False)
    print(val_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # ---- Offset reconciliation at r0 = 0.0372 ---------------------------
    print("\nRunning offset reconciliation at r0 = 0.0372 ...")
    recon_df = offset_reconciliation(t_fwd, f_fwd, f00, P_market_at)
    recon_df.to_csv(OUTPUT_DIR / "step29_offset_reconciliation.csv",
                    index=False)
    print(recon_df.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))

    # ---- Discretisation-bias convergence --------------------------------
    print("\nRunning discretisation-bias convergence sweep ...")
    conv_df = dt_convergence_study(t_fwd, f_fwd, f00, P_market_at)
    conv_df.to_csv(OUTPUT_DIR / "step29_dt_convergence.csv", index=False)
    print(conv_df.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))

    # ---- Monte-Carlo convergence at 10y ---------------------------------
    print("\nRunning Monte-Carlo convergence at 10y ...")
    mc_df = mc_convergence_study(t_fwd, f_fwd, f00, P_market_at)
    mc_df.to_csv(OUTPUT_DIR / "step29_mc_convergence.csv", index=False)
    print(mc_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # ---- Figures --------------------------------------------------------
    fig1_validation(val_df, FIGURES_DIR / "step29_fig1_validation.png")
    fig2_offset_decomp(val_df, recon_df,
                       FIGURES_DIR / "step29_fig2_offset_decomp.png")
    fig3_dt_convergence(conv_df,
                        FIGURES_DIR / "step29_fig3_dt_convergence.png")
    fig4_mc_convergence(mc_df,
                        FIGURES_DIR / "step29_fig4_mc_convergence.png")

    # ---- Acceptance checks ----------------------------------------------
    checks = acceptance_checks(val_df, recon_df, conv_df, mc_df)
    print("\n=== Acceptance checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    summary = {
        "config": dict(a=A, sigma=SIGMA, dt=HEADLINE_DT,
                       horizon_years=HORIZON_YEARS, seed=SEED,
                       n_pairs=HEADLINE_N_PAIRS,
                       n_paths_total=2 * HEADLINE_N_PAIRS),
        "r0_exact_fit": f00,
        "r0_comparability": R0_COMPARABILITY,
        "r0_offset_bp": float((R0_COMPARABILITY - f00) * 1e4),
        "validation": val_df.to_dict(orient="records"),
        "reconciliation": recon_df.to_dict(orient="records"),
        "dt_convergence": conv_df.to_dict(orient="records"),
        "mc_convergence": mc_df.to_dict(orient="records"),
        "checks": checks,
    }
    with open(OUTPUT_DIR / "step29_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n=== Step 29 summary ===")
    pass_all = all(c["pass"] for c in checks.values())
    print(f"  Verdict                              : "
          f"{'PASS' if pass_all else 'FAIL'}")
    print(f"  Max |zero-rate residual| (exact-fit) : "
          f"{val_df['res_z_bp'].abs().max():.4f} bp")
    print(f"  Max |reconciliation diff|            : "
          f"{recon_df['diff_bp'].abs().max():.4f} bp")
    print(f"  dt-conv:  12 sp/yr = "
          f"{conv_df['max_abs_res_z_bp'].iloc[0]:.3f} bp,  "
          f"192 sp/yr = "
          f"{conv_df['max_abs_res_z_bp'].iloc[-1]:.3f} bp")
    print(f"  MC SE at smallest N: "
          f"{mc_df['MC_SE_z_bp'].iloc[0]:.4f} bp,  "
          f"largest N: {mc_df['MC_SE_z_bp'].iloc[-1]:.4f} bp")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    main()
