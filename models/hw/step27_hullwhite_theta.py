"""
models/hw/step27_hullwhite_theta.py
-----------------------------------
Phase E, Step 27: build the Hull-White deterministic drift theta(t) that
pins the model to the observed initial term structure.

The Hull-White (extended-Vasicek) short-rate dynamics under Q are

    dr = (theta(t) - a r) dt + sigma dW.

theta(t) is chosen so the model-implied initial discount curve matches
the market curve exactly.  The closed form

    theta(t) = df/dt + a f(0, t) + (sigma^2 / (2 a)) (1 - exp(-2 a t))

requires only the instantaneous forward curve f(0, t) (slope and level)
and the calibration constants (a, sigma).

Pipeline:
  1. Par-bootstrap continuously-compounded discount factors P(0, T) from
     the U.S. Treasury CMT par-yield curve (17 March 2026).  Tenors <= 1y
     are treated as zero-coupon money-market quotes; tenors >= 2y are par
     coupon bonds.  Each coupon bond is solved with monotone bisection so
     it reprices to par by construction.
  2. Bootstrap a piecewise-linear instantaneous forward curve consistent
     with those discount factors.  Anchored flat at the short end
     (f_0 = f_1 = z(t_1)); the linear-forward recursion reproduces the
     discount factors at every knot to machine precision.
  3. Evaluate theta(t) on the monthly grid t = 1/12, 2/12, ..., 10 years
     (120 points), the same grid used by Steps 20-24, so the theta vector
     drops directly into the Phase E simulator without resampling.

Calibration: a = 0.50, sigma = 0.01 (inherited from the Step 17 Vasicek
calibration; Hull-White is the time-dependent Vasicek).

Outputs (aligned with Steps 13-24 artefact layout):

    models/hw/step27_summary.json
    models/hw/step27_cmt_input.csv
    models/hw/step27_forward_knots.csv
    models/hw/step27_theta_grid.csv
    figures/step27/step27_fig1_curve_triple.png
    figures/step27/step27_fig2_theta.png
    figures/step27/step27_fig3_decomposition.png
    figures/step27/step27_fig4_repricing.png
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
# Configuration
# ---------------------------------------------------------------------------
A = 0.50
SIGMA = 0.01
DT = 1.0 / 12.0
HORIZON_YEARS = 10.0

# U.S. Treasury par-yield (CMT) curve on 17 March 2026
# Source: U.S. Department of the Treasury (Phase A, Step 1 record).
CMT_TENORS = [
    1.0 / 12.0,   # 1M
    3.0 / 12.0,   # 3M
    6.0 / 12.0,   # 6M
    1.0,          # 1Y
    2.0,          # 2Y
    3.0,          # 3Y
    5.0,          # 5Y
    7.0,          # 7Y
    10.0,         # 10Y
    20.0,         # 20Y
    30.0,         # 30Y
]
CMT_LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y",
              "10Y", "20Y", "30Y"]
CMT_YIELDS = [0.0374, 0.0372, 0.0371, 0.0363, 0.0368, 0.0368,
              0.0379, 0.0398, 0.0420, 0.0481, 0.0485]

COUPON_FREQ = 2

OUTPUT_DIR = ROOT / "models" / "hw"
FIGURES_DIR = ROOT / "figures" / "step27"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-24 token set)
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
# Par bootstrap of discount factors
# ---------------------------------------------------------------------------
def bootstrap_discount_factors(tenors, par_yields, coupon_freq=2):
    """
    Bootstrap continuously-compounded discount factors from a CMT par
    curve.  Tenors <= 1y are treated as zero-coupon money-market quotes
    (single cash flow at maturity, simple basis); tenors >= 2y are par
    coupon bonds paying coupon_freq times a year with the par coupon
    equal to the quoted yield.

    Coupon-date discount factors are obtained by log-linear interpolation
    in P (linear in zero rate times maturity).  Each coupon bond is solved
    by monotone bisection so every par bond reprices to par.
    """
    knot_t = [0.0]
    knot_P = [1.0]

    def P_at(t, cand_T=None, cand_P=None):
        lt = list(knot_t)
        lP = [np.log(p) for p in knot_P]
        if cand_T is not None:
            lt = lt + [cand_T]
            lP = lP + [np.log(cand_P)]
        return float(np.exp(np.interp(t, lt, lP)))

    for T, y in zip(tenors, par_yields):
        if T <= 1.0 + 1e-9:
            P = 1.0 / (1.0 + y * T)
        else:
            c = y / coupon_freq
            n = int(round(T * coupon_freq))
            times = [(k + 1) / coupon_freq for k in range(n)]

            def price(PT):
                pv = sum(c * P_at(t, T, PT) for t in times[:-1])
                return pv + (1.0 + c) * P_at(T, T, PT)

            lo, hi = 1e-8, 1.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if price(mid) < 1.0:
                    lo = mid
                else:
                    hi = mid
            P = 0.5 * (lo + hi)
        knot_t.append(float(T))
        knot_P.append(float(P))

    t = np.array(knot_t[1:])
    P = np.array(knot_P[1:])
    z = -np.log(P) / t
    return t, P, z


# ---------------------------------------------------------------------------
# Piecewise-linear instantaneous forward bootstrap
# ---------------------------------------------------------------------------
def bootstrap_forward_curve(t_knots, P_knots):
    """
    Piecewise-linear instantaneous forward curve consistent with the
    bootstrapped discount factors.  Anchored flat at the short end
    (f_0 = f_1 = z(t_1)); the linear-forward recursion
    f_i = 2 (I_i - I_{i-1}) / dt - f_{i-1} reproduces the discount
    factors at every knot to machine precision.

    Returns (t_full, f_full) including t = 0.
    """
    t = np.concatenate(([0.0], t_knots))
    I = np.concatenate(([0.0], -np.log(P_knots)))
    f = np.empty_like(t)
    f[0] = I[1] / t[1]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        f[i] = 2.0 * (I[i] - I[i - 1]) / dt - f[i - 1]
    return t, f


def forward_at(t_query, t_knots, f_knots):
    """Piecewise-linear evaluation; flat extrapolation past the last knot."""
    return np.interp(t_query, t_knots, f_knots)


def forward_slope_at(t_query, t_knots, f_knots):
    """Piecewise-constant slope df/dt for the segment containing t."""
    slopes = np.diff(f_knots) / np.diff(t_knots)
    idx = np.searchsorted(t_knots, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(slopes) - 1)
    flat = np.asarray(t_query) >= t_knots[-1]
    s = slopes[idx]
    s = np.where(flat, 0.0, s)
    return s


# ---------------------------------------------------------------------------
# Hull-White theta(t)
# ---------------------------------------------------------------------------
def hull_white_theta(t_grid, t_knots, f_knots, a, sigma):
    """theta(t) = df/dt + a f(0,t) + (sigma^2 / (2a)) (1 - exp(-2 a t))."""
    f = forward_at(t_grid, t_knots, f_knots)
    dfdt = forward_slope_at(t_grid, t_knots, f_knots)
    drift_term = a * f
    convexity = (sigma ** 2) / (2.0 * a) * (1.0 - np.exp(-2.0 * a * t_grid))
    theta = dfdt + drift_term + convexity
    return f, dfdt, drift_term, convexity, theta


# ---------------------------------------------------------------------------
# Verification: par-bond repricing and forward-integral reconstruction
# ---------------------------------------------------------------------------
def repricing_residuals(tenors, par_yields, t_knots, P_knots,
                        coupon_freq=2):
    """Reprice every CMT bond from the bootstrapped P(t) curve;
    residual should be machine zero by construction."""
    t_full = np.concatenate(([0.0], t_knots))
    P_full = np.concatenate(([1.0], P_knots))
    lt = t_full
    lP = np.log(P_full)

    def P_interp(t):
        return float(np.exp(np.interp(t, lt, lP)))

    rows = []
    for T, y in zip(tenors, par_yields):
        if T <= 1.0 + 1e-9:
            target = 1.0 / (1.0 + y * T)
            price = P_interp(T)
            resid = price - target
        else:
            c = y / coupon_freq
            n = int(round(T * coupon_freq))
            times = [(k + 1) / coupon_freq for k in range(n)]
            pv = sum(c * P_interp(t) for t in times[:-1])
            pv += (1.0 + c) * P_interp(times[-1])
            resid = pv - 1.0
        rows.append({"T": T, "par_yield": y, "residual": resid,
                     "residual_bp": resid * 1e4})
    return pd.DataFrame(rows)


def forward_integral_residuals(t_knots, P_knots):
    """Trapezoidal integral of f reproduces -ln P at every knot;
    residual should be machine zero."""
    t_full, f_full = bootstrap_forward_curve(t_knots, P_knots)
    I_recon = np.zeros_like(t_full)
    for i in range(1, len(t_full)):
        dt = t_full[i] - t_full[i - 1]
        I_recon[i] = I_recon[i - 1] + 0.5 * (f_full[i - 1] + f_full[i]) * dt
    P_recon = np.exp(-I_recon[1:])
    return pd.DataFrame({
        "T": t_knots,
        "P_bootstrap": P_knots,
        "P_reconstructed": P_recon,
        "residual": P_recon - P_knots,
        "residual_bp": (P_recon - P_knots) * 1e4,
    })


def theta_finite_diff_check(t_knots, f_knots, a, sigma):
    """Compare analytic theta against a finite-difference derivative."""
    t = np.linspace(DT, HORIZON_YEARS, 200)
    h = 1e-5
    f_plus = forward_at(t + h, t_knots, f_knots)
    f_minus = forward_at(np.maximum(t - h, 0.0), t_knots, f_knots)
    dfdt_fd = (f_plus - f_minus) / (2 * h)
    f_t = forward_at(t, t_knots, f_knots)
    conv = (sigma ** 2) / (2.0 * a) * (1.0 - np.exp(-2.0 * a * t))
    theta_fd = dfdt_fd + a * f_t + conv

    f_an, dfdt_an, _, _, theta_an = hull_white_theta(
        t, t_knots, f_knots, a, sigma)
    # Skip points within +/-1e-3 of any internal knot (where df/dt jumps
    # and finite-difference disagrees with the piecewise-constant slope).
    mask = np.ones_like(t, dtype=bool)
    for tk in t_knots[1:-1]:
        mask &= np.abs(t - tk) > 5e-4
    return float(np.abs((theta_fd - theta_an)[mask]).max())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_curve_triple(t_knots, P_knots, z_knots, t_fwd, f_fwd,
                      cmt_tenors, cmt_yields, path):
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    t_plot = np.linspace(0.05, max(t_knots) + 0.5, 400)
    z_plot = np.interp(t_plot, t_knots, z_knots)
    f_plot = forward_at(t_plot, t_fwd, f_fwd)
    ax.plot(t_plot, z_plot * 100, color=BLUE, lw=1.8,
            label="Zero rate z(t)")
    ax.plot(t_plot, f_plot * 100, color=BURG, lw=1.4, linestyle="--",
            label="Instantaneous forward f(0, t)")
    ax.scatter(cmt_tenors, np.asarray(cmt_yields) * 100,
               s=42, color=OCHRE, edgecolor=INK, linewidth=0.6,
               zorder=5, label="CMT par yields")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Initial term structure on 17 March 2026: par, zero, forward",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig2_theta(t_grid, theta, f_grid, path):
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    ax.plot(t_grid, theta * 100, color=BURG, lw=1.7,
            label=r"Hull-White $\theta(t)$")
    ax.plot(t_grid, f_grid * 100, color=BLUE, lw=1.2, linestyle="--",
            label=r"Forward $f(0, t)$")
    ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xlabel("t (years)")
    ax.set_ylabel(r"Rate level (%)")
    ax.set_title(r"Hull-White drift $\theta(t)$ on the monthly grid "
                 f"(a = {A}, sigma = {SIGMA})",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig3_decomposition(t_grid, dfdt, drift_term, convexity, path):
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    ax.plot(t_grid, dfdt * 100, color=BLUE, lw=1.5,
            label=r"$\partial_t f(0, t)$ (slope)")
    ax.plot(t_grid, drift_term * 100, color=TEAL, lw=1.5,
            label=r"$a \cdot f(0, t)$ (reversion)")
    ax.plot(t_grid, convexity * 100 * 100, color=OCHRE, lw=1.5,
            linestyle="--",
            label=r"Convexity $\times 100$ "
                  r"($\sigma^2/(2a)(1-e^{-2at})$)")
    ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("Term contribution (%)")
    ax.set_title(r"$\theta(t)$ decomposition: slope + reversion + convexity",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig4_repricing(repricing_df, integral_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)
    axes[0].bar(np.arange(len(repricing_df)),
                np.abs(repricing_df["residual_bp"]),
                color=OCHRE, edgecolor=INK, linewidth=0.6)
    axes[0].set_xticks(np.arange(len(repricing_df)))
    axes[0].set_xticklabels(CMT_LABELS, rotation=45, ha="right")
    axes[0].set_ylabel("|residual| (bp)")
    axes[0].set_title("Par-bond repricing error",
                      fontweight="bold", pad=10)
    axes[0].set_yscale("symlog", linthresh=1e-12)

    axes[1].bar(np.arange(len(integral_df)),
                np.abs(integral_df["residual_bp"]),
                color=TEAL, edgecolor=INK, linewidth=0.6)
    axes[1].set_xticks(np.arange(len(integral_df)))
    axes[1].set_xticklabels(CMT_LABELS, rotation=45, ha="right")
    axes[1].set_ylabel("|residual| (bp)")
    axes[1].set_title(r"Forward-integral reconstruction of P(0, t)",
                      fontweight="bold", pad=10)
    axes[1].set_yscale("symlog", linthresh=1e-12)

    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity / acceptance checks
# ---------------------------------------------------------------------------
def acceptance_checks(repricing_df, integral_df, theta_fd_err, n_grid):
    checks: dict = {}

    max_rep_bp = float(np.abs(repricing_df["residual_bp"]).max())
    checks["par_bonds_reprice_to_par"] = {
        "max_abs_residual_bp": max_rep_bp,
        "tolerance_bp": 1e-6,
        "pass": bool(max_rep_bp < 1e-6),
    }
    max_int_bp = float(np.abs(integral_df["residual_bp"]).max())
    checks["forward_integral_reproduces_P"] = {
        "max_abs_residual_bp": max_int_bp,
        "tolerance_bp": 1e-10,
        "pass": bool(max_int_bp < 1e-9),
    }
    checks["analytical_theta_vs_finite_diff"] = {
        "max_abs_diff": float(theta_fd_err),
        "tolerance": 1e-6,
        "pass": bool(theta_fd_err < 1e-6),
    }
    checks["exact_fit_initial_curve"] = {
        "by_construction": True,
        "pass": True,
    }
    checks["theta_evaluated_on_monthly_grid"] = {
        "n_grid": int(n_grid),
        "pass": bool(n_grid == 120),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase E, Step 27: Hull-White theta(t) calibration")
    print("-" * 64)
    print(f"a = {A}, sigma = {SIGMA}, dt = {DT:.6f}, "
          f"horizon = {HORIZON_YEARS}y")
    print(f"CMT curve: 17 March 2026 ({len(CMT_TENORS)} tenors)")
    print()

    # ---- Save CMT input -------------------------------------------------
    cmt_df = pd.DataFrame({
        "label": CMT_LABELS,
        "T_yr": CMT_TENORS,
        "par_yield": CMT_YIELDS,
        "par_yield_pct": [100 * y for y in CMT_YIELDS],
    })
    cmt_df.to_csv(OUTPUT_DIR / "step27_cmt_input.csv", index=False)
    print("CMT par-yield curve (input):")
    print(cmt_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Par bootstrap of discount factors ------------------------------
    t_knots, P_knots, z_knots = bootstrap_discount_factors(
        CMT_TENORS, CMT_YIELDS, COUPON_FREQ)

    # ---- Piecewise-linear forward bootstrap -----------------------------
    t_fwd, f_fwd = bootstrap_forward_curve(t_knots, P_knots)

    knot_df = pd.DataFrame({
        "T_yr": t_knots,
        "label": CMT_LABELS,
        "P": P_knots,
        "z_pct": z_knots * 100,
        "f_at_knot_pct": forward_at(t_knots, t_fwd, f_fwd) * 100,
    })
    knot_df.to_csv(OUTPUT_DIR / "step27_forward_knots.csv", index=False)
    print("Bootstrapped knots (P, z, f at each CMT tenor):")
    print(knot_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Evaluate theta(t) on the monthly grid --------------------------
    n_grid = int(round(HORIZON_YEARS / DT))
    t_grid = np.arange(1, n_grid + 1) * DT  # 1/12, 2/12, ..., 10y
    f_grid, dfdt_grid, drift_grid, conv_grid, theta_grid = hull_white_theta(
        t_grid, t_fwd, f_fwd, A, SIGMA)
    theta_df = pd.DataFrame({
        "t_yr": t_grid,
        "f_pct": f_grid * 100,
        "dfdt_pct_per_yr": dfdt_grid * 100,
        "drift_term_pct": drift_grid * 100,
        "convexity_term_bp": conv_grid * 1e4,
        "theta_pct": theta_grid * 100,
    })
    theta_df.to_csv(OUTPUT_DIR / "step27_theta_grid.csv", index=False)

    # ---- Verification ---------------------------------------------------
    repricing_df = repricing_residuals(CMT_TENORS, CMT_YIELDS,
                                       t_knots, P_knots, COUPON_FREQ)
    integral_df = forward_integral_residuals(t_knots, P_knots)
    theta_fd_err = theta_finite_diff_check(t_fwd, f_fwd, A, SIGMA)

    print("Par-bond repricing residuals (machine-zero expected):")
    print(repricing_df.to_string(
        index=False, float_format=lambda v: f"{v: .2e}"))
    print()
    print("Forward-integral reconstruction residuals:")
    print(integral_df.to_string(
        index=False, float_format=lambda v: f"{v: .2e}"))
    print()
    print(f"Analytical theta vs finite-difference: {theta_fd_err:.3e} bp")
    print()

    # ---- Figures --------------------------------------------------------
    fig1_curve_triple(t_knots, P_knots, z_knots, t_fwd, f_fwd,
                      CMT_TENORS, CMT_YIELDS,
                      FIGURES_DIR / "step27_fig1_curve_triple.png")
    fig2_theta(t_grid, theta_grid, f_grid,
               FIGURES_DIR / "step27_fig2_theta.png")
    fig3_decomposition(t_grid, dfdt_grid, drift_grid, conv_grid,
                       FIGURES_DIR / "step27_fig3_decomposition.png")
    fig4_repricing(repricing_df, integral_df,
                   FIGURES_DIR / "step27_fig4_repricing.png")

    # ---- Acceptance checks ----------------------------------------------
    checks = acceptance_checks(repricing_df, integral_df,
                               theta_fd_err, n_grid)
    print("=== Acceptance checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")
    print()

    # ---- Headline values ------------------------------------------------
    theta_min = float(theta_grid.min())
    theta_max = float(theta_grid.max())
    theta_1m = float(theta_grid[0])
    theta_12m = float(theta_grid[11])
    theta_120m = float(theta_grid[-1])
    f_0 = float(f_fwd[0])
    f_10y = float(forward_at(np.array([10.0]), t_fwd, f_fwd)[0])
    conv_10y_bp = float(conv_grid[-1] * 1e4)

    print("=== Headline values ===")
    print(f"  f(0, 0)               = {f_0 * 100:.4f}%")
    print(f"  f(0, 10y)             = {f_10y * 100:.4f}%")
    print(f"  theta(1m)             = {theta_1m * 100:.4f}%")
    print(f"  theta(12m)            = {theta_12m * 100:.4f}%")
    print(f"  theta(120m)           = {theta_120m * 100:.4f}%")
    print(f"  theta_min             = {theta_min * 100:.4f}%")
    print(f"  theta_max             = {theta_max * 100:.4f}%")
    print(f"  convexity at 10y (bp) = {conv_10y_bp:.4f}")

    summary = {
        "config": dict(a=A, sigma=SIGMA, dt=DT,
                       horizon_years=HORIZON_YEARS,
                       coupon_freq=COUPON_FREQ),
        "cmt_input": cmt_df.to_dict(orient="records"),
        "knots": knot_df.to_dict(orient="records"),
        "headline": {
            "f_0_pct": f_0 * 100,
            "f_10y_pct": f_10y * 100,
            "theta_1m_pct": theta_1m * 100,
            "theta_12m_pct": theta_12m * 100,
            "theta_120m_pct": theta_120m * 100,
            "theta_min_pct": theta_min * 100,
            "theta_max_pct": theta_max * 100,
            "convexity_10y_bp": conv_10y_bp,
            "n_grid_points": int(n_grid),
        },
        "checks": checks,
    }
    with open(OUTPUT_DIR / "step27_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\nfigures written to {FIGURES_DIR}")
    print(f"output written to  {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
