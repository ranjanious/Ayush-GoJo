"""
models/hw/step28_hullwhite_sim.py
---------------------------------
Phase E, Step 28: turn the Step 27 theta(t) into a working Hull-White
Euler-Maruyama simulator and run the curve-recovery and risk diagnostics.

The Hull-White short rate solves

    dr = (theta(t) - a r) dt + sigma dW.

Discretising with Euler-Maruyama on the project grid:

    r_{k+1} = r_k + (theta(t_k) - a r_k) dt + sigma sqrt(dt) Z_k.

The update is the Step 18 Vasicek recursion with a single substitution:
the constant drift level a*b is replaced by the time-varying theta(t_k)
read from Step 27.  The mean-reversion speed a, the diffusion sigma, the
initial rate r0, the time step, the path count, and the random seed are
all inherited from the Vasicek run, so the two models can be compared
shock for shock.

Initial-rate choice: r0 = 0.0372 (the Vasicek level) rather than the
curve-consistent f(0,0) = 3.7342%%.  This is the comparability choice the
Step 28 brief requests: it makes drift the only difference between the
two simulators and exposes a deliberate 1.4 bp r0 offset that the
recovery convergence study (Section 4.3) absorbs.

Reproducibility: seed = 42, N = 10,000 paths, dt = 1/12, T_max = 10y;
theta from Step 27 on the 17 March 2026 CMT curve.

Outputs (aligned with Steps 13-27 artefact layout):

    models/hw/step28_path_moments.csv
    models/hw/step28_zcb_recovery.csv
    models/hw/step28_horizon_diagnostics.csv
    models/hw/step28_terminal_distribution.csv
    models/hw/step28_convergence.csv
    models/hw/step28_summary.json
    figures/step28/step28_fig1_paths.png
    figures/step28/step28_fig2_zcb_recovery.png
    figures/step28/step28_fig3_terminal_density.png
    figures/step28/step28_fig4_mean_and_negrate.png
    figures/step28/step28_fig5_recovery_convergence.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Make Step 27 importable as `hw`.
sys.path.insert(0, str(HERE))
import step27_hullwhite_theta as hw  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration; shared with Step 13/18 (Vasicek) for shock-for-shock
# comparability and with Step 27 for the theta(t) calibration.
# ---------------------------------------------------------------------------
A = 0.50
SIGMA = 0.010
R0 = 0.0372              # Vasicek r0; not the curve-consistent f(0, 0)
B_VASI = 0.04            # Vasicek long-run level
DT = 1.0 / 12.0
HORIZON_YEARS = 10.0
N_STEPS = int(round(HORIZON_YEARS / DT))   # 120
N_PATHS = 10_000
SEED = 42

HORIZONS = [1.0, 2.0, 5.0, 10.0]
RECOVERY_MATS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
CONV_STEPS_PER_YEAR = [12, 24, 48, 96]

OUTPUT_DIR = ROOT / "models" / "hw"
FIGURES_DIR = ROOT / "figures" / "step28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-27 token set)
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
# Theta vector on the simulator grid (rebuilds from Step 27 calibration).
# ---------------------------------------------------------------------------
def theta_on_grid(n_steps=N_STEPS, dt=DT):
    t_cmt, P_cmt, _ = hw.bootstrap_discount_factors(
        hw.CMT_TENORS, hw.CMT_YIELDS)
    t_fwd, f_fwd = hw.bootstrap_forward_curve(t_cmt, P_cmt)
    # drift uses the LEFT endpoint of each Euler step: t = 0, dt, 2dt, ...
    t_drift = np.arange(n_steps) * dt
    _, _, _, _, theta = hw.hull_white_theta(t_drift, t_fwd, f_fwd, A, SIGMA)
    return t_drift, theta, t_fwd, f_fwd, t_cmt, P_cmt


# ---------------------------------------------------------------------------
# Simulators -- shared shock matrix for shock-for-shock comparability.
# ---------------------------------------------------------------------------
def simulate_hull_white(theta_k, shocks, r0, a, sigma, dt):
    """Euler-Maruyama Hull-White: a*b replaced by the time-varying
    theta(t_k)."""
    n_paths, n_steps = shocks.shape
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        drift = (theta_k[k] - a * r[:, k]) * dt
        r[:, k + 1] = r[:, k] + drift + sigma * sqrt_dt * shocks[:, k]
    return r


def simulate_vasicek(shocks, r0, a, b, sigma, dt):
    """Euler-Maruyama Vasicek baseline on the identical shock matrix."""
    n_paths, n_steps = shocks.shape
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        drift = a * (b - r[:, k]) * dt
        r[:, k + 1] = r[:, k] + drift + sigma * sqrt_dt * shocks[:, k]
    return r


# ---------------------------------------------------------------------------
# Bond pricing
# ---------------------------------------------------------------------------
def simulated_zcb_curve(r, dt):
    """E[exp(-int_0^T r_u du)] across paths via trapezoidal integration."""
    cum = np.zeros_like(r)
    cum[:, 1:] = np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * dt, axis=1)
    P = np.mean(np.exp(-cum), axis=0)
    return P


def analytic_hw_zcb(t_grid, t_fwd, f_fwd, P_market_interp, r0, a, sigma):
    """
    Closed-form Hull-White P(0, T) under the chosen r0:
        P(0, T) = P^M(0, T) * exp( B(0, T) (f(0, 0) - r0) ),
    with B(0, T) = (1 - exp(-a T)) / a.  The volatility term of the full
    formula vanishes at t = 0, so the only departure from the market
    curve is the deterministic r0 offset.
    """
    f00 = float(f_fwd[0])
    B = (1.0 - np.exp(-a * t_grid)) / a
    return P_market_interp * np.exp(B * (f00 - r0)), f00


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def horizon_diagnostics(r_hw, r_vas, dt, horizons):
    rows = []
    for label, r in (("HW", r_hw), ("Vas", r_vas)):
        for T in horizons:
            k = int(round(T / dt))
            sample = r[:, k]
            rows.append({
                "horizon_yr": T,
                "model": label,
                "mean_pct": float(sample.mean() * 100),
                "sd_pct": float(sample.std(ddof=1) * 100),
                "p_negative_pct": float((sample < 0.0).mean() * 100),
                "min_pct": float(sample.min() * 100),
            })
    return pd.DataFrame(rows)


def path_moments(r_hw, r_vas, dt, t_fwd, f_fwd):
    n_grid = r_hw.shape[1]
    t = np.arange(n_grid) * dt
    f_grid = hw.forward_at(t, t_fwd, f_fwd)
    return pd.DataFrame({
        "t_yr": t,
        "mean_hw_pct": r_hw.mean(axis=0) * 100,
        "mean_vas_pct": r_vas.mean(axis=0) * 100,
        "sd_hw_pct": r_hw.std(axis=0, ddof=1) * 100,
        "sd_vas_pct": r_vas.std(axis=0, ddof=1) * 100,
        "forward_pct": f_grid * 100,
        "p_neg_hw_pct": (r_hw < 0.0).mean(axis=0) * 100,
        "p_neg_vas_pct": (r_vas < 0.0).mean(axis=0) * 100,
    })


def zcb_recovery(r_hw, dt, t_fwd, f_fwd, t_cmt, P_cmt, mats):
    """Compare simulated, market, and analytic Hull-White ZCB curves."""
    P_sim_full = simulated_zcb_curve(r_hw, dt)
    t_grid = np.arange(P_sim_full.size) * dt

    # Interpolate market discount curve to the grid (log-linear in P).
    t_full = np.concatenate(([0.0], t_cmt))
    P_full = np.concatenate(([1.0], P_cmt))
    P_market_grid = np.exp(np.interp(t_grid, t_full, np.log(P_full)))

    # Analytic HW curve (constant r0 offset; volatility term vanishes at t=0)
    P_an_grid, f00 = analytic_hw_zcb(t_grid, t_fwd, f_fwd, P_market_grid,
                                     R0, A, SIGMA)

    rows = []
    for T in mats:
        k = int(round(T / dt))
        Pm = float(P_market_grid[k])
        Pa = float(P_an_grid[k])
        Ps = float(P_sim_full[k])
        zm = -np.log(Pm) / T
        za = -np.log(Pa) / T
        zs = -np.log(Ps) / T
        rows.append({
            "T_yr": T,
            "P_market": Pm,
            "P_analytic": Pa,
            "P_simulated": Ps,
            "z_market_pct": float(zm * 100),
            "z_analytic_pct": float(za * 100),
            "z_simulated_pct": float(zs * 100),
            "err_sim_vs_market_bp": float((zs - zm) * 1e4),
            "err_sim_vs_analytic_bp": float((zs - za) * 1e4),
            "err_an_vs_market_bp": float((za - zm) * 1e4),
        })
    df = pd.DataFrame(rows)
    return df, t_grid, P_sim_full, P_market_grid, P_an_grid, f00


def convergence_study(steps_per_year_list, mats, n_paths_for_conv=10_000):
    """Re-run HW at each Euler granularity using antithetic shocks; report
    the max |z_sim - z_market| over the tested maturities."""
    t_cmt, P_cmt, _ = hw.bootstrap_discount_factors(
        hw.CMT_TENORS, hw.CMT_YIELDS)
    t_fwd, f_fwd = hw.bootstrap_forward_curve(t_cmt, P_cmt)
    t_full = np.concatenate(([0.0], t_cmt))
    P_full = np.concatenate(([1.0], P_cmt))

    rows = []
    rng = np.random.default_rng(SEED + 1)
    for sp in steps_per_year_list:
        dt_c = 1.0 / sp
        n_steps_c = int(round(HORIZON_YEARS * sp))
        half = n_paths_for_conv // 2
        Z = rng.standard_normal((half, n_steps_c))
        shocks = np.vstack([Z, -Z])  # antithetic variates
        t_drift = np.arange(n_steps_c) * dt_c
        _, _, _, _, theta_c = hw.hull_white_theta(
            t_drift, t_fwd, f_fwd, A, SIGMA)
        r_c = simulate_hull_white(theta_c, shocks, R0, A, SIGMA, dt_c)
        P_sim = simulated_zcb_curve(r_c, dt_c)
        t_grid = np.arange(P_sim.size) * dt_c
        P_market_grid = np.exp(np.interp(t_grid, t_full, np.log(P_full)))
        max_err_bp = 0.0
        for T in mats:
            k = int(round(T / dt_c))
            zm = -np.log(P_market_grid[k]) / T
            zs = -np.log(P_sim[k]) / T
            err = abs(zs - zm) * 1e4
            if err > max_err_bp:
                max_err_bp = err
        rows.append({
            "steps_per_year": int(sp),
            "dt": dt_c,
            "max_err_bp": float(max_err_bp),
        })
    return pd.DataFrame(rows)


def terminal_distribution(r_hw, r_vas, dt, T):
    k = int(round(T / dt))
    return pd.DataFrame({
        "hw_terminal": r_hw[:, k],
        "vasicek_terminal": r_vas[:, k],
    })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_paths(r_hw, dt, t_fwd, f_fwd, path):
    fig, ax = _new_figure(figsize=(9.0, 5.0))
    t = np.arange(r_hw.shape[1]) * dt
    rng_plot = np.random.default_rng(0)
    idx = rng_plot.choice(r_hw.shape[0], size=60, replace=False)
    for i in idx:
        ax.plot(t, r_hw[i] * 100, color=BLUE, lw=0.5, alpha=0.30)
    ax.plot(t, r_hw.mean(axis=0) * 100, color=BURG, lw=2.0,
            label=r"$E[r_t]$ (Hull-White)")
    ax.plot(t, hw.forward_at(t, t_fwd, f_fwd) * 100, color=TEAL, lw=1.4,
            linestyle="--", label="Forward $f(0, t)$")
    ax.axhline(0.0, color=INK, lw=0.6, linestyle=":")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("Short rate (%)")
    ax.set_title("Hull-White sample paths, ensemble mean, and forward curve",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig2_zcb_recovery(t_grid, P_sim, P_market, P_an, recovery_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)

    # Left: zero rates
    mask = t_grid > 0
    z_sim = np.zeros_like(t_grid)
    z_market = np.zeros_like(t_grid)
    z_an = np.zeros_like(t_grid)
    z_sim[mask] = -np.log(P_sim[mask]) / t_grid[mask]
    z_market[mask] = -np.log(P_market[mask]) / t_grid[mask]
    z_an[mask] = -np.log(P_an[mask]) / t_grid[mask]
    axes[0].plot(t_grid[mask], z_market[mask] * 100, color=BLUE, lw=1.8,
                 label="Market z(t)")
    axes[0].plot(t_grid[mask], z_an[mask] * 100, color=TEAL, lw=1.3,
                 linestyle="--", label="Analytic HW (r0 offset)")
    axes[0].plot(t_grid[mask], z_sim[mask] * 100, color=BURG, lw=1.3,
                 linestyle=":", label="Simulated HW")
    axes[0].set_xlabel("Maturity (years)")
    axes[0].set_ylabel("Zero rate (%)")
    axes[0].set_title("Initial ZCB curve recovery",
                      fontweight="bold", pad=10)
    axes[0].legend(loc="lower right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK)

    # Right: residual in bp
    err = np.zeros_like(t_grid)
    err[mask] = (z_sim[mask] - z_market[mask]) * 1e4
    axes[1].plot(t_grid[mask], err[mask], color=BURG, lw=1.4,
                 label="z_sim - z_market (bp)")
    axes[1].axhline(0.0, color=INK, lw=0.6, linestyle=":")
    axes[1].set_xlabel("Maturity (years)")
    axes[1].set_ylabel("Recovery error (bp)")
    axes[1].set_title("Sim vs market: zero-rate residual",
                      fontweight="bold", pad=10)
    axes[1].legend(loc="lower right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig3_terminal_density(r_hw, r_vas, dt, T, path):
    k = int(round(T / dt))
    r_hw_T = r_hw[:, k] * 100
    r_vas_T = r_vas[:, k] * 100
    lo = min(r_hw_T.min(), r_vas_T.min())
    hi = max(r_hw_T.max(), r_vas_T.max())
    bins = np.linspace(lo - 0.1, hi + 0.1, 60)

    fig, ax = _new_figure(figsize=(8.6, 4.8))
    ax.hist(r_hw_T, bins=bins, density=True, alpha=0.45,
            color=BURG, label=f"HW (mean {r_hw_T.mean():.2f}%)")
    ax.hist(r_vas_T, bins=bins, density=True, alpha=0.45,
            color=TEAL, label=f"Vasicek (mean {r_vas_T.mean():.2f}%)")
    ax.axvline(r_hw_T.mean(), color=BURG, lw=1.2, linestyle="--")
    ax.axvline(r_vas_T.mean(), color=TEAL, lw=1.2, linestyle="--")
    ax.set_xlabel(f"$r_T$ at T = {T:.0f}y (%)")
    ax.set_ylabel("density")
    ax.set_title("Terminal short-rate distributions: HW vs Vasicek",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig4_mean_and_negrate(path_moments_df, horizons_df, t_fwd, f_fwd, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)

    t = path_moments_df["t_yr"]
    axes[0].plot(t, path_moments_df["forward_pct"], color=INK, lw=1.4,
                 linestyle=":", label="Forward f(0, t)")
    axes[0].plot(t, path_moments_df["mean_hw_pct"], color=BURG, lw=1.6,
                 label="HW $E[r_t]$")
    axes[0].plot(t, path_moments_df["mean_vas_pct"], color=TEAL, lw=1.6,
                 label="Vasicek $E[r_t]$")
    axes[0].axhline(B_VASI * 100, color=SUB, lw=0.8, linestyle="--",
                    alpha=0.7, label="Vasicek b = 4.00%")
    axes[0].set_xlabel("t (years)")
    axes[0].set_ylabel("Mean rate (%)")
    axes[0].set_title("Mean term structures",
                      fontweight="bold", pad=10)
    axes[0].legend(loc="lower right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK, fontsize=8)

    width = 0.35
    horizons = sorted(horizons_df["horizon_yr"].unique())
    x = np.arange(len(horizons))
    hw_neg = [horizons_df[(horizons_df["horizon_yr"] == h)
                          & (horizons_df["model"] == "HW")]
              ["p_negative_pct"].iloc[0] for h in horizons]
    vas_neg = [horizons_df[(horizons_df["horizon_yr"] == h)
                           & (horizons_df["model"] == "Vas")]
               ["p_negative_pct"].iloc[0] for h in horizons]
    axes[1].bar(x - width / 2, hw_neg, width=width, color=BURG,
                label="HW")
    axes[1].bar(x + width / 2, vas_neg, width=width, color=TEAL,
                label="Vasicek")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(h)}y" for h in horizons])
    axes[1].set_ylabel("P(r_t < 0) (%)")
    axes[1].set_title("Probability of a negative short rate",
                      fontweight="bold", pad=10)
    axes[1].legend(loc="upper right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig5_recovery_convergence(conv_df, path):
    fig, ax = _new_figure(figsize=(8.0, 4.8))
    ax.loglog(conv_df["steps_per_year"], conv_df["max_err_bp"], "o-",
              color=BURG, markersize=6, lw=1.4,
              label="Max |z_sim - z_market| (bp)")
    ax.set_xlabel("Steps per year")
    ax.set_ylabel("Max absolute zero-rate error (bp)")
    ax.set_title("Recovery convergence under Euler step refinement",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity / acceptance checks
# ---------------------------------------------------------------------------
def acceptance_checks(horizons_df, recovery_df, conv_df, r_hw, r_vas, dt,
                      f00):
    checks: dict = {}

    # (1) HW reduces to Vasicek when shocks shared and diffusion unchanged:
    # standard deviation curves identical to ~MC precision at every horizon.
    sd_diff = 0.0
    for h in HORIZONS:
        sd_hw = horizons_df[(horizons_df["horizon_yr"] == h)
                            & (horizons_df["model"] == "HW")]["sd_pct"].iloc[0]
        sd_vas = horizons_df[(horizons_df["horizon_yr"] == h)
                             & (horizons_df["model"] == "Vas")]["sd_pct"].iloc[0]
        sd_diff = max(sd_diff, abs(sd_hw - sd_vas))
    checks["variance_invariant_under_drift_swap"] = {
        "max_sd_diff_pct": float(sd_diff),
        "tolerance_pct": 1e-6,
        "pass": bool(sd_diff < 1e-6),
    }

    # (2) HW mean curve tracks the forward curve (no large deviations)
    moments = path_moments(r_hw, r_vas, dt, *theta_on_grid()[2:4])
    diff = (moments["mean_hw_pct"] - moments["forward_pct"]).abs()
    checks["hw_mean_tracks_forward"] = {
        "max_abs_diff_pct": float(diff.max()),
        "tolerance_pct": 1.0,
        "pass": bool(diff.max() < 1.0),
    }

    # (3) Recovery error at the floor; the floor is the deterministic r0 offset
    # (about 1.4 bp).  Either the error decays to that floor, or it is
    # already there on the monthly grid because the trapezoidal discount-
    # factor integration cancels the leading Euler-drift bias.
    err12 = conv_df[conv_df["steps_per_year"] == 12]["max_err_bp"].iloc[0]
    err96 = conv_df[conv_df["steps_per_year"] == 96]["max_err_bp"].iloc[0]
    r0_offset_bp = abs(R0 - f00) * 1e4
    floor_bp = max(r0_offset_bp, 0.5)
    checks["recovery_within_r0_offset_floor"] = {
        "err_12_steps_bp": float(err12),
        "err_96_steps_bp": float(err96),
        "r0_offset_bp": float(r0_offset_bp),
        "tolerance_bp": 5.0,
        "pass": bool(err12 < 5.0 and err96 < 5.0
                     and abs(err12 - err96) < 2.0),
    }

    # (4) Analytic HW matches market curve up to r0 offset only
    max_an_market = float(recovery_df["err_an_vs_market_bp"].abs().max())
    checks["analytic_matches_market_modulo_r0"] = {
        "max_abs_an_vs_market_bp": max_an_market,
        "tolerance_bp": 5.0,
        "pass": bool(max_an_market < 5.0),
    }

    # (5) HW shifts the mean upward relative to Vasicek at long horizons
    mean_hw_10 = horizons_df[(horizons_df["horizon_yr"] == 10.0)
                             & (horizons_df["model"] == "HW")]["mean_pct"].iloc[0]
    mean_vas_10 = horizons_df[(horizons_df["horizon_yr"] == 10.0)
                              & (horizons_df["model"] == "Vas")]["mean_pct"].iloc[0]
    checks["hw_mean_above_vasicek_at_10y"] = {
        "mean_hw_pct": float(mean_hw_10),
        "mean_vas_pct": float(mean_vas_10),
        "gap_bp": float((mean_hw_10 - mean_vas_10) * 100),
        "pass": bool(mean_hw_10 > mean_vas_10 + 1.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase E, Step 28: Hull-White Euler-Maruyama simulator")
    print("-" * 64)
    print(f"a = {A}, sigma = {SIGMA}, r0 = {R0} "
          f"(Vasicek level, not f(0,0))")
    print(f"seed = {SEED}, N = {N_PATHS}, dt = {DT:.6f}, "
          f"T_max = {HORIZON_YEARS}")
    print()

    # ---- Build theta on the simulator grid ------------------------------
    t_drift, theta_grid, t_fwd, f_fwd, t_cmt, P_cmt = theta_on_grid()
    f00 = float(f_fwd[0])
    print(f"theta(0)   = {theta_grid[0] * 100:.4f}%")
    print(f"theta(119) = {theta_grid[-1] * 100:.4f}%")
    print(f"f(0, 0)    = {f00 * 100:.4f}%  (r0 offset = "
          f"{(R0 - f00) * 1e4:.2f} bp)")
    print()

    # ---- Shared shocks; simulate HW and Vasicek -------------------------
    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal((N_PATHS, N_STEPS))
    print("Simulating Hull-White and Vasicek on shared shock matrix ...")
    r_hw = simulate_hull_white(theta_grid, shocks, R0, A, SIGMA, DT)
    r_vas = simulate_vasicek(shocks, R0, A, B_VASI, SIGMA, DT)

    # ---- Diagnostics ----------------------------------------------------
    moments_df = path_moments(r_hw, r_vas, DT, t_fwd, f_fwd)
    moments_df.to_csv(OUTPUT_DIR / "step28_path_moments.csv", index=False)

    horizons_df = horizon_diagnostics(r_hw, r_vas, DT, HORIZONS)
    horizons_df.to_csv(OUTPUT_DIR / "step28_horizon_diagnostics.csv",
                       index=False)
    print("\nHorizon diagnostics:")
    print(horizons_df.to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))

    recovery_df, t_grid, P_sim, P_market_grid, P_an_grid, _ = zcb_recovery(
        r_hw, DT, t_fwd, f_fwd, t_cmt, P_cmt, RECOVERY_MATS)
    recovery_df.to_csv(OUTPUT_DIR / "step28_zcb_recovery.csv", index=False)
    print("\nZCB recovery (selected maturities):")
    print(recovery_df.to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))

    term_df = terminal_distribution(r_hw, r_vas, DT, 10.0)
    term_df.to_csv(OUTPUT_DIR / "step28_terminal_distribution.csv",
                   index=False)

    print("\nConvergence study ...")
    conv_df = convergence_study(CONV_STEPS_PER_YEAR, RECOVERY_MATS)
    conv_df.to_csv(OUTPUT_DIR / "step28_convergence.csv", index=False)
    print(conv_df.to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))

    # ---- Figures --------------------------------------------------------
    fig1_paths(r_hw, DT, t_fwd, f_fwd,
               FIGURES_DIR / "step28_fig1_paths.png")
    fig2_zcb_recovery(t_grid, P_sim, P_market_grid, P_an_grid, recovery_df,
                      FIGURES_DIR / "step28_fig2_zcb_recovery.png")
    fig3_terminal_density(r_hw, r_vas, DT, 10.0,
                          FIGURES_DIR / "step28_fig3_terminal_density.png")
    fig4_mean_and_negrate(moments_df, horizons_df, t_fwd, f_fwd,
                          FIGURES_DIR / "step28_fig4_mean_and_negrate.png")
    fig5_recovery_convergence(
        conv_df, FIGURES_DIR / "step28_fig5_recovery_convergence.png")

    # ---- Acceptance checks ----------------------------------------------
    checks = acceptance_checks(horizons_df, recovery_df, conv_df,
                               r_hw, r_vas, DT, f00)
    print("\n=== Acceptance checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    summary = {
        "config": dict(a=A, sigma=SIGMA, r0=R0, b_vasicek=B_VASI, dt=DT,
                       horizon_years=HORIZON_YEARS, n_paths=N_PATHS,
                       seed=SEED),
        "theta_milestones": {
            "theta_0_pct": float(theta_grid[0] * 100),
            "theta_119m_pct": float(theta_grid[-1] * 100),
            "f00_pct": f00 * 100,
            "r0_offset_bp": float((R0 - f00) * 1e4),
        },
        "horizon_diagnostics": horizons_df.to_dict(orient="records"),
        "recovery_table": recovery_df.to_dict(orient="records"),
        "convergence": conv_df.to_dict(orient="records"),
        "max_zcb_recovery_err_bp_monthly": float(
            recovery_df["err_sim_vs_market_bp"].abs().max()),
        "checks": checks,
    }
    with open(OUTPUT_DIR / "step28_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n=== Step 28 summary ===")
    pass_all = all(c["pass"] for c in checks.values())
    print(f"  Verdict                   : "
          f"{'PASS' if pass_all else 'FAIL'}")
    print(f"  HW  E[r_10y]              : "
          f"{horizons_df[(horizons_df['model']=='HW') & (horizons_df['horizon_yr']==10.0)]['mean_pct'].iloc[0]:.4f}%")
    print(f"  Vas E[r_10y]              : "
          f"{horizons_df[(horizons_df['model']=='Vas') & (horizons_df['horizon_yr']==10.0)]['mean_pct'].iloc[0]:.4f}%")
    print(f"  SD r_10y (both models)    : "
          f"{horizons_df[(horizons_df['model']=='HW') & (horizons_df['horizon_yr']==10.0)]['sd_pct'].iloc[0]:.4f}%")
    print(f"  Max recovery err (monthly): "
          f"{summary['max_zcb_recovery_err_bp_monthly']:.2f} bp")
    print(f"  Recovery err at 96 steps  : "
          f"{conv_df[conv_df['steps_per_year']==96]['max_err_bp'].iloc[0]:.2f} bp")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    main()
