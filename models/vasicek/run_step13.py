"""
models/vasicek/run_step13.py
----------------------------
Phase C, Step 13 driver for the Vasicek Euler-Maruyama simulator.

Pipeline:
  1. Simulate 10,000 Vasicek paths over T = 10 years on a monthly grid.
  2. Validate sample mean and variance against the closed-form Vasicek moments.
  3. Generate six diagnostic figures (sample paths, mean/variance overlays,
     terminal cross-section, Vasicek vs DBM dispersion, negative-rate preview).
  4. Run a side-by-side DBM comparison at the same grid; DBM uses SEED + 1 so
     the two models receive independent shock sequences (not perfectly
     correlated, which would understate the contrast in Figure 5).
  5. Preview the Step 18 negative-rate diagnostics for Vasicek.
  6. Persist the numerical bundle to step13_results.json under this folder
     and save figures under figures/step13/*.

Run:
    python models/vasicek/run_step13.py
    # or, with the package layout, from project root:
    python -m models.vasicek.run_step13
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

from models.vasicek.vasicek_simulator import (  # noqa: E402
    A_DEFAULT,
    B_DEFAULT,
    DT_DEFAULT,
    N_PATHS_DEFAULT,
    R0_DEFAULT,
    SEED_DEFAULT,
    SIGMA_DEFAULT,
    T_DEFAULT,
    simulate_vasicek_paths,
    vasicek_stationary_moments,
    vasicek_theoretical_mean,
    vasicek_theoretical_variance,
)


# -----------------------------------------------------------------------------
# Configuration -- pinned to the simulator defaults (placeholder calibration).
# -----------------------------------------------------------------------------
A = A_DEFAULT
B = B_DEFAULT
SIGMA = SIGMA_DEFAULT
R0 = R0_DEFAULT
DT = DT_DEFAULT
T = T_DEFAULT
N_PATHS = N_PATHS_DEFAULT
SEED = SEED_DEFAULT
HORIZONS = [1.0, 2.0, 5.0, 10.0]

FIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "figures", "step13")
)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "step13_results.json")


# -----------------------------------------------------------------------------
# Lightweight DBM simulator for the dispersion comparison (Section 5).
# -----------------------------------------------------------------------------
def simulate_dbm_paths(
    mu: float = 0.0,
    sigma: float = SIGMA,
    r0: float = R0,
    dt: float = DT,
    T: float = T,
    n_paths: int = N_PATHS,
    seed: int = SEED + 1,
) -> np.ndarray:
    """
    Drifted Brownian motion: dr_t = mu*dt + sigma*dW_t.
    Mirror-image of simulate_vasicek_paths but with no mean reversion;
    seed deliberately offset by +1 so the two simulations are independent.
    """
    n_steps = int(round(T / dt))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = r0
    for k in range(n_steps):
        paths[:, k + 1] = paths[:, k] + mu * dt + sigma * sqrt_dt * Z[:, k]
    return paths


# -----------------------------------------------------------------------------
# Numerical summaries
# -----------------------------------------------------------------------------
def moment_validation(paths: np.ndarray, t_grid: np.ndarray) -> dict[str, Any]:
    """Compute the Section 4 closed-form moment validation."""
    sample_mean = paths.mean(axis=0)
    sample_var = paths.var(axis=0, ddof=1)

    theory_mean = vasicek_theoretical_mean(t_grid, a=A, b=B, r0=R0)
    theory_var = vasicek_theoretical_variance(t_grid, a=A, sigma=SIGMA)

    mean_err = sample_mean - theory_mean
    var_err = sample_var - theory_var

    return {
        "sample_mean": sample_mean,
        "sample_var": sample_var,
        "theory_mean": theory_mean,
        "theory_var": theory_var,
        "max_abs_mean_err": float(np.max(np.abs(mean_err))),
        "max_abs_var_err": float(np.max(np.abs(var_err))),
    }


def horizon_table(paths: np.ndarray, horizons: list[float]) -> list[dict[str, Any]]:
    """Cross-sectional summary at the requested horizons."""
    rows = []
    for t_h in horizons:
        k = int(round(t_h / DT))
        col = paths[:, k]
        sample_mean = float(col.mean())
        sample_sd = float(col.std(ddof=1))
        theory_mean = float(B + (R0 - B) * np.exp(-A * t_h))
        theory_var = float((SIGMA ** 2) * (1.0 - np.exp(-2.0 * A * t_h)) / (2.0 * A))
        theory_sd = float(np.sqrt(theory_var))
        q05, q50, q95 = (float(x) for x in np.quantile(col, [0.05, 0.50, 0.95]))
        rows.append({
            "horizon_yr": t_h,
            "sample_mean": sample_mean,
            "theory_mean": theory_mean,
            "sample_sd": sample_sd,
            "theory_sd": theory_sd,
            "q05": q05,
            "median": q50,
            "q95": q95,
        })
    return rows


def negative_rate_diagnostics(paths: np.ndarray) -> dict[str, Any]:
    """Section 6: negative-rate preview for Step 18."""
    n_paths, n_cols = paths.shape
    any_neg_per_path = (paths < 0.0).any(axis=1)
    frac_any_neg = float(any_neg_per_path.mean())
    frac_neg_at_T = float((paths[:, -1] < 0.0).mean())
    global_min = float(paths.min())
    total_neg_obs = int((paths < 0.0).sum())

    # Per-horizon fraction negative (matches Section 6 commentary).
    horizon_neg = []
    for t_h in HORIZONS:
        k = int(round(t_h / DT))
        horizon_neg.append({
            "horizon_yr": t_h,
            "frac_negative": float((paths[:, k] < 0.0).mean()),
        })

    return {
        "frac_any_negative": frac_any_neg,
        "n_paths_any_negative": int(any_neg_per_path.sum()),
        "frac_negative_at_T": frac_neg_at_T,
        "global_min_rate": global_min,
        "total_negative_observations": total_neg_obs,
        "n_paths": int(n_paths),
        "n_total_observations": int(n_paths * n_cols),
        "per_horizon_fraction_negative": horizon_neg,
    }


def dbm_contrast(vasicek_paths: np.ndarray, dbm_paths: np.ndarray) -> dict[str, Any]:
    """Section 5: dispersion contrast at T = 10."""
    vas_T = vasicek_paths[:, -1]
    dbm_T = dbm_paths[:, -1]
    sd_vas = float(vas_T.std(ddof=1))
    sd_dbm = float(dbm_T.std(ddof=1))
    sd_dbm_theory = float(SIGMA * np.sqrt(T))            # sigma * sqrt(T)
    sd_vas_stationary = float(SIGMA / np.sqrt(2.0 * A))  # sigma / sqrt(2 a)
    return {
        "vasicek_sd_at_T": sd_vas,
        "dbm_sd_at_T": sd_dbm,
        "ratio_dbm_over_vasicek": sd_dbm / sd_vas,
        "dbm_theory_sd": sd_dbm_theory,
        "vasicek_stationary_sd": sd_vas_stationary,
    }


def sanity_checks(
    moments: dict[str, Any],
    horizons: list[dict[str, Any]],
    contrast: dict[str, Any],
    neg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Section 7: pass/fail table."""
    long_h = horizons[-1]
    checks: dict[str, dict[str, Any]] = {}

    checks["sample_mean_tracks_theory"] = {
        "criterion": "max|err| < 1e-3",
        "observed": float(moments["max_abs_mean_err"]),
        "pass": bool(moments["max_abs_mean_err"] < 1e-3),
    }
    checks["sample_variance_tracks_theory"] = {
        "criterion": "max|err| < 1e-5",
        "observed": float(moments["max_abs_var_err"]),
        "pass": bool(moments["max_abs_var_err"] < 1e-5),
    }
    checks["long_horizon_mean_approaches_b"] = {
        "criterion": "|mean(T=10) - b| < 1e-3",
        "observed": float(abs(long_h["sample_mean"] - B)),
        "pass": bool(abs(long_h["sample_mean"] - B) < 1e-3),
    }
    stationary_sd = SIGMA / np.sqrt(2.0 * A)
    checks["long_horizon_sd_approaches_stationary"] = {
        "criterion": "|sd(T=10) - sigma/sqrt(2a)| < 1e-3",
        "observed": float(abs(long_h["sample_sd"] - stationary_sd)),
        "pass": bool(abs(long_h["sample_sd"] - stationary_sd) < 1e-3),
    }
    checks["dbm_dispersion_exceeds_vasicek"] = {
        "criterion": "ratio > 1",
        "observed": float(contrast["ratio_dbm_over_vasicek"]),
        "pass": bool(contrast["ratio_dbm_over_vasicek"] > 1.0),
    }
    checks["negative_rates_documented"] = {
        "criterion": "documented; reported in Step 18",
        "observed": float(neg["frac_any_negative"]),
        "pass": True,  # documentation rather than pass/fail
    }
    return checks


# -----------------------------------------------------------------------------
# Figures (Section 4 overlays + Section 5 dispersion + Section 6 preview)
# -----------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def figure1_paths(
    paths: np.ndarray, t_grid: np.ndarray, moments: dict[str, Any]
) -> str:
    """Figure 1: 200 sample paths + conditional mean + stationary 2-sigma band."""
    _, stat_var = vasicek_stationary_moments(a=A, b=B, sigma=SIGMA)
    stat_sd = np.sqrt(stat_var)

    fig, ax = plt.subplots(figsize=(10, 5))
    rng_plot = np.random.default_rng(0)
    indices = rng_plot.choice(paths.shape[0], size=200, replace=False)
    for i in indices:
        ax.plot(t_grid, paths[i] * 100.0, color="#4472C4", alpha=0.10, linewidth=0.6)
    ax.plot(t_grid, moments["theory_mean"] * 100.0,
            color="#C00000", linewidth=1.8, label="Conditional mean")
    ax.axhline(B * 100.0, color="black", linestyle="--", linewidth=1.0,
               label=f"b = {B*100:.2f}%")
    ax.fill_between(
        t_grid,
        (B - 2 * stat_sd) * 100.0,
        (B + 2 * stat_sd) * 100.0,
        color="#FFC000", alpha=0.20,
        label=r"Stationary $\pm 2\sigma$",
    )
    ax.scatter([0.0], [R0 * 100.0], color="black", zorder=5,
               label=f"$r_0$ = {R0*100:.2f}%")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short rate (%)")
    ax.set_title("Figure 1. Vasicek sample paths (200 of 10,000)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    out = os.path.join(FIG_DIR, "fig1_paths.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure2_mean(t_grid: np.ndarray, moments: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_grid, moments["sample_mean"] * 100.0,
            label="Sample mean", color="#4472C4", linewidth=1.6)
    ax.plot(t_grid, moments["theory_mean"] * 100.0,
            label=r"Theory: $b + (r_0 - b)e^{-at}$",
            color="#C00000", linewidth=1.4, linestyle="--")
    ax.axhline(B * 100.0, color="black", linestyle=":", linewidth=0.8,
               label=f"b = {B*100:.2f}%")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"$E[r_t]$ (%)")
    ax.set_title(
        f"Figure 2. Sample vs theoretical mean   "
        f"(max |err| = {moments['max_abs_mean_err']:.2e})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "fig2_mean.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure3_variance(t_grid: np.ndarray, moments: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_grid, moments["sample_var"], label="Sample variance",
            color="#4472C4", linewidth=1.6)
    ax.plot(t_grid, moments["theory_var"],
            label=r"Theory: $\sigma^2(1 - e^{-2at})/(2a)$",
            color="#C00000", linewidth=1.4, linestyle="--")
    _, stat_var = vasicek_stationary_moments(a=A, b=B, sigma=SIGMA)
    ax.axhline(stat_var, color="black", linestyle=":", linewidth=0.8,
               label=fr"Stationary $\sigma^2/(2a) = {stat_var:.2e}$")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"Var$[r_t]$")
    ax.set_title(
        f"Figure 3. Sample vs theoretical variance   "
        f"(max |err| = {moments['max_abs_var_err']:.2e})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "fig3_variance.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure4_terminal_density(paths: np.ndarray) -> str:
    r_T = paths[:, -1]
    theory_mean = float(B + (R0 - B) * np.exp(-A * T))
    theory_var = float((SIGMA ** 2) * (1.0 - np.exp(-2.0 * A * T)) / (2.0 * A))
    theory_sd = float(np.sqrt(theory_var))

    grid = np.linspace(r_T.min() - 0.5 * theory_sd,
                       r_T.max() + 0.5 * theory_sd, 400)
    pdf = stats.norm.pdf(grid, loc=theory_mean, scale=theory_sd)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(r_T * 100.0, bins=80, density=True, alpha=0.55,
            color="#4472C4", label=f"Empirical r_T at T = {T:.0f}")
    ax.plot(grid * 100.0, pdf / 100.0, color="#C00000", linewidth=1.6,
            label=fr"$\mathcal{{N}}(b, \sigma^2/(2a))$ overlay")
    ax.axvline(theory_mean * 100.0, color="black", linestyle="--",
               linewidth=0.9, label=f"theory mean = {theory_mean*100:.2f}%")
    ax.set_xlabel("Short rate at T (%)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 4. Terminal density vs Gaussian closed form")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "fig4_terminal_density.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure5_dispersion(
    vasicek_paths: np.ndarray, dbm_paths: np.ndarray, contrast: dict[str, Any]
) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(
        min(vasicek_paths[:, -1].min(), dbm_paths[:, -1].min()) - 0.005,
        max(vasicek_paths[:, -1].max(), dbm_paths[:, -1].max()) + 0.005,
        80,
    )
    ax.hist(dbm_paths[:, -1] * 100.0, bins=bins * 100.0, density=True,
            alpha=0.55, color="#ED7D31",
            label=f"DBM (sd = {contrast['dbm_sd_at_T']:.4f})")
    ax.hist(vasicek_paths[:, -1] * 100.0, bins=bins * 100.0, density=True,
            alpha=0.55, color="#4472C4",
            label=f"Vasicek (sd = {contrast['vasicek_sd_at_T']:.4f})")
    ax.axvline(B * 100.0, color="black", linestyle="--", linewidth=0.9,
               label=f"b = {B*100:.2f}%")
    ax.axvline(0.0, color="red", linestyle=":", linewidth=0.9,
               label="Zero rate")
    ax.set_xlabel("Short rate at T = 10 (%)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Figure 5. Vasicek vs DBM terminal cross-section "
        f"(ratio = {contrast['ratio_dbm_over_vasicek']:.2f})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "fig5_vas_vs_dbm.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure6_negatives(paths: np.ndarray, t_grid: np.ndarray) -> str:
    frac_neg_t = (paths < 0.0).mean(axis=0)
    min_rate_t = paths.min(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(t_grid, frac_neg_t * 100.0, color="#4472C4", linewidth=1.5)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"P($r_t < 0$) (%)")
    ax.set_title("Figure 6a. Fraction of paths with negative rate")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t_grid, min_rate_t * 100.0, color="#C00000", linewidth=1.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Minimum simulated rate (%)")
    ax2.set_title("Figure 6b. Minimum rate across paths over time")
    ax2.grid(alpha=0.3)

    fig.suptitle("Figure 6. Vasicek negative-rate preview (Step 18 lite)",
                 fontsize=11, fontweight="bold")
    out = os.path.join(FIG_DIR, "fig6_negatives.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> dict[str, Any]:
    print("Phase C, Step 13: Vasicek Euler-Maruyama simulator")
    print("-" * 60)
    print(f"a       = {A}")
    print(f"b       = {B}")
    print(f"sigma   = {SIGMA}")
    print(f"r0      = {R0}")
    print(f"dt      = {DT:.6f}")
    print(f"T       = {T}")
    print(f"n_paths = {N_PATHS}")
    print(f"seed    = {SEED}")
    print()

    n_steps = int(round(T / DT))
    t_grid = np.arange(n_steps + 1) * DT

    # Vasicek simulation
    vasicek_paths = simulate_vasicek_paths(
        a=A, b=B, sigma=SIGMA, r0=R0, dt=DT, T=T,
        n_paths=N_PATHS, seed=SEED,
    )
    print(f"Vasicek paths shape: {vasicek_paths.shape}")

    # DBM comparison run (independent shock sequence)
    dbm_paths = simulate_dbm_paths(
        mu=0.0, sigma=SIGMA, r0=R0, dt=DT, T=T,
        n_paths=N_PATHS, seed=SEED + 1,
    )
    print(f"DBM paths shape:     {dbm_paths.shape}")

    # Numerical analyses
    moments = moment_validation(vasicek_paths, t_grid)
    horizons = horizon_table(vasicek_paths, HORIZONS)
    contrast = dbm_contrast(vasicek_paths, dbm_paths)
    neg = negative_rate_diagnostics(vasicek_paths)
    checks = sanity_checks(moments, horizons, contrast, neg)

    # Console summary
    print(f"\nMax |sample mean - theory mean|     = {moments['max_abs_mean_err']:.4e}")
    print(f"Max |sample variance - theory var|  = {moments['max_abs_var_err']:.4e}")

    print("\nHorizon table:")
    print(f"  {'T':>4}  {'mean(sim)':>10}  {'mean(thy)':>10}  "
          f"{'sd(sim)':>10}  {'sd(thy)':>10}  {'q05':>8}  "
          f"{'q50':>8}  {'q95':>8}")
    for row in horizons:
        print(
            f"  {row['horizon_yr']:>4.1f}  "
            f"{row['sample_mean']:>10.6f}  {row['theory_mean']:>10.6f}  "
            f"{row['sample_sd']:>10.6f}  {row['theory_sd']:>10.6f}  "
            f"{row['q05']:>8.4f}  {row['median']:>8.4f}  {row['q95']:>8.4f}"
        )

    print("\nVasicek vs DBM at T = 10:")
    print(f"  Vasicek SD          = {contrast['vasicek_sd_at_T']:.6f}")
    print(f"  DBM SD              = {contrast['dbm_sd_at_T']:.6f}")
    print(f"  Ratio (DBM/Vas)     = {contrast['ratio_dbm_over_vasicek']:.4f}")
    print(f"  DBM theory SD       = {contrast['dbm_theory_sd']:.6f}")
    print(f"  Vasicek stat. SD    = {contrast['vasicek_stationary_sd']:.6f}")

    print("\nNegative-rate preview:")
    print(f"  Fraction any path negative   = {neg['frac_any_negative']*100:.3f}%"
          f"  ({neg['n_paths_any_negative']} of {neg['n_paths']})")
    print(f"  Fraction negative at T = 10  = {neg['frac_negative_at_T']*100:.3f}%")
    print(f"  Global minimum rate          = {neg['global_min_rate']:.6f}")
    print(f"  Total negative observations  = {neg['total_negative_observations']}"
          f"  of {neg['n_total_observations']}")

    print("\nSanity checks:")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name:<42s}  observed = {info['observed']:.4e}")

    # Figures
    _ensure_dir(FIG_DIR)
    fig_paths = {
        "fig1_paths":             figure1_paths(vasicek_paths, t_grid, moments),
        "fig2_mean":              figure2_mean(t_grid, moments),
        "fig3_variance":          figure3_variance(t_grid, moments),
        "fig4_terminal_density":  figure4_terminal_density(vasicek_paths),
        "fig5_vas_vs_dbm":        figure5_dispersion(vasicek_paths, dbm_paths, contrast),
        "fig6_negatives":         figure6_negatives(vasicek_paths, t_grid),
    }
    print("\nSaved figures:")
    for k, v in fig_paths.items():
        print(f"  {k}: {v}")

    # Persist JSON bundle (drop ndarray fields to keep it serialisable).
    payload: dict[str, Any] = {
        "config": {
            "a": A, "b": B, "sigma": SIGMA, "r0": R0,
            "dt": DT, "T": T, "n_paths": N_PATHS, "seed": SEED,
            "horizons": HORIZONS,
        },
        "moment_validation": {
            "max_abs_mean_err": moments["max_abs_mean_err"],
            "max_abs_var_err": moments["max_abs_var_err"],
        },
        "horizon_table": horizons,
        "dbm_contrast": contrast,
        "negative_rate_preview": neg,
        "sanity_checks": checks,
        "figures": {k: os.path.relpath(v, start=os.path.dirname(__file__))
                    for k, v in fig_paths.items()},
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"\nWrote results to {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
