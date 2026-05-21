"""
models/cir/step20_cir_em.py
---------------------------
Phase D, Step 20: Cox-Ingersoll-Ross (CIR) Euler-Maruyama simulator with
the full-truncation scheme.

The CIR short-rate SDE under Q is

    dr_t = a (b - r_t) dt + sigma * sqrt(r_t) dW_t,

with a > 0 the mean-reversion speed, b > 0 the long-run level, sigma > 0
the volatility coefficient, and r_0 > 0 the initial short rate.  The
Feller condition

    2 a b >= sigma^2

ensures the continuous-time process stays strictly positive almost surely.
Plain Euler-Maruyama discretisation can nevertheless push a path negative
for any positive dt, which would then produce a NaN inside the sqrt(.)
diffusion term.  The Deelstra-Delbaen "full-truncation" remedy keeps the
drift and the diffusion of the discrete process evaluated at the
truncated value:

    r_{k+1} = r_k + a (b - r_k) dt
              + sigma * sqrt(max(r_k, 0)) * sqrt(dt) * Z_k,

while the propagated state is the unprojected r_{k+1}.  The truncation
acts only inside the diffusion square root; the drift still pulls the
path back toward b whenever r_k dips below zero.  This scheme is the
weakly-convergent reference discretisation against which the Phase D
Step 19 exact non-central chi-squared sampler is benchmarked.  Same-seed
coupling between the two drivers isolates the discretisation bias.

The CIR marginal admits a closed-form first and second moment:

    E[r_t]   = b + (r_0 - b) * exp(-a t),
    Var[r_t] = (r_0 sigma^2 / a) (exp(-a t) - exp(-2 a t))
             + (b sigma^2 / (2 a)) (1 - exp(-a t))^2.

The exact transition law is r_t | r_s ~ (c)^{-1} * chi^2_{d, lambda}, with

    c      = 4 a / (sigma^2 (1 - exp(-a (t-s)))),
    d      = 4 a b / sigma^2,
    lambda = c * r_s * exp(-a (t-s)).

Reproducibility: seed = 42 reused from Steps 13, 16, 17, 18 so that the
random-number stream is bitwise consistent with the rest of the project.

Outputs (aligned with Steps 13-18 artefact layout):

    models/cir/step20_em_moments.csv
    models/cir/step20_exact_moments.csv
    models/cir/step20_couple.csv
    models/cir/step20_stress_moments.csv
    models/cir/step20_summary.json
    figures/step20/step20_fig1_paths.png
    figures/step20/step20_fig2_moments.png
    figures/step20/step20_fig3_terminal_density.png
    figures/step20/step20_fig4_truncation.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ncx2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# ---------------------------------------------------------------------------
# Configuration; CIR placeholders satisfy the Feller condition 2ab >= sigma^2.
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

# Feller margin: 2ab = 0.04, sigma^2 = 0.0064, ratio = 6.25 -> comfortable.
FELLER_LHS = 2.0 * A * B_MEAN
FELLER_RHS = SIGMA ** 2
FELLER_OK = FELLER_LHS >= FELLER_RHS
FELLER_RATIO = FELLER_LHS / FELLER_RHS

# Feller-stress configuration; deliberately violates 2ab >= sigma^2 so the
# truncation scheme is exercised.  Reported as a secondary diagnostic.
A_STRESS = 0.30
B_STRESS = 0.04
SIGMA_STRESS = 0.20
R0_STRESS = 0.0372

OUTPUT_DIR = ROOT / "models" / "cir"
FIGURES_DIR = ROOT / "figures" / "step20"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 14-18 token set)
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
# CIR analytic moments and exact marginal density
# ---------------------------------------------------------------------------
def cir_mean(t, a, b, r0):
    return b + (r0 - b) * np.exp(-a * t)


def cir_var(t, a, b, sigma, r0):
    term1 = (r0 * sigma ** 2 / a) * (np.exp(-a * t) - np.exp(-2.0 * a * t))
    term2 = (b * sigma ** 2 / (2.0 * a)) * (1.0 - np.exp(-a * t)) ** 2
    return term1 + term2


def cir_marginal_pdf(x, t, a, b, sigma, r0):
    """Non-central chi-squared density of r_t scaled into level units."""
    c = 4.0 * a / (sigma ** 2 * (1.0 - np.exp(-a * t)))
    d = 4.0 * a * b / sigma ** 2
    lam = c * r0 * np.exp(-a * t)
    return ncx2.pdf(x * c, df=d, nc=lam) * c


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------
def simulate_em_full_truncation(a, b, sigma, r0, horizon, dt, n_paths, seed):
    """Full-truncation Euler-Maruyama discretisation of the CIR SDE.

    Returns the path matrix and the per-step truncation count (number of
    paths where r_k went strictly below zero at step k).
    """
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    trunc_count = np.zeros(n_steps, dtype=np.int64)
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        rk = r[:, k]
        rk_plus = np.maximum(rk, 0.0)
        r[:, k + 1] = (rk + a * (b - rk) * dt
                       + sigma * np.sqrt(rk_plus) * sqrt_dt * Z)
        trunc_count[k] = int((rk < 0.0).sum())
    return r, trunc_count


def simulate_exact_ncx2(a, b, sigma, r0, horizon, dt, n_paths, seed):
    """
    Exact CIR transition via the non-central chi-squared law.  Stub driver
    consistent with the Step 19 exact sampler; same seed protocol as the
    EM driver.  Returns an (n_paths, n_steps+1) path matrix.
    """
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def moment_table(r_paths, horizons, dt, a, b, sigma, r0):
    rows = []
    for T in horizons:
        k = int(round(T / dt))
        sample = r_paths[:, k]
        rows.append({
            "T": T,
            "E_sim": float(sample.mean()),
            "E_ana": float(cir_mean(T, a, b, r0)),
            "Var_sim": float(sample.var(ddof=1)),
            "Var_ana": float(cir_var(T, a, b, sigma, r0)),
            "min_sim": float(sample.min()),
            "max_sim": float(sample.max()),
            "neg_sim": float((sample < 0.0).mean()),
        })
    return pd.DataFrame(rows)


def truncation_summary(trunc_count, dt, horizon):
    n_steps = trunc_count.shape[0]
    activated_any = int((trunc_count > 0).sum())
    return {
        "total_steps": int(n_steps),
        "total_truncations": int(trunc_count.sum()),
        "step_indices_with_truncation": activated_any,
        "fraction_of_steps_with_any_trunc": float(activated_any / n_steps),
        "mean_paths_truncated_per_step": float(trunc_count.mean()),
        "max_paths_truncated_in_any_step": int(trunc_count.max()),
    }


def coupling_table(em_paths, exact_paths, dt, horizons):
    rows = []
    for T in horizons:
        k = int(round(T / dt))
        diff = em_paths[:, k] - exact_paths[:, k]
        rows.append({
            "T_yr": T,
            "mean_gap_bp": float(diff.mean() * 1e4),
            "abs_gap_mean_bp": float(np.abs(diff).mean() * 1e4),
            "abs_gap_p95_bp": float(np.quantile(np.abs(diff), 0.95) * 1e4),
            "abs_gap_max_bp": float(np.abs(diff).max() * 1e4),
        })
    return pd.DataFrame(rows)


def feller_diagnostic(a, b, sigma):
    return {
        "feller_lhs_2ab": 2.0 * a * b,
        "feller_rhs_sigma2": sigma ** 2,
        "feller_ratio": (2.0 * a * b) / sigma ** 2,
        "feller_holds": bool(2.0 * a * b >= sigma ** 2),
        "boundary_attainable": bool(2.0 * a * b < sigma ** 2),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_sample_paths(em_paths, exact_paths, dt, path):
    fig, ax = _new_figure(figsize=(8.0, 4.6))
    n_show = 6
    t_grid = np.arange(em_paths.shape[1]) * dt
    for i in range(n_show):
        ax.plot(t_grid, em_paths[i], color=BLUE, lw=0.9, alpha=0.7,
                label="EM (full trunc.)" if i == 0 else None)
        ax.plot(t_grid, exact_paths[i], color=BURG, lw=0.9, alpha=0.7,
                linestyle="--", label="Exact ncx2" if i == 0 else None)
    ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.axhline(B_MEAN, color=TEAL, lw=0.8, linestyle=":", alpha=0.6)
    ax.set_xlabel("t (years)")
    ax.set_ylabel(r"$r_t$")
    ax.set_title(f"CIR sample paths: EM full truncation vs exact ncx2 "
                 f"(first {n_show} paths)", fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_moment_convergence(em_paths, exact_paths, dt, a, b, sigma, r0, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        _style_axes(ax)
    t_grid = np.arange(em_paths.shape[1]) * dt
    mu_ana = cir_mean(t_grid, a, b, r0)
    var_ana = cir_var(t_grid, a, b, sigma, r0)
    var_ana[0] = 0.0
    ax1.plot(t_grid, em_paths.mean(axis=0), color=BLUE, lw=1.5,
             label="EM mean (sim)")
    ax1.plot(t_grid, exact_paths.mean(axis=0), color=BURG, lw=1.5,
             linestyle="--", label="Exact mean (sim)")
    ax1.plot(t_grid, mu_ana, color=TEAL, lw=1.2, linestyle=":",
             label="Analytic mean")
    ax1.set_xlabel("t (years)")
    ax1.set_ylabel(r"$E[r_t]$")
    ax1.set_title("Mean convergence", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", facecolor=BG, edgecolor=RULE,
               labelcolor=INK)
    ax2.plot(t_grid, em_paths.var(axis=0, ddof=1), color=BLUE, lw=1.5,
             label="EM var (sim)")
    ax2.plot(t_grid, exact_paths.var(axis=0, ddof=1), color=BURG, lw=1.5,
             linestyle="--", label="Exact var (sim)")
    ax2.plot(t_grid, var_ana, color=TEAL, lw=1.2, linestyle=":",
             label="Analytic var")
    ax2.set_xlabel("t (years)")
    ax2.set_ylabel(r"Var$[r_t]$")
    ax2.set_title("Variance convergence", fontweight="bold", pad=10)
    ax2.legend(loc="lower right", facecolor=BG, edgecolor=RULE,
               labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig3_terminal_density(em_paths, exact_paths, a, b, sigma, r0, T,
                          dt, path):
    fig, ax = _new_figure(figsize=(8.0, 4.6))
    k = int(round(T / dt))
    r_em = em_paths[:, k]
    r_ex = exact_paths[:, k]
    xs = np.linspace(max(1e-6, min(r_em.min(), r_ex.min()) - 0.005),
                     max(r_em.max(), r_ex.max()) + 0.005, 400)
    pdf_ana = cir_marginal_pdf(xs, T, a, b, sigma, r0)
    ax.hist(r_em, bins=80, density=True, color=BLUE, alpha=0.35,
            label="EM histogram")
    ax.hist(r_ex, bins=80, density=True, histtype="step", color=BURG,
            lw=1.6, label="Exact histogram")
    ax.plot(xs, pdf_ana, color=TEAL, lw=1.8, label="Analytic ncx2 pdf")
    ax.axvline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xlabel(r"$r_T$")
    ax.set_ylabel("density")
    ax.set_title(f"Terminal short-rate density at T = {T:.0f}y; "
                 "EM vs exact vs analytic",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig4_truncation_activity(trunc_count, dt, path):
    fig, ax = _new_figure(figsize=(8.0, 4.4))
    t_grid = np.arange(trunc_count.shape[0]) * dt + dt
    ax.bar(t_grid, trunc_count, width=dt * 0.9, color=OCHRE, alpha=0.85,
           label=r"Paths with $r_k$ < 0 (truncation triggered)")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("Count of paths truncated at step k")
    ax.set_title("Full-truncation activity per discrete step",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks(em_moments, trunc_summary, stress_summary,
                  stress_min, em_global_min, feller):
    checks: dict = {}
    # (1) Feller condition holds for the baseline
    checks["feller_condition_holds_baseline"] = {
        "ratio_2ab_over_sigma2": feller["feller_ratio"],
        "pass": bool(feller["feller_holds"]),
    }
    # (2) positivity under Feller: zero negative terminals at every horizon
    max_neg = float(em_moments["neg_sim"].max())
    checks["positivity_under_feller"] = {
        "max_neg_fraction_at_horizons": max_neg,
        "global_min_baseline": em_global_min,
        "baseline_truncations": trunc_summary["total_truncations"],
        "pass": bool(max_neg == 0.0
                     and trunc_summary["total_truncations"] == 0
                     and em_global_min >= 0.0),
    }
    # (3) first-moment band: |E_sim - E_ana| <= 5 bp
    mean_gap_bp = (em_moments["E_sim"] - em_moments["E_ana"]).abs() * 1e4
    checks["mean_band_5bp"] = {
        "max_abs_mean_gap_bp": float(mean_gap_bp.max()),
        "tolerance_bp": 5.0,
        "pass": bool(mean_gap_bp.max() <= 5.0),
    }
    # (4) second-moment band: |Var_sim/Var_ana - 1| <= 5%
    var_rel = (em_moments["Var_sim"] / em_moments["Var_ana"] - 1.0).abs()
    checks["variance_band_5pct"] = {
        "max_abs_var_rel": float(var_rel.max()),
        "tolerance_pct": 0.05,
        "pass": bool(var_rel.max() <= 0.05),
    }
    # (5) stress-run activation: truncation engages and bounds the min
    checks["stress_activation"] = {
        "stress_truncations": stress_summary["total_truncations"],
        "stress_steps_with_truncation": stress_summary["step_indices_with_truncation"],
        "stress_global_min": stress_min,
        "pass": bool(stress_summary["total_truncations"] > 0
                     and stress_summary["step_indices_with_truncation"] >= 100
                     and stress_min < 0.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase D, Step 20: CIR Euler-Maruyama with full truncation")
    print("-" * 64)
    print(f"a = {A}, b = {B_MEAN}, sigma = {SIGMA}, r0 = {R0}")
    print(f"seed = {SEED}, N = {N_PATHS}, dt = {DT:.6f}, "
          f"T_max = {HORIZON_YEARS}")
    print(f"Feller: 2ab = {FELLER_LHS:.6f} >= sigma^2 = {FELLER_RHS:.6f}  "
          f"(ratio = {FELLER_RATIO:.2f}, holds = {FELLER_OK})")
    print()

    em_paths, trunc_count = simulate_em_full_truncation(
        A, B_MEAN, SIGMA, R0, HORIZON_YEARS, DT, N_PATHS, SEED)
    exact_paths = simulate_exact_ncx2(
        A, B_MEAN, SIGMA, R0, HORIZON_YEARS, DT, N_PATHS, SEED)

    em_moments = moment_table(em_paths, HORIZONS, DT, A, B_MEAN, SIGMA, R0)
    exact_moments = moment_table(exact_paths, HORIZONS, DT, A, B_MEAN, SIGMA, R0)
    trunc = truncation_summary(trunc_count, DT, HORIZON_YEARS)
    couple = coupling_table(em_paths, exact_paths, DT, HORIZONS)
    feller = feller_diagnostic(A, B_MEAN, SIGMA)

    em_moments.to_csv(OUTPUT_DIR / "step20_em_moments.csv", index=False)
    exact_moments.to_csv(OUTPUT_DIR / "step20_exact_moments.csv", index=False)
    couple.to_csv(OUTPUT_DIR / "step20_couple.csv", index=False)

    print("EM moment table:")
    print(em_moments.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Exact ncx2 moment table:")
    print(exact_moments.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Same-seed pathwise coupling (EM minus exact, basis points):")
    print(couple.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Full-truncation activity summary:")
    for key, val in trunc.items():
        print(f"  {key:38s} = {val}")
    print()
    print("Feller diagnostic:")
    for key, val in feller.items():
        print(f"  {key:24s} = {val}")
    print()
    print(f"EM global min    = {em_paths.min():+.6f}")
    print(f"EM global max    = {em_paths.max():+.6f}")
    print(f"Exact global min = {exact_paths.min():+.6f}")
    print(f"Exact global max = {exact_paths.max():+.6f}")

    # ---- Feller-stress configuration ----
    stress_feller = feller_diagnostic(A_STRESS, B_STRESS, SIGMA_STRESS)
    stress_paths, stress_trunc = simulate_em_full_truncation(
        A_STRESS, B_STRESS, SIGMA_STRESS, R0_STRESS,
        HORIZON_YEARS, DT, N_PATHS, SEED)
    stress_summary = truncation_summary(stress_trunc, DT, HORIZON_YEARS)
    stress_moments = moment_table(
        stress_paths, HORIZONS, DT, A_STRESS, B_STRESS, SIGMA_STRESS, R0_STRESS)
    stress_moments.to_csv(OUTPUT_DIR / "step20_stress_moments.csv",
                          index=False)
    print()
    print("Feller-stress configuration (a, b, sigma):",
          (A_STRESS, B_STRESS, SIGMA_STRESS))
    print(f"  Feller holds = {stress_feller['feller_holds']}, "
          f"ratio = {stress_feller['feller_ratio']:.3f}")
    for key, val in stress_summary.items():
        print(f"  {key:38s} = {val}")
    print(f"  EM global min (stress) = {stress_paths.min():+.6f}")

    # ---- Sanity checks ----
    checks = sanity_checks(em_moments, trunc, stress_summary,
                           float(stress_paths.min()),
                           float(em_paths.min()), feller)
    print("\n=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    # ---- Figures ----
    fig1_sample_paths(em_paths, exact_paths, DT,
                      FIGURES_DIR / "step20_fig1_paths.png")
    fig2_moment_convergence(em_paths, exact_paths, DT, A, B_MEAN, SIGMA, R0,
                            FIGURES_DIR / "step20_fig2_moments.png")
    fig3_terminal_density(em_paths, exact_paths, A, B_MEAN, SIGMA, R0,
                          10.0, DT,
                          FIGURES_DIR / "step20_fig3_terminal_density.png")
    fig4_truncation_activity(stress_trunc, DT,
                             FIGURES_DIR / "step20_fig4_truncation.png")

    summary = {
        "config": dict(a=A, b=B_MEAN, sigma=SIGMA, r0=R0, dt=DT,
                       n_paths=N_PATHS, horizon_years=HORIZON_YEARS,
                       seed=SEED),
        "horizons": HORIZONS,
        "feller": feller,
        "em_moments": em_moments.to_dict(orient="records"),
        "exact_moments": exact_moments.to_dict(orient="records"),
        "coupling": couple.to_dict(orient="records"),
        "truncation": trunc,
        "em_global_min": float(em_paths.min()),
        "em_global_max": float(em_paths.max()),
        "exact_global_min": float(exact_paths.min()),
        "exact_global_max": float(exact_paths.max()),
        "stress_config": dict(a=A_STRESS, b=B_STRESS, sigma=SIGMA_STRESS,
                              r0=R0_STRESS),
        "stress_feller": stress_feller,
        "stress_truncation": stress_summary,
        "stress_moments": stress_moments.to_dict(orient="records"),
        "stress_em_global_min": float(stress_paths.min()),
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step20_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print()
    print(f"figures written to {FIGURES_DIR}")
    print(f"output written to  {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
