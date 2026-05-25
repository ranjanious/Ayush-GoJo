"""
models/cir/step23_cir_validation.py
-----------------------------------
Phase D, Step 23: Validate the CIR analytical bond-price formula against
Monte Carlo, mirroring the Vasicek standard set in Step 17.

For the CIR short rate

    dr_t = a (b - r_t) dt + sigma sqrt(r_t) dW_t,   r_0 > 0,

the zero-coupon bond price admits the exact affine form

    P(0, T) = A(T) exp(-B(T) r_0),
    gamma   = sqrt(a^2 + 2 sigma^2),
    B(T)    = 2 (e^{gamma T} - 1) / ((a + gamma)(e^{gamma T} - 1) + 2 gamma),
    A(T)    = ( 2 gamma e^{(a + gamma) T / 2}
                / ((a + gamma)(e^{gamma T} - 1) + 2 gamma) )^{2 a b / sigma^2}.

Unlike Vasicek's Gaussian closed form, the CIR formula rides on the
non-central chi-squared transition law: r_T conditional on r_0 satisfies
r_T = X / c with X ~ ncx2(d, lam), where

    d   = 4 a b / sigma^2,
    c   = 4 a / (sigma^2 (1 - e^{-a T})),
    lam = c r_0 e^{-a T}.

Two Monte Carlo benchmarks run in parallel:

    MC_exact : exact ncx2 transition sampled at every step dt; no
               discretisation bias in the marginal of r_t.
    MC_EM    : Euler-Maruyama with full-truncation diffusion;
               biased by dt but the production scheme of Steps 18, 20, 22.

For each MC path the discount factor is D_i = exp(-trapz(r_i, dt)); the
Monte Carlo price is mean(D_i).  The >=1% target maps to
rel_err = |P_MC - P_closed| / P_closed <= 0.01 across maturities.

Scenarios cover the base CIR calibration plus the three Feller stress
points from Step 21 (vary sigma so 2ab/sigma^2 in {6.25, 1.5, 1.0, 0.7}).
Same-seed coupling reuses seed = 42 from Steps 13, 16, 17, 18, 20, 21, 22.

Outputs (aligned with Steps 13-22 artefact layout):
    models/cir/step23_validation.csv          one row per (scenario, scheme, T)
    models/cir/step23_scenario_summary.csv    rollup per scenario / scheme
    models/cir/step23_convergence.csv         rel_err vs N (base scenario)
    models/cir/step23_summary.json            config + headline metrics
    figures/step23/step23_fig1_rel_error.png
    figures/step23/step23_fig2_price_overlay.png
    figures/step23/step23_fig3_convergence.png
    figures/step23/step23_fig4_residual_bp.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# ---------------------------------------------------------------------------
# Shared configuration (reuses Steps 20, 21, 22).
# ---------------------------------------------------------------------------
R0 = 0.0372
DT = 1.0 / 12.0
N_PATHS = 10_000
HORIZON_YEARS = 10.0
SEED = 42

A = 0.50
B_MEAN = 0.04
SIGMA_BASE = 0.08    # Feller ratio 6.25

BOND_MATS = [1.0, 2.0, 5.0, 10.0]

# Feller ratio = 2ab / sigma^2; vary sigma to land on the target ratios.
FELLER_LHS = 2.0 * A * B_MEAN
SCENARIOS = [
    {"name": "base",     "sigma": SIGMA_BASE,                       "ratio": FELLER_LHS / SIGMA_BASE ** 2},
    {"name": "above",    "sigma": float(np.sqrt(FELLER_LHS / 1.5)), "ratio": 1.5},
    {"name": "boundary", "sigma": float(np.sqrt(FELLER_LHS / 1.0)), "ratio": 1.0},
    {"name": "below",    "sigma": float(np.sqrt(FELLER_LHS / 0.7)), "ratio": 0.7},
]

# Convergence sweep for the base scenario.
CONV_N = [500, 1_000, 2_500, 5_000, 10_000, 25_000]
CONV_T = 10.0

OUTPUT_DIR = ROOT / "models" / "cir"
FIGURES_DIR = ROOT / "figures" / "step23"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-22).
BG = "#fbf7ec"
INK = "#1c1a15"
SUB = "#5c544a"
RULE = "#c5b8a0"
BLUE = "#2c4a5e"
BURG = "#8b2d3a"
OCHRE = "#b8842a"
TEAL = "#2d6a5f"

SCEN_COLORS = {
    "base":     BURG,
    "above":    TEAL,
    "boundary": OCHRE,
    "below":    BLUE,
}
SCHEME_LINESTYLE = {"exact": "-", "EM": "--"}


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
# Closed-form CIR bond price.
# ---------------------------------------------------------------------------
def cir_B(T, a, sigma):
    gamma = np.sqrt(a ** 2 + 2.0 * sigma ** 2)
    num = 2.0 * (np.exp(gamma * T) - 1.0)
    den = (a + gamma) * (np.exp(gamma * T) - 1.0) + 2.0 * gamma
    return num / den


def cir_A(T, a, b, sigma):
    gamma = np.sqrt(a ** 2 + 2.0 * sigma ** 2)
    num = 2.0 * gamma * np.exp((a + gamma) * T / 2.0)
    den = (a + gamma) * (np.exp(gamma * T) - 1.0) + 2.0 * gamma
    return float((num / den) ** (2.0 * a * b / sigma ** 2))


def cir_price(T, r0, a, b, sigma):
    return cir_A(T, a, b, sigma) * float(np.exp(-cir_B(T, a, sigma) * r0))


# ---------------------------------------------------------------------------
# Simulators (reused from Steps 20, 22).
# ---------------------------------------------------------------------------
def simulate_cir_em(a, b, sigma, r0, horizon, dt, n_paths, seed):
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        rk = r[:, k]
        rk_plus = np.maximum(rk, 0.0)
        r[:, k + 1] = (rk + a * (b - rk) * dt
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


# ---------------------------------------------------------------------------
# Monte Carlo discount factors via trapezoidal integration.
# ---------------------------------------------------------------------------
def mc_discount_factors(paths, dt, maturities):
    """Return {T: ndarray of per-path discount factors} via trapezoid rule."""
    running = np.zeros_like(paths)
    running[:, 1:] = 0.5 * (paths[:, :-1] + paths[:, 1:]) * dt
    integral = np.cumsum(running, axis=1)
    out = {}
    for T in maturities:
        k = int(round(T / dt))
        out[T] = np.exp(-integral[:, k])
    return out


def mc_summary(df, p_closed):
    mean = float(df.mean())
    se = float(df.std(ddof=1) / np.sqrt(df.shape[0]))
    abs_err = abs(mean - p_closed)
    rel_err = abs_err / p_closed
    return mean, se, abs_err, rel_err


# ---------------------------------------------------------------------------
# Per-scenario validation.
# ---------------------------------------------------------------------------
def validate_scenario(scen, maturities):
    a, b, sigma, r0 = A, B_MEAN, scen["sigma"], R0
    paths_ex = simulate_cir_exact(a, b, sigma, r0, HORIZON_YEARS, DT, N_PATHS, SEED)
    paths_em = simulate_cir_em(a, b, sigma, r0, HORIZON_YEARS, DT, N_PATHS, SEED)
    dfs_ex = mc_discount_factors(paths_ex, DT, maturities)
    dfs_em = mc_discount_factors(paths_em, DT, maturities)

    rows = []
    for T in maturities:
        p_closed = cir_price(T, r0, a, b, sigma)
        for scheme, dfs in (("exact", dfs_ex), ("EM", dfs_em)):
            mean, se, abs_err, rel_err = mc_summary(dfs[T], p_closed)
            rows.append({
                "scenario": scen["name"],
                "feller_ratio": scen["ratio"],
                "sigma": sigma,
                "scheme": scheme,
                "T_yr": T,
                "P_closed": p_closed,
                "P_MC": mean,
                "se": se,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "rel_err_pct": rel_err * 100,
                "bp_residual": (mean - p_closed) * 1e4,
                "se_bp": se * 1e4,
                "within_1pct": bool(rel_err <= 0.01),
            })
    return rows


# ---------------------------------------------------------------------------
# Convergence sweep (base scenario only).
# ---------------------------------------------------------------------------
def convergence_sweep(maturities, n_grid):
    sigma = SIGMA_BASE
    rows = []
    for N in n_grid:
        paths_ex = simulate_cir_exact(A, B_MEAN, sigma, R0, HORIZON_YEARS, DT, N, SEED)
        paths_em = simulate_cir_em(A, B_MEAN, sigma, R0, HORIZON_YEARS, DT, N, SEED)
        dfs_ex = mc_discount_factors(paths_ex, DT, [CONV_T])
        dfs_em = mc_discount_factors(paths_em, DT, [CONV_T])
        p_closed = cir_price(CONV_T, R0, A, B_MEAN, sigma)
        for scheme, dfs in (("exact", dfs_ex), ("EM", dfs_em)):
            mean, se, abs_err, rel_err = mc_summary(dfs[CONV_T], p_closed)
            rows.append({
                "N": N,
                "scheme": scheme,
                "T_yr": CONV_T,
                "P_closed": p_closed,
                "P_MC": mean,
                "se": se,
                "rel_err": rel_err,
                "rel_err_pct": rel_err * 100,
                "bp_residual": (mean - p_closed) * 1e4,
                "se_bp": se * 1e4,
            })
    return rows


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------
def fig1_rel_error(df, path):
    fig, ax = _new_figure(figsize=(8.2, 4.8))
    for scen in SCENARIOS:
        for scheme in ("exact", "EM"):
            sub = df[(df["scenario"] == scen["name"])
                     & (df["scheme"] == scheme)]
            ax.plot(sub["T_yr"], sub["rel_err_pct"],
                    color=SCEN_COLORS[scen["name"]],
                    linestyle=SCHEME_LINESTYLE[scheme],
                    marker="o", markersize=4, lw=1.4,
                    label=f"{scen['name']} ({scheme})")
    ax.axhline(1.0, color=INK, lw=1.0, linestyle=":", alpha=0.7)
    ax.text(BOND_MATS[-1], 1.05, "1% target", color=INK, fontsize=8,
            ha="right", va="bottom")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Relative price error (%)")
    ax.set_title("CIR closed form vs Monte Carlo: relative price error by maturity",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_price_overlay(df, path):
    fig, ax = _new_figure(figsize=(8.2, 4.8))
    base = df[df["scenario"] == "base"]
    Ts = sorted(base["T_yr"].unique())
    p_closed = [base[base["T_yr"] == T]["P_closed"].iloc[0] for T in Ts]
    p_ex = [base[(base["T_yr"] == T) & (base["scheme"] == "exact")]["P_MC"].iloc[0]
            for T in Ts]
    p_em = [base[(base["T_yr"] == T) & (base["scheme"] == "EM")]["P_MC"].iloc[0]
            for T in Ts]
    ax.plot(Ts, p_closed, color=INK, lw=2.0, marker="o", markersize=5,
            label="Closed form")
    ax.plot(Ts, p_ex, color=TEAL, lw=1.4, marker="s", markersize=4,
            linestyle="-", label="MC exact (ncx2)")
    ax.plot(Ts, p_em, color=BURG, lw=1.4, marker="^", markersize=4,
            linestyle="--", label="MC EM (truncated)")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Zero-coupon bond price P(0, T)")
    ax.set_title("Base CIR scenario: closed-form price vs Monte Carlo",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig3_convergence(df, path):
    fig, ax = _new_figure(figsize=(8.2, 4.8))
    for scheme in ("exact", "EM"):
        sub = df[df["scheme"] == scheme]
        ax.plot(sub["N"], sub["rel_err_pct"],
                color=TEAL if scheme == "exact" else BURG,
                linestyle=SCHEME_LINESTYLE[scheme],
                marker="o", markersize=4, lw=1.6,
                label=f"MC {scheme}")
        upper = sub["rel_err_pct"] + sub["se_bp"] / 1e4 / sub["P_closed"] * 100
        lower = (sub["rel_err_pct"]
                 - sub["se_bp"] / 1e4 / sub["P_closed"] * 100).clip(lower=0)
        ax.fill_between(sub["N"], lower, upper,
                        color=TEAL if scheme == "exact" else BURG, alpha=0.15)
    ax.axhline(1.0, color=INK, lw=1.0, linestyle=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo paths N (log scale)")
    ax.set_ylabel("Relative price error at T = 10y (%)")
    ax.set_title("Convergence of MC bond price to CIR closed form (base scenario)",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig4_residual_bp(df, path):
    fig, ax = _new_figure(figsize=(8.2, 4.8))
    scen_names = [s["name"] for s in SCENARIOS]
    n_scen = len(scen_names)
    x = np.arange(len(BOND_MATS))
    width = 0.10
    for i, name in enumerate(scen_names):
        for j, scheme in enumerate(("exact", "EM")):
            sub = df[(df["scenario"] == name)
                     & (df["scheme"] == scheme)].sort_values("T_yr")
            offset = ((i - (n_scen - 1) / 2) * 2 + j - 0.5) * width
            ax.bar(x + offset, sub["bp_residual"], width=width,
                   color=SCEN_COLORS[name],
                   alpha=0.85 if scheme == "exact" else 0.40,
                   label=f"{name} ({scheme})")
            ax.errorbar(x + offset, sub["bp_residual"], yerr=sub["se_bp"],
                        fmt="none", ecolor=INK, elinewidth=0.6, capsize=2)
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(T)}y" for T in BOND_MATS])
    ax.set_ylabel("Signed residual (P_MC - P_closed) in basis points")
    ax.set_title("Per-scenario residuals with +/- 1 MC standard error",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks.
# ---------------------------------------------------------------------------
def sanity_checks(df, conv_df):
    checks: dict = {}

    # (1) all 32 comparisons within the 1% bound
    checks["all_within_1pct"] = {
        "n_comparisons": int(len(df)),
        "n_within": int(df["within_1pct"].sum()),
        "pass": bool(df["within_1pct"].all()),
    }

    # (2) worst residual inside ~3 MC standard errors of zero
    worst = df.loc[df["rel_err_pct"].idxmax()]
    z = abs(worst["bp_residual"]) / worst["se_bp"] if worst["se_bp"] > 0 else 0.0
    checks["worst_within_3se"] = {
        "worst_rel_err_pct": float(worst["rel_err_pct"]),
        "worst_scenario": str(worst["scenario"]),
        "worst_scheme": str(worst["scheme"]),
        "worst_T": float(worst["T_yr"]),
        "z_residual_over_se": float(z),
        "pass": bool(z < 3.0),
    }

    # (3) convergence: error decays roughly as 1/sqrt(N) -> se_bp halves
    # each time N quadruples.  Check the exact-scheme se_bp ratio.
    conv_ex = conv_df[conv_df["scheme"] == "exact"].sort_values("N")
    se_500 = conv_ex[conv_ex["N"] == 500]["se_bp"].iloc[0]
    se_10000 = conv_ex[conv_ex["N"] == 10000]["se_bp"].iloc[0]
    ratio = se_500 / se_10000
    expected = np.sqrt(10000 / 500)  # ~4.47
    checks["convergence_root_n"] = {
        "se_ratio_500_to_10000": float(ratio),
        "expected_sqrt_ratio": float(expected),
        "pass": bool(abs(ratio - expected) / expected < 0.15),
    }

    # (4) production N=10,000 at least 10x inside the target on both schemes
    base_T10 = conv_df[conv_df["N"] == 10000]
    checks["production_10x_margin"] = {
        "max_rel_err_pct_at_N10000": float(base_T10["rel_err_pct"].max()),
        "pass": bool(base_T10["rel_err_pct"].max() < 0.1),
    }

    # (5) closed-form prices strictly increasing in sigma at fixed T
    # (higher vol -> higher bond price via the convexity of exp(-int r))
    mono_ok = True
    for T in BOND_MATS:
        prices_by_ratio = [
            df[(df["scenario"] == s["name"]) & (df["T_yr"] == T)]["P_closed"].iloc[0]
            for s in sorted(SCENARIOS, key=lambda s: s["sigma"])
        ]
        if not all(prices_by_ratio[i] < prices_by_ratio[i + 1]
                   for i in range(len(prices_by_ratio) - 1)):
            mono_ok = False
    checks["price_monotone_in_sigma"] = {
        "pass": bool(mono_ok),
    }
    return checks


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Phase D, Step 23: CIR closed form vs Monte Carlo validation")
    print("-" * 64)
    print(f"r0 = {R0}, dt = {DT:.6f}, N = {N_PATHS}, "
          f"T_max = {HORIZON_YEARS}, seed = {SEED}")
    print(f"CIR base: a = {A}, b = {B_MEAN}, sigma = {SIGMA_BASE}")
    for sc in SCENARIOS:
        print(f"  scenario {sc['name']:8s}: sigma = {sc['sigma']:.6f}, "
              f"Feller ratio 2ab/sigma^2 = {sc['ratio']:.4f}")
    print()

    # ---- Per-scenario validation ----------------------------------------
    all_rows = []
    for sc in SCENARIOS:
        all_rows.extend(validate_scenario(sc, BOND_MATS))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "step23_validation.csv", index=False)
    print("Per-(scenario, scheme, maturity) validation:")
    print(df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Scenario-level rollup ------------------------------------------
    rollup = (df.groupby(["scenario", "feller_ratio", "scheme"], as_index=False)
                .agg(max_rel_err_pct=("rel_err_pct", "max"),
                     mean_rel_err_pct=("rel_err_pct", "mean"),
                     max_abs_bp=("bp_residual", lambda s: float(np.abs(s).max())),
                     all_within_1pct=("within_1pct", "all")))
    rollup.to_csv(OUTPUT_DIR / "step23_scenario_summary.csv", index=False)
    print("Scenario rollup:")
    print(rollup.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Convergence ----------------------------------------------------
    conv_rows = convergence_sweep([CONV_T], CONV_N)
    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(OUTPUT_DIR / "step23_convergence.csv", index=False)
    print(f"Convergence sweep at T = {CONV_T:.0f}y (base scenario):")
    print(conv_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Pass / fail verdict --------------------------------------------
    worst = df["rel_err_pct"].max()
    worst_row = df.loc[df["rel_err_pct"].idxmax()]
    verdict_pass = bool(df["within_1pct"].all())
    print(f"Worst relative error: {worst:.4f}% at "
          f"(scenario={worst_row['scenario']}, scheme={worst_row['scheme']}, "
          f"T={worst_row['T_yr']:.0f}y)")
    print(f"All comparisons within 1% target: {verdict_pass}")
    print()

    # ---- Figures --------------------------------------------------------
    fig1_rel_error(df, FIGURES_DIR / "step23_fig1_rel_error.png")
    fig2_price_overlay(df, FIGURES_DIR / "step23_fig2_price_overlay.png")
    fig3_convergence(conv_df, FIGURES_DIR / "step23_fig3_convergence.png")
    fig4_residual_bp(df, FIGURES_DIR / "step23_fig4_residual_bp.png")

    # ---- Sanity checks --------------------------------------------------
    checks = sanity_checks(df, conv_df)
    print("=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")
    print()

    # ---- JSON summary ---------------------------------------------------
    summary = {
        "config": dict(r0=R0, dt=DT, n_paths=N_PATHS,
                       horizon_years=HORIZON_YEARS, seed=SEED, a=A, b=B_MEAN,
                       sigma_base=SIGMA_BASE, maturities=BOND_MATS),
        "scenarios": [{"name": s["name"], "sigma": s["sigma"], "ratio": s["ratio"]}
                      for s in SCENARIOS],
        "rollup": rollup.to_dict(orient="records"),
        "convergence": conv_df.to_dict(orient="records"),
        "worst_rel_err_pct": float(worst),
        "worst_case": worst_row.to_dict(),
        "all_within_1pct": verdict_pass,
        "sanity_checks": checks,
        "runtime_s": round(time.time() - t0, 2),
    }
    with open(OUTPUT_DIR / "step23_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"figures written to {FIGURES_DIR}")
    print(f"output written to  {OUTPUT_DIR}")
    print(f"runtime: {summary['runtime_s']}s")
    return summary


if __name__ == "__main__":
    main()
