"""
models/cir/step21_feller_sensitivity.py
---------------------------------------
Phase D, Step 21: Verify and stress-test the Feller condition for the
Cox-Ingersoll-Ross (CIR) short-rate model used in Phase D, Step 20.

The CIR SDE under Q is

    dr_t = a (b - r_t) dt + sigma * sqrt(r_t) dW_t,

and the Feller condition

    2 a b >= sigma^2

is the strict-positivity criterion: when it holds the boundary r = 0 is
unattainable in continuous time; when it fails the boundary is hit with
positive probability over any positive horizon.  This step (a) confirms
the chosen calibration sits well above the boundary and (b) runs a
controlled one-knob sensitivity in which sigma is moved so that
2ab / sigma^2 takes three values: 1.5 (well above the boundary), 1.0
(exactly on the boundary), and 0.7 (below the boundary).  For each
scenario both the full-truncation Euler-Maruyama (EM) driver and the
exact non-central chi-squared (ncx2) driver from Step 20 are run on the
same seed; the frequency of zero-rate paths is recorded as

    touch-frequency    = fraction of paths with min_{0<=t<=T} r_t <= 0,
    terminal-frequency = fraction of paths with r_T <= 0.

The EM scheme can register negative r_k purely from the discretisation,
even when Feller holds, because the diffusion is evaluated at max(r_k, 0)
while the propagated state is unprojected.  The exact ncx2 driver cannot
return r < 0 by construction and instead returns vanishingly small
strictly-positive values when the Feller boundary is approached.  The
gap between the two drivers decomposes the zero-touch count into a
genuine boundary-attainment component (visible in both drivers) and a
discretisation component (visible only in EM).

Reproducibility: seed = 42 reused from Steps 13, 16, 17, 18, 20 so the
random-number streams are bitwise consistent with Step 20.  This file is
a dedicated mini-analysis; it does not alter the Step 20 configuration or
its committed CSVs/JSON.

Outputs (aligned with Steps 13-20 artefact layout):

    models/cir/step21_scenario_summary.csv
    models/cir/step21_horizon_table.csv
    models/cir/step21_summary.json
    figures/step21/step21_fig1_paths.png
    figures/step21/step21_fig2_pathwise_min.png
    figures/step21/step21_fig3_zero_freq.png
    figures/step21/step21_fig4_truncation.png
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
# Baseline configuration, identical to Phase D, Step 20.
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

# Baseline Feller margin: 2ab = 0.04, sigma^2 = 0.0064, ratio = 6.25.
FELLER_LHS = 2.0 * A * B_MEAN
FELLER_RHS = SIGMA ** 2
FELLER_RATIO = FELLER_LHS / FELLER_RHS
FELLER_OK = FELLER_LHS >= FELLER_RHS

# Sensitivity sweep. Vary sigma only so the long-run level b and the
# mean-reversion speed a (both calibrated in Phase B) stay fixed. The
# required sigma at a target ratio is sigma = sqrt(2ab / ratio).
SCENARIOS = [
    {"name": "above",    "ratio": 1.5},
    {"name": "boundary", "ratio": 1.0},
    {"name": "below",    "ratio": 0.7},
]
for sc in SCENARIOS:
    sc["a"] = A
    sc["b"] = B_MEAN
    sc["sigma"] = float(np.sqrt(FELLER_LHS / sc["ratio"]))
    sc["r0"] = R0

OUTPUT_DIR = ROOT / "models" / "cir"
FIGURES_DIR = ROOT / "figures" / "step21"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette; matches Steps 16-20.
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
# Simulators (kept local so this file is self-contained and reproducible
# from a clean checkout; byte-identical to the Step 20 drivers).
# ---------------------------------------------------------------------------
def simulate_em_full_truncation(a, b, sigma, r0, horizon, dt, n_paths, seed):
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
def feller_diagnostic(a, b, sigma):
    return {
        "feller_lhs_2ab": 2.0 * a * b,
        "feller_rhs_sigma2": sigma ** 2,
        "feller_ratio": (2.0 * a * b) / sigma ** 2,
        "feller_holds": bool(2.0 * a * b >= sigma ** 2),
        "boundary_attainable": bool(2.0 * a * b < sigma ** 2),
    }


def zero_frequency(paths, eps=0.0):
    """Touch and terminal frequencies of zero-rate events."""
    pathwise_min = paths.min(axis=1)
    return {
        "touch_freq": float((pathwise_min <= eps).mean()),
        "terminal_freq": float((paths[:, -1] <= eps).mean()),
        "pathwise_min_mean": float(pathwise_min.mean()),
        "pathwise_min_p05": float(np.quantile(pathwise_min, 0.05)),
        "pathwise_min_min": float(pathwise_min.min()),
    }


def horizon_zero_table(paths, dt, horizons, label):
    rows = []
    for T in horizons:
        k = int(round(T / dt))
        sub = paths[:, : k + 1]
        z = zero_frequency(sub)
        rows.append({
            "scenario": label,
            "T_yr": T,
            "touch_freq": z["touch_freq"],
            "terminal_freq": z["terminal_freq"],
            "pathwise_min_mean": z["pathwise_min_mean"],
            "pathwise_min_p05": z["pathwise_min_p05"],
            "pathwise_min_min": z["pathwise_min_min"],
        })
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_sample_paths(em_by_scenario, exact_by_scenario, dt, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    fig.patch.set_facecolor(BG)
    n_show = 8
    t_grid = np.arange(next(iter(em_by_scenario.values())).shape[1]) * dt
    titles = {"above": "Above boundary (2ab = 1.5 sigma^2)",
              "boundary": "On boundary (2ab = sigma^2)",
              "below": "Below boundary (2ab = 0.7 sigma^2)"}
    for ax, name in zip(axes, ("above", "boundary", "below")):
        _style_axes(ax)
        em = em_by_scenario[name]
        ex = exact_by_scenario[name]
        for i in range(n_show):
            ax.plot(t_grid, em[i], color=BLUE, lw=0.8, alpha=0.65,
                    label="EM full trunc." if i == 0 else None)
            ax.plot(t_grid, ex[i], color=BURG, lw=0.8, alpha=0.65,
                    linestyle="--", label="Exact ncx2" if i == 0 else None)
        ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
        ax.set_xlabel("t (years)")
        ax.set_ylabel(r"$r_t$")
        ax.set_title(titles[name], fontsize=9)
        ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
                  labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_pathwise_min_density(em_by_scenario, exact_by_scenario, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        _style_axes(ax)
    colors = {"above": TEAL, "boundary": OCHRE, "below": BURG}
    titles = {"above": "2ab = 1.5 sigma^2", "boundary": "2ab = sigma^2",
              "below": "2ab = 0.7 sigma^2"}
    for ax, label, paths_by in [
        (axes[0], "EM full truncation", em_by_scenario),
        (axes[1], "Exact ncx2", exact_by_scenario),
    ]:
        for name in ("above", "boundary", "below"):
            pm = paths_by[name].min(axis=1)
            ax.hist(pm, bins=80, density=True, color=colors[name],
                    alpha=0.45, label=titles[name])
        ax.axvline(0.0, color=INK, lw=1.0, linestyle=":")
        ax.set_xlabel(r"pathwise minimum of $r_t$ over [0, T]")
        ax.set_ylabel("density")
        ax.set_title(label, fontweight="bold", pad=8)
        ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
                  labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig3_zero_frequency_bars(em_freqs, exact_freqs, path):
    labels = ["Above\n2ab = 1.5 sigma^2", "Boundary\n2ab = sigma^2",
              "Below\n2ab = 0.7 sigma^2"]
    order = ("above", "boundary", "below")
    em_touch = [em_freqs[n]["touch_freq"] for n in order]
    em_term = [em_freqs[n]["terminal_freq"] for n in order]
    ex_touch = [exact_freqs[n]["touch_freq"] for n in order]
    ex_term = [exact_freqs[n]["terminal_freq"] for n in order]

    fig, ax = _new_figure(figsize=(9.5, 4.8))
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 1.5 * w, em_touch, width=w, color=BLUE, label="EM, touch freq.")
    ax.bar(x - 0.5 * w, em_term, width=w, color=BLUE, alpha=0.45,
           label="EM, terminal freq.")
    ax.bar(x + 0.5 * w, ex_touch, width=w, color=BURG,
           label="Exact, touch freq.")
    ax.bar(x + 1.5 * w, ex_term, width=w, color=BURG, alpha=0.45,
           label="Exact, terminal freq.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of paths")
    ax.set_title("Zero-rate path frequency at T = 10y across Feller scenarios",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig4_truncation_activity(trunc_by_scenario, dt, path):
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), sharex=True)
    fig.patch.set_facecolor(BG)
    titles = {"above": "Above boundary (2ab = 1.5 sigma^2)",
              "boundary": "On boundary (2ab = sigma^2)",
              "below": "Below boundary (2ab = 0.7 sigma^2)"}
    for ax, name in zip(axes, ("above", "boundary", "below")):
        _style_axes(ax)
        tc = trunc_by_scenario[name]
        t_grid = np.arange(tc.shape[0]) * dt + dt
        ax.bar(t_grid, tc, width=dt * 0.9, color=OCHRE, alpha=0.85)
        ax.set_ylabel("Paths truncated")
        ax.set_title(titles[name], fontsize=9)
    axes[-1].set_xlabel("t (years)")
    fig.suptitle("EM full-truncation activity per step under each Feller scenario",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks(em_freq_by, exact_freq_by, horizon_df, scenarios):
    checks: dict = {}

    # (1) Baseline Feller margin
    base = feller_diagnostic(A, B_MEAN, SIGMA)
    checks["baseline_feller_margin"] = {
        "ratio": base["feller_ratio"],
        "holds": base["feller_holds"],
        "pass": bool(base["feller_holds"] and base["feller_ratio"] > 1.0),
    }

    # (2) One-knob arithmetic: recomputed sigma hits target ratio exactly
    max_dev = 0.0
    for sc in scenarios:
        achieved = (2.0 * sc["a"] * sc["b"]) / sc["sigma"] ** 2
        max_dev = max(max_dev, abs(achieved - sc["ratio"]))
    checks["one_knob_arithmetic"] = {
        "max_ratio_deviation": float(max_dev),
        "pass": bool(max_dev < 1e-9),
    }

    # (3) Monotonicity: EM touch-freq strictly increasing across scenarios at
    # every horizon; exact pathwise-min p05 strictly decreasing.
    order = ("above", "boundary", "below")
    touch_mono = True
    p05_mono = True
    for T in HORIZONS:
        em_touch = [
            horizon_df[(horizon_df["scenario"] == f"{n}/EM")
                       & (horizon_df["T_yr"] == T)]["touch_freq"].iloc[0]
            for n in order
        ]
        ex_p05 = [
            horizon_df[(horizon_df["scenario"] == f"{n}/Exact")
                       & (horizon_df["T_yr"] == T)]["pathwise_min_p05"].iloc[0]
            for n in order
        ]
        if not (em_touch[0] < em_touch[1] < em_touch[2]):
            touch_mono = False
        if not (ex_p05[0] > ex_p05[1] > ex_p05[2]):
            p05_mono = False
    checks["monotonicity"] = {
        "em_touch_increasing": touch_mono,
        "exact_p05_decreasing": p05_mono,
        "pass": bool(touch_mono and p05_mono),
    }

    # (4) Exact driver never strictly negative across scenarios
    exact_nonneg = all(
        exact_freq_by[n]["pathwise_min_min"] >= 0.0 for n in order)
    checks["exact_driver_nonnegative"] = {
        "min_exact_pathwise_min": float(
            min(exact_freq_by[n]["pathwise_min_min"] for n in order)),
        "pass": bool(exact_nonneg),
    }

    # (5) Boundary-attainment signature: below-boundary exact p05 collapses
    # to within 1e-3 of zero at T = 10y
    below_p05_T10 = horizon_df[
        (horizon_df["scenario"] == "below/Exact")
        & (horizon_df["T_yr"] == 10.0)]["pathwise_min_p05"].iloc[0]
    checks["boundary_attainment_signature"] = {
        "below_exact_p05_T10": float(below_p05_T10),
        "tolerance": 1e-3,
        "pass": bool(abs(below_p05_T10) < 1e-3),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase D, Step 21: Verify and stress-test the Feller condition")
    print("-" * 64)
    print(f"Baseline a = {A}, b = {B_MEAN}, sigma = {SIGMA}, r0 = {R0}")
    print(f"Baseline 2ab = {FELLER_LHS:.6f}, sigma^2 = {FELLER_RHS:.6f},  "
          f"ratio = {FELLER_RATIO:.4f}, holds = {FELLER_OK}")
    print()
    print("Sensitivity scenarios (sigma chosen to hit each target ratio):")
    for sc in SCENARIOS:
        print(f"  {sc['name']:8s}  ratio = {sc['ratio']:.2f}  "
              f"=> sigma = {sc['sigma']:.6f}  "
              f"(2ab = {2.0*sc['a']*sc['b']:.6f}, "
              f"sigma^2 = {sc['sigma']**2:.6f})")
    print()

    em_paths_by = {}
    exact_paths_by = {}
    trunc_by = {}
    em_freq_by = {}
    exact_freq_by = {}
    feller_by = {}

    horizon_rows = []
    summary_rows = []

    for sc in SCENARIOS:
        name = sc["name"]
        a, b, sigma, r0 = sc["a"], sc["b"], sc["sigma"], sc["r0"]
        em, tc = simulate_em_full_truncation(
            a, b, sigma, r0, HORIZON_YEARS, DT, N_PATHS, SEED)
        ex = simulate_exact_ncx2(
            a, b, sigma, r0, HORIZON_YEARS, DT, N_PATHS, SEED)
        em_paths_by[name] = em
        exact_paths_by[name] = ex
        trunc_by[name] = tc
        em_freq_by[name] = zero_frequency(em)
        exact_freq_by[name] = zero_frequency(ex)
        feller_by[name] = feller_diagnostic(a, b, sigma)

        horizon_rows.extend(horizon_zero_table(em, DT, HORIZONS, f"{name}/EM"))
        horizon_rows.extend(
            horizon_zero_table(ex, DT, HORIZONS, f"{name}/Exact"))

        summary_rows.append({
            "scenario": name,
            "ratio": sc["ratio"],
            "sigma": sigma,
            "2ab": 2.0 * a * b,
            "sigma^2": sigma ** 2,
            "feller_holds": feller_by[name]["feller_holds"],
            "EM_touch_freq": em_freq_by[name]["touch_freq"],
            "EM_terminal_freq": em_freq_by[name]["terminal_freq"],
            "Exact_touch_freq": exact_freq_by[name]["touch_freq"],
            "Exact_terminal_freq": exact_freq_by[name]["terminal_freq"],
            "EM_pathwise_min_min": em_freq_by[name]["pathwise_min_min"],
            "Exact_pathwise_min_min": exact_freq_by[name]["pathwise_min_min"],
            "EM_total_truncations": int(tc.sum()),
            "EM_steps_with_any_trunc": int((tc > 0).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    summary_df.to_csv(OUTPUT_DIR / "step21_scenario_summary.csv", index=False)
    horizon_df.to_csv(OUTPUT_DIR / "step21_horizon_table.csv", index=False)

    print("Scenario summary (T = 10y):")
    print(summary_df.to_string(index=False,
                               float_format=lambda v: f"{v:.6f}"))
    print()
    print("Horizon-by-horizon zero-rate frequencies:")
    print(horizon_df.to_string(index=False,
                               float_format=lambda v: f"{v:.6f}"))
    print()

    fig1_sample_paths(em_paths_by, exact_paths_by, DT,
                      FIGURES_DIR / "step21_fig1_paths.png")
    fig2_pathwise_min_density(em_paths_by, exact_paths_by,
                              FIGURES_DIR / "step21_fig2_pathwise_min.png")
    fig3_zero_frequency_bars(em_freq_by, exact_freq_by,
                             FIGURES_DIR / "step21_fig3_zero_freq.png")
    fig4_truncation_activity(trunc_by, DT,
                             FIGURES_DIR / "step21_fig4_truncation.png")

    checks = sanity_checks(em_freq_by, exact_freq_by, horizon_df, SCENARIOS)
    print("=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")
    print()

    summary = {
        "config": dict(a=A, b=B_MEAN, sigma=SIGMA, r0=R0, dt=DT,
                       n_paths=N_PATHS, horizon_years=HORIZON_YEARS,
                       seed=SEED),
        "baseline_feller": feller_diagnostic(A, B_MEAN, SIGMA),
        "scenarios": [
            {
                "name": sc["name"],
                "ratio": sc["ratio"],
                "a": sc["a"], "b": sc["b"], "sigma": sc["sigma"],
                "r0": sc["r0"],
                "feller": feller_by[sc["name"]],
                "EM": em_freq_by[sc["name"]],
                "Exact": exact_freq_by[sc["name"]],
                "EM_total_truncations": int(trunc_by[sc["name"]].sum()),
                "EM_steps_with_any_trunc": int((trunc_by[sc["name"]] > 0).sum()),
            }
            for sc in SCENARIOS
        ],
        "horizon_table": horizon_df.to_dict(orient="records"),
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step21_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"figures written to {FIGURES_DIR}")
    print(f"output written to  {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
