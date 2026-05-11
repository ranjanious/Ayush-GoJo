"""
models/vasicek/step15_validation.py
-----------------------------------
Phase C, Step 15: Validate the Vasicek Monte Carlo simulator against the
closed-form zero-coupon bond price.

Acceptance criterion: |P_MC(0, T) - P_closed(0, T)| / P_closed(0, T) <= 1%
for every test maturity.  Step 13 supplies the simulator; the closed-form
prices come from shared/vasicek_closed_form.py.

Outputs (written under models/vasicek/ and figures/step15/ to stay aligned
with the Step 13 / Step 14 artefact layout):

    models/vasicek/step15_validation.csv     per-maturity table (8 rows)
    models/vasicek/step15_convergence.csv    error vs. n_paths at T = 5
    models/vasicek/step15_dt_sensitivity.csv error vs. Delta_t at T = 5
    models/vasicek/step15_summary.json       machine-readable summary
    figures/step15/fig15_1_price_curve.png       MC vs. closed form
    figures/step15/fig15_2_relative_error.png    rel. error with 1% band
    figures/step15/fig15_3_path_convergence.png  N convergence (log-log)
    figures/step15/fig15_4_dt_bias.png           dt sensitivity (log-log)

Reproducibility: seed = 42 throughout; numpy default_rng.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.vasicek_closed_form import (  # noqa: E402
    vasicek_B,
    vasicek_lnA,
    vasicek_zcb_price,
)
from models.vasicek.vasicek_simulator import zcb_price_mc  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARAMS = dict(a=0.5, b=0.04, sigma=0.01, r0=0.0372)
DT_BASE = 1.0 / 12.0
N_PATHS_BASE = 10_000
SEED = 42

MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
CONVERGENCE_NS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
DT_GRID = [1.0 / 4.0, 1.0 / 12.0, 1.0 / 24.0, 1.0 / 52.0, 1.0 / 252.0]

OUTPUT_DIR = ROOT / "models" / "vasicek"
FIGURES_DIR = ROOT / "figures" / "step15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Project palette (Step 14 token set)
# ---------------------------------------------------------------------------
BG = "#fbf7ec"
INK = "#1c1a15"
SUB = "#5c544a"
RULE = "#c5b8a0"
BLUE = "#2c4a5e"   # closed-form / primary
BURG = "#8b2d3a"   # MC / error
OCHRE = "#b8842a"  # benchmark / acceptance band
TEAL = "#2d6a5f"   # secondary marker


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
# 1. Per-maturity validation table
# ---------------------------------------------------------------------------
def per_maturity_table() -> pd.DataFrame:
    rows = []
    for T in MATURITIES:
        P_cf = vasicek_zcb_price(
            0.0, T, PARAMS["r0"],
            PARAMS["a"], PARAMS["b"], PARAMS["sigma"],
        )
        P_mc, se, _ = zcb_price_mc(
            a=PARAMS["a"], b=PARAMS["b"], sigma=PARAMS["sigma"], r0=PARAMS["r0"],
            T=T, dt=DT_BASE, n_paths=N_PATHS_BASE, seed=SEED,
        )
        abs_err = P_mc - P_cf
        rel_err = abs_err / P_cf
        z_stat = abs_err / se if se > 0 else np.nan
        rows.append({
            "T_years": T,
            "P_closed_form": P_cf,
            "P_mc": P_mc,
            "mc_std_error": se,
            "abs_error": abs_err,
            "rel_error_pct": 100.0 * rel_err,
            "abs_rel_error_pct": 100.0 * abs(rel_err),
            "z_stat": z_stat,
            "passes_1pct": bool(abs(rel_err) <= 0.01),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step15_validation.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Path-count convergence
# ---------------------------------------------------------------------------
def path_convergence_table(T: float = 5.0) -> pd.DataFrame:
    P_cf = vasicek_zcb_price(
        0.0, T, PARAMS["r0"],
        PARAMS["a"], PARAMS["b"], PARAMS["sigma"],
    )
    rows = []
    for n in CONVERGENCE_NS:
        P_mc, se, _ = zcb_price_mc(
            a=PARAMS["a"], b=PARAMS["b"], sigma=PARAMS["sigma"], r0=PARAMS["r0"],
            T=T, dt=DT_BASE, n_paths=n, seed=SEED,
        )
        rows.append({
            "n_paths": n,
            "P_closed_form": P_cf,
            "P_mc": P_mc,
            "mc_std_error": se,
            "abs_error": P_mc - P_cf,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step15_convergence.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Time-step sensitivity
# ---------------------------------------------------------------------------
def dt_sensitivity_table(T: float = 5.0) -> pd.DataFrame:
    P_cf = vasicek_zcb_price(
        0.0, T, PARAMS["r0"],
        PARAMS["a"], PARAMS["b"], PARAMS["sigma"],
    )
    rows = []
    for dt in DT_GRID:
        P_mc, se, _ = zcb_price_mc(
            a=PARAMS["a"], b=PARAMS["b"], sigma=PARAMS["sigma"], r0=PARAMS["r0"],
            T=T, dt=dt, n_paths=N_PATHS_BASE, seed=SEED,
        )
        rows.append({
            "dt": dt,
            "dt_inv": int(round(1.0 / dt)),
            "P_closed_form": P_cf,
            "P_mc": P_mc,
            "mc_std_error": se,
            "abs_error": P_mc - P_cf,
            "rel_error_pct": 100.0 * (P_mc - P_cf) / P_cf,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "step15_dt_sensitivity.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_price_curve(df_main: pd.DataFrame) -> None:
    fig, ax = _new_figure((8.0, 4.8))
    ax.plot(df_main["T_years"], df_main["P_closed_form"], "-",
            color=BLUE, linewidth=2.0,
            label="Closed form $P(0, T)$", zorder=3)
    ax.errorbar(df_main["T_years"], df_main["P_mc"],
                yerr=1.96 * df_main["mc_std_error"], fmt="o",
                color=BURG, ecolor=BURG, elinewidth=1.0,
                capsize=3, markersize=6,
                label="Monte Carlo (10,000 paths, 95% CI)", zorder=4)
    ax.set_xlabel("Maturity $T$ (years)")
    ax.set_ylabel("Zero-coupon bond price $P(0, T)$")
    ax.set_title("Vasicek ZCB price: closed form vs. Monte Carlo",
                 fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
              loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_1_price_curve.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_relative_error(df_main: pd.DataFrame) -> None:
    fig, ax = _new_figure((8.0, 4.8))
    half = 100.0 * 1.96 * df_main["mc_std_error"] / df_main["P_closed_form"]
    ax.axhline(0.0, color=SUB, linewidth=0.8)
    ax.axhline(1.0, color=OCHRE, linewidth=1.0, linestyle="--",
               label="1% acceptance band")
    ax.axhline(-1.0, color=OCHRE, linewidth=1.0, linestyle="--")
    ax.errorbar(df_main["T_years"], df_main["rel_error_pct"],
                yerr=half, fmt="o-", color=BURG, ecolor=BURG,
                elinewidth=1.0, capsize=3, markersize=6, linewidth=1.2,
                label="$(P_{MC} - P_{cf}) / P_{cf}$, 95% MC CI")
    ax.set_xlabel("Maturity $T$ (years)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Monte Carlo relative error vs. closed form",
                 fontweight="bold", pad=12)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
              loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_2_relative_error.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_path_convergence(df_conv: pd.DataFrame) -> None:
    fig, ax = _new_figure((8.0, 4.8))
    abs_err = df_conv["abs_error"].abs()
    se = df_conv["mc_std_error"]
    ax.loglog(df_conv["n_paths"], abs_err, "o-", color=BURG, linewidth=1.4,
              markersize=7, label="$|P_{MC} - P_{cf}|$ at $T = 5$")
    ax.loglog(df_conv["n_paths"], se, "s--", color=BLUE, linewidth=1.2,
              markersize=6, label="MC standard error")
    n_arr = np.array(df_conv["n_paths"], dtype=float)
    ref = se.iloc[0] * np.sqrt(n_arr[0] / n_arr)
    ax.loglog(n_arr, ref, ":", color=SUB, linewidth=1.0,
              label="$N^{-1/2}$ reference")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Absolute error")
    ax.set_title("Monte Carlo path-count convergence (T = 5)",
                 fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
              loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_3_path_convergence.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


def figure_dt_bias(df_dt: pd.DataFrame) -> None:
    fig, ax = _new_figure((8.0, 4.8))
    ax.loglog(df_dt["dt"], df_dt["abs_error"].abs(), "o-", color=TEAL,
              linewidth=1.4, markersize=7,
              label="$|P_{MC} - P_{cf}|$ at $T = 5$")
    ax.loglog(df_dt["dt"], df_dt["mc_std_error"], "s--", color=BLUE,
              linewidth=1.2, markersize=6, label="MC standard error")
    dt_arr = np.array(df_dt["dt"], dtype=float)
    base = df_dt["abs_error"].abs().iloc[0]
    ref = base * (dt_arr / dt_arr[0])
    ax.loglog(dt_arr, ref, ":", color=SUB, linewidth=1.0,
              label=r"$O(\Delta t)$ reference")
    ax.set_xlabel(r"Time step $\Delta t$ (years)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Discretization bias vs. time step (T = 5)",
                 fontweight="bold", pad=12)
    ax.invert_xaxis()
    ax.legend(facecolor=BG, edgecolor=RULE, labelcolor=INK,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_4_dt_bias.png",
                dpi=220, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks (Section 5)
# ---------------------------------------------------------------------------
def sanity_checks(df_main: pd.DataFrame, df_conv: pd.DataFrame) -> dict:
    """Four internal cross-checks: sigma -> 0 limit, B(t,T) monotone,
    N^{-1/2} slope, dt bias dominated by MC noise."""
    checks: dict = {}

    # (1) sigma -> 0 limit: P = exp(-integ) where integ = b T + (r0 - b) B(0, T)
    a, b, r0 = PARAMS["a"], PARAMS["b"], PARAMS["r0"]
    T_sanity = 5.0
    P_sigma0 = vasicek_zcb_price(0.0, T_sanity, r0, a, b, 0.0)
    integ = b * T_sanity + (r0 - b) * vasicek_B(0.0, T_sanity, a)
    P_expected = float(np.exp(-integ))
    checks["sigma_zero_limit"] = {
        "P_sigma0": P_sigma0,
        "exp_minus_integ": P_expected,
        "abs_diff": abs(P_sigma0 - P_expected),
        "pass": abs(P_sigma0 - P_expected) < 1e-12,
    }

    # (2) B(t, T) monotone-increasing with limits B(t,t)=0, B->1/a
    B_vals = [vasicek_B(0.0, T, a) for T in MATURITIES]
    is_increasing = all(B_vals[i] < B_vals[i + 1] for i in range(len(B_vals) - 1))
    checks["B_monotone_increasing"] = {
        "B_values": B_vals,
        "limit_1_over_a": 1.0 / a,
        "pass": bool(is_increasing and B_vals[0] > 0 and B_vals[-1] < 1.0 / a),
    }

    # (3) MC SE slope on log-log axes against N: expect -0.5
    logN = np.log(df_conv["n_paths"].to_numpy(dtype=float))
    logSE = np.log(df_conv["mc_std_error"].to_numpy(dtype=float))
    slope, _intercept = np.polyfit(logN, logSE, 1)
    checks["mc_se_slope_vs_N"] = {
        "fitted_slope": float(slope),
        "target": -0.5,
        "abs_dev": float(abs(slope + 0.5)),
        "pass": bool(abs(slope + 0.5) < 0.05),
    }

    # (4) bias |err| / SE < 1 at every dt
    df_dt = pd.read_csv(OUTPUT_DIR / "step15_dt_sensitivity.csv")
    ratios = (df_dt["abs_error"].abs() / df_dt["mc_std_error"]).to_numpy()
    checks["bias_dominated_by_mc_noise"] = {
        "ratios_err_over_SE": ratios.tolist(),
        "max_ratio": float(ratios.max()),
        "pass": bool(ratios.max() < 1.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Phase C, Step 15: closed-form Vasicek validation")
    print("-" * 60)
    print(f"seed = {SEED} | dt = {DT_BASE:.6f} | n_paths = {N_PATHS_BASE}")
    print(f"params = a={PARAMS['a']}, b={PARAMS['b']}, "
          f"sigma={PARAMS['sigma']}, r0={PARAMS['r0']}")
    print()

    print("Building per-maturity validation table ...")
    df_main = per_maturity_table()

    print("Path-count convergence sweep (T=5) ...")
    df_conv = path_convergence_table(T=5.0)

    print("Time-step sensitivity sweep (T=5) ...")
    df_dt = dt_sensitivity_table(T=5.0)

    print("Rendering figures ...")
    figure_price_curve(df_main)
    figure_relative_error(df_main)
    figure_path_convergence(df_conv)
    figure_dt_bias(df_dt)

    print("Running sanity checks ...")
    checks = sanity_checks(df_main, df_conv)

    summary = {
        "parameters": PARAMS,
        "dt_base": DT_BASE,
        "n_paths_base": N_PATHS_BASE,
        "seed": SEED,
        "n_maturities": int(len(df_main)),
        "max_abs_rel_error_pct": float(df_main["abs_rel_error_pct"].max()),
        "median_abs_rel_error_pct": float(df_main["abs_rel_error_pct"].median()),
        "all_pass_1pct": bool(df_main["passes_1pct"].all()),
        "max_abs_z_stat": float(df_main["z_stat"].abs().max()),
        "median_abs_z_stat": float(df_main["z_stat"].abs().median()),
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step15_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, allow_nan=False)

    # --- Console summary -----------------------------------------------------
    print("\n=== Per-maturity validation table ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 140):
        print(df_main.to_string(index=False))

    print("\n=== Path-count convergence (T = 5) ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 140):
        print(df_conv.to_string(index=False))

    print("\n=== Time-step sensitivity (T = 5) ===")
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 140):
        print(df_dt.to_string(index=False))

    print("\n=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    print("\n=== Step 15 summary ===")
    verdict = "PASS" if summary["all_pass_1pct"] else "FAIL"
    print(f"  Verdict                : {verdict}")
    print(f"  Maturities tested      : {summary['n_maturities']}")
    print(f"  Max |rel. err.| (%)    : {summary['max_abs_rel_error_pct']:.4f}")
    print(f"  Median |rel. err.| (%) : {summary['median_abs_rel_error_pct']:.4f}")
    print(f"  Max |z|                : {summary['max_abs_z_stat']:.2f}")
    print(f"  Acceptance criterion   : <= 1.000% at every maturity")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
