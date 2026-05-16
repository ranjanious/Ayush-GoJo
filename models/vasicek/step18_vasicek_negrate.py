"""
models/vasicek/step18_vasicek_negrate.py
----------------------------------------
Phase C, Step 18: negative-rate diagnostics for the Vasicek simulator.

The Vasicek SDE under Q is

    dr_t = a (b - r_t) dt + sigma dW_t,

which admits the exact Gaussian marginal law

    r_t | r_0 ~ N(mu_t, v_t),
        mu_t = b + (r_0 - b) * exp(-a * t),
        v_t  = sigma**2 * (1 - exp(-2 a t)) / (2 a).

Because the marginal is Gaussian, the negative-rate probability at any fixed
horizon has a clean closed form:

    P(r_t < 0) = Phi(-mu_t / sqrt(v_t)).

This step mirrors Phase B, Step 10 (the DBM negative-rate diagnostics) but
swaps the DBM simulator for two Vasicek simulators: the Step 13 Euler-
Maruyama driver and the Step 17 exact-transition driver.  Same-seed coupling
across the two drivers eliminates Monte Carlo noise from the side-by-side
comparison; the only remaining gap is discretisation.  The Step 10 DBM
numbers are loaded as a constant table so the headline three-way comparison
(Vasicek-EM vs Vasicek-exact vs DBM) is reproducible from the artefacts
filed in Phase B without re-running the DBM simulator.

The first-passage probability P(min_{0<=t<=T} r_t < 0) does not have an
elementary closed form for the Vasicek process; we report the simulated
grid estimate only.  The DBM analytical first-passage benchmark from
Step 10 is still printed for comparison.

Reproducibility: seed = 42 reused from Steps 13, 16, 17, and 10 so the
Brownian increment stream is bitwise identical to those modules.

Outputs (aligned with Steps 13-17 artefact layout):

    models/vasicek/step18_main.csv
    models/vasicek/step18_couple.csv
    models/vasicek/step18_analytic.csv
    models/vasicek/step18_summary.json
    figures/step18/step18_fig1_p_any.png
    figures/step18/step18_fig2_p_term.png
    figures/step18/step18_fig3_terminal_density.png
    figures/step18/step18_fig4_min_rate.png
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

# ---------------------------------------------------------------------------
# Configuration; identical to Steps 13, 16, 17
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
DBM_SIGMA = 0.01  # Step 10 parameters
DBM_MU = 0.0

OUTPUT_DIR = ROOT / "models" / "vasicek"
FIGURES_DIR = ROOT / "figures" / "step18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Step 10 DBM headline numbers, filed verbatim from
# models/dbm/step10_dbm_diagnostics.py reproducible run.
# Columns: T_years, P_any_sim, P_any_ana, P_term_sim, P_term_ana, min_rate
DBM_STEP10 = {
    1.0:  dict(any_sim=0.00010, any_ana=0.00020,
               term_sim=0.00010, term_ana=0.00010, min_rate=-0.00092),
    2.0:  dict(any_sim=0.00510, any_ana=0.00853,
               term_sim=0.00360, term_ana=0.00426, min_rate=-0.01599),
    5.0:  dict(any_sim=0.08080, any_ana=0.09619,
               term_sim=0.04750, term_ana=0.04809, min_rate=-0.04292),
    10.0: dict(any_sim=0.21940, any_ana=0.23945,
               term_sim=0.11510, term_ana=0.11972, min_rate=-0.08421),
}

# Project palette (matches Steps 14-17)
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
# Vasicek analytic marginal-moment and negative-rate formulas
# ---------------------------------------------------------------------------
def vasicek_mean(t, a, b, r0):
    return b + (r0 - b) * np.exp(-a * t)


def vasicek_var(t, a, sigma):
    return sigma ** 2 * (1.0 - np.exp(-2.0 * a * t)) / (2.0 * a)


def vasicek_p_terminal_negative(t, a, b, sigma, r0):
    """Closed-form P(r_t < 0) for the Vasicek marginal at fixed t."""
    mu = vasicek_mean(t, a, b, r0)
    var = vasicek_var(t, a, sigma)
    return float(norm.cdf(-mu / np.sqrt(var)))


def vasicek_stationary_p_negative(a, b, sigma):
    """Stationary P(r < 0) under the Vasicek invariant law N(b, sigma^2 / (2a))."""
    std_inf = sigma / np.sqrt(2.0 * a)
    return float(norm.cdf(-b / std_inf))


# ---------------------------------------------------------------------------
# Simulators (reused from Step 17, locked to identical seed and RNG order)
# ---------------------------------------------------------------------------
def simulate_em(a, b, sigma, r0, horizon, dt, n_paths, seed):
    """Step 13 Euler-Maruyama Vasicek paths."""
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
    """Step 17 exact-transition Vasicek paths; same RNG order as simulate_em."""
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    mu_decay = np.exp(-a * dt)
    cond_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        r[:, k + 1] = b + (r[:, k] - b) * mu_decay + cond_std * Z
    return r


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def negative_rate_diagnostics(r_paths, horizons, dt):
    """Compute the three Step 10 metrics on a single path matrix."""
    out = {
        "horizons": list(horizons),
        "frac_any": [],
        "frac_term": [],
        "min_rate": [],
    }
    for T in horizons:
        k = int(round(T / dt))
        window = r_paths[:, : k + 1]
        mins = window.min(axis=1)
        out["frac_any"].append(float((mins < 0.0).mean()))
        out["frac_term"].append(float((r_paths[:, k] < 0.0).mean()))
        out["min_rate"].append(float(mins.min()))
    out["global_min"] = float(r_paths.min())
    out["global_any"] = float((r_paths.min(axis=1) < 0.0).mean())
    return out


def analytic_table(horizons, a, b, sigma, r0):
    rows = []
    for T in horizons:
        mu = vasicek_mean(T, a, b, r0)
        var = vasicek_var(T, a, sigma)
        rows.append({
            "T": T,
            "E_rt": mu,
            "Var_rt": var,
            "std_rt": np.sqrt(var),
            "z_score": -mu / np.sqrt(var),
            "P_term_ana": float(norm.cdf(-mu / np.sqrt(var))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_main_table(em_diag, exact_diag, ana, dbm):
    rows = []
    for i, T in enumerate(em_diag["horizons"]):
        rows.append({
            "T_yr": T,
            "P_any_em": em_diag["frac_any"][i],
            "P_any_exact": exact_diag["frac_any"][i],
            "P_term_em": em_diag["frac_term"][i],
            "P_term_exact": exact_diag["frac_term"][i],
            "P_term_ana_vasi": float(ana.iloc[i]["P_term_ana"]),
            "min_em": em_diag["min_rate"][i],
            "min_exact": exact_diag["min_rate"][i],
            "DBM_P_any_step10": dbm[T]["any_sim"],
            "DBM_P_term_step10": dbm[T]["term_sim"],
            "DBM_min_step10": dbm[T]["min_rate"],
        })
    return pd.DataFrame(rows)


def build_couple_table(em_paths, exact_paths, dt, horizons):
    """Same-seed pathwise coupling diagnostics on the negative-rate event."""
    rows = []
    for T in horizons:
        k = int(round(T / dt))
        mins_em = em_paths[:, : k + 1].min(axis=1)
        mins_ex = exact_paths[:, : k + 1].min(axis=1)
        neg_em = mins_em < 0.0
        neg_ex = mins_ex < 0.0
        agree = float((neg_em == neg_ex).mean())
        em_only = int(((neg_em) & (~neg_ex)).sum())
        ex_only = int(((~neg_em) & (neg_ex)).sum())
        rows.append({
            "T_yr": T,
            "agreement": agree,
            "em_only_count": em_only,
            "exact_only_count": ex_only,
            "min_gap_bp": float((mins_em - mins_ex).mean() * 1e4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_p_any_vs_horizon(em_diag, exact_diag, dbm, path):
    fig, ax = _new_figure()
    horizons = em_diag["horizons"]
    ax.plot(horizons, em_diag["frac_any"], "o-", color=BLUE, lw=1.8,
            label="Vasicek EM (sim)")
    ax.plot(horizons, exact_diag["frac_any"], "s--", color=BURG, lw=1.8,
            label="Vasicek exact (sim)")
    ax.plot(horizons, [dbm[T]["any_sim"] for T in horizons], "^:",
            color=OCHRE, lw=1.8, label="DBM Step 10 (sim)")
    ax.set_xlabel("Horizon T (years)")
    ax.set_ylabel(r"P(min $r_t$ < 0 over [0, T])")
    ax.set_title("Negative-rate touch frequency: Vasicek vs DBM",
                 fontweight="bold", pad=10)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_p_term_vs_horizon(em_diag, exact_diag, ana, dbm, path):
    fig, ax = _new_figure()
    horizons = em_diag["horizons"]
    ax.plot(horizons, em_diag["frac_term"], "o-", color=BLUE, lw=1.8,
            label="Vasicek EM (sim)")
    ax.plot(horizons, exact_diag["frac_term"], "s--", color=BURG, lw=1.8,
            label="Vasicek exact (sim)")
    ax.plot(horizons, ana["P_term_ana"].values, "x:", color=TEAL, lw=1.8,
            label=r"Vasicek analytic $\Phi(-\mu/\sigma)$")
    ax.plot(horizons, [dbm[T]["term_sim"] for T in horizons], "^:",
            color=OCHRE, lw=1.8, label="DBM Step 10 (sim)")
    ax.set_xlabel("Horizon T (years)")
    ax.set_ylabel(r"P($r_T$ < 0)")
    ax.set_title("Terminal-marginal negative probability",
                 fontweight="bold", pad=10)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
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

    mu = vasicek_mean(T, a, b, r0)
    var = vasicek_var(T, a, sigma)
    xs = np.linspace(
        min(r_ex.min(), r_em.min()) - 0.005,
        max(r_ex.max(), r_em.max()) + 0.005, 400,
    )
    pdf_vasi = norm.pdf(xs, loc=mu, scale=np.sqrt(var))
    pdf_dbm = norm.pdf(xs, loc=R0, scale=DBM_SIGMA * np.sqrt(T))

    ax.hist(r_em, bins=80, density=True, color=BLUE, alpha=0.35,
            label="Vasicek EM histogram")
    ax.hist(r_ex, bins=80, density=True, histtype="step", color=BURG,
            lw=1.6, label="Vasicek exact histogram")
    ax.plot(xs, pdf_vasi, color=TEAL, lw=1.8, label="Vasicek analytic pdf")
    ax.plot(xs, pdf_dbm, color=OCHRE, lw=1.4, linestyle="--",
            label="DBM analytic pdf")

    neg_mask = xs < 0.0
    if neg_mask.any():
        ax.fill_between(xs[neg_mask], 0.0, pdf_vasi[neg_mask],
                        color=BURG, alpha=0.35,
                        label="Vasicek negative tail")

    ax.axvline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xlabel(r"$r_T$")
    ax.set_ylabel("density")
    ax.set_title(f"Terminal short-rate density at T = {T:.0f}y; "
                 "Vasicek vs DBM",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig4_min_rate_extremes(em_diag, exact_diag, dbm, path):
    fig, ax = _new_figure()
    horizons = em_diag["horizons"]
    ax.plot(horizons, em_diag["min_rate"], "o-", color=BLUE, lw=1.8,
            label="Vasicek EM (min across paths)")
    ax.plot(horizons, exact_diag["min_rate"], "s--", color=BURG, lw=1.8,
            label="Vasicek exact (min across paths)")
    ax.plot(horizons, [dbm[T]["min_rate"] for T in horizons], "^:",
            color=OCHRE, lw=1.8, label="DBM Step 10 (min)")
    ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xlabel("Horizon T (years)")
    ax.set_ylabel(r"min $r_t$ (across 10,000 paths)")
    ax.set_title("Sampled extreme of the short rate",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks(em_diag, exact_diag, couple_table):
    checks = {}

    # (1) stationary closed-form: Phi(-b/(sigma/sqrt(2a))) = Phi(-4) ~ 3.17e-5
    stat_p = vasicek_stationary_p_negative(A, B_MEAN, SIGMA)
    expected_phi_minus4 = float(norm.cdf(-4.0))
    checks["stationary_closed_form"] = {
        "vasicek_stationary_p_negative": stat_p,
        "phi_minus4_reference": expected_phi_minus4,
        "abs_diff": abs(stat_p - expected_phi_minus4),
        "pass": bool(abs(stat_p - expected_phi_minus4) < 1e-12),
    }

    # (2) marginal-moment agreement at every horizon (re-confirmed from Step 17)
    # Allow >= 99.5% coupling agreement on the negative-rate indicator.
    min_agree = float(couple_table["agreement"].min())
    checks["coupling_agreement_above_995pct"] = {
        "min_agreement": min_agree,
        "threshold": 0.995,
        "pass": bool(min_agree >= 0.995),
    }

    # (3) sigma -> 0 collapse: deterministic ODE, no negative rates.
    r_zero = simulate_exact(A, B_MEAN, 0.0, R0,
                            HORIZON_YEARS, DT, 200, SEED)
    any_neg_zero = float((r_zero.min(axis=1) < 0.0).mean())
    min_zero = float(r_zero.min())
    checks["sigma_zero_collapse"] = {
        "any_neg_fraction": any_neg_zero,
        "global_min": min_zero,
        "pass": bool(any_neg_zero == 0.0 and min_zero > 0.0),
    }

    # (4) high-sigma stress: sigma = 4x calibrated value, expect ~15.87%
    # stationary P(r<0) and a comparable terminal MC estimate.
    r_high = simulate_exact(A, B_MEAN, 0.04, R0,
                            HORIZON_YEARS, DT, N_PATHS, SEED)
    p_term_high = float((r_high[:, -1] < 0.0).mean())
    stat_high = vasicek_stationary_p_negative(A, B_MEAN, 0.04)
    checks["high_sigma_stress"] = {
        "p_term_sim_T10": p_term_high,
        "stationary_p_negative": stat_high,
        "tol_pct": 0.02,
        "pass": bool(abs(p_term_high - stat_high) < 0.02),
    }

    # (5) two-order-of-magnitude DBM vs Vasicek headline ratio at T = 10y
    em_T10 = em_diag["frac_any"][-1]
    dbm_T10 = DBM_STEP10[10.0]["any_sim"]
    ratio = dbm_T10 / max(em_T10, 1e-9)
    checks["two_orders_of_magnitude_T10"] = {
        "dbm_p_any_T10": dbm_T10,
        "vasicek_em_p_any_T10": em_T10,
        "ratio": ratio,
        "threshold": 100.0,
        "pass": bool(ratio >= 100.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase C, Step 18: Vasicek negative-rate diagnostics")
    print("-" * 64)
    print(f"a = {A}, b = {B_MEAN}, sigma = {SIGMA}, r0 = {R0}")
    print(f"seed = {SEED}, N = {N_PATHS}, dt = {DT:.6f}, "
          f"T_max = {HORIZON_YEARS}")
    print()

    em_paths = simulate_em(A, B_MEAN, SIGMA, R0, HORIZON_YEARS,
                           DT, N_PATHS, SEED)
    exact_paths = simulate_exact(A, B_MEAN, SIGMA, R0, HORIZON_YEARS,
                                 DT, N_PATHS, SEED)

    em_diag = negative_rate_diagnostics(em_paths, HORIZONS, DT)
    exact_diag = negative_rate_diagnostics(exact_paths, HORIZONS, DT)
    ana = analytic_table(HORIZONS, A, B_MEAN, SIGMA, R0)

    main_table = build_main_table(em_diag, exact_diag, ana, DBM_STEP10)
    couple_table = build_couple_table(em_paths, exact_paths, DT, HORIZONS)

    main_table.to_csv(OUTPUT_DIR / "step18_main.csv", index=False)
    couple_table.to_csv(OUTPUT_DIR / "step18_couple.csv", index=False)
    ana.to_csv(OUTPUT_DIR / "step18_analytic.csv", index=False)

    # Console report
    print("Main diagnostic table "
          "(sim_em / sim_exact / Vasicek analytical / DBM Step 10):")
    print(main_table.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Same-seed coupling diagnostics:")
    print(couple_table.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    stat_p = vasicek_stationary_p_negative(A, B_MEAN, SIGMA)
    print(f"Vasicek stationary P(r < 0) = {stat_p:.3e}")
    print(f"Vasicek global min EM    = {em_diag['global_min']:+.5f}")
    print(f"Vasicek global min exact = {exact_diag['global_min']:+.5f}")
    print(f"Vasicek global any-neg EM    = {em_diag['global_any']:.5f}")
    print(f"Vasicek global any-neg exact = {exact_diag['global_any']:.5f}")
    print(f"DBM     global any-neg Step10 = {DBM_STEP10[10.0]['any_sim']:.5f}")
    print(f"DBM     global min   Step10 = {DBM_STEP10[10.0]['min_rate']:+.5f}")

    # Sanity checks
    checks = sanity_checks(em_diag, exact_diag, couple_table)
    print("\n=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    # Figures
    fig1_p_any_vs_horizon(em_diag, exact_diag, DBM_STEP10,
                          FIGURES_DIR / "step18_fig1_p_any.png")
    fig2_p_term_vs_horizon(em_diag, exact_diag, ana, DBM_STEP10,
                           FIGURES_DIR / "step18_fig2_p_term.png")
    fig3_terminal_density(em_paths, exact_paths, A, B_MEAN, SIGMA, R0,
                          10.0, DT,
                          FIGURES_DIR / "step18_fig3_terminal_density.png")
    fig4_min_rate_extremes(em_diag, exact_diag, DBM_STEP10,
                           FIGURES_DIR / "step18_fig4_min_rate.png")

    summary = {
        "config": dict(a=A, b=B_MEAN, sigma=SIGMA, r0=R0, dt=DT,
                       n_paths=N_PATHS, horizon_years=HORIZON_YEARS,
                       seed=SEED),
        "horizons": HORIZONS,
        "vasicek_em": em_diag,
        "vasicek_exact": exact_diag,
        "vasicek_analytic": ana.to_dict(orient="records"),
        "stationary_p_negative": stat_p,
        "dbm_step10": {str(k): v for k, v in DBM_STEP10.items()},
        "coupling": couple_table.to_dict(orient="records"),
        "sanity_checks": checks,
        "main_table": main_table.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "step18_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print()
    print(f"figures written to {FIGURES_DIR}")
    print(f"output written to  {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
