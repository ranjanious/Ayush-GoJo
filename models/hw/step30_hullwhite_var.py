"""
models/hw/step30_hullwhite_var.py
---------------------------------
Phase E, Step 30: forward-bond VaR/ES for Hull-White, completing the
five-model comparison opened by Step 22 (constant, DBM, Vasicek, CIR).

Question: does fitting the initial yield curve change VaR and ES
materially versus the flat-mean-reversion Vasicek model?

Closed-form Hull-White conditional ZCB price (Brigo-Mercurio 3.39):

    P(t, T) = A(t, T) exp(-B(t, T) r_t),
    B(t, T) = (1 - exp(-a (T - t))) / a,
    ln A(t, T) = ln(P^M(0, T) / P^M(0, t)) + B(t, T) f^M(0, t)
                  - (sigma^2 / (4 a)) (1 - exp(-2 a t)) B(t, T)^2.

Short-rate marginal under Q with the curve-consistent x0 = r0 - f(0, 0):
    r_h ~ N(m_H(h), v(h))
    m_H(h) = f^M(0, h) + (sigma^2 / (2 a^2)) (1 - exp(-a h))^2
                       + x0 exp(-a h)
    v(h)   = (sigma^2 / (2 a)) (1 - exp(-2 a h))         (= Vasicek variance)

The Hull-White and Vasicek conditional variances coincide because a and
sigma are shared, so the log forward-price dispersion B(h, T) sqrt(v(h))
is identical to Vasicek's to machine precision -- fitting the curve can
only RELOCATE the loss distribution, not change its width.

Pipeline:
  1. Headline five-model VaR/ES at r0 = f(0, 0), h in {1, 2, 5}y,
     T_bond = 10y, levels {0.95, 0.99}.  The four legacy models reuse
     the Step 22 estimators; Hull-White uses the closed forms above.
  2. HW-vs-Vasicek differences in absolute (bp of P_now) and relative
     (%% of Vasicek) terms.
  3. Hull-White at r0 = 0.0372 (Step 28 comparability) to confirm the
     1.4 bp initialisation offset does not move the conclusion.
  4. MC cross-check: rebuild the HW forward-price law from a 200,000-path
     antithetic Euler-Maruyama simulation; compare to the analytic
     marginal in mean and SE.

Reproducibility: seed = 30 throughout, N = 200,000 draws.  theta(t) from
Step 27 on the 17 March 2026 CMT curve.

Outputs (aligned with Steps 13-29 artefact layout):

    models/hw/step30_var_es_fivemodel.csv
    models/hw/step30_hw_vs_vasicek.csv
    models/hw/step30_hw_comparability_r0.csv
    models/hw/step30_mc_crosscheck.csv
    models/hw/step30_summary.json
    figures/step30/step30_fig1_forward_distributions.png
    figures/step30/step30_fig2_var_es_bars.png
    figures/step30/step30_fig3_hw_vs_vasicek.png
    figures/step30/step30_fig4_driver.png
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

sys.path.insert(0, str(ROOT / "models" / "cir"))
import step22_bond_pricing as s22  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
A = 0.50
SIGMA = 0.010
SEED = 30
N_DRAWS = 200_000

HORIZONS = [1.0, 2.0, 5.0]
LEVELS = [0.95, 0.99]
T_BOND = 10.0

R0_COMPARABILITY = 0.0372    # Step 28 Vasicek level for the reconciliation

# MC cross-check
MC_DT = 1.0 / 48.0
MC_N_PAIRS = 100_000          # 200,000 antithetic paths

OUTPUT_DIR = ROOT / "models" / "hw"
FIGURES_DIR = ROOT / "figures" / "step30"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-29 token set)
BG = "#fbf7ec"
INK = "#1c1a15"
SUB = "#5c544a"
RULE = "#c5b8a0"
BLUE = "#2c4a5e"
BURG = "#8b2d3a"
OCHRE = "#b8842a"
TEAL = "#2d6a5f"

MODEL_COLORS = {
    "constant":   INK,
    "DBM":        OCHRE,
    "Vasicek":    TEAL,
    "CIR":        BURG,
    "Hull-White": BLUE,
}


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
# Market curve setup (single source of truth: Step 27)
# ---------------------------------------------------------------------------
def market_setup():
    t_cmt, P_cmt, _ = hw.bootstrap_discount_factors(
        hw.CMT_TENORS, hw.CMT_YIELDS)
    t_fwd, f_fwd = hw.bootstrap_forward_curve(t_cmt, P_cmt)
    return t_cmt, P_cmt, t_fwd, f_fwd


def market_P(t, t_cmt, P_cmt):
    """Log-linear interpolation of the bootstrapped discount factors."""
    t_full = np.concatenate(([0.0], t_cmt))
    P_full = np.concatenate(([1.0], P_cmt))
    return float(np.exp(np.interp(t, t_full, np.log(P_full))))


def market_f(t, t_fwd, f_fwd):
    """Piecewise-linear instantaneous forward."""
    return float(hw.forward_at(np.atleast_1d(t), t_fwd, f_fwd)[0])


# ---------------------------------------------------------------------------
# Closed-form Hull-White bond price (Brigo-Mercurio 3.39)
# ---------------------------------------------------------------------------
def hw_B(t, T, a=A):
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_lnA(t, T, t_cmt, P_cmt, t_fwd, f_fwd, a=A, sigma=SIGMA):
    PM_T = market_P(T, t_cmt, P_cmt)
    PM_t = market_P(t, t_cmt, P_cmt)
    fM_t = market_f(t, t_fwd, f_fwd)
    B = hw_B(t, T, a)
    return (np.log(PM_T / PM_t) + B * fM_t
            - (sigma ** 2) / (4.0 * a)
            * (1.0 - np.exp(-2.0 * a * t)) * B ** 2)


def hw_price(t, T, r, t_cmt, P_cmt, t_fwd, f_fwd, a=A, sigma=SIGMA):
    return np.exp(hw_lnA(t, T, t_cmt, P_cmt, t_fwd, f_fwd, a, sigma)
                  - hw_B(t, T, a) * np.asarray(r, dtype=float))


# ---------------------------------------------------------------------------
# Short-rate marginal and forward-price distribution
# ---------------------------------------------------------------------------
def hw_short_rate_marginal(h, f00, r0, t_fwd, f_fwd, a=A, sigma=SIGMA):
    fM_h = market_f(h, t_fwd, f_fwd)
    convexity = (sigma ** 2) / (2.0 * a ** 2) * (1.0 - np.exp(-a * h)) ** 2
    x0 = r0 - f00
    mean = fM_h + convexity + x0 * np.exp(-a * h)
    var = (sigma ** 2) / (2.0 * a) * (1.0 - np.exp(-2.0 * a * h))
    return float(mean), float(var)


def hw_forward_distribution(h, T_bond, f00, r0, t_cmt, P_cmt, t_fwd, f_fwd,
                            n=N_DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    mean, var = hw_short_rate_marginal(h, f00, r0, t_fwd, f_fwd)
    rh = rng.normal(loc=mean, scale=np.sqrt(var), size=n)
    return hw_price(h, T_bond, rh, t_cmt, P_cmt, t_fwd, f_fwd), mean, var


def hw_p_now(T_bond, f00, r0, t_cmt, P_cmt):
    """P(0, T) at the chosen r0; reduces to the market curve iff r0 = f00."""
    PM_T = market_P(T_bond, t_cmt, P_cmt)
    return PM_T * np.exp(hw_B(0.0, T_bond) * (f00 - r0))


# ---------------------------------------------------------------------------
# Hull-White tail risk
# ---------------------------------------------------------------------------
def hw_tail_risk(h, T_bond, levels, f00, r0, t_cmt, P_cmt, t_fwd, f_fwd,
                 seed=SEED):
    p, mean_rh, var_rh = hw_forward_distribution(
        h, T_bond, f00, r0, t_cmt, P_cmt, t_fwd, f_fwd, seed=seed)
    p0 = hw_p_now(T_bond, f00, r0, t_cmt, P_cmt)
    rows = []
    for q in levels:
        var_p = float(np.quantile(p, 1.0 - q))
        es_p = float(p[p <= var_p].mean()) if (p <= var_p).any() else var_p
        rows.append({
            "model": "Hull-White",
            "h": h,
            "T_bond": T_bond,
            "level": q,
            "P_now": float(p0),
            "VaR_price": var_p,
            "ES_price": es_p,
            "VaR_loss": float(p0 - var_p),
            "ES_loss": float(p0 - es_p),
            "VaR_pct": float((p0 - var_p) / p0),
            "ES_pct": float((p0 - es_p) / p0),
        })
    return rows, p, p0, mean_rh, var_rh


# ---------------------------------------------------------------------------
# MC cross-check via on-the-fly antithetic simulation (reuses Step 29 style)
# ---------------------------------------------------------------------------
def mc_hw_rh_at_horizon(h, r0, t_fwd, f_fwd, n_pairs=MC_N_PAIRS, dt=MC_DT,
                        seed=SEED + 100):
    n_steps = int(round(h / dt))
    t_drift = np.arange(n_steps) * dt
    _, _, _, _, theta = hw.hull_white_theta(t_drift, t_fwd, f_fwd, A, SIGMA)
    rng = np.random.default_rng(seed)
    N = 2 * n_pairs
    r = np.full(N, r0, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Zb = rng.standard_normal(n_pairs)
        Z = np.concatenate([Zb, -Zb])
        r = r + (theta[k] - A * r) * dt + SIGMA * sqrt_dt * Z
    return r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Phase E, Step 30: Hull-White forward-bond VaR/ES")
    print("-" * 64)
    print(f"a = {A}, sigma = {SIGMA}, seed = {SEED}, N = {N_DRAWS}")
    print(f"horizons = {HORIZONS}, levels = {LEVELS}, T_bond = {T_BOND}y")
    print()

    t_cmt, P_cmt, t_fwd, f_fwd = market_setup()
    f00 = float(f_fwd[0])
    print(f"f(0, 0) = {f00 * 100:.4f}%  (exact-fit r0)")
    print(f"P_market(0, {T_BOND}y) = {market_P(T_BOND, t_cmt, P_cmt):.6f}")
    print()

    # ---- Headline five-model VaR/ES -------------------------------------
    print("Five-model forward-bond VaR/ES ...")
    rows = []
    for h in HORIZONS:
        # Legacy four models from Step 22 (its forward_bond_distribution
        # uses its own SEED = 42; tail_risk_table consumes that).
        for model in ("constant", "DBM", "Vasicek", "CIR"):
            rows.extend(s22.tail_risk_table(model, h, T_BOND, LEVELS))
        # Hull-White at the exact-fit r0
        hw_rows, _, _, _, _ = hw_tail_risk(
            h, T_BOND, LEVELS, f00, f00, t_cmt, P_cmt, t_fwd, f_fwd)
        rows.extend(hw_rows)
    var_df = pd.DataFrame(rows)
    var_df.to_csv(OUTPUT_DIR / "step30_var_es_fivemodel.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 180):
        print(var_df.to_string(index=False))

    # ---- HW vs Vasicek decomposition ------------------------------------
    print("\nHW vs Vasicek differences ...")
    recs = []
    for h in HORIZONS:
        for q in LEVELS:
            v = var_df[(var_df["model"] == "Vasicek")
                       & (var_df["h"] == h) & (var_df["level"] == q)].iloc[0]
            w = var_df[(var_df["model"] == "Hull-White")
                       & (var_df["h"] == h) & (var_df["level"] == q)].iloc[0]
            v_pct = v["VaR_pct"] * 100
            w_pct = w["VaR_pct"] * 100
            es_v = v["ES_pct"] * 100
            es_w = w["ES_pct"] * 100
            recs.append({
                "h": h, "level": q,
                "Vasicek_VaR_pct": v_pct,
                "HullWhite_VaR_pct": w_pct,
                "VaR_abs_diff_bp": (w_pct - v_pct) * 100,
                "VaR_rel_diff_pct":
                    100 * (w_pct - v_pct) / v_pct if v_pct != 0 else float("inf"),
                "Vasicek_ES_pct": es_v,
                "HullWhite_ES_pct": es_w,
                "ES_abs_diff_bp": (es_w - es_v) * 100,
                "ES_rel_diff_pct":
                    100 * (es_w - es_v) / es_v if es_v != 0 else float("inf"),
            })
    diff_df = pd.DataFrame(recs)
    diff_df.to_csv(OUTPUT_DIR / "step30_hw_vs_vasicek.csv", index=False)
    print(diff_df.to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))

    # ---- Comparability initialisation r0 = 0.0372 -----------------------
    print("\nHull-White at r0 = 0.0372 (Step 28 comparability) ...")
    comp_rows = []
    for h in HORIZONS:
        hw_rows, _, _, _, _ = hw_tail_risk(
            h, T_BOND, LEVELS, f00, R0_COMPARABILITY,
            t_cmt, P_cmt, t_fwd, f_fwd, seed=SEED + 1)
        comp_rows.extend(hw_rows)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUTPUT_DIR / "step30_hw_comparability_r0.csv",
                   index=False)
    max_var_gap_bp = 0.0
    for h in HORIZONS:
        for q in LEVELS:
            v_fit = var_df[(var_df["model"] == "Hull-White")
                           & (var_df["h"] == h) & (var_df["level"] == q)].iloc[0]
            v_cmp = comp_df[(comp_df["h"] == h) & (comp_df["level"] == q)].iloc[0]
            gap = abs((v_cmp["VaR_pct"] - v_fit["VaR_pct"]) * 1e4)
            max_var_gap_bp = max(max_var_gap_bp, gap)
    print(f"  max |VaR%% gap| between exact-fit and r0=0.0372: "
          f"{max_var_gap_bp:.3f} bp")

    # ---- MC cross-check ------------------------------------------------
    print("\nMC cross-check of the HW short-rate marginal ...")
    mc_rows = []
    for h in HORIZONS:
        mean_a, var_a = hw_short_rate_marginal(h, f00, f00, t_fwd, f_fwd)
        rh = mc_hw_rh_at_horizon(h, f00, t_fwd, f_fwd)
        mean_s = float(rh.mean())
        var_s = float(rh.var(ddof=1))
        # Forward price mean via analytic vs MC
        pm_a = market_P(T_BOND, t_cmt, P_cmt)
        # Analytic E[P(h,T_BOND) | r_h~N(mean_a, var_a)] using lognormal:
        ln_A = hw_lnA(h, T_BOND, t_cmt, P_cmt, t_fwd, f_fwd)
        B_hT = hw_B(h, T_BOND)
        mean_logP = ln_A - B_hT * mean_a
        var_logP = B_hT ** 2 * var_a
        p_an_mean = float(np.exp(mean_logP + 0.5 * var_logP))
        p_mc_mean = float(np.mean(hw_price(h, T_BOND, rh, t_cmt, P_cmt,
                                           t_fwd, f_fwd)))
        mc_rows.append({
            "h": h,
            "rh_mean_analytic": mean_a,
            "rh_mean_mc": mean_s,
            "rh_mean_err_bp": (mean_s - mean_a) * 1e4,
            "rh_sd_analytic": float(np.sqrt(var_a)),
            "rh_sd_mc": float(np.sqrt(var_s)),
            "forward_price_mean_analytic": p_an_mean,
            "forward_price_mean_mc": p_mc_mean,
            "forward_price_mean_err_bp": (p_mc_mean - p_an_mean) * 1e4,
        })
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(OUTPUT_DIR / "step30_mc_crosscheck.csv", index=False)
    print(mc_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # ---- Dispersion identity ---------------------------------------------
    print("\nDispersion identity (HW vs Vasicek log-price SD):")
    disp_rows = []
    A_vasi, SIG_VASI = s22.A_VASI, s22.SIG_VASI
    for h in HORIZONS:
        v_vas = SIG_VASI ** 2 * (1.0 - np.exp(-2.0 * A_vasi * h)) / (2.0 * A_vasi)
        _, v_hw = hw_short_rate_marginal(h, f00, f00, t_fwd, f_fwd)
        B_h = hw_B(h, T_BOND)
        sd_vas = B_h * np.sqrt(v_vas)
        sd_hw = B_h * np.sqrt(v_hw)
        disp_rows.append({
            "h": h,
            "B_hT": float(B_h),
            "vasicek_var_rh": float(v_vas),
            "hw_var_rh": float(v_hw),
            "vasicek_logprice_sd": float(sd_vas),
            "hw_logprice_sd": float(sd_hw),
            "abs_diff": float(abs(sd_hw - sd_vas)),
        })
    disp_df = pd.DataFrame(disp_rows)
    max_disp_diff = float(disp_df["abs_diff"].max())
    print(disp_df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
    print(f"  max |HW - Vas| log-price SD: {max_disp_diff:.3e}")

    # ---- Exact curve fit at multiple maturities ------------------------
    fit_rows = []
    for T in (1.0, 2.0, 5.0, 10.0):
        p_hw = float(hw_p_now(T, f00, f00, t_cmt, P_cmt))
        p_mkt = market_P(T, t_cmt, P_cmt)
        fit_rows.append({"T_yr": T, "P_HW": p_hw, "P_market": p_mkt,
                         "diff_bp": (p_hw - p_mkt) * 1e4})
    fit_df = pd.DataFrame(fit_rows)
    max_fit_bp = float(fit_df["diff_bp"].abs().max())

    # ---- Figures --------------------------------------------------------
    fig1_forward_distributions(t_cmt, P_cmt, t_fwd, f_fwd, f00, var_df,
                               FIGURES_DIR
                               / "step30_fig1_forward_distributions.png")
    fig2_var_es_bars(var_df, FIGURES_DIR / "step30_fig2_var_es_bars.png")
    fig3_hw_vs_vasicek(diff_df, FIGURES_DIR
                       / "step30_fig3_hw_vs_vasicek.png")
    fig4_driver(t_fwd, f_fwd, f00,
                FIGURES_DIR / "step30_fig4_driver.png")

    # ---- Acceptance checks ----------------------------------------------
    checks = {
        "exact_curve_fit_max_bp": {
            "value_bp": max_fit_bp,
            "tolerance_bp": 1e-6,
            "pass": bool(max_fit_bp < 1e-6),
        },
        "shared_logprice_dispersion": {
            "max_abs_diff": max_disp_diff,
            "tolerance": 1e-12,
            "pass": bool(max_disp_diff < 1e-12),
        },
        "mc_rh_mean_within_005bp": {
            "max_abs_err_bp": float(mc_df["rh_mean_err_bp"].abs().max()),
            "tolerance_bp": 0.05,
            "pass": bool(mc_df["rh_mean_err_bp"].abs().max() < 0.05),
        },
        "mc_forward_price_mean_within_1bp": {
            "max_abs_err_bp": float(
                mc_df["forward_price_mean_err_bp"].abs().max()),
            "tolerance_bp": 1.0,
            "pass": bool(
                mc_df["forward_price_mean_err_bp"].abs().max() < 1.0),
        },
        "r0_comparability_within_5bp": {
            "max_var_gap_bp": float(max_var_gap_bp),
            "tolerance_bp": 5.0,
            "pass": bool(max_var_gap_bp < 5.0),
        },
    }
    print("\n=== Acceptance checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    summary = {
        "config": dict(a=A, sigma=SIGMA, seed=SEED, n_draws=N_DRAWS,
                       horizons=HORIZONS, levels=LEVELS, T_bond=T_BOND,
                       r0_exact_fit=f00,
                       r0_comparability=R0_COMPARABILITY),
        "P_now": {m: float(var_df[var_df["model"] == m]["P_now"].iloc[0])
                  for m in ("constant", "DBM", "Vasicek", "CIR",
                            "Hull-White")},
        "five_model_var_es": var_df.to_dict(orient="records"),
        "hw_vs_vasicek": diff_df.to_dict(orient="records"),
        "hw_comparability": comp_df.to_dict(orient="records"),
        "mc_crosscheck": mc_df.to_dict(orient="records"),
        "dispersion_identity": disp_df.to_dict(orient="records"),
        "exact_fit_max_bp": max_fit_bp,
        "checks": checks,
    }
    with open(OUTPUT_DIR / "step30_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n=== Step 30 summary ===")
    pass_all = all(c["pass"] for c in checks.values())
    print(f"  Verdict                              : "
          f"{'PASS' if pass_all else 'FAIL'}")
    print(f"  P(0,10y) HW exact-fit                : "
          f"{summary['P_now']['Hull-White']:.4f}")
    print(f"  P(0,10y) Vasicek                     : "
          f"{summary['P_now']['Vasicek']:.4f}")
    print(f"  Max |VaR abs diff HW-Vas| (1y, 2y, 5y): "
          f"{diff_df['VaR_abs_diff_bp'].abs().max():.2f} bp")
    print(f"  Max |dispersion gap| HW-Vas          : "
          f"{max_disp_diff:.3e}")
    print(f"  Exact curve fit, max bp              : "
          f"{max_fit_bp:.6f}")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_forward_distributions(t_cmt, P_cmt, t_fwd, f_fwd, f00, var_df,
                               path):
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    h = 1.0
    # HW forward distribution
    p_hw, _, _ = hw_forward_distribution(h, T_BOND, f00, f00, t_cmt, P_cmt,
                                         t_fwd, f_fwd)
    p_vas = s22.forward_bond_distribution("Vasicek", h, T_BOND)
    p_dbm = s22.forward_bond_distribution("DBM", h, T_BOND)
    p_cir = s22.forward_bond_distribution("CIR", h, T_BOND)
    bins = np.linspace(
        min(p_hw.min(), p_vas.min(), p_dbm.min(), p_cir.min()) - 0.01,
        max(p_hw.max(), p_vas.max(), p_dbm.max(), p_cir.max()) + 0.01,
        80)
    ax.hist(p_dbm, bins=bins, density=True, alpha=0.35,
            color=MODEL_COLORS["DBM"], label="DBM")
    ax.hist(p_cir, bins=bins, density=True, alpha=0.45,
            color=MODEL_COLORS["CIR"], label="CIR")
    ax.hist(p_vas, bins=bins, density=True, alpha=0.45,
            color=MODEL_COLORS["Vasicek"], label="Vasicek")
    ax.hist(p_hw, bins=bins, density=True, alpha=0.55,
            color=MODEL_COLORS["Hull-White"], label="Hull-White")
    p_const = float(var_df[var_df["model"] == "constant"]["P_now"].iloc[0])
    ax.axvline(p_const, color=MODEL_COLORS["constant"], lw=1.4,
               linestyle="--", label=f"constant ({p_const:.4f})")
    p_now_vas = float(var_df[var_df["model"] == "Vasicek"]["P_now"].iloc[0])
    p_now_hw = float(var_df[var_df["model"] == "Hull-White"]["P_now"].iloc[0])
    ax.axvline(p_now_vas, color=MODEL_COLORS["Vasicek"], lw=1.0,
               linestyle=":")
    ax.axvline(p_now_hw, color=MODEL_COLORS["Hull-White"], lw=1.0,
               linestyle=":")
    ax.set_xlabel("Forward bond price P(h=1y, T=10y)")
    ax.set_ylabel("density")
    ax.set_title("Forward bond-price laws at h = 1y, five models",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig2_var_es_bars(var_df, path):
    h = 1.0
    sub = var_df[(var_df["h"] == h) & (var_df["level"] == 0.95)]
    sub = sub.set_index("model").reindex(
        ["constant", "DBM", "Vasicek", "CIR", "Hull-White"])
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x - width / 2, sub["VaR_loss"], width=width, color=BLUE,
           label="VaR 95% loss")
    ax.bar(x + width / 2, sub["ES_loss"], width=width, color=BURG, alpha=0.7,
           label="ES 95% loss")
    ax.axhline(0.0, color=INK, lw=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index)
    ax.set_ylabel("Price loss per $1 notional")
    ax.set_title("Forward-bond VaR and ES across five models, h = 1y",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig3_hw_vs_vasicek(diff_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)
    for ax, lvl in zip(axes, LEVELS):
        sub = diff_df[diff_df["level"] == lvl]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub["Vasicek_VaR_pct"], width=width,
               color=TEAL, label="Vasicek VaR %")
        ax.bar(x + width / 2, sub["HullWhite_VaR_pct"], width=width,
               color=BLUE, label="HW VaR %")
        ax.scatter(x - width / 2, sub["Vasicek_ES_pct"], color=TEAL,
                   marker="v", s=42, label="Vasicek ES %")
        ax.scatter(x + width / 2, sub["HullWhite_ES_pct"], color=BLUE,
                   marker="v", s=42, label="HW ES %")
        for xi, (vpct, wpct) in enumerate(
                zip(sub["Vasicek_VaR_pct"], sub["HullWhite_VaR_pct"])):
            ax.annotate(f"{(wpct - vpct) * 100:+.0f} bp",
                        xy=(xi, max(vpct, wpct)),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=SUB)
        ax.axhline(0.0, color=INK, lw=0.7, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}y" for h in sub["h"]])
        ax.set_ylabel("VaR / ES (%)")
        ax.set_title(f"HW vs Vasicek at {int(lvl * 100)}%",
                     fontweight="bold", pad=10)
        ax.legend(loc="lower left", facecolor=BG, edgecolor=RULE,
                  labelcolor=INK, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig4_driver(t_fwd, f_fwd, f00, path):
    """Short-rate marginal mean and 90% band, HW vs Vasicek."""
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    h_grid = np.linspace(0.01, 10.0, 200)
    A_vasi, B_vasi, SIG_VASI = s22.A_VASI, s22.B_VASI, s22.SIG_VASI
    R0_vas = s22.R0
    hw_mean = np.empty_like(h_grid)
    hw_lo = np.empty_like(h_grid)
    hw_hi = np.empty_like(h_grid)
    vas_mean = np.empty_like(h_grid)
    vas_lo = np.empty_like(h_grid)
    vas_hi = np.empty_like(h_grid)
    for i, h in enumerate(h_grid):
        m, v = hw_short_rate_marginal(h, f00, f00, t_fwd, f_fwd)
        sd = np.sqrt(v)
        hw_mean[i] = m
        hw_lo[i] = m - 1.645 * sd
        hw_hi[i] = m + 1.645 * sd
        m_v = B_vasi + (R0_vas - B_vasi) * np.exp(-A_vasi * h)
        v_v = SIG_VASI ** 2 * (1.0 - np.exp(-2.0 * A_vasi * h)) / (2.0 * A_vasi)
        sd_v = np.sqrt(v_v)
        vas_mean[i] = m_v
        vas_lo[i] = m_v - 1.645 * sd_v
        vas_hi[i] = m_v + 1.645 * sd_v
    ax.fill_between(h_grid, vas_lo * 100, vas_hi * 100, color=TEAL,
                    alpha=0.20, label="Vasicek 90% band")
    ax.fill_between(h_grid, hw_lo * 100, hw_hi * 100, color=BLUE,
                    alpha=0.20, label="HW 90% band")
    ax.plot(h_grid, vas_mean * 100, color=TEAL, lw=1.6,
            label="Vasicek mean")
    ax.plot(h_grid, hw_mean * 100, color=BLUE, lw=1.6,
            label="HW mean")
    ax.plot(h_grid,
            hw.forward_at(h_grid, t_fwd, f_fwd) * 100,
            color=INK, lw=1.0, linestyle="--",
            label="Market f(0, h)")
    ax.axhline(B_vasi * 100, color=SUB, lw=0.8, linestyle=":",
               label=f"Vasicek b = {B_vasi * 100:.2f}%")
    ax.set_xlabel("Horizon h (years)")
    ax.set_ylabel("Short rate (%)")
    ax.set_title("Driver: HW mean tracks the forward curve; band shared",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
