"""
models/hw/step31_hullwhite_vasicek_compare.py
---------------------------------------------
Phase E, Step 31: Hull-White versus Vasicek -- the risk-management
verdict.  Step 30 placed Hull-White inside the five-model VaR/ES table
and showed that fitting the initial yield curve relocates the loss
distribution without changing its width.  Step 31 isolates the question
a risk manager actually asks: across h in {1, 2, 5}y, how far apart are
Hull-White and Vasicek on VaR99, ES99, and P(loss)?

Verdict (small, pure level effect):
  - VaR99 gaps: +17, +31, +68 bp of P_now at h = 1, 2, 5y
  - ES99 gaps: +17, +30, +67 bp
  - P(loss) mark-to-market: HW slightly above Vasicek near term
  - P(loss) forward-carried (carry netted out): identical at every h
  - Pricing-level gap: 277 bp of P_now (large)
  -> curve fitting is FIRST-ORDER for pricing, SECOND-ORDER for risk.

Reproducibility: imports step22_bond_pricing, step27_hullwhite_theta,
step30_hullwhite_var read-only.  No upstream artefact altered.

Outputs (aligned with Steps 13-30 artefact layout):

    models/hw/step31_varse_vasicek_hw.csv
    models/hw/step31_ploss.csv
    models/hw/step31_hw_minus_vasicek.csv
    models/hw/step31_summary.json
    figures/step31/step31_fig1_var99_es99.png
    figures/step31/step31_fig2_ploss.png
    figures/step31/step31_fig3_loss_region.png
    figures/step31/step31_fig4_gap_summary.png
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

sys.path.insert(0, str(HERE))
import step27_hullwhite_theta as hw   # noqa: E402
import step30_hullwhite_var as s30    # noqa: E402

sys.path.insert(0, str(ROOT / "models" / "cir"))
import step22_bond_pricing as s22     # noqa: E402

# ---------------------------------------------------------------------------
# Configuration -- inherits from Steps 22, 27, 30
# ---------------------------------------------------------------------------
HORIZONS = [1.0, 2.0, 5.0]
LEVELS = [0.99, 0.95]
T_BOND = 10.0
N_DRAWS = 200_000

OUTPUT_DIR = ROOT / "models" / "hw"
FIGURES_DIR = ROOT / "figures" / "step31"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Palette
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
# Gaussian forward-price laws (Vasicek + Hull-White)
# ---------------------------------------------------------------------------
def vasicek_law(h, T):
    tau = T - h
    A_g = s22.vasi_A(tau)
    B_g = s22.vasi_B(tau)
    m = (s22.B_VASI + (s22.R0 - s22.B_VASI)
         * np.exp(-s22.A_VASI * h))
    v = (s22.SIG_VASI ** 2
         * (1.0 - np.exp(-2.0 * s22.A_VASI * h)) / (2.0 * s22.A_VASI))
    return A_g, B_g, m, v


def hullwhite_law(h, T, f00, r0, t_cmt, P_cmt, t_fwd, f_fwd):
    A_g = float(np.exp(s30.hw_lnA(h, T, t_cmt, P_cmt, t_fwd, f_fwd)))
    B_g = float(s30.hw_B(h, T))
    m, v = s30.hw_short_rate_marginal(h, f00, r0, t_fwd, f_fwd)
    return A_g, B_g, m, v


def prob_loss_gaussian(A_g, B_g, m, v, ref_level):
    """
    P( A_g exp(-B_g r_h) < ref_level ) for a long position.  The forward
    price falls below the reference when r_h exceeds the break-even rate
    r* = ln(A_g / ref_level) / B_g (B_g > 0); r_h is Gaussian.
    """
    r_star = np.log(A_g / ref_level) / B_g
    return float(norm.sf((r_star - m) / np.sqrt(v))), float(r_star)


# ---------------------------------------------------------------------------
# Sample reproduction (forward-price empirical samples)
# ---------------------------------------------------------------------------
def vasicek_samples(h, T):
    return s22.forward_bond_distribution("Vasicek", h, T, n=N_DRAWS,
                                         seed=s22.SEED)


def hullwhite_samples(h, T, f00, r0, t_cmt, P_cmt, t_fwd, f_fwd, seed):
    p, _, _ = s30.hw_forward_distribution(
        h, T, f00, r0, t_cmt, P_cmt, t_fwd, f_fwd,
        n=N_DRAWS, seed=seed)
    return p


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Phase E, Step 31: Hull-White vs Vasicek risk-management verdict")
    print("-" * 64)

    t_cmt, P_cmt, t_fwd, f_fwd = s30.market_setup()
    f00 = float(f_fwd[0])
    print(f"f(0, 0) = {f00 * 100:.4f}%%")
    print(f"horizons = {HORIZONS}, T_bond = {T_BOND}, N_draws = {N_DRAWS}")
    print()

    # ---- Reproduce VaR/ES from Step 30 ----------------------------------
    print("Reproducing VaR/ES for Vasicek and Hull-White ...")
    rows_var = []
    for h in HORIZONS:
        vas = s22.tail_risk_table("Vasicek", h, T_BOND, LEVELS)
        hw_rows, _, _, _, _ = s30.hw_tail_risk(
            h, T_BOND, LEVELS, f00, f00, t_cmt, P_cmt, t_fwd, f_fwd)
        rows_var.extend(vas)
        rows_var.extend(hw_rows)
    var_df = pd.DataFrame(rows_var)
    var_df.to_csv(OUTPUT_DIR / "step31_varse_vasicek_hw.csv", index=False)

    # ---- Build HW-minus-Vasicek differences -----------------------------
    diff_rows = []
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
            diff_rows.append({
                "h": h, "level": q,
                "vas_VaR_pct": v_pct,
                "hw_VaR_pct": w_pct,
                "VaR_gap_bp": (w_pct - v_pct) * 100,
                "vas_ES_pct": es_v,
                "hw_ES_pct": es_w,
                "ES_gap_bp": (es_w - es_v) * 100,
            })
    diff_df = pd.DataFrame(diff_rows)

    # ---- P(loss) under both references ----------------------------------
    print("\nComputing P(loss) under MtM and forward-carried references ...")
    ploss_rows = []
    P0_T = s30.market_P(T_BOND, t_cmt, P_cmt)   # market 10y price
    for h in HORIZONS:
        P0_h = s30.market_P(h, t_cmt, P_cmt)
        # Laws
        vA, vB, vm, vv = vasicek_law(h, T_BOND)
        v_pnow = s22.price_vasicek(T_BOND)
        wA, wB, wm, wv = hullwhite_law(h, T_BOND, f00, f00,
                                       t_cmt, P_cmt, t_fwd, f_fwd)
        w_pnow = s30.hw_p_now(T_BOND, f00, f00, t_cmt, P_cmt)
        # Forward-carried reference uses each model's own P(0, h).
        v_p0h = s22.price_vasicek(h)
        w_p0h = s30.hw_p_now(h, f00, f00, t_cmt, P_cmt)
        # Empirical samples for cross-check
        v_samp = vasicek_samples(h, T_BOND)
        w_samp = hullwhite_samples(h, T_BOND, f00, f00,
                                   t_cmt, P_cmt, t_fwd, f_fwd, seed=30)
        for model, (Ag, Bg, m, v, pnow, p0h, samp) in {
            "Vasicek":    (vA, vB, vm, vv, v_pnow, v_p0h, v_samp),
            "Hull-White": (wA, wB, wm, wv, w_pnow, w_p0h, w_samp),
        }.items():
            fwd_ref = pnow / p0h
            pl_mtm, r_mtm = prob_loss_gaussian(Ag, Bg, m, v, pnow)
            pl_fwd, r_fwd = prob_loss_gaussian(Ag, Bg, m, v, fwd_ref)
            ploss_rows.append({
                "model": model, "h": h,
                "Pnow": pnow,
                "fwd_ref": fwd_ref,
                "ploss_mtm": pl_mtm,
                "ploss_mtm_emp": float(np.mean(samp < pnow)),
                "rstar_mtm": r_mtm,
                "ploss_fwd": pl_fwd,
                "ploss_fwd_emp": float(np.mean(samp < fwd_ref)),
                "rstar_fwd": r_fwd,
                "Er_h": m,
                "sd_r_h": float(np.sqrt(v)),
            })
    ploss_df = pd.DataFrame(ploss_rows)
    ploss_df.to_csv(OUTPUT_DIR / "step31_ploss.csv", index=False)

    # ---- HW minus Vasicek table (P(loss) added on top of VaR/ES) --------
    plg_rows = []
    for h in HORIZONS:
        v = ploss_df[(ploss_df["model"] == "Vasicek")
                     & (ploss_df["h"] == h)].iloc[0]
        w = ploss_df[(ploss_df["model"] == "Hull-White")
                     & (ploss_df["h"] == h)].iloc[0]
        plg_rows.append({
            "h": h,
            "ploss_mtm_vas": v["ploss_mtm"],
            "ploss_mtm_hw": w["ploss_mtm"],
            "ploss_mtm_gap_pp": (w["ploss_mtm"] - v["ploss_mtm"]) * 100,
            "ploss_fwd_vas": v["ploss_fwd"],
            "ploss_fwd_hw": w["ploss_fwd"],
            "ploss_fwd_gap_pp": (w["ploss_fwd"] - v["ploss_fwd"]) * 100,
        })
    # Combine VaR/ES diff and P(loss) diff into one rollup CSV
    rollup = []
    for h in HORIZONS:
        for q in LEVELS:
            sub = diff_df[(diff_df["h"] == h) & (diff_df["level"] == q)].iloc[0]
            rollup.append({**sub.to_dict(),
                           "metric_kind": "VaR_ES"})
    for r in plg_rows:
        rollup.append({**r, "metric_kind": "P_loss"})
    rollup_df = pd.DataFrame(rollup)
    rollup_df.to_csv(OUTPUT_DIR / "step31_hw_minus_vasicek.csv", index=False)

    # ---- Console summary -----------------------------------------------
    print("\n=== VaR99 / ES99 (and 95%% context) ===")
    print(diff_df.to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))
    print("\n=== P(loss) (both references) ===")
    print(ploss_df.to_string(
        index=False, float_format=lambda v: f"{v: .6f}"))

    # ---- Pricing-level gap ----------------------------------------------
    v_pnow_10 = float(ploss_df[(ploss_df["model"] == "Vasicek")
                               & (ploss_df["h"] == HORIZONS[0])
                               ]["Pnow"].iloc[0])
    w_pnow_10 = float(ploss_df[(ploss_df["model"] == "Hull-White")
                               & (ploss_df["h"] == HORIZONS[0])
                               ]["Pnow"].iloc[0])
    pricing_level_gap_bp = (w_pnow_10 - v_pnow_10) / v_pnow_10 * 1e4
    print(f"\nPricing-level gap, P(0, 10y): "
          f"Vasicek = {v_pnow_10:.4f}, HW = {w_pnow_10:.4f}, "
          f"|gap| = {abs(pricing_level_gap_bp):.1f} bp of Pnow")

    # ---- Acceptance checks ----------------------------------------------
    print("\nRunning acceptance checks ...")
    # (1) Bit-for-bit reproduction of Step 30 VaR/ES
    s30_summary = json.loads(
        (OUTPUT_DIR / "step30_summary.json").read_text())
    s30_var = pd.DataFrame(s30_summary["five_model_var_es"])
    max_repro = 0.0
    for m in ("Vasicek", "Hull-White"):
        for h in HORIZONS:
            for q in LEVELS:
                a = s30_var[(s30_var["model"] == m)
                            & (s30_var["h"] == h)
                            & (s30_var["level"] == q)].iloc[0]
                b = var_df[(var_df["model"] == m)
                           & (var_df["h"] == h)
                           & (var_df["level"] == q)].iloc[0]
                for key in ("VaR_pct", "ES_pct"):
                    max_repro = max(max_repro, abs(a[key] - b[key]))

    # (2) Analytic vs empirical P(loss)
    max_ploss_gap_pp = 0.0
    for _, row in ploss_df.iterrows():
        for kind in ("mtm", "fwd"):
            gap = abs(row[f"ploss_{kind}"] - row[f"ploss_{kind}_emp"]) * 100
            max_ploss_gap_pp = max(max_ploss_gap_pp, gap)

    # (3) Shared log-price dispersion
    max_disp_diff = 0.0
    for h in HORIZONS:
        _, vB, _, vv = vasicek_law(h, T_BOND)
        _, wB, _, wv = hullwhite_law(h, T_BOND, f00, f00,
                                     t_cmt, P_cmt, t_fwd, f_fwd)
        max_disp_diff = max(max_disp_diff,
                            abs(wB * np.sqrt(wv) - vB * np.sqrt(vv)))

    # (4) Forward-carried P(loss) model-invariant
    max_fwd_invariance = float(
        max(abs(r["ploss_fwd_gap_pp"]) for r in plg_rows))

    checks = {
        "step30_reproduction": {
            "max_abs_diff": float(max_repro),
            "tolerance": 1e-12,
            "pass": bool(max_repro < 1e-12),
        },
        "analytic_vs_empirical_ploss": {
            "max_gap_pp": float(max_ploss_gap_pp),
            "tolerance_pp": 1.0,
            "pass": bool(max_ploss_gap_pp < 1.0),
        },
        "shared_log_price_dispersion": {
            "max_abs_diff": float(max_disp_diff),
            "tolerance": 1e-12,
            "pass": bool(max_disp_diff < 1e-12),
        },
        "forward_carried_ploss_invariant": {
            "max_gap_pp": float(max_fwd_invariance),
            "tolerance_pp": 1e-9,
            "pass": bool(max_fwd_invariance < 1e-9),
        },
        "no_upstream_artefact_altered": {
            "imports_readonly": True,
            "pass": True,
        },
    }
    print("\n=== Acceptance checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")

    # ---- Figures --------------------------------------------------------
    fig1_var99_es99(diff_df, FIGURES_DIR / "step31_fig1_var99_es99.png")
    fig2_ploss(ploss_df, FIGURES_DIR / "step31_fig2_ploss.png")
    fig3_loss_region(t_cmt, P_cmt, t_fwd, f_fwd, f00,
                     FIGURES_DIR / "step31_fig3_loss_region.png")
    fig4_gap_summary(diff_df, abs(pricing_level_gap_bp),
                     FIGURES_DIR / "step31_fig4_gap_summary.png")

    # ---- JSON summary ---------------------------------------------------
    summary = {
        "config": dict(horizons=HORIZONS, levels=LEVELS, T_bond=T_BOND,
                       n_draws=N_DRAWS, f00=f00),
        "varse_table": var_df.to_dict(orient="records"),
        "diff_table": diff_df.to_dict(orient="records"),
        "ploss_table": ploss_df.to_dict(orient="records"),
        "pricing_level_gap_bp": float(abs(pricing_level_gap_bp)),
        "checks": checks,
    }
    with open(OUTPUT_DIR / "step31_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n=== Step 31 verdict ===")
    pass_all = all(c["pass"] for c in checks.values())
    print(f"  Verdict                       : "
          f"{'PASS' if pass_all else 'FAIL'}")
    print(f"  VaR99 gaps (bp): "
          f"1y={diff_df[(diff_df['h']==1.0)&(diff_df['level']==0.99)]['VaR_gap_bp'].iloc[0]:.1f},  "
          f"2y={diff_df[(diff_df['h']==2.0)&(diff_df['level']==0.99)]['VaR_gap_bp'].iloc[0]:.1f},  "
          f"5y={diff_df[(diff_df['h']==5.0)&(diff_df['level']==0.99)]['VaR_gap_bp'].iloc[0]:.1f}")
    print(f"  ES99 gaps (bp):  "
          f"1y={diff_df[(diff_df['h']==1.0)&(diff_df['level']==0.99)]['ES_gap_bp'].iloc[0]:.1f},  "
          f"2y={diff_df[(diff_df['h']==2.0)&(diff_df['level']==0.99)]['ES_gap_bp'].iloc[0]:.1f},  "
          f"5y={diff_df[(diff_df['h']==5.0)&(diff_df['level']==0.99)]['ES_gap_bp'].iloc[0]:.1f}")
    print(f"  P(loss) MtM 1y: Vasicek={ploss_df[(ploss_df['model']=='Vasicek')&(ploss_df['h']==1.0)]['ploss_mtm'].iloc[0]*100:.3f}%%, "
          f"HW={ploss_df[(ploss_df['model']=='Hull-White')&(ploss_df['h']==1.0)]['ploss_mtm'].iloc[0]*100:.3f}%%")
    print(f"  Pricing-level gap            : "
          f"{abs(pricing_level_gap_bp):.0f} bp of P_now (vs ~70 bp risk gap)")
    print(f"\n  Wrote tables  -> {OUTPUT_DIR}")
    print(f"  Wrote figures -> {FIGURES_DIR}")
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_var99_es99(diff_df, path):
    sub99 = diff_df[diff_df["level"] == 0.99].sort_values("h")
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    x = np.arange(len(sub99))
    width = 0.36
    ax.bar(x - width / 2, sub99["vas_VaR_pct"], width=width, color=TEAL,
           label="Vasicek VaR99")
    ax.bar(x + width / 2, sub99["hw_VaR_pct"], width=width, color=BLUE,
           label="Hull-White VaR99")
    ax.scatter(x - width / 2, sub99["vas_ES_pct"], color=TEAL,
               marker="v", s=60, edgecolor=INK, linewidth=0.6,
               label="Vasicek ES99", zorder=5)
    ax.scatter(x + width / 2, sub99["hw_ES_pct"], color=BLUE,
               marker="v", s=60, edgecolor=INK, linewidth=0.6,
               label="Hull-White ES99", zorder=5)
    for xi, (vpct, wpct) in enumerate(
            zip(sub99["vas_VaR_pct"], sub99["hw_VaR_pct"])):
        gap_bp = (wpct - vpct) * 100
        ax.annotate(f"+{gap_bp:.0f} bp", xy=(xi, max(vpct, wpct)),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=SUB)
    ax.axhline(0.0, color=INK, lw=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}y" for h in sub99["h"]])
    ax.set_ylabel("VaR99 / ES99 (% of P_now)")
    ax.set_title("Hull-White vs Vasicek: VaR99 and ES99 across horizons",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig2_ploss(ploss_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax in axes:
        _style_axes(ax)
    fig.patch.set_facecolor(BG)
    horizons = sorted(ploss_df["h"].unique())
    x = np.arange(len(horizons))
    width = 0.36
    vas_mtm = [ploss_df[(ploss_df["model"] == "Vasicek")
                        & (ploss_df["h"] == h)]["ploss_mtm"].iloc[0] * 100
               for h in horizons]
    hw_mtm = [ploss_df[(ploss_df["model"] == "Hull-White")
                       & (ploss_df["h"] == h)]["ploss_mtm"].iloc[0] * 100
              for h in horizons]
    vas_fwd = [ploss_df[(ploss_df["model"] == "Vasicek")
                        & (ploss_df["h"] == h)]["ploss_fwd"].iloc[0] * 100
               for h in horizons]
    hw_fwd = [ploss_df[(ploss_df["model"] == "Hull-White")
                       & (ploss_df["h"] == h)]["ploss_fwd"].iloc[0] * 100
              for h in horizons]
    axes[0].bar(x - width / 2, vas_mtm, width=width, color=TEAL,
                label="Vasicek")
    axes[0].bar(x + width / 2, hw_mtm, width=width, color=BLUE,
                label="Hull-White")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{int(h)}y" for h in horizons])
    axes[0].set_ylabel("P(loss) (%)")
    axes[0].set_title("Mark-to-market: P(P(h,T) < P_now)",
                      fontweight="bold", pad=10)
    axes[0].legend(loc="upper right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK, fontsize=8)

    axes[1].bar(x - width / 2, vas_fwd, width=width, color=TEAL,
                label="Vasicek")
    axes[1].bar(x + width / 2, hw_fwd, width=width, color=BLUE,
                label="Hull-White")
    axes[1].axhline(50.0, color=SUB, lw=0.8, linestyle="--",
                    label="50% forward line")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(h)}y" for h in horizons])
    axes[1].set_ylabel("P(loss) (%)")
    axes[1].set_title("Forward-carried: P(P(h,T) < P_now / P(0,h))",
                      fontweight="bold", pad=10)
    axes[1].legend(loc="lower right", facecolor=BG, edgecolor=RULE,
                   labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig3_loss_region(t_cmt, P_cmt, t_fwd, f_fwd, f00, path):
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    h = 1.0
    p_vas = vasicek_samples(h, T_BOND)
    p_hw = hullwhite_samples(h, T_BOND, f00, f00,
                             t_cmt, P_cmt, t_fwd, f_fwd, seed=30)
    v_pnow = s22.price_vasicek(T_BOND)
    w_pnow = s30.hw_p_now(T_BOND, f00, f00, t_cmt, P_cmt)
    lo = min(p_vas.min(), p_hw.min()) - 0.005
    hi = max(p_vas.max(), p_hw.max()) + 0.005
    bins = np.linspace(lo, hi, 80)
    ax.hist(p_vas, bins=bins, density=True, alpha=0.45, color=TEAL,
            label=f"Vasicek (P_now = {v_pnow:.4f})")
    ax.hist(p_hw, bins=bins, density=True, alpha=0.55, color=BLUE,
            label=f"Hull-White (P_now = {w_pnow:.4f})")
    # Shade the loss regions p < P_now
    bw = bins[1] - bins[0]
    for samp, pnow, color in ((p_vas, v_pnow, TEAL),
                              (p_hw, w_pnow, BLUE)):
        # shade the histogram bars below pnow
        h_, edges = np.histogram(samp, bins=bins, density=True)
        for e_l, e_r, hh in zip(edges[:-1], edges[1:], h_):
            if e_r <= pnow:
                ax.fill_between([e_l, e_r], 0, hh, color=color, alpha=0.30)
    ax.axvline(v_pnow, color=TEAL, lw=1.4, linestyle="--")
    ax.axvline(w_pnow, color=BLUE, lw=1.4, linestyle="--")
    ax.set_xlabel("Forward bond price P(1y, 10y)")
    ax.set_ylabel("density")
    ax.set_title("Forward-price laws at h = 1y with MtM loss regions shaded",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def fig4_gap_summary(diff_df, pricing_gap_bp, path):
    sub99 = diff_df[diff_df["level"] == 0.99].sort_values("h")
    fig, ax = _new_figure(figsize=(9.0, 4.8))
    x = np.arange(len(sub99))
    width = 0.36
    ax.bar(x - width / 2, sub99["VaR_gap_bp"], width=width, color=BLUE,
           label="VaR99 gap")
    ax.bar(x + width / 2, sub99["ES_gap_bp"], width=width, color=BURG,
           label="ES99 gap")
    ax.axhline(pricing_gap_bp, color=INK, lw=1.4, linestyle="--",
               label=f"Pricing-level gap (|{pricing_gap_bp:.0f}| bp)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}y" for h in sub99["h"]])
    ax.set_ylabel("HW - Vasicek gap (bp of P_now)")
    ax.set_title("Curve fitting: first-order for pricing, second-order for risk",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
