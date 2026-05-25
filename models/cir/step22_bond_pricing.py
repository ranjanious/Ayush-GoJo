"""
models/cir/step22_bond_pricing.py
---------------------------------
Phase D, Step 22: CIR bond pricing and risk metrics, with the four-model
comparison DBM vs Vasicek vs CIR vs constant rate.

For each model the zero-coupon bond price P(0, T) is

    P(0, T) = E^Q [ exp( - int_0^T r_s ds ) ],

evaluated either in closed form (when one exists) or by Monte Carlo on the
simulated short-rate paths shared with Phase B/C/D.  The four models in
side-by-side notation are

    constant   r_t = r_0                                       (deterministic)
    DBM        dr_t = sigma_d dW_t                             (Gaussian increments)
    Vasicek    dr_t = a (b - r_t) dt + sigma_v dW_t            (Gaussian marginal)
    CIR        dr_t = a (b - r_t) dt + sigma_c sqrt(r_t) dW_t  (ncx2 marginal)

Closed forms:

    constant:  P(0, T) = exp(-r_0 T)
    DBM:       int_0^T r_t dt = r_0 T + sigma_d int_0^T W_t dt is
               N(r_0 T, sigma_d^2 T^3 / 3), so
               P(0, T) = exp(-r_0 T + sigma_d^2 T^3 / 6).
    Vasicek:   P(t, T) = A_v(tau) exp(-B_v(tau) r_t)
    CIR:       P(t, T) = A_c(tau) exp(-B_c(tau) r_t)

Reproducibility: seed = 42 reused from Steps 13, 16, 17, 18, 20, 21 so the
EM Brownian streams are bitwise identical to those modules and the ncx2
streams to Step 20.

Outputs (aligned with Steps 13-21 artefact layout):

    models/cir/step22_bond_metrics.csv
    models/cir/step22_coupling.csv
    models/cir/step22_terminal_distribution.csv
    models/cir/step22_var_es.csv
    models/cir/step22_summary.json
    figures/step22/step22_fig1_yield_curve.png
    figures/step22/step22_fig2_terminal_density.png
    figures/step22/step22_fig3_forward_price.png
    figures/step22/step22_fig4_var_es.png
    figures/step22/step22_fig5_skew.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ncx2, norm, skew, kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# ---------------------------------------------------------------------------
# Shared configuration; reused verbatim from Steps 13, 17, 18, 20, 21.
# ---------------------------------------------------------------------------
R0 = 0.0372
DT = 1.0 / 12.0
N_PATHS = 10_000
HORIZON_YEARS = 10.0
SEED = 42

# Vasicek calibration (Steps 13, 17, 18)
A_VASI = 0.50
B_VASI = 0.04
SIG_VASI = 0.01

# CIR calibration (Steps 20, 21) - Feller ratio 6.25
A_CIR = 0.50
B_CIR = 0.04
SIG_CIR = 0.08

# DBM calibration (Steps 10, 18)
SIG_DBM = 0.01

# Constant-rate baseline (anchored to r_0 per the chosen scope)
R_CONST = R0

BOND_MATS = [1.0, 2.0, 5.0, 10.0]
VAR_HORIZONS = [1.0, 2.0, 5.0]
VAR_BOND_MAT = 10.0
VAR_LEVELS = [0.95, 0.99]

OUTPUT_DIR = ROOT / "models" / "cir"
FIGURES_DIR = ROOT / "figures" / "step22"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (Steps 16-21 token set)
BG = "#fbf7ec"
INK = "#1c1a15"
SUB = "#5c544a"
RULE = "#c5b8a0"
BLUE = "#2c4a5e"
BURG = "#8b2d3a"
OCHRE = "#b8842a"
TEAL = "#2d6a5f"

MODEL_COLORS = {
    "constant": INK,
    "DBM": OCHRE,
    "Vasicek": TEAL,
    "CIR": BURG,
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
# Closed-form bond pricing
# ---------------------------------------------------------------------------
def price_constant(T, r0=R_CONST):
    return float(np.exp(-r0 * T))


def price_dbm(T, r0=R0, sigma=SIG_DBM):
    """Closed-form bond price under DBM r_t = r_0 + sigma W_t.

    int_0^T r_t dt = r_0 T + sigma int_0^T W_t dt is Gaussian with mean
    r_0 T and variance sigma^2 T^3 / 3, so P(0,T) = exp(-r_0 T + sigma^2 T^3 / 6).
    """
    return float(np.exp(-r0 * T + 0.5 * sigma ** 2 * T ** 3 / 3.0))


def vasi_B(tau, a=A_VASI):
    return (1.0 - np.exp(-a * tau)) / a


def vasi_A(tau, a=A_VASI, b=B_VASI, sigma=SIG_VASI):
    B = vasi_B(tau, a)
    return float(np.exp((B - tau) * (a ** 2 * b - 0.5 * sigma ** 2) / a ** 2
                        - sigma ** 2 * B ** 2 / (4.0 * a)))


def price_vasicek(T, r=R0, a=A_VASI, b=B_VASI, sigma=SIG_VASI):
    return vasi_A(T, a, b, sigma) * float(np.exp(-vasi_B(T, a) * r))


def cir_B(tau, a=A_CIR, sigma=SIG_CIR):
    gamma = np.sqrt(a ** 2 + 2.0 * sigma ** 2)
    num = 2.0 * (np.exp(gamma * tau) - 1.0)
    den = (a + gamma) * (np.exp(gamma * tau) - 1.0) + 2.0 * gamma
    return num / den


def cir_A(tau, a=A_CIR, b=B_CIR, sigma=SIG_CIR):
    gamma = np.sqrt(a ** 2 + 2.0 * sigma ** 2)
    num = 2.0 * gamma * np.exp((a + gamma) * tau / 2.0)
    den = (a + gamma) * (np.exp(gamma * tau) - 1.0) + 2.0 * gamma
    return float((num / den) ** (2.0 * a * b / sigma ** 2))


def price_cir(T, r=R0, a=A_CIR, b=B_CIR, sigma=SIG_CIR):
    return cir_A(T, a, b, sigma) * float(np.exp(-cir_B(T, a, sigma) * r))


# ---------------------------------------------------------------------------
# Simulators (reused from Steps 13, 17, 18, 20)
# ---------------------------------------------------------------------------
def simulate_dbm(r0, sigma, horizon, dt, n_paths, seed):
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        r[:, k + 1] = r[:, k] + sigma * sqrt_dt * Z
    return r


def simulate_vasicek_em(a, b, sigma, r0, horizon, dt, n_paths, seed):
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r[:, 0] = r0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal(n_paths)
        r[:, k + 1] = r[:, k] + a * (b - r[:, k]) * dt + sigma * sqrt_dt * Z
    return r


def simulate_vasicek_exact(a, b, sigma, r0, horizon, dt, n_paths, seed):
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
# Monte Carlo bond pricing on a path matrix
# ---------------------------------------------------------------------------
def mc_bond_prices(paths, dt, maturities):
    """Trapezoidal integration of r_s over [0, T] for each requested T.

    Returns {T: (price_mean, df_paths)} where df_paths is the per-path
    discount factor exp(-int_0^T r_s ds).
    """
    running = np.zeros_like(paths)
    running[:, 1:] = 0.5 * (paths[:, :-1] + paths[:, 1:]) * dt
    integral = np.cumsum(running, axis=1)
    out = {}
    for T in maturities:
        k = int(round(T / dt))
        df = np.exp(-integral[:, k])
        out[T] = (float(df.mean()), df)
    return out


# ---------------------------------------------------------------------------
# Risk metrics derived from the closed-form bond prices
# ---------------------------------------------------------------------------
def risk_metrics(model, T, price, r=R0):
    if model in ("constant", "DBM"):
        B = T
    elif model == "Vasicek":
        B = vasi_B(T)
    elif model == "CIR":
        B = cir_B(T)
    else:
        raise ValueError(model)
    yield_T = -np.log(price) / T
    duration = B
    convexity = B ** 2
    dv01 = duration * price * 1e-4
    return {
        "yield": float(yield_T),
        "duration": float(duration),
        "convexity": float(convexity),
        "DV01": float(dv01),
    }


# ---------------------------------------------------------------------------
# Tail-risk on forward bond prices
# ---------------------------------------------------------------------------
def vasicek_rh_sample(h, a, b, sigma, r0, n, rng):
    mu = b + (r0 - b) * np.exp(-a * h)
    var = sigma ** 2 * (1.0 - np.exp(-2.0 * a * h)) / (2.0 * a)
    return rng.normal(loc=mu, scale=np.sqrt(var), size=n)


def cir_rh_sample(h, a, b, sigma, r0, n, rng):
    d = 4.0 * a * b / sigma ** 2
    c = 4.0 * a / (sigma ** 2 * (1.0 - np.exp(-a * h)))
    lam = c * r0 * np.exp(-a * h)
    chi = rng.noncentral_chisquare(df=d, nonc=lam, size=n)
    return chi / c


def forward_bond_distribution(model, h, T_bond, n=200_000, seed=SEED):
    rng = np.random.default_rng(seed)
    tau = T_bond - h
    if model == "constant":
        return np.full(n, np.exp(-R_CONST * tau))
    if model == "Vasicek":
        rh = vasicek_rh_sample(h, A_VASI, B_VASI, SIG_VASI, R0, n, rng)
        return vasi_A(tau) * np.exp(-vasi_B(tau) * rh)
    if model == "CIR":
        rh = cir_rh_sample(h, A_CIR, B_CIR, SIG_CIR, R0, n, rng)
        return cir_A(tau) * np.exp(-cir_B(tau) * rh)
    if model == "DBM":
        # r_h ~ N(r0, sigma^2 h); conditional on r_h the integrated short
        # rate from h to T is Gaussian (mean r_h tau, var sigma^2 tau^3/3)
        # and independent of r_h, so forward P is lognormal in r_h.
        rh = rng.normal(loc=R0, scale=SIG_DBM * np.sqrt(h), size=n)
        log_p = -rh * tau + 0.5 * SIG_DBM ** 2 * tau ** 3 / 3.0
        return np.exp(log_p)
    raise ValueError(model)


def tail_risk_table(model, h, T_bond, levels):
    p = forward_bond_distribution(model, h, T_bond)
    p0 = {
        "constant": price_constant(T_bond),
        "DBM":      price_dbm(T_bond),
        "Vasicek":  price_vasicek(T_bond),
        "CIR":      price_cir(T_bond),
    }[model]
    rows = []
    for q in levels:
        var_p = float(np.quantile(p, 1.0 - q))
        es_p = float(p[p <= var_p].mean()) if (p <= var_p).any() else var_p
        rows.append({
            "model": model,
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
    return rows


# ---------------------------------------------------------------------------
# Terminal-distribution skewness / kurtosis / tails for r_T
# ---------------------------------------------------------------------------
def terminal_distribution_table(rT_by_model, ana_skew_kurt):
    rows = []
    for name, sample in rT_by_model.items():
        rows.append({
            "model": name,
            "mean": float(sample.mean()),
            "std": float(sample.std(ddof=1)),
            "skew_sim": float(skew(sample, bias=False)),
            "ex_kurt_sim": float(kurtosis(sample, fisher=True, bias=False)),
            "skew_ana": ana_skew_kurt[name][0],
            "ex_kurt_ana": ana_skew_kurt[name][1],
            "p01": float(np.quantile(sample, 0.01)),
            "p05": float(np.quantile(sample, 0.05)),
            "p50": float(np.quantile(sample, 0.50)),
            "p95": float(np.quantile(sample, 0.95)),
            "p99": float(np.quantile(sample, 0.99)),
        })
    return pd.DataFrame(rows)


def cir_terminal_moments(T, a=A_CIR, b=B_CIR, sigma=SIG_CIR, r0=R0):
    """Analytical skewness / excess kurtosis of r_T under CIR via the ncx2
    cumulants k_n(X) = 2^(n-1) (n-1)! (d + n lam).  Scaling by 1/c leaves
    skew and excess kurtosis unchanged.
    """
    d = 4.0 * a * b / sigma ** 2
    decay = np.exp(-a * T)
    c = 4.0 * a / (sigma ** 2 * (1.0 - decay))
    lam = c * r0 * decay
    k2 = 2.0 * (d + 2.0 * lam)
    k3 = 8.0 * (d + 3.0 * lam)
    k4 = 48.0 * (d + 4.0 * lam)
    sk = k3 / k2 ** 1.5
    ekk = k4 / k2 ** 2
    return float(sk), float(ekk)


# ---------------------------------------------------------------------------
# Same-seed coupling diagnostics: MC bond price vs closed-form bond price
# ---------------------------------------------------------------------------
def coupling_table(model, paths_em, paths_exact, dt, maturities):
    mc_em = mc_bond_prices(paths_em, dt, maturities)
    mc_ex = mc_bond_prices(paths_exact, dt, maturities)
    rows = []
    for T in maturities:
        cf = price_vasicek(T) if model == "Vasicek" else price_cir(T)
        df_em = mc_em[T][1]
        df_ex = mc_ex[T][1]
        rows.append({
            "model": model,
            "T_yr": T,
            "P_closed": float(cf),
            "P_MC_EM": float(df_em.mean()),
            "P_MC_exact": float(df_ex.mean()),
            "MC_EM_minus_CF_bp": float((df_em.mean() - cf) * 1e4),
            "MC_exact_minus_CF_bp": float((df_ex.mean() - cf) * 1e4),
            "MC_EM_se_bp": float(df_em.std(ddof=1) / np.sqrt(df_em.shape[0]) * 1e4),
            "MC_exact_se_bp": float(df_ex.std(ddof=1) / np.sqrt(df_ex.shape[0]) * 1e4),
            "pathwise_abs_gap_EM_bp": float(np.abs(df_em - df_ex).mean() * 1e4),
        })
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_yield_curve(prices_by_model_dense, path):
    fig, ax = _new_figure(figsize=(8.0, 4.8))
    taus = np.linspace(0.25, 10.0, 80)
    for name in ("constant", "DBM", "Vasicek", "CIR"):
        prices = prices_by_model_dense[name](taus)
        yields = -np.log(prices) / taus
        ax.plot(taus, yields * 100, color=MODEL_COLORS[name], lw=1.8,
                label=name)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Zero-coupon yield y(T) (%)")
    ax.set_title("Closed-form zero-coupon yield curves, four models",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_terminal_density(rT_by_model, ana_pdfs, path):
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    lo = min(s.min() for s in rT_by_model.values())
    hi = max(s.max() for s in rT_by_model.values())
    span = hi - lo
    xs = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 400)
    bins = np.linspace(lo - 0.01 * span, hi + 0.01 * span, 80)
    for name in ("DBM", "Vasicek", "CIR"):
        ax.hist(rT_by_model[name], bins=bins, density=True,
                color=MODEL_COLORS[name], alpha=0.30, label=f"{name} (MC)")
        ax.plot(xs, ana_pdfs[name](xs), color=MODEL_COLORS[name], lw=1.6,
                label=f"{name} (analytic)")
    ax.axvline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.axvline(R_CONST, color=MODEL_COLORS["constant"], lw=1.4,
               linestyle="--", label="constant r_0")
    ax.set_xlabel(r"$r_T$ at T = 10y")
    ax.set_ylabel("density")
    ax.set_title("Terminal short-rate distribution: symmetric Gaussian vs ncx2 skew",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig3_forward_price_distribution(samples_by_model, h, T_bond, path):
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    lo = min(s.min() for s in samples_by_model.values() if s.std() > 0)
    hi = max(s.max() for s in samples_by_model.values() if s.std() > 0)
    span = hi - lo
    bins = np.linspace(lo - 0.02 * span, hi + 0.02 * span, 80)
    for name in ("DBM", "Vasicek", "CIR"):
        s = samples_by_model[name]
        ax.hist(s, bins=bins, density=True, color=MODEL_COLORS[name],
                alpha=0.40, label=name)
    p_const = samples_by_model["constant"][0]
    ax.axvline(p_const, color=MODEL_COLORS["constant"], lw=1.4,
               linestyle="--", label=f"constant ({p_const:.4f})")
    ax.set_xlabel(f"P(h={h:.0f}y, T={T_bond:.0f}y)")
    ax.set_ylabel("density")
    ax.set_title(f"Forward bond-price distribution, h = {h:.0f}y, T = {T_bond:.0f}y",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig4_var_es_bars(var_df, path):
    sub = var_df[var_df["h"] == 1.0]
    levels = sorted(sub["level"].unique())
    models = ["DBM", "Vasicek", "CIR"]
    x = np.arange(len(models))
    width = 0.36
    fig, ax = _new_figure(figsize=(8.5, 4.8))
    for i, q in enumerate(levels):
        sub_q = sub[sub["level"] == q].set_index("model")
        loss = [sub_q.loc[m, "VaR_loss"] for m in models]
        es = [sub_q.loc[m, "ES_loss"] for m in models]
        offset = (i - 0.5) * width
        ax.bar(x + offset, loss, width=width * 0.45,
               color=BLUE if q == 0.95 else BURG,
               label=f"VaR {int(q*100)}%")
        ax.bar(x + offset + width * 0.45, es, width=width * 0.45,
               color=BLUE if q == 0.95 else BURG, alpha=0.45,
               label=f"ES {int(q*100)}%")
    ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Price loss per $1 notional, T = 10y, h = 1y")
    ax.set_title("Forward-bond VaR and ES across stochastic-rate models",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE,
              labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig5_skew_summary(term_df, path):
    fig, ax = _new_figure(figsize=(8.0, 4.6))
    models = ["DBM", "Vasicek", "CIR"]
    sim_sk = [term_df.set_index("model").loc[m, "skew_sim"] for m in models]
    ana_sk = [term_df.set_index("model").loc[m, "skew_ana"] for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, sim_sk, width=width, color=BLUE, label="MC skew")
    ax.bar(x + width / 2, ana_sk, width=width, color=BURG, alpha=0.85,
           label="Analytic skew")
    ax.axhline(0.0, color=INK, lw=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(r"Skewness of $r_T$ at T = 10y")
    ax.set_title("CIR positive skew vs symmetric Gaussian baselines",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", facecolor=BG, edgecolor=RULE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks(coupling_df, term_df):
    checks: dict = {}

    # (1) Vasicek MC within ~3 SE of closed form at every maturity
    vasi = coupling_df[coupling_df["model"] == "Vasicek"]
    vasi_ok = bool((vasi["MC_EM_minus_CF_bp"].abs()
                    <= 3.0 * vasi["MC_EM_se_bp"] + 0.5).all())
    checks["vasicek_mc_within_3se"] = {
        "max_abs_gap_bp": float(vasi["MC_EM_minus_CF_bp"].abs().max()),
        "pass": vasi_ok,
    }

    # (2) CIR MC marginal price within 5 bp of closed form (both drivers)
    cir = coupling_df[coupling_df["model"] == "CIR"]
    cir_ok = bool((cir["MC_EM_minus_CF_bp"].abs() <= 5.0).all()
                  and (cir["MC_exact_minus_CF_bp"].abs() <= 5.0).all())
    checks["cir_marginal_price_within_5bp"] = {
        "max_abs_em_gap_bp": float(cir["MC_EM_minus_CF_bp"].abs().max()),
        "max_abs_exact_gap_bp": float(cir["MC_exact_minus_CF_bp"].abs().max()),
        "pass": cir_ok,
    }

    # (3) CIR skew matches ncx2 analytic within MC noise (~0.05 at N=10k)
    cir_row = term_df.set_index("model").loc["CIR"]
    skew_gap = abs(cir_row["skew_sim"] - cir_row["skew_ana"])
    checks["cir_skew_matches_ncx2"] = {
        "skew_sim": float(cir_row["skew_sim"]),
        "skew_ana": float(cir_row["skew_ana"]),
        "abs_gap": float(skew_gap),
        "pass": bool(skew_gap < 0.05),
    }

    # (4) Gaussian baselines symmetric (|skew| < 0.05)
    dbm_skew = abs(term_df.set_index("model").loc["DBM", "skew_sim"])
    vasi_skew = abs(term_df.set_index("model").loc["Vasicek", "skew_sim"])
    checks["gaussian_baselines_symmetric"] = {
        "dbm_abs_skew": float(dbm_skew),
        "vasicek_abs_skew": float(vasi_skew),
        "pass": bool(dbm_skew < 0.05 and vasi_skew < 0.05),
    }

    # (5) CIR strictly positive left tail: p01 > 0
    cir_p01 = term_df.set_index("model").loc["CIR", "p01"]
    checks["cir_positive_left_tail"] = {
        "cir_p01": float(cir_p01),
        "pass": bool(cir_p01 > 0.0),
    }
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase D, Step 22: CIR bond pricing and risk metrics")
    print("-" * 64)
    print(f"r0 = {R0}, dt = {DT:.6f}, N = {N_PATHS}, "
          f"T_max = {HORIZON_YEARS}, seed = {SEED}")
    print(f"Vasicek: a = {A_VASI}, b = {B_VASI}, sigma_v = {SIG_VASI}")
    print(f"CIR:     a = {A_CIR}, b = {B_CIR}, sigma_c = {SIG_CIR}")
    print(f"DBM:     sigma_d = {SIG_DBM}")
    print(f"Const:   r = {R_CONST}")
    print()

    # ---- Simulate -------------------------------------------------------
    dbm_paths = simulate_dbm(R0, SIG_DBM, HORIZON_YEARS, DT, N_PATHS, SEED)
    vasi_em = simulate_vasicek_em(A_VASI, B_VASI, SIG_VASI, R0,
                                  HORIZON_YEARS, DT, N_PATHS, SEED)
    vasi_ex = simulate_vasicek_exact(A_VASI, B_VASI, SIG_VASI, R0,
                                     HORIZON_YEARS, DT, N_PATHS, SEED)
    cir_em = simulate_cir_em(A_CIR, B_CIR, SIG_CIR, R0,
                             HORIZON_YEARS, DT, N_PATHS, SEED)
    cir_ex = simulate_cir_exact(A_CIR, B_CIR, SIG_CIR, R0,
                                HORIZON_YEARS, DT, N_PATHS, SEED)

    # ---- Closed-form bond prices and risk metrics -----------------------
    bond_rows = []
    for T in BOND_MATS:
        for name, price_fn in (
            ("constant", price_constant),
            ("DBM", price_dbm),
            ("Vasicek", price_vasicek),
            ("CIR", price_cir),
        ):
            p = price_fn(T)
            rm = risk_metrics(name, T, p)
            bond_rows.append({
                "model": name, "T_yr": T, "P0T": p,
                "yield_pct": rm["yield"] * 100,
                "duration": rm["duration"],
                "convexity": rm["convexity"],
                "DV01": rm["DV01"],
            })
    bond_df = pd.DataFrame(bond_rows)
    bond_df.to_csv(OUTPUT_DIR / "step22_bond_metrics.csv", index=False)
    print("Closed-form bond prices and risk metrics:")
    print(bond_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- MC vs closed-form coupling -------------------------------------
    coupling_rows = []
    coupling_rows.extend(coupling_table("Vasicek", vasi_em, vasi_ex, DT, BOND_MATS))
    coupling_rows.extend(coupling_table("CIR", cir_em, cir_ex, DT, BOND_MATS))
    mc_dbm = mc_bond_prices(dbm_paths, DT, BOND_MATS)
    for T in BOND_MATS:
        cf = price_dbm(T)
        df = mc_dbm[T][1]
        coupling_rows.append({
            "model": "DBM", "T_yr": T,
            "P_closed": cf,
            "P_MC_EM": float(df.mean()),
            "P_MC_exact": float(df.mean()),
            "MC_EM_minus_CF_bp": float((df.mean() - cf) * 1e4),
            "MC_exact_minus_CF_bp": float((df.mean() - cf) * 1e4),
            "MC_EM_se_bp": float(df.std(ddof=1) / np.sqrt(df.shape[0]) * 1e4),
            "MC_exact_se_bp": float(df.std(ddof=1) / np.sqrt(df.shape[0]) * 1e4),
            "pathwise_abs_gap_EM_bp": 0.0,
        })
    for T in BOND_MATS:
        cf = price_constant(T)
        coupling_rows.append({
            "model": "constant", "T_yr": T,
            "P_closed": cf, "P_MC_EM": cf, "P_MC_exact": cf,
            "MC_EM_minus_CF_bp": 0.0, "MC_exact_minus_CF_bp": 0.0,
            "MC_EM_se_bp": 0.0, "MC_exact_se_bp": 0.0,
            "pathwise_abs_gap_EM_bp": 0.0,
        })
    coupling_df = pd.DataFrame(coupling_rows)
    coupling_df.to_csv(OUTPUT_DIR / "step22_coupling.csv", index=False)
    print("Same-seed MC vs closed-form coupling:")
    print(coupling_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Terminal r_T distribution: skew, kurtosis, tails ----------------
    k_term = int(round(HORIZON_YEARS / DT))
    rT_by_model = {
        "DBM": dbm_paths[:, k_term],
        "Vasicek": vasi_ex[:, k_term],
        "CIR": cir_ex[:, k_term],
    }
    ana_skew_kurt = {
        "DBM": (0.0, 0.0),
        "Vasicek": (0.0, 0.0),
        "CIR": cir_terminal_moments(HORIZON_YEARS),
    }
    term_df = terminal_distribution_table(rT_by_model, ana_skew_kurt)
    term_df.to_csv(OUTPUT_DIR / "step22_terminal_distribution.csv", index=False)
    print("Terminal-distribution diagnostics (T = 10y):")
    print(term_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- VaR / ES on forward bond price ---------------------------------
    var_rows = []
    for h in VAR_HORIZONS:
        for name in ("constant", "DBM", "Vasicek", "CIR"):
            var_rows.extend(tail_risk_table(name, h, VAR_BOND_MAT, VAR_LEVELS))
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(OUTPUT_DIR / "step22_var_es.csv", index=False)
    print("Forward-bond VaR / ES (P(h, T = 10y)):")
    print(var_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    # ---- Figures --------------------------------------------------------
    def vec_price(price_fn):
        return lambda taus: np.array([price_fn(float(t)) for t in taus])
    price_dense = {
        "constant": vec_price(price_constant),
        "DBM":      vec_price(price_dbm),
        "Vasicek":  vec_price(price_vasicek),
        "CIR":      vec_price(price_cir),
    }
    c_cir = 4.0 * A_CIR / (SIG_CIR ** 2 * (1.0 - np.exp(-A_CIR * HORIZON_YEARS)))
    ana_pdfs = {
        "DBM": lambda x: norm.pdf(
            x, loc=R0, scale=SIG_DBM * np.sqrt(HORIZON_YEARS)),
        "Vasicek": lambda x: norm.pdf(
            x,
            loc=B_VASI + (R0 - B_VASI) * np.exp(-A_VASI * HORIZON_YEARS),
            scale=np.sqrt(SIG_VASI ** 2
                          * (1 - np.exp(-2 * A_VASI * HORIZON_YEARS))
                          / (2 * A_VASI))),
        "CIR": lambda x: (
            c_cir * ncx2.pdf(
                np.clip(x, 1e-12, None) * c_cir,
                df=4.0 * A_CIR * B_CIR / SIG_CIR ** 2,
                nc=c_cir * R0 * np.exp(-A_CIR * HORIZON_YEARS))),
    }
    fig1_yield_curve(price_dense, FIGURES_DIR / "step22_fig1_yield_curve.png")
    fig2_terminal_density(rT_by_model, ana_pdfs,
                          FIGURES_DIR / "step22_fig2_terminal_density.png")
    fwd_samples = {name: forward_bond_distribution(name, 1.0, VAR_BOND_MAT)
                   for name in ("constant", "DBM", "Vasicek", "CIR")}
    fig3_forward_price_distribution(fwd_samples, 1.0, VAR_BOND_MAT,
                                    FIGURES_DIR / "step22_fig3_forward_price.png")
    fig4_var_es_bars(var_df, FIGURES_DIR / "step22_fig4_var_es.png")
    fig5_skew_summary(term_df, FIGURES_DIR / "step22_fig5_skew.png")

    # ---- Sanity checks ---------------------------------------------------
    checks = sanity_checks(coupling_df, term_df)
    print("=== Sanity checks ===")
    for name, info in checks.items():
        marker = "PASS" if info["pass"] else "FAIL"
        print(f"  [{marker}]  {name}")
    print()

    # ---- JSON summary ----------------------------------------------------
    summary = {
        "config": dict(r0=R0, dt=DT, n_paths=N_PATHS,
                       horizon_years=HORIZON_YEARS, seed=SEED,
                       a_vasi=A_VASI, b_vasi=B_VASI, sigma_vasi=SIG_VASI,
                       a_cir=A_CIR, b_cir=B_CIR, sigma_cir=SIG_CIR,
                       sigma_dbm=SIG_DBM, r_const=R_CONST),
        "bond_metrics": bond_df.to_dict(orient="records"),
        "coupling": coupling_df.to_dict(orient="records"),
        "terminal_distribution": term_df.to_dict(orient="records"),
        "var_es": var_df.to_dict(orient="records"),
        "feller": {"2ab": 2 * A_CIR * B_CIR, "sigma2": SIG_CIR ** 2,
                   "ratio": 2 * A_CIR * B_CIR / SIG_CIR ** 2,
                   "holds": bool(2 * A_CIR * B_CIR >= SIG_CIR ** 2)},
        "sanity_checks": checks,
    }
    with open(OUTPUT_DIR / "step22_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"figures written to {FIGURES_DIR}")
    print(f"output written to  {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
