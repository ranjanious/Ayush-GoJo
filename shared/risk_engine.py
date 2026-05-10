"""
shared/risk_engine.py
---------------------
Shared risk-metric engine used by all four short-rate models (DBM, Vasicek,
CIR, Hull-White). Built in Phase A, Step 5. Extended in Phase B, Step 9 to
report simulation standard errors for all tail estimates via both the delta
method with Silverman-bandwidth KDE and a nonparametric bootstrap.
"""

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
# Core risk metrics
# ----------------------------------------------------------------------
def var_empirical(dist, alpha):
    """Empirical Value-at-Risk: the lower alpha-quantile of the distribution."""
    return float(np.quantile(dist, alpha))


def expected_shortfall(dist, alpha):
    """
    Expected Shortfall (CVaR): the mean of the distribution conditional on
    falling at or below VaR_alpha.
    """
    var_level = var_empirical(dist, alpha)
    tail = dist[dist <= var_level]
    return float(np.mean(tail)) if tail.size else float(var_level)


def prob_loss(dist, threshold=100.0):
    """Probability of a nominal loss: P(V_T < threshold)."""
    return float(np.mean(dist < threshold))


# ----------------------------------------------------------------------
# Simulation standard errors -- Method A: delta / influence-function
# ----------------------------------------------------------------------
def _silverman_bandwidth(dist):
    """Silverman's rule-of-thumb bandwidth for a Gaussian kernel."""
    n = len(dist)
    s = np.std(dist, ddof=1)
    iqr = np.subtract(*np.percentile(dist, [75, 25]))
    scale = min(s, iqr / 1.34) if iqr > 0 else s
    return 0.9 * scale * n ** (-1.0 / 5.0)


def var_se_delta(dist, alpha):
    """
    Delta-method VaR standard error:
        SE[VaR_alpha] = sqrt(alpha * (1 - alpha) / N) / f_hat(VaR_alpha)
    with f_hat a Silverman-bandwidth Gaussian KDE evaluated at the VaR point.
    """
    n = len(dist)
    if np.std(dist) < 1e-12:
        return float("nan")
    var_level = var_empirical(dist, alpha)
    h = _silverman_bandwidth(dist)
    if h < 1e-15:
        return float("nan")
    f_hat = np.mean(stats.norm.pdf((dist - var_level) / h)) / h
    if f_hat < 1e-10:
        return float("nan")
    return float(np.sqrt(alpha * (1.0 - alpha) / n) / f_hat)


def es_se_iid(dist, alpha):
    """
    Influence-function SE for ES under the i.i.d. assumption:
        SE[ES_alpha] ~ std(X | X <= VaR_alpha) / sqrt(alpha * N).
    Follows Acerbi and Tasche (2002).
    """
    n = len(dist)
    var_level = var_empirical(dist, alpha)
    tail = dist[dist <= var_level]
    if tail.size < 2:
        return float("nan")
    return float(np.std(tail, ddof=1) / np.sqrt(alpha * n))


# ----------------------------------------------------------------------
# Simulation standard errors -- Method B: nonparametric bootstrap
# ----------------------------------------------------------------------
def var_se_bootstrap(dist, alpha, n_boot=1000, seed=42):
    """Nonparametric bootstrap SE for VaR with B = n_boot resamples."""
    rng = np.random.default_rng(seed)
    n = len(dist)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(dist, size=n, replace=True)
        boot[b] = var_empirical(sample, alpha)
    return float(np.std(boot, ddof=1))


def es_se_bootstrap(dist, alpha, n_boot=1000, seed=42):
    """Nonparametric bootstrap SE for ES with B = n_boot resamples."""
    rng = np.random.default_rng(seed)
    n = len(dist)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(dist, size=n, replace=True)
        boot[b] = expected_shortfall(sample, alpha)
    return float(np.std(boot, ddof=1))


# ----------------------------------------------------------------------
# Legacy SE (Step 8 backward compat -- uses scipy.stats.gaussian_kde)
# ----------------------------------------------------------------------
def sim_std_error(dist, alpha):
    """Delta-method SE via scipy gaussian_kde (kept for Step 8 compatibility)."""
    N = len(dist)
    q = var_empirical(dist, alpha)
    if np.std(dist) < 1e-12:
        return np.nan
    kde = stats.gaussian_kde(dist, bw_method='silverman')
    f_hat = kde(q)[0]
    if f_hat < 1e-10:
        return np.nan
    return np.sqrt(alpha * (1 - alpha) / N) / f_hat


# ----------------------------------------------------------------------
# Batch interfaces
# ----------------------------------------------------------------------
def summary_stats(dist, label="", V0=100.0):
    """
    Full risk metric set for a terminal value distribution (Step 8 API).
    """
    return {
        "Label"    : label,
        "Mean"     : np.mean(dist),
        "Std"      : np.std(dist),
        "Min"      : np.min(dist),
        "Max"      : np.max(dist),
        "Skewness" : float(stats.skew(dist)),
        "Kurtosis" : float(stats.kurtosis(dist)),
        "VaR_95"   : var_empirical(dist, 0.05),
        "VaR_99"   : var_empirical(dist, 0.01),
        "ES_95"    : expected_shortfall(dist, 0.05),
        "ES_99"    : expected_shortfall(dist, 0.01),
        "P_loss"   : prob_loss(dist, V0),
        "SE_VaR95" : sim_std_error(dist, 0.05),
        "SE_VaR99" : sim_std_error(dist, 0.01),
    }


def full_metric_set(dist, threshold=100.0):
    """
    Step 14 reporting API.  Returns the field names used in the Step 14
    completion-log tables exactly:

        mean, std, skew, kurt, min, max,
        VaR95, ES95, VaR95_SE, VaR99, ES99, VaR99_SE,
        p_loss, n_paths

    SE for the VaR estimates is the delta-method (Silverman-bandwidth KDE)
    standard error from `var_se_delta`.
    """
    dist = np.asarray(dist, dtype=float)
    n = int(dist.size)
    out = {
        "mean":     float(np.mean(dist)),
        "std":      float(np.std(dist, ddof=1)) if n > 1 else 0.0,
        "skew":     float(stats.skew(dist)) if n > 1 else 0.0,
        "kurt":     float(stats.kurtosis(dist)) if n > 1 else 0.0,
        "min":      float(np.min(dist)),
        "max":      float(np.max(dist)),
        "VaR95":    var_empirical(dist, 0.05),
        "ES95":     expected_shortfall(dist, 0.05),
        "VaR95_SE": var_se_delta(dist, 0.05),
        "VaR99":    var_empirical(dist, 0.01),
        "ES99":     expected_shortfall(dist, 0.01),
        "VaR99_SE": var_se_delta(dist, 0.01),
        "p_loss":   prob_loss(dist, threshold),
        "n_paths":  n,
    }
    return out


def risk_report(dist, threshold=100.0, alphas=(0.05, 0.01),
                n_boot=1000, seed=42):
    """
    Full risk-metric set with dual SE methods (Step 9+ API).
    """
    out = {
        "mean": float(np.mean(dist)),
        "std":  float(np.std(dist, ddof=1)),
        "skew": float(stats.skew(dist)),
        "kurt": float(stats.kurtosis(dist)),
        "min":  float(np.min(dist)),
        "max":  float(np.max(dist)),
        "p_loss": prob_loss(dist, threshold),
    }
    for a in alphas:
        label = int(round((1 - a) * 100))
        out[f"VaR{label}"]          = var_empirical(dist, a)
        out[f"ES{label}"]           = expected_shortfall(dist, a)
        out[f"VaR{label}_SE_delta"] = var_se_delta(dist, a)
        out[f"VaR{label}_SE_boot"]  = var_se_bootstrap(dist, a, n_boot, seed)
        out[f"ES{label}_SE_iid"]    = es_se_iid(dist, a)
        out[f"ES{label}_SE_boot"]   = es_se_bootstrap(dist, a, n_boot, seed)
    return out
