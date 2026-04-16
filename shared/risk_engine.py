"""
Shared risk metric engine (Phase A).
Takes any array of terminal portfolio values; returns the full metric set.
Called identically across all four models.
"""

import numpy as np
from scipy import stats


def var_empirical(dist, alpha):
    """Lower alpha-quantile (e.g. alpha=0.05 for VaR95)."""
    return np.percentile(dist, alpha * 100)


def expected_shortfall(dist, alpha):
    """Mean of the tail at or below the VaR threshold."""
    threshold = var_empirical(dist, alpha)
    tail = dist[dist <= threshold]
    return np.mean(tail) if len(tail) > 0 else threshold


def prob_loss(dist, threshold=100.0):
    """Fraction of paths where terminal value falls below threshold."""
    return np.mean(dist < threshold)


def sim_std_error(dist, alpha):
    """
    Delta-method standard error for the empirical quantile estimator.
    Uses Silverman's bandwidth for the kernel density estimate.
    """
    N   = len(dist)
    q   = var_empirical(dist, alpha)
    if np.std(dist) < 1e-12:
        return np.nan
    kde = stats.gaussian_kde(dist, bw_method='silverman')
    f_hat = kde(q)[0]
    if f_hat < 1e-10:
        return np.nan
    return np.sqrt(alpha * (1 - alpha) / N) / f_hat


def summary_stats(dist, label="", V0=100.0):
    """
    Full risk metric set for a terminal value distribution.

    Returns a dict with: Mean, Std, Min, Max, Skewness, Kurtosis,
    VaR_95, VaR_99, ES_95, ES_99, P_loss, SE_VaR95, SE_VaR99.
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
