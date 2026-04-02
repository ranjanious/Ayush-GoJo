"""
Step 5: Independent test suite for risk_metrics.py
Validates correctness, numerical stability, and edge-case handling
under controlled synthetic inputs before Phase B begins.

Run with:
    python -m pytest tests/test_risk_metrics.py -v
"""

import numpy as np
import pytest
from scipy import stats

from models.shared.risk_metrics import risk_metrics


# --- Test 1: Constant Input ---
def test_constant_input():
    """All terminal values identical (degenerate distribution)."""
    v = np.full(1000, 95.0)
    r = risk_metrics(v)
    assert abs(r['var'] - 95.0) < 1e-9
    assert abs(r['es'] - 95.0) < 1e-9
    assert abs(r['prob_loss'] - 1.0) < 1e-9
    assert abs(r['mean'] - 95.0) < 1e-9
    assert r['std'] < 1e-9
    assert r['n_paths'] == 1000


# --- Test 2: All Above Initial ---
def test_all_above_initial():
    """prob_loss = 0 when no path ends below 100."""
    v = np.full(500, 105.0)
    r = risk_metrics(v)
    assert abs(r['prob_loss']) < 1e-9
    assert r['n_paths'] == 500


# --- Test 3: Known Normal ---
def test_normal_distribution():
    """VaR, ES, mean, std against closed-form normal quantiles."""
    np.random.seed(42)
    v = np.random.normal(100.0, 5.0, 100_000)
    r = risk_metrics(v)
    var_cf = 100.0 + 5.0 * stats.norm.ppf(0.05)
    es_cf = 100.0 - 5.0 * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
    assert abs(r['mean'] - 100.0) < 0.05
    assert abs(r['std'] - 5.0) < 0.05
    assert abs(r['var'] - var_cf) < 0.15
    assert abs(r['es'] - es_cf) < 0.20
    # P(N(100,5) < 100) = Phi(0) = 0.5 (not 0.0228 as in original spec;
    # 0.0228 = P(Z < -2) which would require initial_value=110)
    assert abs(r['prob_loss'] - 0.5) < 0.01


# --- Test 4: Custom Alpha ---
def test_custom_alpha():
    """Function respects non-default alpha=0.01."""
    np.random.seed(0)
    v = np.random.normal(100.0, 5.0, 50_000)
    r = risk_metrics(v, alpha=0.01)
    var_cf = 100.0 + 5.0 * stats.norm.ppf(0.01)
    assert abs(r['var'] - var_cf) < 0.20


# --- Test 5: Skewness and Kurtosis Signs ---
def test_skewness_kurtosis():
    """Correct signs for normal (symmetric) and lognormal (right-skewed)."""
    np.random.seed(42)
    normal = np.random.normal(100.0, 5.0, 100_000)
    r_norm = risk_metrics(normal)
    assert abs(r_norm['skewness']) < 0.05
    assert abs(r_norm['kurtosis']) < 0.10

    lognormal = np.exp(np.random.normal(0, 1, 100_000))
    r_log = risk_metrics(lognormal, initial_value=0.0)
    assert r_log['skewness'] > 0
    assert r_log['kurtosis'] > 0


# --- Test 6: std_error Finite and Positive ---
def test_std_error():
    """KDE-based delta-method SE is positive and finite."""
    np.random.seed(42)
    v = np.random.normal(100.0, 5.0, 10_000)
    r = risk_metrics(v)
    assert np.isfinite(r['std_error'])
    assert r['std_error'] > 0


# --- Test 7: n_paths ---
def test_n_paths():
    """n_paths echoes actual array size."""
    v = np.random.normal(100.0, 3.0, 7777)
    r = risk_metrics(v)
    assert r['n_paths'] == 7777


# --- Test 8: ES <= VaR ---
def test_es_leq_var():
    """Expected Shortfall must always be <= VaR (monotonicity)."""
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        v = np.random.normal(100.0, 5.0, 20_000)
        r = risk_metrics(v)
        assert r['es'] <= r['var'] + 1e-9


# --- Test 9: Custom initial_value ---
def test_custom_initial_value():
    """prob_loss shifts correctly with non-default initial_value."""
    v = np.full(1000, 50.0)
    assert abs(risk_metrics(v, initial_value=40.0)['prob_loss'] - 0.0) < 1e-9
    assert abs(risk_metrics(v, initial_value=60.0)['prob_loss'] - 1.0) < 1e-9
