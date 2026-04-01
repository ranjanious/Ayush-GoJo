"""
Unit and integration tests for the shared pricing and risk infrastructure.
Covers Steps 4 validation gates from the project record.
"""

import math
import numpy as np
import pytest

from models.shared.riemann_discount import riemann_discount
from models.shared.price_bond import price_bond
from models.shared.portfolio_value import portfolio_value
from models.shared.risk_metrics import risk_metrics


# -- riemann_discount tests ------------------------------------------------

class TestRiemannDiscount:
    def test_zero_rates_return_one(self):
        d = riemann_discount(np.zeros(120), 1 / 12)
        assert d == 1.0

    def test_flat_rate(self):
        d = riemann_discount(np.full(120, 0.05), 1 / 12)
        expected = math.exp(-0.05 * 120 * (1 / 12))
        assert abs(d - expected) < 1e-12


# -- price_bond tests ------------------------------------------------------

class TestPriceBond:
    def test_zero_coupon_bond(self):
        dt = 1 / 12
        pv = price_bond(np.full(24, 0.05), dt, coupon=0.0, face=100.0,
                        payment_steps=[24])
        expected = 100.0 * math.exp(-0.05 * 24 * dt)
        assert abs(pv - expected) < 1e-10

    def test_par_bond_a_continuous(self):
        dt = 1 / 12
        r = 0.0368
        steps_a = [6, 12, 18, 24]
        dfs_a = [math.exp(-r * s * dt) for s in steps_a]
        c_par_a = 100.0 * (1.0 - dfs_a[-1]) / sum(dfs_a)
        pv = price_bond(np.full(24, r), dt, coupon=c_par_a, face=100.0,
                        payment_steps=steps_a)
        assert abs(pv - 100.0) < 1e-10

    def test_par_bond_b_continuous(self):
        dt = 1 / 12
        r = 0.042
        steps_b = list(range(6, 121, 6))
        dfs_b = [math.exp(-r * s * dt) for s in steps_b]
        c_par_b = 100.0 * (1.0 - dfs_b[-1]) / sum(dfs_b)
        pv = price_bond(np.full(120, r), dt, coupon=c_par_b, face=100.0,
                        payment_steps=steps_b)
        assert abs(pv - 100.0) < 1e-10

    def test_treasury_coupon_not_par(self):
        dt = 1 / 12
        steps_a = [6, 12, 18, 24]
        c_t = 100.0 * 0.0368 / 2  # 1.84
        pv = price_bond(np.full(24, 0.0368), dt, coupon=c_t, face=100.0,
                        payment_steps=steps_a)
        # Under continuous compounding, Treasury coupon does NOT price at par
        assert pv < 100.0
        assert pv > 99.0


# -- portfolio_value tests -------------------------------------------------

class TestPortfolioValue:
    def test_equal_par(self):
        assert portfolio_value(100.0, 100.0) == 100.0

    def test_symmetric_cancel(self):
        assert portfolio_value(90.0, 110.0) == 100.0

    def test_custom_weights(self):
        v = portfolio_value(90.0, 110.0, weight_a=0.8, weight_b=0.2)
        assert abs(v - 94.0) < 1e-12

    def test_loss_scenario(self):
        assert portfolio_value(95.0, 92.0) < 100.0


# -- risk_metrics tests ----------------------------------------------------

class TestRiskMetrics:
    @pytest.fixture
    def normal_metrics(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(loc=100.0, scale=5.0, size=100_000)
        return risk_metrics(vals, alpha=0.05, initial_value=100.0)

    def test_var_range(self, normal_metrics):
        assert 91.0 < normal_metrics['var'] < 92.5

    def test_es_less_than_var(self, normal_metrics):
        assert normal_metrics['es'] <= normal_metrics['var']

    def test_prob_loss_near_half(self, normal_metrics):
        assert 0.48 < normal_metrics['prob_loss'] < 0.52

    def test_low_skewness(self, normal_metrics):
        assert abs(normal_metrics['skewness']) < 0.05

    def test_valid_std_error(self, normal_metrics):
        assert normal_metrics['std_error'] > 0
        assert not np.isnan(normal_metrics['std_error'])

    def test_all_keys_present(self, normal_metrics):
        expected_keys = {
            'var', 'es', 'prob_loss', 'std_error',
            'mean', 'std', 'skewness', 'kurtosis', 'n_paths'
        }
        assert expected_keys == set(normal_metrics.keys())

    def test_n_paths(self, normal_metrics):
        assert normal_metrics['n_paths'] == 100_000


# -- Integration test ------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: simulate paths -> price bonds -> portfolio -> risk."""
        rng = np.random.default_rng(42)
        dt = 1 / 12
        N_PATHS = 1_000
        N_STEPS = 120
        r0 = 0.0372
        sigma = 0.01

        increments = rng.standard_normal((N_PATHS, N_STEPS))
        rate_paths = r0 + sigma * np.cumsum(increments, axis=1)

        bond_a_steps = list(range(6, 25, 6))
        bond_b_steps = list(range(6, 121, 6))
        bond_a_coupon = 100.0 * 0.0368 / 2
        bond_b_coupon = 100.0 * 0.0420 / 2

        terminal_values = np.empty(N_PATHS)
        for i in range(N_PATHS):
            pa = price_bond(rate_paths[i], dt, bond_a_coupon, 100.0,
                            bond_a_steps)
            pb = price_bond(rate_paths[i], dt, bond_b_coupon, 100.0,
                            bond_b_steps)
            terminal_values[i] = portfolio_value(pa, pb)

        m = risk_metrics(terminal_values, alpha=0.05, initial_value=100.0)

        assert m['n_paths'] == N_PATHS
        assert 50.0 < m['var'] < 150.0
        assert m['es'] <= m['var']
