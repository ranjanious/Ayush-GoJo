"""
Phase B Step 7: Tests for DBM Euler-Maruyama simulator.
Validates path shape, moments, bond pricing, and risk measures.
"""

import numpy as np
import pytest

from models.dbm.dbm_simulator import simulate_dbm, bond_price_along_path


class TestSimulateDBM:
    def test_output_shape(self):
        """Paths array has shape (N, steps+1)."""
        paths = simulate_dbm(N=100, T=2.0, dt=1 / 12, seed=0)
        assert paths.shape == (100, 25)

    def test_initial_condition(self):
        """All paths start at r0."""
        paths = simulate_dbm(r0=0.0372, N=500, seed=1)
        assert np.all(paths[:, 0] == 0.0372)

    def test_mean_terminal_rate(self):
        """E[r_T] = r0 + mu*T. For mu=0, E[r_T] = r0."""
        paths = simulate_dbm(r0=0.0372, mu=0.0, N=10_000, T=2.0, seed=42)
        r_T = paths[:, -1]
        assert abs(r_T.mean() - 0.0372) < 0.001

    def test_std_terminal_rate(self):
        """Std[r_T] = sigma * sqrt(T). For sigma=0.01, T=2: 0.01414."""
        paths = simulate_dbm(r0=0.0372, sigma=0.01, N=10_000, T=2.0, seed=42)
        r_T = paths[:, -1]
        expected_std = 0.01 * np.sqrt(2.0)
        assert abs(r_T.std() - expected_std) < 0.001

    def test_nonzero_drift(self):
        """With positive drift, E[r_T] > r0."""
        paths = simulate_dbm(r0=0.04, mu=0.01, N=10_000, T=2.0, seed=42)
        r_T = paths[:, -1]
        expected_mean = 0.04 + 0.01 * 2.0  # r0 + mu*T = 0.06
        assert abs(r_T.mean() - expected_mean) < 0.001

    def test_reproducibility(self):
        """Same seed produces identical paths."""
        p1 = simulate_dbm(seed=99, N=50)
        p2 = simulate_dbm(seed=99, N=50)
        assert np.array_equal(p1, p2)

    def test_different_seeds(self):
        """Different seeds produce different paths."""
        p1 = simulate_dbm(seed=1, N=50)
        p2 = simulate_dbm(seed=2, N=50)
        assert not np.array_equal(p1, p2)


class TestBondPriceAlongPath:
    def test_constant_rate_zero_coupon(self):
        """Flat rate path: ZCB price = face * exp(-r*T)."""
        import math
        N, steps = 100, 24
        r = 0.05
        dt = 1 / 12
        paths = np.full((N, steps + 1), r)
        prices = bond_price_along_path(
            paths, coupon=0.0, face=100.0,
            payment_times=np.array([2.0]), dt=dt,
        )
        expected = 100.0 * math.exp(-r * 2.0)
        assert np.allclose(prices, expected, atol=1e-8)

    def test_constant_rate_bond_a(self):
        """Flat rate: Bond A price is deterministic and near par."""
        N, steps = 100, 24
        r = 0.0368
        dt = 1 / 12
        paths = np.full((N, steps + 1), r)
        payment_times = np.arange(0.5, 2.5, 0.5)
        prices = bond_price_along_path(
            paths, coupon=1.84, face=100.0,
            payment_times=payment_times, dt=dt,
        )
        # All paths identical -> all prices identical
        assert prices.std() < 1e-10
        # Price near par under continuous compounding
        assert 99.0 < prices[0] < 101.0

    def test_stochastic_prices_vary(self):
        """With stochastic rates, bond prices should vary across paths."""
        paths = simulate_dbm(N=1_000, sigma=0.01, T=2.0, seed=42)
        payment_times = np.arange(0.5, 2.5, 0.5)
        prices = bond_price_along_path(
            paths, coupon=1.84, face=100.0,
            payment_times=payment_times, dt=1 / 12,
        )
        assert prices.std() > 0.1

    def test_prices_in_reasonable_range(self):
        """Bond A prices should be between $90 and $110 (sanity check)."""
        paths = simulate_dbm(N=10_000, sigma=0.01, T=2.0, seed=42)
        payment_times = np.arange(0.5, 2.5, 0.5)
        prices = bond_price_along_path(
            paths, coupon=1.84, face=100.0,
            payment_times=payment_times, dt=1 / 12,
        )
        assert prices.mean() > 90.0
        assert prices.mean() < 110.0


class TestRiskMeasures:
    @pytest.fixture
    def bond_a_prices(self):
        paths = simulate_dbm(N=10_000, sigma=0.01, T=2.0, seed=42)
        payment_times = np.arange(0.5, 2.5, 0.5)
        return bond_price_along_path(
            paths, coupon=1.84, face=100.0,
            payment_times=payment_times, dt=1 / 12,
        )

    def test_var95_below_par(self, bond_a_prices):
        """VaR at 95% should be below par (stochastic rates add risk)."""
        var95 = np.percentile(bond_a_prices, 5)
        assert var95 < 100.0

    def test_es95_below_var95(self, bond_a_prices):
        """ES must be <= VaR."""
        var95 = np.percentile(bond_a_prices, 5)
        es95 = bond_a_prices[bond_a_prices <= var95].mean()
        assert es95 <= var95

    def test_var99_below_var95(self, bond_a_prices):
        """99% VaR should be more extreme (lower) than 95% VaR."""
        var95 = np.percentile(bond_a_prices, 5)
        var99 = np.percentile(bond_a_prices, 1)
        assert var99 < var95
