"""
vasicek_closed_form.py
----------------------
Closed-form zero-coupon bond price under the Vasicek (1977) short-rate model.

The Vasicek SDE under the (assumed) risk-neutral measure is

    dr_t = a (b - r_t) dt + sigma dW_t,

with constant parameters a > 0, b in R, sigma > 0.

The zero-coupon bond price at time t for maturity T admits the affine form

    P(t, T) = A(t, T) * exp(-B(t, T) * r_t),

with

    B(t, T)   = (1 - exp(-a (T - t))) / a,
    ln A(t,T) = (b - sigma^2 / (2 a^2)) * (B(t, T) - (T - t))
                - sigma^2 * B(t, T)^2 / (4 a).

This module is the reference implementation used by Phase C Step 15
to validate the Monte Carlo simulator written for Step 13.

Project: Stochastic Modeling of Interest Rates and Its Impact on
         Bond Portfolio Risk (Ranjan and Jinka, 2026).
"""

from __future__ import annotations

import numpy as np


def vasicek_B(t: float, T: float, a: float) -> float:
    """B(t, T) = (1 - exp(-a (T - t))) / a."""
    tau = T - t
    if a <= 0.0:
        raise ValueError("Vasicek mean-reversion speed 'a' must be positive.")
    return (1.0 - np.exp(-a * tau)) / a


def vasicek_lnA(t: float, T: float, a: float, b: float, sigma: float) -> float:
    """ln A(t, T) under the Vasicek model."""
    tau = T - t
    B = vasicek_B(t, T, a)
    term1 = (b - sigma ** 2 / (2.0 * a ** 2)) * (B - tau)
    term2 = -(sigma ** 2) * B ** 2 / (4.0 * a)
    return term1 + term2


def vasicek_zcb_price(
    t: float,
    T: float,
    r_t: float,
    a: float,
    b: float,
    sigma: float,
) -> float:
    """Closed-form zero-coupon bond price P(t, T) under Vasicek."""
    B = vasicek_B(t, T, a)
    lnA = vasicek_lnA(t, T, a, b, sigma)
    return float(np.exp(lnA - B * r_t))


def vasicek_zcb_yield(
    t: float,
    T: float,
    r_t: float,
    a: float,
    b: float,
    sigma: float,
) -> float:
    """Continuously compounded zero yield y(t, T) = -ln P(t, T) / (T - t)."""
    tau = T - t
    if tau <= 0.0:
        raise ValueError("Maturity T must exceed valuation date t.")
    lnP = vasicek_lnA(t, T, a, b, sigma) - vasicek_B(t, T, a) * r_t
    return float(-lnP / tau)


def vasicek_zcb_price_vec(
    tau: float,
    r: np.ndarray,
    a: float,
    b: float,
    sigma: float,
) -> np.ndarray:
    """
    Vectorised Vasicek ZCB price for a scalar tau = T - t and an array of
    short rates r (one per Monte Carlo path).

    Returns
    -------
    ndarray with the same shape as `r`: P(t, t + tau) for each rate value.

    Vasicek is time-homogeneous, so B(t, T) and lnA(t, T) depend only on
    tau = T - t.  This helper is used by Step 16 to value each bond's
    remaining cash-flow strip along every simulated path.
    """
    if tau <= 0.0:
        return np.ones_like(np.asarray(r, dtype=float))
    B = vasicek_B(0.0, tau, a)
    lnA = vasicek_lnA(0.0, tau, a, b, sigma)
    return np.exp(lnA - B * np.asarray(r, dtype=float))


if __name__ == "__main__":
    # Sigma -> 0 sanity: the diffusion vanishes, r_t = b + (r0 - b) exp(-a t),
    # and P(0, T) = exp(- integral_0^T r_s ds).
    a, b, sigma, r0, T = 0.5, 0.04, 0.0, 0.0372, 5.0
    P = vasicek_zcb_price(0.0, T, r0, a, b, sigma)
    integ = b * T + (r0 - b) * vasicek_B(0.0, T, a)
    print(f"sigma = 0 sanity: P = {P:.6f}, exp(-integ) = {np.exp(-integ):.6f}")
