"""
Shared bond pricing infrastructure (Phase A).
Reused identically by all four models: DBM, Vasicek, CIR, Hull-White.
"""

import numpy as np


def riemann_discount(rate_paths, pay_time, dt):
    """
    Path-specific discount factor to a payment at pay_time from t=0.
    Uses left-endpoint Riemann sum: exp(-sum(r_k * dt)).

    Args:
        rate_paths : ndarray (n_paths, n_steps+1)
        pay_time   : float, payment date in years
        dt         : float, time step in years

    Returns:
        ndarray (n_paths,)
    """
    n_steps = int(round(pay_time / dt))
    return np.exp(-np.sum(rate_paths[:, :n_steps], axis=1) * dt)


def price_bond(rate_paths, cash_flows, pay_dates, origin_time, dt):
    """
    Present value at origin_time of all cash flows occurring after origin_time.
    Discounts each cash flow using path-specific rates from origin_time onward.

    Args:
        rate_paths  : ndarray (n_paths, n_steps+1)
        cash_flows  : ndarray (n_payments,)  -- includes face value in final entry
        pay_dates   : ndarray (n_payments,)  -- payment dates in years
        origin_time : float, valuation horizon in years
        dt          : float, time step in years

    Returns:
        ndarray (n_paths,)
    """
    n_paths = rate_paths.shape[0]
    prices  = np.zeros(n_paths)
    origin_step = int(round(origin_time / dt))

    for j, pd in enumerate(pay_dates):
        if pd <= origin_time + 1e-9:
            continue
        pay_step = int(round(pd / dt))
        segment  = rate_paths[:, origin_step:pay_step]
        df       = np.exp(-np.sum(segment, axis=1) * dt)
        prices  += cash_flows[j] * df

    return prices


def price_bond_flat(r_flat, cash_flows, pay_dates, origin_time):
    """
    Price a bond at a single flat continuously compounded rate.
    Used for the constant-rate benchmark.

    Args:
        r_flat      : float
        cash_flows  : ndarray (n_payments,)
        pay_dates   : ndarray (n_payments,)
        origin_time : float

    Returns:
        float
    """
    price = 0.0
    for j, pd in enumerate(pay_dates):
        if pd <= origin_time + 1e-9:
            continue
        tau    = pd - origin_time
        price += cash_flows[j] * np.exp(-r_flat * tau)
    return price


def portfolio_value(price_A, price_B, wA=0.5, wB=0.5):
    """
    Weighted portfolio value. Default 50/50; V0 = wA*100 + wB*100 = 100.

    Args:
        price_A, price_B : ndarray (n_paths,) or float
        wA, wB           : portfolio weights

    Returns:
        ndarray (n_paths,) or float
    """
    return wA * price_A + wB * price_B
