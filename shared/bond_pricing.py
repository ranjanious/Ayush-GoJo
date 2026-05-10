"""
Shared bond pricing infrastructure (Phase A).
Reused identically by all four models: DBM, Vasicek, CIR, Hull-White.

Step 14 added the canonical horizon-valuation identity
    V(T_h) = sum_j cf_j * exp(I[h] - I[j])
implemented as `price_bond_at_horizon`, plus the helper `path_integrals`
that pre-computes I[k] = integral of r_u du from 0 to k*dt (left-endpoint
Riemann), and `constant_rate_bond_price` for the deterministic baseline.
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


def portfolio_value(price_A, price_B, wA=0.5, wB=0.5, w_a=None, w_b=None):
    """
    Weighted portfolio value. Default 50/50; V0 = wA*100 + wB*100 = 100.

    `w_a` / `w_b` aliases supplied for Step 14 spec compatibility; if either
    is provided it overrides the corresponding wA / wB argument.

    Args:
        price_A, price_B : ndarray (n_paths,) or float
        wA, wB           : portfolio weights (legacy positional names)
        w_a, w_b         : portfolio weights (Step 14 keyword names)

    Returns:
        ndarray (n_paths,) or float
    """
    if w_a is not None:
        wA = w_a
    if w_b is not None:
        wB = w_b
    return wA * price_A + wB * price_B


# ----------------------------------------------------------------------
# Step 14 additions: canonical horizon-valuation pipeline
# ----------------------------------------------------------------------
def path_integrals(rate_paths, dt):
    """
    Cumulative integral of the short rate, left-endpoint Riemann.

        I[p, k] = sum_{i < k} r[p, i] * dt   for k = 0, 1, ..., n_steps
        I[p, 0] = 0

    Returns
    -------
    ndarray (n_paths, n_steps + 1)
    """
    I = np.zeros_like(rate_paths)
    I[:, 1:] = np.cumsum(rate_paths[:, :-1] * dt, axis=1)
    return I


def price_bond_at_horizon(
    rate_paths,
    coupon,
    face,
    payment_idx,
    dt,
    horizon_idx,
):
    """
    Path-specific time-T_h value of a fixed-coupon bond.

    Implements the canonical identity (Step 14 spec, Section 3.2):
        V(T_h) = sum_j cf_j * exp( I[h] - I[j] )

    Past cash flows (idx <= h) are accrued forward at path rates;
    future cash flows (idx > h) are discounted back at path rates.
    Vectorised across paths -- one matrix product per call.

    Args
    ----
    rate_paths  : ndarray (n_paths, n_steps + 1)
    coupon      : float, semiannual coupon amount (e.g. 1.84)
    face        : float, face value (e.g. 100.0)
    payment_idx : ndarray of int grid indices for cash flow dates
    dt          : float, grid spacing in years
    horizon_idx : int, grid index of the valuation horizon T_h

    Returns
    -------
    ndarray (n_paths,)
    """
    payment_idx = np.asarray(payment_idx, dtype=int)
    I = path_integrals(rate_paths, dt)
    I_h = I[:, horizon_idx]

    values = np.zeros(rate_paths.shape[0], dtype=float)
    last = int(payment_idx[-1])
    for j in payment_idx:
        cf = coupon + face if int(j) == last else coupon
        values += cf * np.exp(I_h - I[:, int(j)])
    return values


def constant_rate_bond_price(coupon, face, payment_times, r):
    """
    Deterministic bond price at t = 0 under continuous compounding at
    flat rate r.

        P0 = sum_j cf_j * exp(-r * t_j)

    Args
    ----
    coupon         : float, semiannual coupon amount
    face           : float, face value (added to the final cash flow)
    payment_times  : iterable of payment times in years
    r              : float, flat annualised rate

    Returns
    -------
    float
    """
    payment_times = np.asarray(payment_times, dtype=float)
    n = payment_times.size
    cfs = np.full(n, coupon, dtype=float)
    cfs[-1] += face
    return float(np.sum(cfs * np.exp(-r * payment_times)))

