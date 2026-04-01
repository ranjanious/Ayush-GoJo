"""
riemann_discount.py

Computes the path-specific discount factor for one simulated
short-rate path using a left-endpoint Riemann sum approximation
of the integral of r_s from 0 to T.

    Discount factor: D = exp( -sum_{k=0}^{N-1} r_k * dt )

Imported by all four models: DBM, Vasicek, CIR, Hull-White.
"""

import numpy as np


def riemann_discount(rate_path: np.ndarray, dt: float) -> float:
    """
    Parameters
    ----------
    rate_path : np.ndarray, shape (N,)
        Simulated short rates at each grid step along one path.
        Units: annualised decimal (e.g. 0.04 means 4 percent).
    dt : float
        Length of one time step in years (e.g. 1/12 for monthly).

    Returns
    -------
    float
        Path-specific discount factor in the range (0, 1].
        Returns exactly 1.0 when all rates are zero.
    """
    integral = np.sum(rate_path) * dt
    return float(np.exp(-integral))
