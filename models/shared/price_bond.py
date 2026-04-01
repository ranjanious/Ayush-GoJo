"""
price_bond.py

Prices a fixed-coupon bond along a single simulated short-rate path
by discounting each cash flow at its payment date.

The discount factor at payment step j is computed by calling
riemann_discount on the slice rate_path[:j], which covers
the rates at steps 0, 1, ..., j-1.

Imported by all four models: DBM, Vasicek, CIR, Hull-White.
"""

import numpy as np
from models.shared.riemann_discount import riemann_discount


def price_bond(
    rate_path: np.ndarray,
    dt: float,
    coupon: float,
    face: float,
    payment_steps: list[int],
) -> float:
    """
    Parameters
    ----------
    rate_path : np.ndarray, shape (N,)
        Simulated short rates along one path.
        Must be at least payment_steps[-1] elements long.
        Units: annualised decimal.
    dt : float
        Length of one time step in years (e.g. 1/12 for monthly).
    coupon : float
        Cash amount paid at each payment date.
        Example: for face=100, annual coupon rate 3.68%,
        semiannual payments: coupon = 100.0 * 0.0368 / 2 = 1.84
    face : float
        Principal repaid at the final payment step (e.g. 100.0).
    payment_steps : list[int]
        Grid step indices at which cash flows occur.
        The final entry is the maturity step where face is also paid.
        Example for Bond A (2-year, monthly grid, semiannual):
            [6, 12, 18, 24]
        Example for Bond B (10-year, monthly grid, semiannual):
            list(range(6, 121, 6))

    Returns
    -------
    float
        Present value of all cash flows along this path.
    """
    pv = 0.0
    for step in payment_steps[:-1]:
        df = riemann_discount(rate_path[:step], dt)
        pv += coupon * df
    final_step = payment_steps[-1]
    df_final = riemann_discount(rate_path[:final_step], dt)
    pv += (coupon + face) * df_final
    return pv
