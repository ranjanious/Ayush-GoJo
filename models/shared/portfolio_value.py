"""
portfolio_value.py

Computes the normalized portfolio terminal value from the prices
of Bond A (2-year) and Bond B (10-year) along one simulated path.

The portfolio is initialized at V_0 = 100.0 with equal weights
of 0.5 in each bond. Terminal values below 100.0 represent a loss.

Imported by all four models: DBM, Vasicek, CIR, Hull-White.
"""


def portfolio_value(
    price_a: float,
    price_b: float,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> float:
    """
    Parameters
    ----------
    price_a : float
        Terminal price of Bond A (2-year) along this path.
    price_b : float
        Terminal price of Bond B (10-year) along this path.
    weight_a : float
        Portfolio weight for Bond A. Default 0.5.
    weight_b : float
        Portfolio weight for Bond B. Default 0.5.

    Returns
    -------
    float
        Weighted portfolio terminal value V_T.
        V_T < 100.0 indicates a loss relative to the initial investment.
    """
    return weight_a * price_a + weight_b * price_b
