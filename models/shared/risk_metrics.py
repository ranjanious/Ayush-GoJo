"""
risk_metrics.py

Computes the full set of portfolio risk metrics from the empirical
distribution of terminal portfolio values across all simulated paths.

All metrics are computed from the raw terminal_values array.
The caller is responsible for passing values on the correct scale
(face value 100.0, initial portfolio value 100.0).

Imported by all four models: DBM, Vasicek, CIR, Hull-White.
"""

import numpy as np
from scipy import stats


def risk_metrics(
    terminal_values: np.ndarray,
    alpha: float = 0.05,
    initial_value: float = 100.0,
) -> dict:
    """
    Parameters
    ----------
    terminal_values : np.ndarray, shape (N_paths,)
        Terminal portfolio value on each simulated path.
    alpha : float
        Tail probability for VaR and ES.
        0.05 gives the 95% confidence level (default).
        0.01 gives the 99% confidence level.
    initial_value : float
        Portfolio value at t=0. Default 100.0.
        Used to compute prob_loss.

    Returns
    -------
    dict with keys:
        var        : Value at Risk (alpha-quantile of distribution)
        es         : Expected Shortfall (mean of values <= VaR)
        prob_loss  : Fraction of paths where V_T < initial_value
        std_error  : Simulation standard error for the VaR estimate
        mean       : Mean of the distribution
        std        : Standard deviation (ddof=1)
        skewness   : Skewness (0 for a symmetric distribution)
        kurtosis   : Excess kurtosis (0 for a normal distribution)
        n_paths    : Number of paths used
    """
    n = len(terminal_values)

    var = float(np.quantile(terminal_values, alpha))

    tail = terminal_values[terminal_values <= var]
    es   = float(np.mean(tail)) if len(tail) > 0 else var

    prob_loss = float(np.mean(terminal_values < initial_value))

    # Simulation standard error for the VaR quantile estimate.
    # SE(VaR) = sqrt(alpha * (1 - alpha) / n) / f_hat
    # where f_hat is the empirical PDF at VaR estimated with a
    # Gaussian kernel using Silverman's bandwidth rule.
    bandwidth = 1.06 * float(np.std(terminal_values)) * (n ** (-0.2))
    f_hat = float(
        np.mean(np.exp(-0.5 * ((terminal_values - var) / bandwidth) ** 2))
        / (bandwidth * np.sqrt(2.0 * np.pi))
    )
    std_error = (
        float(np.sqrt(alpha * (1.0 - alpha) / n) / f_hat)
        if f_hat > 0 else float('nan')
    )

    return {
        'var':       var,
        'es':        es,
        'prob_loss': prob_loss,
        'std_error': std_error,
        'mean':      float(np.mean(terminal_values)),
        'std':       float(np.std(terminal_values, ddof=1)),
        'skewness':  float(stats.skew(terminal_values)),
        'kurtosis':  float(stats.kurtosis(terminal_values)),
        'n_paths':   n,
    }
