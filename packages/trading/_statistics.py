"""
packages/trading/_statistics.py
---------------------------------
Shared statistical primitives used by :mod:`trading.metrics` and
:mod:`trading.walk_forward`.

Motivation
----------
Both modules need the inverse standard normal CDF (Acklam approximation) and
the forward standard normal CDF (via ``math.erf``).  Defining these in a
dedicated private module eliminates duplication and prevents circular imports:

    _statistics.py  ← no trading imports
        ↑
    walk_forward.py  ← imports _inv_norm_cdf for DSR
    metrics.py       ← imports _norm_cdf for PSR

The ``_`` prefix signals these are package-internal helpers; external
consumers should use the public functions in ``metrics.py`` and
``walk_forward.py``.

Currently exports
-----------------
- ``_inv_norm_cdf``: Acklam rational approximation of Φ⁻¹(p),
  max absolute error ~1.15e-9.
- ``_norm_cdf``: Standard normal CDF via ``math.erf`` (exact to float64).
"""

from __future__ import annotations

import math

__all__ = ["_inv_norm_cdf", "_norm_cdf"]


def _inv_norm_cdf(p: float) -> float:
    """Inverse of the standard normal CDF for ``0 < p < 1``.

    Uses the Acklam rational approximation.  Max absolute error ~1.15e-9 —
    well below the precision required for Sharpe haircut calculations.

    Parameters
    ----------
    p:
        Probability in the open unit interval ``(0, 1)``.

    Returns
    -------
    float
        The ``z`` value such that ``Φ(z) = p``.

    Raises
    ------
    ValueError
        If ``p`` is outside the open unit interval.

    References
    ----------
    P. J. Acklam, "An algorithm for computing the inverse normal cumulative
    distribution function".  Numerical Recipes companion.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Acklam coefficients
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    # Upper tail
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via ``math.erf``.

    Exact to float64 precision for all finite inputs.  Used by
    :func:`trading.metrics.compute_psr` to convert the standardised
    test statistic to a probability.

    Parameters
    ----------
    z:
        The standardised value (z-score).

    Returns
    -------
    float
        Probability Φ(z) in [0, 1].
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
