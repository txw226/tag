"""Helpers shared by the example scripts."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x): series for x < a + 1, else continued fraction."""
    if x <= 0.0:
        return 0.0
    log_pre = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        n = a
        while abs(term) > 1e-16 * abs(total):
            n += 1.0
            term *= x / n
            total += term
        return total * math.exp(log_pre)
    # Upper tail Q(a, x) by modified Lentz on the Legendre continued fraction.
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(log_pre) * h


def chi2_ppf(p: float, k: float) -> float:
    """Exact quantile of χ²(k): the x with CDF P(k/2, x/2) = p, found by bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    cdf = lambda x: _gammainc(0.5 * k, 0.5 * x)  # noqa: E731
    lo, hi = 0.0, max(1.0, k)
    while cdf(hi) < p:
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * hi:
            break
    return 0.5 * (lo + hi)


def chi2_bounds(dof: int, n_avg: int = 1, prob: float = 0.95) -> tuple[float, float]:
    """Two-sided ``prob`` bounds for the mean of ``n_avg`` independent χ²(dof) samples.

    That mean is χ²(dof·n_avg) / n_avg; the quantiles are exact (see :func:`chi2_ppf`).
    """
    k = dof * n_avg
    tail = 0.5 * (1.0 - prob)
    return chi2_ppf(tail, k) / n_avg, chi2_ppf(1.0 - tail, k) / n_avg


def band(fig, t, center, sigma, color, name, row, col):
    """Shaded ±2σ region around `center`."""
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([center + 2 * sigma, (center - 2 * sigma)[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line={"width": 0},
            hoverinfo="skip",
            name=name,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
