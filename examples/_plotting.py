"""Helpers shared by the example scripts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


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
