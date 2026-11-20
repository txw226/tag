"""Measurement models shared by the simulations and the filters.

Conventions:
    * Position is at state indices 0 and 1 (``x[:2] = [x, y]``); any further
      state entries are ignored, and Jacobians are zero-padded to ``len(x)``.
    * ``measure_series`` returns NaN rows for steps without a measurement
      (off-rate or not visible); the filters skip those rows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Sensor(ABC):
    """A sensor with additive Gaussian noise of standard deviation ``noise_std``.

    Args:
        noise_std: (m,) per-component noise standard deviations; ``R = diag(noise_std**2)``.
        every: measure on every ``every``-th step only (multi-rate streams).
        name: label for plots and messages.
    """

    def __init__(self, noise_std: ArrayLike, every: int = 1, name: str | None = None) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.noise_std = np.array(noise_std, dtype=float).reshape(-1)
        self.R = np.diag(self.noise_std**2)
        self.m = self.noise_std.size
        self.every = every
        self.name = name or type(self).__name__

    @abstractmethod
    def h(self, x: NDArray) -> NDArray:
        """Noise-free measurement of state ``x``."""

    @abstractmethod
    def H_jac(self, x: NDArray) -> NDArray:
        """(m, len(x)) Jacobian of :meth:`h`."""

    def residual(self, z: NDArray, z_pred: NDArray) -> NDArray:
        return np.asarray(z, dtype=float) - z_pred

    def visible(self, x: NDArray) -> bool:
        """Whether the sensor can measure state ``x`` at all."""
        return True

    def wrap(self, z: NDArray) -> NDArray:
        """Normalize measurement(s) after adding noise, e.g. wrap angles. Acts on the last axis."""
        return z

    def measure_series(self, truth: ArrayLike, rng: np.random.Generator) -> NDArray:
        """Noisy measurements of every row of ``truth``; NaN where there is none.

        Noise is drawn for all rows, so the RNG stream does not depend on
        which rows are masked.
        """
        truth = np.atleast_2d(np.asarray(truth, dtype=float))
        clean = np.array([self.h(x) for x in truth])
        z = self.wrap(clean + rng.normal(0.0, 1.0, size=clean.shape) * self.noise_std)
        keep = np.array([k % self.every == 0 and self.visible(x) for k, x in enumerate(truth)])
        z[~keep] = np.nan
        return z

    def _pad(self, J: NDArray, x: NDArray) -> NDArray:
        """Embed a Jacobian w.r.t. position (m, 2) into one w.r.t. the full state."""
        H = np.zeros((self.m, len(x)))
        H[:, :2] = J
        return H

    def __repr__(self) -> str:
        return f"{self.name}(noise_std={self.noise_std.tolist()}, every={self.every})"


def wrap_angle(a: NDArray | float) -> NDArray | float:
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - a, 2 * np.pi)
