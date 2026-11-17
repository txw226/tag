"""Pieces shared by the Kalman filter variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class FilterResult:
    """Output of a filter's ``filter`` method for N steps and n states."""

    x_prior: NDArray  # (N, n) predicted states
    P_prior: NDArray  # (N, n, n) predicted covariances
    x_post: NDArray  # (N, n) updated states
    P_post: NDArray  # (N, n, n) updated covariances


def _as_matrix(name: str, value: ArrayLike, shape: tuple[int, int]) -> NDArray:
    arr = np.atleast_2d(np.array(value, dtype=float))
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def joseph_update(
    x: NDArray, P: NDArray, y: NDArray, H: NDArray, R: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Kalman measurement update given innovation ``y`` and measurement Jacobian ``H``.

    Returns the updated ``(x, P)`` and the innovation covariance ``S`` and gain ``K``.
    """
    S = H @ P @ H.T + R
    # K = P H^T S^-1, computed as a solve against symmetric S.
    K = np.linalg.solve(S, H @ P).T

    x = x + K @ y
    # Joseph form: stays symmetric positive semi-definite despite round-off.
    IKH = np.eye(len(x)) - K @ H
    P = IKH @ P @ IKH.T + K @ R @ K.T
    return x, 0.5 * (P + P.T), S, K


class KalmanFilterBase:
    """Predict/update driver loop. Subclasses set ``x``, ``P``, ``n`` and define
    ``predict(u)`` and ``update(z)``."""

    x: NDArray
    P: NDArray
    n: int

    def predict(self, u: ArrayLike | None = None) -> None:
        raise NotImplementedError

    def update(self, z: ArrayLike | None) -> None:
        raise NotImplementedError

    def step(self, z: ArrayLike | None, u: ArrayLike | None = None) -> None:
        """One predict + update cycle."""
        self.predict(u)
        self.update(z)

    def filter(self, zs: ArrayLike, us: ArrayLike | None = None) -> FilterResult:
        """Run predict/update over a sequence of measurements.

        Args:
            zs: (N, m) measurements; rows may contain NaN to mark a dropout.
            us: optional (N, p) control inputs, or a single (p,) input used every step.
        """
        zs = np.asarray(zs, dtype=float)
        if zs.ndim == 1:
            zs = zs.reshape(-1, 1)
        N = zs.shape[0]

        if us is not None:
            us = np.asarray(us, dtype=float)
            if us.ndim <= 1:
                us = np.broadcast_to(us.reshape(1, -1), (N, us.size))

        n = self.n
        out = FilterResult(
            x_prior=np.empty((N, n)),
            P_prior=np.empty((N, n, n)),
            x_post=np.empty((N, n)),
            P_post=np.empty((N, n, n)),
        )
        for k in range(N):
            self.predict(None if us is None else us[k])
            out.x_prior[k], out.P_prior[k] = self.x, self.P
            z = zs[k]
            self.update(None if np.isnan(z).any() else z)
            out.x_post[k], out.P_post[k] = self.x, self.P
        return out
