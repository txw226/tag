"""Linear Kalman filter.

Model:
    x_k = F x_{k-1} + B u_k + w_k,   w_k ~ N(0, Q)
    z_k = H x_k + v_k,               v_k ~ N(0, R)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tag.filtering.kalman._base import (
    AcceptPosterior,
    FilterResult,
    KalmanFilterBase,
    UpdateOutcome,
    _as_matrix,
    joseph_update,
)

__all__ = ["FilterResult", "LinearKalmanFilter"]


class LinearKalmanFilter(KalmanFilterBase):
    """Discrete-time linear Kalman filter with fixed model matrices.

    Args:
        F: (n, n) state transition matrix.
        H: (m, n) measurement matrix.
        Q: (n, n) process noise covariance.
        R: (m, m) measurement noise covariance.
        x0: (n,) initial state estimate.
        P0: (n, n) initial state covariance.
        B: optional (n, p) control input matrix.
        accept_posterior: optional ``accept_posterior(x, P) -> bool``; False makes the update a
            rejected update (the prior is kept). See :class:`KalmanFilterBase`.
        max_consecutive_unabsorbed: optional lock-out limit; ``None`` (default) only reports.
    """

    def __init__(
        self,
        F: ArrayLike,
        H: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
        B: ArrayLike | None = None,
        accept_posterior: AcceptPosterior | None = None,
        max_consecutive_unabsorbed: int | None = None,
    ) -> None:
        F = np.atleast_2d(np.array(F, dtype=float))
        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            raise ValueError(f"F must be square (n, n), got {F.shape}")
        n = F.shape[0]

        H = np.atleast_2d(np.array(H, dtype=float))
        if H.ndim != 2 or H.shape[1] != n:
            raise ValueError(f"H must have shape (m, {n}), got {H.shape}")
        m = H.shape[0]

        self.F = F
        self.H = H
        self.Q = _as_matrix("Q", Q, (n, n))
        self.R = _as_matrix("R", R, (m, m))
        self.P = _as_matrix("P0", P0, (n, n))

        x0 = np.array(x0, dtype=float).reshape(-1)
        if x0.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {x0.shape}")
        self.x = x0

        if B is None:
            self.B = None
        else:
            B = np.array(B, dtype=float)
            if B.ndim == 1:
                B = B.reshape(-1, 1)
            if B.ndim != 2 or B.shape[0] != n:
                raise ValueError(f"B must have shape ({n}, p), got {B.shape}")
            self.B = B

        self.n, self.m = n, m
        self._init_acceptance(accept_posterior, max_consecutive_unabsorbed)

        # Diagnostics from the most recent update.
        self.y: NDArray | None = None  # innovation
        self.S: NDArray | None = None  # innovation covariance
        self.K: NDArray | None = None  # Kalman gain

    def predict(self, u: ArrayLike | None = None) -> None:
        """Propagate the state and covariance one step forward."""
        self.x = self.F @ self.x
        if u is not None:
            if self.B is None:
                raise ValueError("control input u given but filter has no B matrix")
            u = np.asarray(u, dtype=float).reshape(-1)
            if u.shape != (self.B.shape[1],):
                raise ValueError(f"u must have shape ({self.B.shape[1]},), got {u.shape}")
            self.x = self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: ArrayLike | None) -> UpdateOutcome:
        """Correct the state with measurement z. ``None`` skips the update.

        The diagnostics ``y``, ``S`` and ``K`` are kept even for a rejected update.
        """
        if z is None:
            return UpdateOutcome.SKIPPED
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (self.m,):
            raise ValueError(f"z must have shape ({self.m},), got {z.shape}")

        y = z - self.H @ self.x
        x, P, self.S, self.K = joseph_update(self.x, self.P, y, self.H, self.R)
        self.y = y
        return self._accept(x, P)
