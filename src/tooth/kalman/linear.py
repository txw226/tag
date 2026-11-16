"""Linear Kalman filter.

Model:
    x_k = F x_{k-1} + B u_k + w_k,   w_k ~ N(0, Q)
    z_k = H x_k + v_k,               v_k ~ N(0, R)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class FilterResult:
    """Output of :meth:`LinearKalmanFilter.filter` for N steps and n states."""

    x_prior: NDArray  # (N, n) predicted states
    P_prior: NDArray  # (N, n, n) predicted covariances
    x_post: NDArray  # (N, n) updated states
    P_post: NDArray  # (N, n, n) updated covariances


def _as_matrix(name: str, value: ArrayLike, shape: tuple[int, int]) -> NDArray:
    arr = np.atleast_2d(np.asarray(value, dtype=float))
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


class LinearKalmanFilter:
    """Discrete-time linear Kalman filter with fixed model matrices.

    Args:
        F: (n, n) state transition matrix.
        H: (m, n) measurement matrix.
        Q: (n, n) process noise covariance.
        R: (m, m) measurement noise covariance.
        x0: (n,) initial state estimate.
        P0: (n, n) initial state covariance.
        B: optional (n, p) control input matrix.
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
    ) -> None:
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            raise ValueError(f"F must be square (n, n), got {F.shape}")
        n = F.shape[0]

        H = np.atleast_2d(np.asarray(H, dtype=float))
        if H.ndim != 2 or H.shape[1] != n:
            raise ValueError(f"H must have shape (m, {n}), got {H.shape}")
        m = H.shape[0]

        self.F = F
        self.H = H
        self.Q = _as_matrix("Q", Q, (n, n))
        self.R = _as_matrix("R", R, (m, m))
        self.P = _as_matrix("P0", P0, (n, n))

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {x0.shape}")
        self.x = x0

        if B is None:
            self.B = None
        else:
            B = np.asarray(B, dtype=float)
            if B.ndim == 1:
                B = B.reshape(-1, 1)
            if B.ndim != 2 or B.shape[0] != n:
                raise ValueError(f"B must have shape ({n}, p), got {B.shape}")
            self.B = B

        self.n, self.m = n, m
        self._I = np.eye(n)

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

    def update(self, z: ArrayLike | None) -> None:
        """Correct the state with measurement z. ``None`` skips the update."""
        if z is None:
            return
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (self.m,):
            raise ValueError(f"z must have shape ({self.m},), got {z.shape}")

        H, P, R = self.H, self.P, self.R
        y = z - H @ self.x
        S = H @ P @ H.T + R
        # K = P H^T S^-1, computed as a solve against symmetric S.
        K = np.linalg.solve(S, H @ P).T

        self.x = self.x + K @ y
        # Joseph form: stays symmetric positive semi-definite despite round-off.
        IKH = self._I - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)

        self.y, self.S, self.K = y, S, K

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
