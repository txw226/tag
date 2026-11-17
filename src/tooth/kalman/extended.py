"""Extended Kalman filter.

Model:
    x_k = f(x_{k-1}, u_k) + w_k,   w_k ~ N(0, Q)
    z_k = h(x_k) + v_k,            v_k ~ N(0, R)

The filter linearizes f around the previous posterior and h around the
prior. Jacobians come from user-supplied callables when given, otherwise
from central finite differences.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tooth.kalman._base import KalmanFilterBase, _as_matrix, joseph_update

TransitionFn = Callable[[NDArray, NDArray | None], ArrayLike]
MeasurementFn = Callable[[NDArray], ArrayLike]
ResidualFn = Callable[[NDArray, NDArray], ArrayLike]


def numerical_jacobian(fn: Callable[[NDArray], ArrayLike], x: ArrayLike, eps: float = 1e-6) -> NDArray:
    """Jacobian of ``fn`` at ``x`` by central differences.

    The step for component i is ``eps * max(1, |x_i|)``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    cols = []
    for i in range(x.size):
        step = eps * max(1.0, abs(x[i]))
        dx = np.zeros_like(x)
        dx[i] = step
        hi = np.asarray(fn(x + dx), dtype=float).reshape(-1)
        lo = np.asarray(fn(x - dx), dtype=float).reshape(-1)
        cols.append((hi - lo) / (2 * step))
    return np.column_stack(cols)


class ExtendedKalmanFilter(KalmanFilterBase):
    """Discrete-time extended Kalman filter.

    Args:
        f: transition ``f(x, u) -> x_next``; ``u`` is None when no control is given.
        h: measurement ``h(x) -> z_pred``.
        Q: (n, n) process noise covariance.
        R: (m, m) measurement noise covariance.
        x0: (n,) initial state estimate.
        P0: (n, n) initial state covariance.
        F_jac: optional ``F_jac(x, u) -> (n, n)`` Jacobian of f; numerical if omitted.
        H_jac: optional ``H_jac(x) -> (m, n)`` Jacobian of h; numerical if omitted.
        residual: optional ``residual(z, z_pred) -> y``, e.g. to wrap angles.
            Defaults to ``z - z_pred``.
        eps: relative finite-difference step for numerical Jacobians.
    """

    def __init__(
        self,
        f: TransitionFn,
        h: MeasurementFn,
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
        F_jac: Callable[[NDArray, NDArray | None], ArrayLike] | None = None,
        H_jac: Callable[[NDArray], ArrayLike] | None = None,
        residual: ResidualFn | None = None,
        eps: float = 1e-6,
    ) -> None:
        x0 = np.array(x0, dtype=float).reshape(-1)
        n = x0.size
        R = np.atleast_2d(np.array(R, dtype=float))
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be square (m, m), got {R.shape}")
        m = R.shape[0]

        self.f, self.h = f, h
        self.F_jac, self.H_jac = F_jac, H_jac
        self.residual = residual
        self.eps = eps
        self.Q = _as_matrix("Q", Q, (n, n))
        self.R = R
        self.P = _as_matrix("P0", P0, (n, n))
        self.x = x0
        self.n, self.m = n, m

        # Diagnostics from the most recent predict / update.
        self.F: NDArray | None = None  # transition Jacobian
        self.H: NDArray | None = None  # measurement Jacobian
        self.y: NDArray | None = None  # innovation
        self.S: NDArray | None = None  # innovation covariance
        self.K: NDArray | None = None  # Kalman gain

    @staticmethod
    def _checked(name: str, value: ArrayLike, shape: tuple[int, ...]) -> NDArray:
        arr = np.asarray(value, dtype=float)
        if len(shape) == 1:
            arr = arr.reshape(-1)
        if arr.shape != shape:
            raise ValueError(f"{name} must return shape {shape}, got {arr.shape}")
        return arr

    def predict(self, u: ArrayLike | None = None) -> None:
        """Propagate the state through f and the covariance through its Jacobian."""
        n = self.n
        if u is not None:
            u = np.asarray(u, dtype=float).reshape(-1)
        x = self.x

        x_next = self._checked("f", self.f(x, u), (n,))
        if self.F_jac is not None:
            F = self._checked("F_jac", self.F_jac(x, u), (n, n))
        else:
            F = numerical_jacobian(lambda s: self.f(s, u), x, self.eps)

        self.x = x_next
        self.P = F @ self.P @ F.T + self.Q
        self.F = F

    def update(self, z: ArrayLike | None) -> None:
        """Correct the state with measurement z. ``None`` skips the update."""
        if z is None:
            return
        n, m = self.n, self.m
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (m,):
            raise ValueError(f"z must have shape ({m},), got {z.shape}")

        x = self.x
        z_pred = self._checked("h", self.h(x), (m,))
        if self.H_jac is not None:
            H = self._checked("H_jac", self.H_jac(x), (m, n))
        else:
            H = numerical_jacobian(self.h, x, self.eps)

        if self.residual is None:
            y = z - z_pred
        else:
            y = self._checked("residual", self.residual(z, z_pred), (m,))

        self.x, self.P, self.S, self.K = joseph_update(x, self.P, y, H, self.R)
        self.y, self.H = y, H
