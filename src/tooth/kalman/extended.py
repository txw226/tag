"""Extended Kalman filter.

Model:
    x_k = f(x_{k-1}, u_k) + w_k,   w_k ~ N(0, Q)
    z_k = h(x_k) + v_k,            v_k ~ N(0, R)

The filter linearizes f around the previous posterior and h around the
prior. Jacobians come from user-supplied callables when given, otherwise
from central finite differences.

Several sensors can be fused: pass a measurement model (anything with ``h``,
``H_jac``, ``R`` and ``residual``, such as a :class:`tooth.sensors.Sensor`) to
``update(z, sensor)``, or run :meth:`ExtendedKalmanFilter.filter_sensors`
over one measurement stream per sensor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tooth.kalman._base import (
    FilterResult,
    KalmanFilterBase,
    _as_matrix,
    _controls,
    joseph_update,
)

TransitionFn = Callable[[NDArray, NDArray | None], ArrayLike]
MeasurementFn = Callable[[NDArray], ArrayLike]
ResidualFn = Callable[[NDArray, NDArray], ArrayLike]


class MeasurementModel(Protocol):
    """What ``update(z, sensor)`` needs from a sensor."""

    R: NDArray

    def h(self, x: NDArray) -> ArrayLike: ...
    def H_jac(self, x: NDArray) -> ArrayLike: ...
    def residual(self, z: NDArray, z_pred: NDArray) -> ArrayLike: ...


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
        h: default measurement ``h(x) -> z_pred``. Optional: without it, every
            update needs a ``sensor``.
        Q: (n, n) process noise covariance.
        R: (m, m) measurement noise covariance for ``h``; required with ``h``.
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
        h: MeasurementFn | None,
        Q: ArrayLike,
        R: ArrayLike | None,
        x0: ArrayLike,
        P0: ArrayLike,
        F_jac: Callable[[NDArray, NDArray | None], ArrayLike] | None = None,
        H_jac: Callable[[NDArray], ArrayLike] | None = None,
        residual: ResidualFn | None = None,
        eps: float = 1e-6,
    ) -> None:
        x0 = np.array(x0, dtype=float).reshape(-1)
        n = x0.size
        if (h is None) != (R is None):
            raise ValueError("h and R must be given together (or both omitted to use sensors)")
        if R is not None:
            R = np.atleast_2d(np.array(R, dtype=float))
            if R.ndim != 2 or R.shape[0] != R.shape[1]:
                raise ValueError(f"R must be square (m, m), got {R.shape}")

        self.f, self.h = f, h
        self.F_jac, self.H_jac = F_jac, H_jac
        self.residual = residual
        self.eps = eps
        self.Q = _as_matrix("Q", Q, (n, n))
        self.R = R
        self.P = _as_matrix("P0", P0, (n, n))
        self.x = x0
        self.n = n
        self.m = None if R is None else R.shape[0]

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

    def update(self, z: ArrayLike | None, sensor: MeasurementModel | None = None) -> None:
        """Correct the state with measurement z. ``None`` skips the update.

        With ``sensor``, its ``h``, ``H_jac``, ``R`` and ``residual`` replace the
        filter's defaults for this update.
        """
        if z is None:
            return
        h: MeasurementFn
        H_jac: MeasurementFn | None
        residual: ResidualFn | None
        R: NDArray
        if sensor is not None:
            h, H_jac, residual = (
                sensor.h,
                getattr(sensor, "H_jac", None),
                getattr(sensor, "residual", None),
            )
            R = np.atleast_2d(np.asarray(sensor.R, dtype=float))
        elif self.h is not None and self.R is not None:  # __init__ guarantees both or neither
            h, H_jac, residual, R = self.h, self.H_jac, self.residual, self.R
        else:
            raise ValueError("filter has no default h; pass a sensor to update()")

        n, m = self.n, R.shape[0]
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (m,):
            raise ValueError(f"z must have shape ({m},), got {z.shape}")

        x = self.x
        z_pred = self._checked("h", h(x), (m,))
        if H_jac is not None:
            H = self._checked("H_jac", H_jac(x), (m, n))
        else:
            H = numerical_jacobian(h, x, self.eps)

        if residual is None:
            y = z - z_pred
        else:
            y = self._checked("residual", residual(z, z_pred), (m,))

        self.x, self.P, self.S, self.K = joseph_update(x, self.P, y, H, R)
        self.y, self.H = y, H

    def filter_sensors(
        self,
        streams: Sequence[tuple[MeasurementModel, ArrayLike]],
        us: ArrayLike | None = None,
    ) -> FilterResult:
        """Fuse several measurement streams.

        Each step predicts once, then updates with every stream whose row is
        not NaN, in the order given. Sequential updates are exact when the
        sensors' noises are independent.

        Args:
            streams: ``(sensor, zs)`` pairs, with ``zs`` of shape (N, m_sensor)
                and the same N for every stream; NaN rows mean no measurement.
            us: optional (N, p) control inputs, or a single (p,) input used every step.
        """
        if not streams:
            raise ValueError("filter_sensors needs at least one (sensor, zs) stream")
        prepared = []
        for sensor, zs in streams:
            zs = np.asarray(zs, dtype=float)
            if zs.ndim == 1:
                zs = zs.reshape(-1, 1)
            prepared.append((sensor, zs))
        N = prepared[0][1].shape[0]
        for sensor, zs in prepared:
            if zs.shape[0] != N:
                raise ValueError(f"all streams need {N} rows; {sensor!r} has {zs.shape[0]}")
        us = _controls(us, N)

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
            for sensor, zs in prepared:
                if not np.isnan(zs[k]).any():
                    self.update(zs[k], sensor)
            out.x_post[k], out.P_post[k] = self.x, self.P
        return out
