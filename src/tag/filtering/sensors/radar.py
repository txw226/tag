"""Ground radar measuring range and bearing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tag.filtering.sensors.base import Sensor, wrap_angle


class RangeBearingRadar(Sensor):
    """Radar at ``position`` measuring z = [range, bearing], bearing = atan2(dy, dx) in radians."""

    def __init__(
        self,
        position: ArrayLike,
        range_std: float,
        bearing_std: float,
        every: int = 1,
        name: str = "radar",
    ) -> None:
        super().__init__([range_std, bearing_std], every, name)
        self.position = np.array(position, dtype=float).reshape(2)

    def h(self, x: NDArray) -> NDArray:
        dx, dy = x[0] - self.position[0], x[1] - self.position[1]
        return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])

    def H_jac(self, x: NDArray) -> NDArray:
        dx, dy = x[0] - self.position[0], x[1] - self.position[1]
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2)
        return self._pad(np.array([[dx / r, dy / r], [-dy / r2, dx / r2]]), x)

    def residual(self, z: NDArray, z_pred: NDArray) -> NDArray:
        """Residual with the bearing difference wrapped to (-pi, pi]."""
        y = np.asarray(z, dtype=float) - z_pred
        y[1] = wrap_angle(y[1])
        return y

    def wrap(self, z: NDArray) -> NDArray:
        z = np.array(z, dtype=float)
        z[..., 1] = wrap_angle(z[..., 1])
        return z

    def to_position(self, z: ArrayLike) -> NDArray:
        """Convert [range, bearing] measurement(s) to Cartesian position(s)."""
        z = np.asarray(z, dtype=float)
        r, th = z[..., 0], z[..., 1]
        p = self.position
        return np.stack([p[0] + r * np.cos(th), p[1] + r * np.sin(th)], axis=-1)

    def position_cov(self, z: ArrayLike) -> NDArray:
        """Covariance J R J^T of :meth:`to_position` at one measurement (linearized)."""
        r, th = np.asarray(z, dtype=float)
        J = np.array([[np.cos(th), -r * np.sin(th)], [np.sin(th), r * np.cos(th)]])
        return J @ self.R @ J.T
