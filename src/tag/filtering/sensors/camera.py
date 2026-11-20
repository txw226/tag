"""Pinhole camera in the 2D plane: a 1D image coordinate per target."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tag.filtering.sensors.base import Sensor


class PinholeCamera(Sensor):
    """Camera at ``position`` with optical axis at angle ``yaw`` (radians from +x).

    The target's offset ``d = p - position`` splits into depth along the axis
    and lateral offset across it (positive to the left of the axis, i.e.
    counter-clockwise). The measurement is the image coordinate

        u = focal_px * lateral / depth     (pixels; 0 on the axis)

    A single camera observes only the direction to the target, not its range;
    two cameras at different positions (stereo) triangulate it.

    Args:
        position: (2,) camera position (m).
        yaw: optical-axis angle (rad).
        focal_px: focal length in pixels.
        pixel_std: image-coordinate noise (px).
        width_px: image width; targets with |u| > width_px / 2 are not visible.
    """

    def __init__(
        self,
        position: ArrayLike,
        yaw: float,
        focal_px: float,
        pixel_std: float,
        width_px: float,
        every: int = 1,
        name: str = "camera",
    ) -> None:
        super().__init__([pixel_std], every, name)
        self.position = np.array(position, dtype=float).reshape(2)
        self.yaw = float(yaw)
        self.focal_px = float(focal_px)
        self.width_px = float(width_px)
        self.axis = np.array([np.cos(yaw), np.sin(yaw)])
        self.normal = np.array([-np.sin(yaw), np.cos(yaw)])

    @property
    def half_fov(self) -> float:
        """Half the horizontal field of view (rad)."""
        return float(np.arctan(0.5 * self.width_px / self.focal_px))

    def _depth_lateral(self, x: NDArray) -> tuple[float, float]:
        d = np.asarray(x[:2], dtype=float) - self.position
        return float(self.axis @ d), float(self.normal @ d)

    def h(self, x: NDArray) -> NDArray:
        depth, lateral = self._depth_lateral(x)
        return np.array([self.focal_px * lateral / depth])

    def H_jac(self, x: NDArray) -> NDArray:
        depth, lateral = self._depth_lateral(x)
        J = self.focal_px * (self.normal * depth - lateral * self.axis) / depth**2
        return self._pad(J.reshape(1, 2), x)

    def visible(self, x: NDArray) -> bool:
        depth, lateral = self._depth_lateral(x)
        return depth > 0 and abs(self.focal_px * lateral / depth) <= 0.5 * self.width_px
