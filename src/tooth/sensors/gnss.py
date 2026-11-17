"""Onboard GNSS receiver giving position fixes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tooth.sensors.base import Sensor


class GnssPosition(Sensor):
    """Position fix z = [x, y] with independent noise ``std`` per axis.

    Linear (H = [I 0]) but usually slow: ``every=10`` gives 2 Hz at a 20 Hz
    filter rate. Blank rows of its series to NaN to simulate an outage.
    """

    def __init__(self, std: float, every: int = 10, name: str = "GNSS") -> None:
        super().__init__([std, std], every, name)

    def h(self, x: NDArray) -> NDArray:
        return np.array(x[:2], dtype=float)

    def H_jac(self, x: NDArray) -> NDArray:
        return self._pad(np.eye(2), x)
