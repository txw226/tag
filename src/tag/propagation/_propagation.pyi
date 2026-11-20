"""Compiled core of tag.propagation."""

import numpy
import numpy.typing

class HyperbolicOrbitError(ValueError): ...

EARTH_MU: float = 398600.4418

def eccentric_anomaly(mean_anomaly: float, e: float) -> float:
    """
    Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (radians).

    E is in the same revolution as M. Raises ValueError unless 0 <= e < 1 and M is finite.
    """

class TwoBodyOrbit:
    """
    Two-body propagation of an elliptic orbit from an osculating Cartesian state.

    ``rv = [x, y, z, vx, vy, vz]`` is the state at the epoch, in units consistent with ``mu``
    (default EARTH_MU: km, km/s, s). Near-circular, near-equatorial and GEO orbits are
    accurate. Raises HyperbolicOrbitError if e >= 1 (hyperbolic, parabolic or rectilinear),
    and ValueError for a non-finite state, zero position or mu <= 0.
    """

    def __init__(self, rv: numpy.typing.ArrayLike, mu: float = EARTH_MU) -> None: ...
    def propagate(self, dt: float) -> numpy.typing.NDArray[numpy.float64]:
        """
        The Cartesian state ``dt`` seconds (any sign) after the epoch, as a float64 array of shape (6,).

        Precision doesn't decay with long ``dt``: the mean anomaly is reduced modulo 2 pi.
        """

    @property
    def mu(self) -> float:
        """Gravitational parameter."""

    @property
    def semi_major_axis(self) -> float:
        """Semi-major axis a."""

    @property
    def eccentricity(self) -> float:
        """Eccentricity e, in [0, 1)."""

    @property
    def period(self) -> float:
        """Orbital period, 2 pi / n."""
