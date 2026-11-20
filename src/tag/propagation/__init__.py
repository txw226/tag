"""Orbit propagation, backed by a compiled C++ core.

Units are whatever the caller uses consistently with ``mu``; the default ``EARTH_MU`` is in km^3/s^2,
so states are then in km and km/s and times in seconds.
"""

from tag.propagation._propagation import EARTH_MU, HyperbolicOrbitError, eccentric_anomaly

__all__ = ["EARTH_MU", "HyperbolicOrbitError", "eccentric_anomaly"]
