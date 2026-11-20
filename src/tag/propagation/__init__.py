"""Orbit propagation, backed by a compiled C++ core.

Units are whatever the caller uses consistently with ``mu``; the default ``EARTH_MU`` is in km^3/s^2,
so states are then in km and km/s and times in seconds.
"""

from typing import Literal

from tag.propagation._propagation import (
    EARTH_MU,
    EQ_A,
    EQ_H,
    EQ_K,
    EQ_LAMBDA,
    EQ_P,
    EQ_Q,
    KEP_A,
    KEP_ANOMALY,
    KEP_ARGP,
    KEP_E,
    KEP_I,
    KEP_RAAN,
    HyperbolicOrbitError,
    cartesian_to_equinoctial,
    cartesian_to_keplerian,
    eccentric_anomaly,
    equinoctial_to_cartesian,
    keplerian_to_cartesian,
)

# The anomaly kinds the Keplerian converters accept.
type AnomalyKind = Literal["mean", "true", "eccentric"]

__all__ = [
    "EARTH_MU",
    "EQ_A",
    "EQ_H",
    "EQ_K",
    "EQ_LAMBDA",
    "EQ_P",
    "EQ_Q",
    "KEP_A",
    "KEP_ANOMALY",
    "KEP_ARGP",
    "KEP_E",
    "KEP_I",
    "KEP_RAAN",
    "AnomalyKind",
    "HyperbolicOrbitError",
    "cartesian_to_equinoctial",
    "cartesian_to_keplerian",
    "eccentric_anomaly",
    "equinoctial_to_cartesian",
    "keplerian_to_cartesian",
]
