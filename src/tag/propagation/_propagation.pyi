"""Compiled core of tag.propagation."""

from typing import Literal

import numpy
import numpy.typing

class HyperbolicOrbitError(ValueError): ...

EARTH_MU: float = 398600.4418

KEP_A: int = 0

KEP_E: int = 1

KEP_I: int = 2

KEP_RAAN: int = 3

KEP_ARGP: int = 4

KEP_ANOMALY: int = 5

EQ_A: int = 0

EQ_H: int = 1

EQ_K: int = 2

EQ_P: int = 3

EQ_Q: int = 4

EQ_LAMBDA: int = 5

def cartesian_to_keplerian(
    rv: numpy.typing.ArrayLike, mu: float = EARTH_MU, anomaly: Literal["mean", "true", "eccentric"] = "mean"
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Convert a Cartesian state [r, v] to Keplerian elements [a, e, i, raan, argp, anomaly].

    Angles are in radians, in [0, 2pi) (i in [0, pi]); index with the KEP_* constants. anomaly selects
    the last element: 'mean', 'true' or 'eccentric'.

    Singular cases (e or sin(i) below 1e-11) follow the usual conventions:
    - circular: argp = 0 and the anomaly is measured from the ascending node;
    - equatorial: raan = 0 and argp is measured from the x axis;
    - circular and equatorial: raan = argp = 0 and the anomaly is the true longitude.
    Near these cases raan, argp and the anomaly are ill-conditioned individually; use
    cartesian_to_equinoctial instead.

    Raises HyperbolicOrbitError if e >= 1, ValueError for invalid input.
    """

def keplerian_to_cartesian(
    elements: numpy.typing.ArrayLike, mu: float = EARTH_MU, anomaly: Literal["mean", "true", "eccentric"] = "mean"
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Convert Keplerian elements [a, e, i, raan, argp, anomaly] to a Cartesian state [r, v].

    Angles are in radians (any value); index with the KEP_* constants. anomaly says what the last
    element is: 'mean', 'true' or 'eccentric'. Raises HyperbolicOrbitError if e >= 1, ValueError for
    invalid input.
    """

def eccentric_anomaly(mean_anomaly: float, e: float) -> float:
    """
    Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (radians).

    E is in the same revolution as M. Raises ValueError unless 0 <= e < 1 and M is finite.
    """
