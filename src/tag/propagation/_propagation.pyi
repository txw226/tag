"""Compiled core of tag.propagation."""

from typing import Literal

import numpy
import numpy.typing

class HyperbolicOrbitError(ValueError): ...

EARTH_MU: float = 398600.4418

EARTH_RE: float = 6378.137

EARTH_J2: float = 0.00108262668

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

def cartesian_to_equinoctial(rv: numpy.typing.ArrayLike, mu: float = EARTH_MU) -> numpy.typing.NDArray[numpy.float64]:
    """
    Convert a Cartesian state [r, v] to equinoctial elements [a, h, k, p, q, lambda].

    Direct set: h = e sin(argp + raan), k = e cos(argp + raan), p = tan(i/2) sin(raan),
    q = tan(i/2) cos(raan), lambda = M + argp + raan (mean longitude, radians, in [0, 2pi)); index with
    the EQ_* constants. Well defined for circular and equatorial orbits; singular only for retrograde
    equatorial orbits (i = 180 deg), which raise ValueError. Raises HyperbolicOrbitError if e >= 1, ValueError
    for invalid input.
    """

def equinoctial_to_cartesian(
    elements: numpy.typing.ArrayLike, mu: float = EARTH_MU
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Convert equinoctial elements [a, h, k, p, q, lambda] to a Cartesian state [r, v].

    See cartesian_to_equinoctial for the definitions; index with the EQ_* constants. Raises
    HyperbolicOrbitError if e = sqrt(h^2 + k^2) >= 1, ValueError for invalid input.
    """

def osculating_to_mean(
    rv: numpy.typing.ArrayLike, mu: float = EARTH_MU, re: float = EARTH_RE, j2: float = EARTH_J2
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Convert an osculating Cartesian state [r, v] to J2 mean equinoctial elements [a, h, k, p, q, lambda].

    Mean elements are Brouwer-Lyddane, first order in J2: the osculating elements less the short-period
    terms (and Brouwer's long-period terms), in Lyddane's form, so circular and equatorial orbits are
    fine. Index the result with the EQ_* constants (see cartesian_to_equinoctial); mean lambda is in
    [0, 2pi). There is no closed form: the inverse of mean_to_osculating is iterated to a fixed point,
    usually in 3-5 iterations, so that mean_to_osculating gives rv back to well below a millimetre.

    re is the equatorial radius (units of rv) and j2 the unnormalized zonal harmonic; defaults are Earth's.
    The theory is singular at the critical inclination (63.43 deg, or 116.57 deg retrograde), which
    Molniya orbits sit at: nearby, the corrections are large and unreliable, and when the iteration
    fails to converge it raises ValueError naming the inclination rather than return garbage.
    Raises HyperbolicOrbitError if e >= 1, ValueError for invalid input or a retrograde equatorial orbit.
    """

def mean_to_osculating(
    elements: numpy.typing.ArrayLike, mu: float = EARTH_MU, re: float = EARTH_RE, j2: float = EARTH_J2
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Convert J2 mean equinoctial elements [a, h, k, p, q, lambda] to an osculating Cartesian state [r, v].

    The inverse of osculating_to_mean, in closed form: the mean elements plus the first-order
    Brouwer-Lyddane short-period and long-period terms. Accuracy is first order: errors are O(J2^2).
    The theory is singular at the critical inclination (63.43 deg, or 116.57 deg retrograde): nearby,
    the corrections are large and unreliable; exactly there they are infinite, which raises ValueError.
    Raises HyperbolicOrbitError if the mean or osculating e >= 1, ValueError for invalid input.
    """

def eccentric_anomaly(mean_anomaly: float, e: float) -> float:
    """
    Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (radians).

    E is in the same revolution as M. Raises ValueError unless 0 <= e < 1 and M is finite.
    """

def propagate_many(
    rvs: numpy.typing.ArrayLike,
    dts: numpy.typing.ArrayLike,
    mu: float = EARTH_MU,
    out: numpy.typing.NDArray[numpy.float64] | None = None,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Propagate many orbits to many times: shape (M, N, 6) for rvs (M, 6) and dts (N,).

    ``out[m, j]`` equals ``TwoBodyOrbit(rvs[m], mu).propagate(dts[j])`` exactly. Pass ``out`` (a
    writable, C-contiguous float64 array of shape (M, N, 6)) to fill it in place; it is returned.
    Every state and time is validated before anything is written (HyperbolicOrbitError or
    ValueError, naming the bad row or time), and the GIL is released while propagating.
    """

def _two_body_central_difference_stm(
    rv: numpy.typing.ArrayLike, dt: float, mu: float = EARTH_MU
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Internal test oracle: the two-body state transition matrix by central differences in C++.

    Not public API (not exported from tag.propagation); tests compare analytic state transition
    matrices against it. Entries are good to about 1e-10 of the largest at low e, 1e-7 at e = 0.9
    and only about 1e-5 at e = 0.99.
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

    def propagate_many(
        self, dts: numpy.typing.ArrayLike, out: numpy.typing.NDArray[numpy.float64] | None = None
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        The Cartesian states at each of ``dts`` (a 1-D array of times after the epoch), shape (N, 6).

        Row i equals ``propagate(dts[i])`` exactly. Pass ``out`` (a writable, C-contiguous float64 array
        of shape (N, 6)) to fill it in place; it is returned. Every dt is checked before anything is
        written, and the GIL is released while propagating.
        """

    def propagate_with_stm(
        self, dt: float
    ) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        The state ``dt`` seconds after the epoch, shape (6,), and the state transition matrix
        d x(t) / d x(epoch), shape (6, 6).

        The state equals ``propagate(dt)`` exactly. The matrix is in closed form and reuses its Kepler
        solution, costing about as much again as ``propagate``, which is unaffected.
        """

    def propagate_many_with_stm(
        self,
        dts: numpy.typing.ArrayLike,
        out: numpy.typing.NDArray[numpy.float64] | None = None,
        stm_out: numpy.typing.NDArray[numpy.float64] | None = None,
    ) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        The states at each of ``dts``, shape (N, 6), and their state transition matrices, shape (N, 6, 6).

        Row i equals ``propagate_with_stm(dts[i])`` exactly. ``out`` and ``stm_out`` (writable,
        C-contiguous float64 arrays of those shapes, not overlapping) are filled in place and returned
        when given. Every dt is checked before anything is written, and the GIL is released while
        propagating.
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
