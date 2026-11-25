import math

import numpy as np
import pytest
from numpy.typing import NDArray

from tag.propagation import (
    EARTH_J2,
    EARTH_MU,
    EARTH_RE,
    EQ_A,
    EQ_H,
    EQ_K,
    EQ_LAMBDA,
    EQ_P,
    EQ_Q,
    HyperbolicOrbitError,
    cartesian_to_equinoctial,
    keplerian_to_cartesian,
    mean_to_osculating,
    osculating_to_mean,
)

DEG = math.pi / 180

# Osculating states as Keplerian [a, e, i, raan, argp, M] (km, rad), across regimes and the
# singular cases e = 0 and i = 0 of classical elements.
REGIMES = {
    "leo": [6878.0, 0.001, 51.6 * DEG, 30 * DEG, 40 * DEG, 50 * DEG],
    "leo_circular": [7000.0, 0.0, 98.0 * DEG, 10 * DEG, 0.0, 200 * DEG],
    "leo_equatorial": [7200.0, 0.01, 0.0, 0.0, 70 * DEG, 10 * DEG],
    "leo_circular_equatorial": [7100.0, 0.0, 0.0, 0.0, 0.0, 300 * DEG],
    "leo_eccentric": [9000.0, 0.2, 28.5 * DEG, 100 * DEG, 250 * DEG, 5 * DEG],
    "meo_gps": [26560.0, 0.01, 55.0 * DEG, 120 * DEG, 30 * DEG, 160 * DEG],
    "gto": [24400.0, 0.73, 7.0 * DEG, 50 * DEG, 180 * DEG, 30 * DEG],
    "geo": [42164.17, 1e-5, 0.01 * DEG, 75 * DEG, 10 * DEG, 20 * DEG],
    "geo_exact": [42164.17, 0.0, 0.0, 0.0, 0.0, 1.0],
    # A degree either side of the critical inclination still converges.
    "below_critical": [7500.0, 0.1, 62.43 * DEG, 20 * DEG, 30 * DEG, 40 * DEG],
    "above_critical": [7500.0, 0.1, 64.43 * DEG, 20 * DEG, 30 * DEG, 40 * DEG],
}
CRITICAL_INCLINATION = math.acos(math.sqrt(0.2))  # 1 - 5 cos^2 i = 0


def state(name: str) -> NDArray[np.float64]:
    return keplerian_to_cartesian(REGIMES[name])


def test_earth_constants() -> None:
    assert EARTH_RE == 6378.137  # WGS-84 equatorial radius, km
    assert EARTH_J2 == pytest.approx(1.0826e-3, rel=1e-4)
    assert EARTH_MU == 398600.4418


@pytest.mark.parametrize("name", REGIMES)
def test_osculating_to_mean_to_osculating_round_trip_is_sub_millimetre(name: str) -> None:
    rv = state(name)
    back = mean_to_osculating(osculating_to_mean(rv))
    assert np.linalg.norm(back[:3] - rv[:3]) < 1e-6  # km: 1 mm
    assert np.linalg.norm(back[3:] - rv[3:]) < 1e-9  # km/s: 1 um/s


def test_zero_j2_is_the_identity() -> None:
    rv = state("leo_eccentric")
    np.testing.assert_allclose(mean_to_osculating(osculating_to_mean(rv, j2=0.0), j2=0.0), rv, rtol=0, atol=1e-9)


def j2_acceleration(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Point mass plus J2: the true acceleration, independent of the Brouwer-Lyddane theory."""
    x, y, z = r
    rn = float(np.linalg.norm(r))
    k = 1.5 * EARTH_J2 * EARTH_MU * EARTH_RE**2 / rn**5
    zz = 5 * z * z / rn**2
    return -EARTH_MU / rn**3 * r + k * np.array([x * (zz - 1), y * (zz - 1), z * (zz - 3)])


def j2_trajectory(rv: NDArray[np.float64], dt: float, steps: int) -> NDArray[np.float64]:
    """RK4 integration of the true J2 dynamics, every step returned."""

    def deriv(s: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate([s[3:], j2_acceleration(s[:3])])

    out = [rv]
    for _ in range(steps):
        k1 = deriv(rv)
        k2 = deriv(rv + dt / 2 * k1)
        k3 = deriv(rv + dt / 2 * k2)
        k4 = deriv(rv + dt * k3)
        rv = rv + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(rv)
    return np.array(out)


def spreads(elements: NDArray[np.float64]) -> NDArray[np.float64]:
    """Peak-to-peak of a, e and i over a series of equinoctial elements."""
    e = np.hypot(elements[:, EQ_H], elements[:, EQ_K])
    i = 2 * np.arctan(np.hypot(elements[:, EQ_P], elements[:, EQ_Q]))
    return np.ptp(np.stack([elements[:, EQ_A], e, i]), axis=1)


@pytest.mark.parametrize("name", REGIMES)
def test_mean_elements_remove_the_short_period_terms_of_a_true_j2_orbit(name: str) -> None:
    # Along one revolution of an independently integrated J2 orbit, osculating a, e and i swing by the
    # short-period terms (kilometres in a in LEO), while the mean ones hold steady up to O(J2^2) and the
    # slow long-period terms: here at least 25 times steadier, typically 300 times (sweep: worst GEO, whose
    # osculating swings are themselves tiny, then GTO).
    a = REGIMES[name][0]
    period = 2 * math.pi * math.sqrt(a**3 / EARTH_MU)
    truth = j2_trajectory(state(name), period / 400, 400)
    osculating = spreads(np.array([cartesian_to_equinoctial(rv) for rv in truth]))
    mean = spreads(np.array([osculating_to_mean(rv) for rv in truth]))
    assert np.all(mean <= osculating / 25)
    # The short-period swing in e is first order, of order J2 (Re/a)^2 in every regime (a's vanishes for
    # circular equatorial orbits), so the check above isn't vacuous.
    assert osculating[1] > 0.5 * EARTH_J2 * (EARTH_RE / a) ** 2


@pytest.mark.parametrize("e", [0.0, 0.01, 0.1, 0.74])
def test_osculating_to_mean_raises_at_the_critical_inclination(e: float) -> None:
    # e = 0.74 at 26600 km is a Molniya orbit.
    a = 26600.0 if e > 0.5 else 7000.0
    rv = keplerian_to_cartesian([a, e, CRITICAL_INCLINATION - 0.01 * DEG, 0.3, 4.712, 0.5])
    with pytest.raises(ValueError, match=r"did not converge .* inclination 63\.42.*critical inclination"):
        osculating_to_mean(rv)


def test_mean_to_osculating_exactly_at_the_critical_inclination_raises_or_is_finite() -> None:
    # Exactly at it, 1 - 5 cos^2 i rounds to a tiny nonzero number: the result is finite but huge.
    # The guarantee is only that it never returns non-finite numbers.
    mean = cartesian_to_equinoctial(keplerian_to_cartesian([7000.0, 0.1, CRITICAL_INCLINATION, 0.3, 0.7, 0.5]))
    try:
        assert np.all(np.isfinite(mean_to_osculating(mean)))
    except ValueError as err:
        assert "critical inclination" in str(err)


def test_invalid_input_raises() -> None:
    rv = state("leo")
    with pytest.raises(HyperbolicOrbitError):
        osculating_to_mean([7000.0, 0, 0, 0, 12.0, 0])
    with pytest.raises(ValueError, match="re must be finite and positive"):
        osculating_to_mean(rv, re=0.0)
    with pytest.raises(ValueError, match="j2 must be finite"):
        osculating_to_mean(rv, j2=math.nan)
    with pytest.raises(ValueError, match="mu must be finite and positive"):
        osculating_to_mean(rv, mu=-1.0)
    with pytest.raises(ValueError, match="retrograde equatorial"):
        osculating_to_mean([7000.0, 0, 0, 0, -7.5, 0])
    with pytest.raises(HyperbolicOrbitError):
        mean_to_osculating([7000.0, 0.8, 0.8, 0, 0, 0])
    with pytest.raises(ValueError, match="semi-major axis must be positive"):
        mean_to_osculating([-7000.0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="elements must be finite"):
        mean_to_osculating([7000.0, 0, 0, 0, 0, math.inf])


def angle_wobbles(elements: NDArray[np.float64], t: NDArray[np.float64], e: float, i: float) -> NDArray[np.float64]:
    """Peak-to-peak of lambda, varpi and raan about their best-fit line in t (the secular drift), over a series
    of equinoctial elements. varpi is NaN for near-circular orbits and raan for near-equatorial ones, where
    those angles are undefined."""
    lam = np.unwrap(elements[:, EQ_LAMBDA])
    varpi = np.unwrap(np.arctan2(elements[:, EQ_H], elements[:, EQ_K]))
    raan = np.unwrap(np.arctan2(elements[:, EQ_P], elements[:, EQ_Q]))
    out = []
    for angle, defined in ((lam, True), (varpi, e >= 0.01), (raan, i >= 1 * DEG)):
        out.append(np.ptp(angle - np.polyval(np.polyfit(t, angle, 1), t)) if defined else np.nan)
    return np.array(out)


@pytest.mark.parametrize("name", REGIMES)
def test_mean_angles_remove_the_short_period_terms_of_a_true_j2_orbit(name: str) -> None:
    # The same check for the angles: about their secular drift, mean lambda, varpi and raan hold steady along
    # one revolution of the true J2 orbit, while the osculating ones swing by the short-period terms. Sweep:
    # at least 34 times steadier (GEO, whose osculating swings are ~1e-8 rad), typically several hundred.
    # This caught a missing O(e) term in delta lambda, which left lambda and varpi only ~6 times steadier
    # at e = 0.2 and in GTO.
    a, e, i = REGIMES[name][:3]
    period = 2 * math.pi * math.sqrt(a**3 / EARTH_MU)
    t = np.linspace(0, period, 401)
    truth = j2_trajectory(state(name), period / 400, 400)
    osculating = angle_wobbles(np.array([cartesian_to_equinoctial(rv) for rv in truth]), t, e, i)
    mean = angle_wobbles(np.array([osculating_to_mean(rv) for rv in truth]), t, e, i)
    defined = ~np.isnan(mean)
    assert np.all(mean[defined] <= osculating[defined] / 25), (mean, osculating)
