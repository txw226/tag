import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from tag.propagation import (
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
    AnomalyKind,
    HyperbolicOrbitError,
    cartesian_to_equinoctial,
    cartesian_to_keplerian,
    equinoctial_to_cartesian,
    keplerian_to_cartesian,
)

# Vallado, Fundamentals of Astrodynamics and Applications, Example 2-5 (RV2COE).
VALLADO_RV = [6524.834, 6862.875, 6448.296, 4.901327, 5.533756, -1.976341]


def test_cartesian_to_keplerian_matches_vallado_example_2_5() -> None:
    kep = cartesian_to_keplerian(VALLADO_RV, anomaly="true")
    a, e = kep[KEP_A], kep[KEP_E]
    assert a == pytest.approx(36127.343, abs=1e-2)
    assert e == pytest.approx(0.832853, abs=1e-6)
    assert a * (1 - e**2) == pytest.approx(11067.790, abs=1e-2)  # semi-latus rectum p
    assert math.degrees(kep[KEP_I]) == pytest.approx(87.870, abs=1e-3)
    assert math.degrees(kep[KEP_RAAN]) == pytest.approx(227.89, abs=1e-2)
    assert math.degrees(kep[KEP_ARGP]) == pytest.approx(53.38, abs=1e-2)
    assert math.degrees(kep[KEP_ANOMALY]) == pytest.approx(92.335, abs=1e-3)


def test_keplerian_to_cartesian_matches_vallado_example_2_6() -> None:
    # Example 2-6 (COE2RV): Example 2-5's elements, rounded, back to a state. It gives p rather than a.
    p, e = 11067.790, 0.83285
    kep = np.zeros(6)
    kep[KEP_A], kep[KEP_E] = p / (1 - e**2), e
    kep[KEP_I], kep[KEP_RAAN], kep[KEP_ARGP], kep[KEP_ANOMALY] = np.radians([87.87, 227.89, 53.38, 92.335])
    rv = keplerian_to_cartesian(kep, anomaly="true")
    np.testing.assert_allclose(rv[:3], [6525.368, 6861.532, 6449.119], atol=1e-3)
    np.testing.assert_allclose(rv[3:], [4.902279, 5.533140, -1.975710], atol=1e-6)


MU = 398600.4418
R_GEO = 42164.17
V_GEO = math.sqrt(MU / R_GEO)


def kep_state(a: float, e: float, i_deg: float, raan_deg: float, argp_deg: float, m_deg: float) -> np.ndarray:
    kep = np.zeros(6)
    kep[KEP_A], kep[KEP_E] = a, e
    kep[KEP_I], kep[KEP_RAAN], kep[KEP_ARGP], kep[KEP_ANOMALY] = np.radians([i_deg, raan_deg, argp_deg, m_deg])
    return keplerian_to_cartesian(kep)


# Cartesian states across regimes, including the Keplerian singular cases (circular, equatorial).
STATES = {
    "vallado": np.array(VALLADO_RV),
    "leo": kep_state(6778.0, 5e-4, 51.6, 30.0, 60.0, 57.0),
    "molniya": kep_state(26600.0, 0.74, 63.4, 100.0, 270.0, 17.0),
    "retrograde": kep_state(7200.0, 0.01, 98.0, 200.0, 10.0, 300.0),
    "geo": np.array([R_GEO, 0.0, 0.0, 0.0, V_GEO, 0.0]),
    "geo_rotated": np.array([R_GEO * 0.6, R_GEO * 0.8, 0.0, -V_GEO * 0.8, V_GEO * 0.6, 0.0]),
    "geo_tiny_e_i": kep_state(R_GEO, 1e-9, 1e-7, 75.0, 40.0, 123.0),
    "circular_inclined": np.array([7000.0, 0.0, 0.0, 0.0, 0.6 * math.sqrt(MU / 7000), 0.8 * math.sqrt(MU / 7000)]),
    "equatorial_elliptic": np.array([7000.0, 0.0, 0.0, 0.0, 8.5, 0.0]),
}


def assert_same_state(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-12) -> None:
    np.testing.assert_allclose(actual[:3], expected[:3], rtol=0, atol=rtol * np.linalg.norm(expected[:3]))
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=0, atol=rtol * np.linalg.norm(expected[3:]))


def angle_diff(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


@pytest.mark.parametrize("anomaly", ["mean", "true", "eccentric"])
@pytest.mark.parametrize("name", STATES)
def test_keplerian_round_trip_reproduces_state(name: str, anomaly: AnomalyKind) -> None:
    rv = STATES[name]
    kep = cartesian_to_keplerian(rv, anomaly=anomaly)
    assert_same_state(keplerian_to_cartesian(kep, anomaly=anomaly), rv)


@pytest.mark.parametrize("anomaly", ["mean", "true", "eccentric"])
@pytest.mark.parametrize("name", ["vallado", "leo", "molniya", "retrograde"])
def test_keplerian_elements_round_trip_away_from_singular_cases(name: str, anomaly: AnomalyKind) -> None:
    kep = cartesian_to_keplerian(STATES[name], anomaly=anomaly)
    again = cartesian_to_keplerian(keplerian_to_cartesian(kep, anomaly=anomaly), anomaly=anomaly)
    np.testing.assert_allclose(again[KEP_A], kep[KEP_A], rtol=1e-12)
    # e is small for leo, so its absolute error is the meaningful one.
    np.testing.assert_allclose(again[KEP_E], kep[KEP_E], rtol=0, atol=1e-13)
    angles = [KEP_I, KEP_RAAN, KEP_ARGP, KEP_ANOMALY]
    np.testing.assert_allclose(angle_diff(again[angles], kep[angles]), 0, atol=1e-9 if name == "leo" else 1e-12)


def test_keplerian_singular_conventions() -> None:
    # Circular equatorial: raan = argp = 0 and the anomaly is the true longitude, here 53.13 deg.
    kep = cartesian_to_keplerian(STATES["geo_rotated"], anomaly="true")
    assert kep[KEP_RAAN] == kep[KEP_ARGP] == 0.0
    assert kep[KEP_ANOMALY] == pytest.approx(math.atan2(0.8, 0.6), abs=1e-14)
    # Circular inclined: argp = 0, anomaly measured from the ascending node (here the position itself).
    kep = cartesian_to_keplerian(STATES["circular_inclined"], anomaly="true")
    assert kep[KEP_ARGP] == 0.0
    assert angle_diff(kep[KEP_ANOMALY], 0.0) == pytest.approx(0.0, abs=1e-14)
    assert kep[KEP_I] == pytest.approx(math.acos(0.6), abs=1e-14)
    # Equatorial elliptic, at periapsis on the x axis: raan = 0 and argp (longitude of periapsis) = 0.
    kep = cartesian_to_keplerian(STATES["equatorial_elliptic"], anomaly="true")
    assert kep[KEP_I] == 0.0 and kep[KEP_RAAN] == 0.0
    assert angle_diff(kep[KEP_ARGP], 0.0) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("name", STATES)
def test_equinoctial_round_trip_reproduces_state(name: str) -> None:
    rv = STATES[name]
    assert_same_state(equinoctial_to_cartesian(cartesian_to_equinoctial(rv)), rv)


@pytest.mark.parametrize("name", ["vallado", "leo", "molniya", "retrograde"])
def test_equinoctial_matches_keplerian_definition(name: str) -> None:
    kep = cartesian_to_keplerian(STATES[name], anomaly="mean")
    eq = cartesian_to_equinoctial(STATES[name])
    e, i, raan, argp, m = kep[[KEP_E, KEP_I, KEP_RAAN, KEP_ARGP, KEP_ANOMALY]]
    np.testing.assert_allclose(eq[EQ_A], kep[KEP_A], rtol=1e-12)
    np.testing.assert_allclose(
        eq[[EQ_H, EQ_K, EQ_P, EQ_Q]],
        [
            e * math.sin(argp + raan),
            e * math.cos(argp + raan),
            math.tan(i / 2) * math.sin(raan),
            math.tan(i / 2) * math.cos(raan),
        ],
        rtol=0,
        atol=1e-13,
    )
    assert angle_diff(eq[EQ_LAMBDA], m + argp + raan) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("lam", [0.0, 0.3, 2.0, math.pi, 5.5])
def test_equinoctial_exact_for_circular_equatorial_orbit(lam: float) -> None:
    # e = 0 and i = 0 exactly: the classical singular case, where equinoctial elements stay exact.
    eq = np.zeros(6)
    eq[EQ_A], eq[EQ_LAMBDA] = R_GEO, lam
    rv = equinoctial_to_cartesian(eq)
    expected = [R_GEO * math.cos(lam), R_GEO * math.sin(lam), 0.0, -V_GEO * math.sin(lam), V_GEO * math.cos(lam), 0.0]
    assert_same_state(rv, np.array(expected), rtol=1e-15)
    again = cartesian_to_equinoctial(rv)
    assert again[EQ_A] == pytest.approx(R_GEO, rel=1e-14)
    np.testing.assert_allclose(again[[EQ_H, EQ_K, EQ_P, EQ_Q]], 0.0, rtol=0, atol=1e-15)
    assert angle_diff(again[EQ_LAMBDA], lam) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "eq_in",
    [
        [26600.0, 0.5, -0.5, 0.3, 0.4, 1.0],  # e = 0.71
        [7000.0, 0.0, 0.0, 0.2, -0.1, 4.0],  # circular, inclined
        [42164.0, 1e-9, -2e-9, 0.0, 0.0, 2.5],  # nearly circular, equatorial
        [7000.0, 0.01, 0.02, 1e-9, 1e-9, 6.0],  # nearly equatorial
    ],
)
def test_equinoctial_elements_round_trip(eq_in: list[float]) -> None:
    again = cartesian_to_equinoctial(equinoctial_to_cartesian(eq_in))
    np.testing.assert_allclose(again[EQ_A], eq_in[EQ_A], rtol=1e-13)
    np.testing.assert_allclose(
        again[[EQ_H, EQ_K, EQ_P, EQ_Q]], np.array(eq_in)[[EQ_H, EQ_K, EQ_P, EQ_Q]], rtol=0, atol=1e-14
    )
    assert angle_diff(again[EQ_LAMBDA], eq_in[EQ_LAMBDA]) == pytest.approx(0.0, abs=1e-13)


V_ESCAPE = math.sqrt(2 * MU / 7000)
NOT_ELLIPTIC = {
    "hyperbolic": [7000.0, 0.0, 0.0, 0.0, 1.2 * V_ESCAPE, 0.0],
    "parabolic": [7000.0, 0.0, 0.0, 0.0, V_ESCAPE, 0.0],
    "rectilinear": [7000.0, 0.0, 0.0, 1.0, 0.0, 0.0],
}


@pytest.mark.parametrize("convert", [cartesian_to_keplerian, cartesian_to_equinoctial])
@pytest.mark.parametrize("name", NOT_ELLIPTIC)
def test_non_elliptic_state_raises_hyperbolic_orbit_error(
    convert: Callable[[ArrayLike], np.ndarray], name: str
) -> None:
    with pytest.raises(HyperbolicOrbitError, match="eccentricity"):
        convert(NOT_ELLIPTIC[name])


@pytest.mark.parametrize("e", [1.0, 1.5])
def test_keplerian_elements_with_e_at_least_one_raise(e: float) -> None:
    with pytest.raises(HyperbolicOrbitError, match="eccentricity"):
        keplerian_to_cartesian([7000.0, e, 0.1, 0.2, 0.3, 0.4])


def test_equinoctial_elements_with_e_at_least_one_raise() -> None:
    with pytest.raises(HyperbolicOrbitError, match="eccentricity"):
        equinoctial_to_cartesian([7000.0, 0.6, 0.8, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "elements",
    [
        [-7000.0, 0.1, 0.1, 0.2, 0.3, 0.4],  # negative a
        [7000.0, -0.1, 0.1, 0.2, 0.3, 0.4],  # negative e
        [7000.0, 0.1, math.nan, 0.2, 0.3, 0.4],
    ],
)
def test_invalid_keplerian_elements_raise_value_error(elements: list[float]) -> None:
    with pytest.raises(ValueError):
        keplerian_to_cartesian(elements)


def test_unknown_anomaly_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="anomaly"):
        cartesian_to_keplerian(VALLADO_RV, anomaly="hyperbolic")  # type: ignore[arg-type]


def test_invalid_state_raises_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        cartesian_to_equinoctial([7000.0, 0.0, math.inf, 0.0, 7.5, 0.0])
    with pytest.raises(ValueError, match="mu"):
        cartesian_to_keplerian(VALLADO_RV, mu=-1.0)
