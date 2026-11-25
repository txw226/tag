import numpy as np
import pytest

from tag.propagation import HyperbolicOrbitError, TwoBodyOrbit


def test_matches_vallado_example_2_4() -> None:
    # Vallado, Fundamentals of Astrodynamics and Applications, Example 2-4 (KEPLER): 40 minutes on.
    orbit = TwoBodyOrbit([1131.340, -2282.343, 6672.423, -5.64305, 4.30333, 2.42879])
    rv = orbit.propagate(40 * 60.0)
    np.testing.assert_allclose(rv[:3], [-4219.7527, 4363.0292, -3958.7666], atol=1e-3)
    np.testing.assert_allclose(rv[3:], [3.689866, -1.916735, -6.112511], atol=1e-6)
    assert rv.dtype == np.float64
    assert rv.shape == (6,)


MU = 398600.4418
# Osculating states at the epoch, km and km/s, across regimes.
LEO = [6778.0, 0.0, 0.0, 0.0, 4.72, 6.04]  # e ~ 1e-3, i ~ 52 deg
MOLNIYA = [-1500.0, -6300.0, 2200.0, 8.2, -2.4, 4.9]  # high e, inclined
VALLADO = [1131.340, -2282.343, 6672.423, -5.64305, 4.30333, 2.42879]
REGIMES = {"leo": LEO, "molniya": MOLNIYA, "vallado": VALLADO}


def energy(rv: np.ndarray) -> float:
    return float(0.5 * rv[3:] @ rv[3:] - MU / np.linalg.norm(rv[:3]))


def angular_momentum(rv: np.ndarray) -> np.ndarray:
    return np.cross(rv[:3], rv[3:])


@pytest.mark.parametrize("rv0", REGIMES.values(), ids=REGIMES.keys())
def test_conserves_energy_and_angular_momentum_over_long_propagations(rv0: list[float]) -> None:
    orbit = TwoBodyOrbit(rv0)
    e0, h0 = energy(np.array(rv0)), angular_momentum(np.array(rv0))
    for dt in np.linspace(-30 * 86400.0, 30 * 86400.0, 2001):
        rv = orbit.propagate(dt)
        assert energy(rv) == pytest.approx(e0, rel=1e-14)
        np.testing.assert_allclose(angular_momentum(rv), h0, rtol=0, atol=1e-14 * np.linalg.norm(h0))


@pytest.mark.parametrize("rv0", REGIMES.values(), ids=REGIMES.keys())
@pytest.mark.parametrize("dt", [1.0, 5000.0, 86400.0, 30 * 86400.0])
def test_propagating_forward_then_back_returns_initial_state(rv0: list[float], dt: float) -> None:
    orbit = TwoBodyOrbit(rv0)
    back = TwoBodyOrbit(orbit.propagate(dt)).propagate(-dt)
    # Rebuilding the orbit from a rounded state perturbs n by ~1e-16 relative (several times more near
    # a high-e perigee, where 2/r - v^2/mu cancels, whatever the method). That is a timing error
    # growing with the angle swept, n |dt|: per radian, |v| / n in position and |r''| / n in velocity.
    n = 2 * np.pi / orbit.period
    dt_err = 1e-14 * (1 + n * abs(dt)) / n
    np.testing.assert_allclose(back[:3], rv0[:3], rtol=0, atol=dt_err * np.linalg.norm(rv0[3:]))
    np.testing.assert_allclose(back[3:], rv0[3:], rtol=0, atol=dt_err * MU / np.linalg.norm(rv0[:3]) ** 2)


def circular_state(radius: float, inclination: float, raan: float, u: float) -> np.ndarray:
    """State on a circular orbit at argument of latitude u, from its closed form."""
    speed = np.sqrt(MU / radius)
    node = np.array([np.cos(raan), np.sin(raan), 0.0])
    normal = np.array([np.sin(raan) * np.sin(inclination), -np.cos(raan) * np.sin(inclination), np.cos(inclination)])
    across = np.cross(normal, node)
    r = radius * (np.cos(u) * node + np.sin(u) * across)
    v = speed * (-np.sin(u) * node + np.cos(u) * across)
    return np.concatenate([r, v])


GEO_RADIUS = 42164.1696
CIRCULAR = {
    "geo": (GEO_RADIUS, 0.0, 0.0),
    "circular-equatorial-leo": (6778.0, 0.0, 1.0),
    "circular-inclined": (7000.0, np.radians(98.0), 2.0),
    "circular-retrograde-equatorial": (8000.0, np.pi, 0.5),
}


@pytest.mark.parametrize(("radius", "inclination", "raan"), CIRCULAR.values(), ids=CIRCULAR.keys())
def test_circular_and_equatorial_orbits_match_closed_form(radius: float, inclination: float, raan: float) -> None:
    orbit = TwoBodyOrbit(circular_state(radius, inclination, raan, 0.3))
    n = np.sqrt(MU / radius**3)
    # Over a year (up to ~6000 rad of mean anomaly for LEO), the error stays at the size set by the
    # rounding of n itself, ~1e-16 relative per radian swept: no decay from the growing angle.
    for dt in np.linspace(-365 * 86400.0, 365 * 86400.0, 2001):
        expected = circular_state(radius, inclination, raan, 0.3 + n * dt)
        np.testing.assert_allclose(
            orbit.propagate(dt)[:3], expected[:3], rtol=0, atol=4e-15 * radius * (1 + n * abs(dt))
        )


def two_body_rk4(rv0: np.ndarray, duration: float, steps: int) -> np.ndarray:
    """Independent reference: classic RK4 on r'' = -mu r / |r|^3."""

    def deriv(x: np.ndarray) -> np.ndarray:
        r = x[:3]
        return np.concatenate([x[3:], -MU * r / np.linalg.norm(r) ** 3])

    x, h = np.array(rv0, dtype=float), duration / steps
    for _ in range(steps):
        k1 = deriv(x)
        k2 = deriv(x + h / 2 * k1)
        k3 = deriv(x + h / 2 * k2)
        k4 = deriv(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


NEARLY_SINGULAR = {
    # GEO-like with tiny eccentricity and inclination: the perifocal axes and node are ill-defined.
    "near-geo": [GEO_RADIUS, 1e-6, 1e-7, -1e-9, 3.0746761, 3e-9],
    "near-circular-leo": [6778.0, 0.0, 0.0, 0.0, 4.7200001, 6.04],
    "near-equatorial-eccentric": [7000.0, 0.0, 1e-9, 0.0, 8.5, 1e-10],
    "molniya": MOLNIYA,
}


@pytest.mark.parametrize("rv0", NEARLY_SINGULAR.values(), ids=NEARLY_SINGULAR.keys())
def test_matches_numerical_integration_in_every_regime(rv0: list[float]) -> None:
    orbit = TwoBodyOrbit(rv0)
    duration = orbit.period * 1.3
    expected = two_body_rk4(np.array(rv0), duration, 20000)
    scale = np.linalg.norm(rv0[:3])
    np.testing.assert_allclose(orbit.propagate(duration)[:3], expected[:3], rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(
        orbit.propagate(-duration)[:3], two_body_rk4(np.array(rv0), -duration, 20000)[:3], rtol=0, atol=1e-9 * scale
    )


def test_hyperbolic_orbit_raises_naming_eccentricity() -> None:
    # Periapsis at 7000 km with 1.5x the circular speed: e = 1.5^2 - 1 = 1.25.
    speed = 1.5 * np.sqrt(MU / 7000.0)
    with pytest.raises(HyperbolicOrbitError, match=r"eccentricity 1\.2[45]"):
        TwoBodyOrbit([7000.0, 0.0, 0.0, 0.0, speed, 0.0])


def test_hyperbolic_orbit_error_is_a_value_error() -> None:
    assert issubclass(HyperbolicOrbitError, ValueError)


def test_parabolic_orbit_raises() -> None:
    # mu = 2, r = 1, v = 2: escape speed exactly, so zero energy and e = 1.
    with pytest.raises(HyperbolicOrbitError, match="eccentricity 1"):
        TwoBodyOrbit([1.0, 0.0, 0.0, 0.0, 2.0, 0.0], mu=2.0)


def test_rectilinear_orbit_raises() -> None:
    # Falling straight in: bound, but zero angular momentum, so e = 1.
    with pytest.raises(HyperbolicOrbitError, match="eccentricity 1"):
        TwoBodyOrbit([7000.0, 0.0, 0.0, -1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("rv", "mu", "match"),
    [
        ([np.nan, 0, 0, 0, 7.5, 0], MU, "finite"),
        ([7000.0, 0, 0, 0, np.inf, 0], MU, "finite"),
        ([0.0, 0, 0, 0, 7.5, 0], MU, "position"),
        (LEO, 0.0, "mu"),
        (LEO, -MU, "mu"),
        (LEO, np.nan, "mu"),
    ],
)
def test_invalid_state_raises_value_error(rv: list[float], mu: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TwoBodyOrbit(rv, mu=mu)


@pytest.mark.parametrize("dt", [np.nan, np.inf, -np.inf])
def test_non_finite_dt_raises(dt: float) -> None:
    with pytest.raises(ValueError, match="dt"):
        TwoBodyOrbit(LEO).propagate(dt)


def test_accepts_numpy_state_and_reports_orbit_constants() -> None:
    # At periapsis r_p = 7000 km with 1.2x the circular speed: e = 1.2^2 - 1 = 0.44, a = r_p / (1 - e).
    orbit = TwoBodyOrbit(np.array([0.0, 0.0, 7000.0, 1.2 * np.sqrt(MU / 7000.0), 0.0, 0.0]), mu=MU)
    assert orbit.mu == MU
    assert orbit.eccentricity == pytest.approx(0.44, rel=1e-14)
    assert orbit.semi_major_axis == pytest.approx(7000.0 / 0.56, rel=1e-14)
    assert orbit.period == pytest.approx(2 * np.pi * np.sqrt((7000.0 / 0.56) ** 3 / MU), rel=1e-14)
