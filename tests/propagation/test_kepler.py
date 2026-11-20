import math

import pytest

from tag.propagation import eccentric_anomaly


def test_matches_vallado_example_2_1() -> None:
    # Vallado, Fundamentals of Astrodynamics and Applications, Example 2-1: M = 235.4 deg, e = 0.4.
    E = eccentric_anomaly(math.radians(235.4), 0.4)
    assert math.degrees(E) == pytest.approx(220.512074767522, abs=1e-11)


ECCENTRICITIES = [0.0, 1e-12, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 1 - 1e-6, 1 - 1e-9, 1 - 1e-12]


def mean_anomaly_sweep() -> list[float]:
    """Evenly over [-pi, pi], plus values near 0 (hardest for e near 1) and near +-pi."""
    even = [-math.pi + 2 * math.pi * k / 720 for k in range(721)]
    tiny = [s * 10.0**p for p in range(-12, 0) for s in (1.0, -1.0)]
    near_pi = [s * (math.pi - 10.0**p) for p in range(-12, 0) for s in (1.0, -1.0)]
    return even + tiny + near_pi


def kepler_residual(M: float, e: float) -> float:
    E = eccentric_anomaly(M, e)
    return E - e * math.sin(E) - M


@pytest.mark.parametrize("e", ECCENTRICITIES)
def test_satisfies_keplers_equation_across_sweep(e: float) -> None:
    assert max(abs(kepler_residual(M, e)) for M in mean_anomaly_sweep()) < 1e-15


@pytest.mark.parametrize("e", [0.0, 0.5, 0.99, 1 - 1e-9])
@pytest.mark.parametrize("revolutions", [-1000, -3, 1, 7, 1000])
def test_mean_anomaly_outside_one_revolution_keeps_revolution(e: float, revolutions: int) -> None:
    for M0 in [-3.0, -0.4, 0.0, 1e-6, 2.5]:
        M = M0 + 2 * math.pi * revolutions
        E = eccentric_anomaly(M, e)
        assert E - e * math.sin(E) == pytest.approx(M, rel=4e-16, abs=1e-15)
        # Same revolution as M: E differs from M by at most e.
        assert abs(E - M) <= e + 1e-12


@pytest.mark.parametrize("e", [-1e-12, 1.0, 1.5, math.nan, math.inf])
def test_rejects_eccentricity_outside_elliptic_range(e: float) -> None:
    with pytest.raises(ValueError, match="eccentricity"):
        eccentric_anomaly(1.0, e)


@pytest.mark.parametrize("M", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_mean_anomaly(M: float) -> None:
    with pytest.raises(ValueError, match="mean anomaly"):
        eccentric_anomaly(M, 0.5)
