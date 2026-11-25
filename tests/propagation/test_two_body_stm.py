"""The two-body state transition matrix: propagate_with_stm and propagate_many_with_stm."""

from collections.abc import Callable

import numpy as np
import pytest

from tag.propagation import TwoBodyOrbit
from tag.propagation._propagation import _two_body_central_difference_stm

MU = 398600.4418
# Osculating states at the epoch, km and km/s, across regimes.
LEO = [6778.0, 0.0, 0.0, 0.0, 4.72, 6.04]  # e ~ 1e-3, i ~ 52 deg
MOLNIYA = [-1500.0, -6300.0, 2200.0, 8.2, -2.4, 4.9]  # high e, inclined
GEO = [42164.1696, 0.0, 0.0, 0.0, 3.0746761, 0.0]  # e = 0, i = 0
NEAR_CIRCULAR_EQUATORIAL = [7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 1e-9]  # e ~ 1e-12, i ~ 1e-10
VALLADO = [1131.340, -2282.343, 6672.423, -5.64305, 4.30333, 2.42879]
# e = 0.9 from a 6678 km perigee, inclined: where the oracle's own precision is down to ~1e-7.
HIGHLY_ECCENTRIC = [6678.0, 0.0, 0.0, 0.0, 6.8453, 8.1579]
REGIMES = {
    "leo": LEO,
    "molniya": MOLNIYA,
    "geo": GEO,
    "near-circular-equatorial": NEAR_CIRCULAR_EQUATORIAL,
    "vallado": VALLADO,
    "highly-eccentric": HIGHLY_ECCENTRIC,
}


def scaled(stm: np.ndarray, rv0: list[float]) -> np.ndarray:
    """The state transition matrix made dimensionless by the epoch radius and speed, so its blocks
    compare on one scale."""
    scale = np.repeat([np.linalg.norm(rv0[:3]), np.linalg.norm(rv0[3:])], 3)
    return stm * scale[np.newaxis, :] / scale[:, np.newaxis]


def scalar_stm(orbit: TwoBodyOrbit, dt: float) -> np.ndarray:
    return orbit.propagate_with_stm(dt)[1]


def batched_stm(orbit: TwoBodyOrbit, dt: float) -> np.ndarray:
    return orbit.propagate_many_with_stm(np.array([0.0, dt, -dt]))[1][1]


@pytest.mark.parametrize("stm_of", [scalar_stm, batched_stm], ids=["scalar", "batched"])
@pytest.mark.parametrize("rv0", REGIMES.values(), ids=REGIMES.keys())
@pytest.mark.parametrize("periods", [-10.0, -1.7, -0.25, 1e-4, 0.3, 1.0, 2.6])
def test_stm_matches_central_differences(
    stm_of: Callable[[TwoBodyOrbit, float], np.ndarray], rv0: list[float], periods: float
) -> None:
    orbit = TwoBodyOrbit(rv0)
    dt = periods * orbit.period
    stm = scaled(stm_of(orbit, dt), rv0)
    reference = scaled(_two_body_central_difference_stm(rv0, dt), rv0)
    assert stm.shape == (6, 6)
    assert stm.dtype == np.float64
    # The oracle is good to about 1e-10 of the largest entry, a few 1e-8 when the end point is at a
    # high-e perigee (Molniya, e = 0.9 after whole periods); an algebra error would be O(1).
    np.testing.assert_allclose(stm, reference, rtol=0, atol=1e-7 * np.abs(reference).max())


J = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])


@pytest.mark.parametrize("rv0", REGIMES.values(), ids=REGIMES.keys())
@pytest.mark.parametrize("periods", [-30.0, -0.6, 1e-6, 0.45, 1.0, 7.3, 300.0])
def test_stm_is_symplectic(rv0: list[float], periods: float) -> None:
    orbit = TwoBodyOrbit(rv0)
    # Dimensionless, so the residual compares to eps times the squared size of the entries.
    stm = scaled(orbit.propagate_with_stm(periods * orbit.period)[1], rv0)
    residual = stm.T @ J @ stm - J
    assert np.abs(residual).max() <= 1e-14 * max(1.0, np.abs(stm).max() ** 2)


@pytest.mark.parametrize("rv0", REGIMES.values(), ids=REGIMES.keys())
@pytest.mark.parametrize("periods", [-30.0, -0.6, 1e-6, 0.45, 1.0, 7.3, 300.0])
def test_stm_forward_then_back_composes_to_identity(rv0: list[float], periods: float) -> None:
    orbit = TwoBodyOrbit(rv0)
    dt = periods * orbit.period
    rv, forward = orbit.propagate_with_stm(dt)
    back = TwoBodyOrbit(rv).propagate_with_stm(-dt)[1]
    product = scaled(back, rv0) @ scaled(forward, rv0)
    assert np.abs(product - np.eye(6)).max() <= 1e-14 * max(1.0, np.abs(scaled(forward, rv0)).max() ** 2)
