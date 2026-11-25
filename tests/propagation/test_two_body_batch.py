"""Batched two-body propagation: TwoBodyOrbit.propagate_many and the module-level propagate_many."""

import numpy as np
import numpy.typing as npt
import pytest

from tag.propagation import TwoBodyOrbit, propagate_many

# Osculating states at the epoch, km and km/s, across regimes.
LEO = [6778.0, 0.0, 0.0, 0.0, 4.72, 6.04]  # e ~ 1e-3, i ~ 52 deg
MOLNIYA = [-1500.0, -6300.0, 2200.0, 8.2, -2.4, 4.9]  # high e, inclined
GEO = [42164.1696, 0.0, 0.0, 0.0, 3.0746761, 0.0]  # e = 0, i = 0
REGIMES = {"leo": LEO, "molniya": MOLNIYA, "geo": GEO}

# Both signs, zero, and long spans, unsorted.
DTS = np.array([0.0, 1.0, -1.0, 5000.0, -86400.0, 30 * 86400.0, 123.456, -7.5e6])


@pytest.mark.parametrize("rv0", REGIMES.values(), ids=REGIMES.keys())
def test_orbit_batch_equals_scalar_propagation(rv0: list[float]) -> None:
    orbit = TwoBodyOrbit(rv0)
    states = orbit.propagate_many(DTS)
    assert states.dtype == np.float64
    assert states.shape == (len(DTS), 6)
    np.testing.assert_array_equal(states, [orbit.propagate(dt) for dt in DTS])


def test_many_orbits_batch_equals_scalar_propagation_on_a_grid() -> None:
    rvs = np.array(list(REGIMES.values()))
    mu = 398600.0  # not the default, to check it is passed through
    states = propagate_many(rvs, DTS, mu=mu)
    assert states.dtype == np.float64
    assert states.shape == (len(rvs), len(DTS), 6)
    expected = [[TwoBodyOrbit(rv, mu).propagate(dt) for dt in DTS] for rv in rvs]
    np.testing.assert_array_equal(states, expected)


def test_out_is_filled_in_place_and_returned() -> None:
    orbit = TwoBodyOrbit(LEO)
    out = np.full((len(DTS), 6), np.nan)
    assert orbit.propagate_many(DTS, out=out) is out
    np.testing.assert_array_equal(out, orbit.propagate_many(DTS))

    rvs = np.array([LEO, GEO])
    grid = np.full((2, len(DTS), 6), np.nan)
    assert propagate_many(rvs, DTS, out=grid) is grid
    np.testing.assert_array_equal(grid, propagate_many(rvs, DTS))


def test_array_like_inputs_are_converted() -> None:
    orbit = TwoBodyOrbit(LEO)
    expected = orbit.propagate_many(np.array([0.0, 60.0, -60.0]))
    np.testing.assert_array_equal(orbit.propagate_many([0, 60, -60]), expected)
    np.testing.assert_array_equal(orbit.propagate_many(np.array([0, 60, -60], dtype=np.int32)), expected)
    np.testing.assert_array_equal(orbit.propagate_many(np.array([0.0, 0.0, 60.0, -60.0])[1:]), expected)
    np.testing.assert_array_equal(propagate_many([LEO], (0.0, 60.0, -60.0))[0], expected)
    np.testing.assert_array_equal(
        propagate_many(np.asfortranarray([LEO, GEO]), [60.0])[:, 0],
        [
            TwoBodyOrbit(LEO).propagate(60.0),
            TwoBodyOrbit(GEO).propagate(60.0),
        ],
    )


def test_empty_batches_give_empty_arrays() -> None:
    assert TwoBodyOrbit(LEO).propagate_many(np.empty(0)).shape == (0, 6)
    assert propagate_many(np.empty((0, 6)), DTS).shape == (0, len(DTS), 6)


@pytest.mark.parametrize(
    ("dts", "message"),
    [
        (np.zeros((2, 3)), r"dts must be 1-D, got shape \(2, 3\)"),
        (np.float64(1.0), r"dts must be 1-D, got shape \(\)"),
    ],
)
def test_wrong_times_shape_raises_value_error(dts: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TwoBodyOrbit(LEO).propagate_many(dts)
    with pytest.raises(ValueError, match=message):
        propagate_many([LEO], dts)


@pytest.mark.parametrize("rvs", [np.zeros((2, 5)), np.zeros(6), np.zeros((1, 2, 6))])
def test_wrong_states_shape_raises_value_error(rvs: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"rvs must have shape \(M, 6\), got \("):
        propagate_many(rvs, DTS)


@pytest.mark.parametrize(
    "values",
    [["a", "b"], np.array(["1.5"]), np.array([True, False]), np.array([1 + 2j]), np.array([None], dtype=object)],
    ids=["str-list", "str-array", "bool", "complex", "object"],
)
def test_non_real_dtype_raises_type_error(values: npt.ArrayLike) -> None:
    with pytest.raises(TypeError, match=r"dts must be real numbers, got dtype"):
        TwoBodyOrbit(LEO).propagate_many(values)
    with pytest.raises(TypeError, match=r"dts must be real numbers, got dtype"):
        propagate_many([LEO], values)
    with pytest.raises(TypeError, match=r"rvs must be real numbers, got dtype"):
        propagate_many([LEO[:5] + ["a"]], DTS)


def test_out_with_wrong_shape_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"out must have shape \(8, 6\), got \(6, 8\)"):
        TwoBodyOrbit(LEO).propagate_many(DTS, out=np.empty((6, 8)))
    with pytest.raises(ValueError, match=r"out must have shape \(2, 8, 6\), got \(16, 6\)"):
        propagate_many([LEO, GEO], DTS, out=np.empty((16, 6)))


def read_only(shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape)
    out.flags.writeable = False
    return out


@pytest.mark.parametrize(
    "make_out",
    [
        lambda shape: np.empty(shape, dtype=np.float32),
        lambda shape: np.empty(shape, dtype=np.float64, order="F"),
        lambda shape: np.empty((*shape[:-1], 12))[..., ::2],
        read_only,
        lambda shape: np.empty(shape).tolist(),
    ],
    ids=["float32", "fortran-order", "strided", "read-only", "list"],
)
def test_out_that_cannot_be_written_in_place_raises_type_error(make_out) -> None:
    with pytest.raises(TypeError, match="out must be a writable, C-contiguous numpy array of dtype float64"):
        TwoBodyOrbit(LEO).propagate_many(DTS, out=make_out((len(DTS), 6)))
    with pytest.raises(TypeError, match="out must be a writable, C-contiguous numpy array of dtype float64"):
        propagate_many([LEO], DTS, out=make_out((1, len(DTS), 6)))
