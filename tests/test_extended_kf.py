import numpy as np
import pytest

from tooth.kalman import ExtendedKalmanFilter, LinearKalmanFilter, numerical_jacobian
from tooth.sims.projectile import launch_state, make_filter, simulate


def linear_run():
    rng = np.random.default_rng(1)
    x0 = launch_state(30.0, 45.0)
    run = simulate(x0, 0.05, 0.5, 2.0, rng)
    zs = run.measurements[1:].copy()
    zs[10:15] = np.nan  # dropouts exercise the skip path too
    return run, x0, zs


@pytest.mark.parametrize("analytic", [True, False])
def test_ekf_reduces_to_linear_kf(analytic):
    """With linear f and h, the EKF must reproduce the linear Kalman filter."""
    run, x0, zs = linear_run()
    kf = make_filter(run, velocity_guess=x0[2:], velocity_std=5.0)
    m = run.model
    ekf = ExtendedKalmanFilter(
        f=lambda x, u: m.F @ x + m.B @ u,
        h=lambda x: m.H @ x,
        Q=m.Q,
        R=m.R,
        x0=kf.x,
        P0=kf.P,
        F_jac=(lambda x, u: m.F) if analytic else None,
        H_jac=(lambda x: m.H) if analytic else None,
    )
    ref = kf.filter(zs, m.u)
    res = ekf.filter(zs, m.u)
    # Finite differences of a linear map are exact up to round-off.
    tol = 1e-10 if analytic else 1e-6
    for name in ("x_prior", "P_prior", "x_post", "P_post"):
        np.testing.assert_allclose(getattr(res, name), getattr(ref, name), rtol=tol, atol=tol)


def test_numerical_jacobian_matches_analytic():
    def fn(x):
        return np.array([np.sin(x[0]) * x[1], x[0] ** 2 + np.exp(x[1]), x[1] / x[0]])

    x = np.array([0.7, -1.3])
    expected = np.array(
        [
            [np.cos(x[0]) * x[1], np.sin(x[0])],
            [2 * x[0], np.exp(x[1])],
            [-x[1] / x[0] ** 2, 1 / x[0]],
        ]
    )
    np.testing.assert_allclose(numerical_jacobian(fn, x), expected, rtol=1e-8, atol=1e-9)


def wrap(a):
    return np.pi - np.mod(np.pi - a, 2 * np.pi)


def angle_filter(residual=None):
    """Scalar random-walk angle observed directly."""
    return ExtendedKalmanFilter(
        f=lambda x, u: x,
        h=lambda x: x,
        Q=[[0.0]],
        R=[[0.01]],
        x0=[np.pi - 0.05],
        P0=[[0.01]],
        residual=residual,
    )


def test_custom_residual_wraps_angles():
    z = -np.pi + 0.05  # 0.1 rad from the estimate across the ±pi seam
    naive = angle_filter()
    naive.step(z)
    wrapped = angle_filter(residual=lambda z, zp: wrap(z - zp))
    wrapped.step(z)

    np.testing.assert_allclose(wrapped.y, [0.1])
    # Equal prior and measurement variances: the estimate moves halfway, to the seam.
    np.testing.assert_allclose(abs(wrapped.x[0]), np.pi, atol=1e-12)
    assert naive.y[0] < -6  # without wrapping the innovation is ~ -2pi + 0.1


BASE = dict(f=lambda x, u: x, h=lambda x: x[:1], Q=np.eye(2), R=np.eye(1), x0=np.zeros(2), P0=np.eye(2))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"Q": np.eye(3)}, "Q must have shape"),
        ({"R": np.ones((1, 2))}, "R must be square"),
        ({"P0": np.eye(3)}, "P0 must have shape"),
    ],
)
def test_constructor_shape_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ExtendedKalmanFilter(**(BASE | kwargs))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"f": lambda x, u: np.zeros(3)}, "f must return shape"),
        ({"F_jac": lambda x, u: np.eye(3)}, "F_jac must return shape"),
        ({"h": lambda x: x}, "h must return shape"),
        ({"H_jac": lambda x: np.eye(2)}, "H_jac must return shape"),
        ({"residual": lambda z, zp: np.zeros(2)}, "residual must return shape"),
    ],
)
def test_callable_output_validation(kwargs, match):
    ekf = ExtendedKalmanFilter(**(BASE | kwargs))
    with pytest.raises(ValueError, match=match):
        ekf.step([1.0])


def test_update_none_skips():
    ekf = ExtendedKalmanFilter(**BASE)
    ekf.predict()
    x, P = ekf.x.copy(), ekf.P.copy()
    ekf.update(None)
    np.testing.assert_array_equal(ekf.x, x)
    np.testing.assert_array_equal(ekf.P, P)


def test_filter_copies_its_inputs():
    """In-place edits to a filter's matrices must not leak into the caller's arrays, e.g. run.model.Q."""
    arrays = dict(Q=np.eye(2), R=np.eye(1), x0=np.zeros(2), P0=np.eye(2))
    originals = {k: v.copy() for k, v in arrays.items()}
    ekf = ExtendedKalmanFilter(**(BASE | arrays))
    ekf.Q *= 3.0
    ekf.R *= 3.0
    ekf.P *= 3.0
    ekf.x += 1.0
    for k, v in arrays.items():
        np.testing.assert_array_equal(v, originals[k], err_msg=k)
