import numpy as np
import pytest

from tooth.kalman import LinearKalmanFilter


def scalar_filter(sigma: float, x0: float) -> LinearKalmanFilter:
    return LinearKalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.0]], R=[[sigma**2]], x0=[x0], P0=[[sigma**2]])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"F": np.eye(2, 3)}, "F must be square"),
        ({"H": np.ones((1, 3))}, "H must have shape"),
        ({"Q": np.eye(3)}, "Q must have shape"),
        ({"R": np.eye(2)}, "R must have shape"),
        ({"x0": np.zeros(3)}, "x0 must have shape"),
        ({"P0": np.eye(3)}, "P0 must have shape"),
        ({"B": np.ones((3, 1))}, "B must have shape"),
    ],
)
def test_shape_validation(kwargs, match):
    base = dict(F=np.eye(2), H=np.ones((1, 2)), Q=np.eye(2), R=np.eye(1), x0=np.zeros(2), P0=np.eye(2))
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LinearKalmanFilter(**base)


def test_control_without_B_raises():
    kf = scalar_filter(1.0, 0.0)
    with pytest.raises(ValueError, match="no B matrix"):
        kf.predict(u=[1.0])


def test_scalar_constant_matches_running_mean():
    """Static scalar with Q=0: the KF estimate is the sample mean, P = sigma^2 / n."""
    rng = np.random.default_rng(0)
    sigma, true_value = 3.0, 7.5
    zs = true_value + sigma * rng.standard_normal(200)

    kf = scalar_filter(sigma, zs[0])
    res = kf.filter(zs[1:])

    n = np.arange(2, len(zs) + 1)
    np.testing.assert_allclose(res.x_post[:, 0], np.cumsum(zs)[1:] / n, rtol=1e-10)
    np.testing.assert_allclose(res.P_post[:, 0, 0], sigma**2 / n, rtol=1e-10)


def test_nan_measurement_skips_update():
    kf = scalar_filter(1.0, 0.0)
    res = kf.filter([[np.nan], [1.0]])
    np.testing.assert_array_equal(res.x_post[0], res.x_prior[0])
    assert res.P_post[1, 0, 0] < res.P_prior[1, 0, 0]


def test_covariance_stays_symmetric_positive_definite():
    rng = np.random.default_rng(1)
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-4 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    R = np.array([[1e-6]])  # very precise sensor stresses the covariance update
    kf = LinearKalmanFilter(F, H, Q, R, x0=[0, 0], P0=1e6 * np.eye(2))

    res = kf.filter(rng.standard_normal((5000, 1)))
    for P in res.P_post:
        np.testing.assert_allclose(P, P.T, atol=0)
        assert np.all(np.linalg.eigvalsh(P) > 0)
