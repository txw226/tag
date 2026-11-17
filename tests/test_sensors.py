import numpy as np
import pytest

from tooth.kalman import numerical_jacobian
from tooth.sensors import GnssPosition, PinholeCamera, RangeBearingRadar

RADAR = RangeBearingRadar([-100.0, 0.0], range_std=2.0, bearing_std=0.01)
CAMERA = PinholeCamera([-50.0, 1.0], yaw=np.radians(20), focal_px=1000.0, pixel_std=3.0, width_px=600.0)
GNSS = GnssPosition(std=2.0, every=10)
SENSORS = [RADAR, CAMERA, GNSS]


@pytest.mark.parametrize("sensor", SENSORS, ids=lambda s: s.name)
@pytest.mark.parametrize("state", [[30.0, 25.0, 20.0, 10.0], [80.0, 5.0, -3.0, -30.0, 0.008]])
def test_jacobian_matches_numerical_and_is_padded(sensor, state):
    x = np.array(state)
    H = sensor.H_jac(x)
    assert H.shape == (sensor.m, len(x))
    np.testing.assert_allclose(H, numerical_jacobian(sensor.h, x), atol=1e-8)
    np.testing.assert_array_equal(H[:, 2:], 0.0)  # sensors only see position


NOISELESS = [
    RangeBearingRadar([-100.0, 0.0], range_std=0.0, bearing_std=0.0),
    PinholeCamera([-50.0, 1.0], yaw=np.radians(20), focal_px=1000.0, pixel_std=0.0, width_px=600.0),
    GnssPosition(std=0.0, every=10),
]


@pytest.mark.parametrize("sensor", NOISELESS, ids=lambda s: s.name)
def test_noise_free_series_equals_h(sensor):
    # The track leaves the camera's field of view part-way, so NaN masking is covered too.
    truth = np.array([[10.0 + 5 * k, 20.0 + 5 * k, 0.0, 0.0] for k in range(25)])
    z = sensor.measure_series(truth, np.random.default_rng(0))
    for k, x in enumerate(truth):
        if k % sensor.every == 0 and sensor.visible(x):
            np.testing.assert_allclose(z[k], sensor.h(x), atol=1e-12)
        else:
            assert np.isnan(z[k]).all()


def test_every_masks_off_rate_steps():
    truth = np.zeros((25, 4))
    z = GNSS.measure_series(truth, np.random.default_rng(0))
    measured = ~np.isnan(z).any(axis=1)
    np.testing.assert_array_equal(np.flatnonzero(measured), [0, 10, 20])


def test_noise_has_requested_std():
    truth = np.tile([20.0, 30.0, 0.0, 0.0], (20000, 1))
    z = RADAR.measure_series(truth, np.random.default_rng(1))
    np.testing.assert_allclose((z - RADAR.h(truth[0])).std(axis=0), RADAR.noise_std, rtol=0.03)
    np.testing.assert_allclose(RADAR.R, np.diag(RADAR.noise_std**2))


def test_radar_wraps_bearing_and_inverts():
    behind = RangeBearingRadar([0.0, 0.0], range_std=0.1, bearing_std=0.5)
    truth = np.tile([-50.0, 1e-3, 0.0, 0.0], (2000, 1))  # bearing just below pi
    z = behind.measure_series(truth, np.random.default_rng(2))
    assert np.all(np.abs(z[:, 1]) <= np.pi)
    assert (z[:, 1] < 0).any()  # some noisy bearings wrapped to -pi side
    np.testing.assert_allclose(
        behind.residual(np.array([1.0, -np.pi + 0.1]), np.array([1.0, np.pi - 0.1])),
        [0.0, 0.2],
    )

    x = np.array([30.0, 25.0, 0.0, 0.0])
    np.testing.assert_allclose(RADAR.to_position(RADAR.h(x)), x[:2])


def test_camera_geometry():
    cam = PinholeCamera([0.0, 0.0], yaw=0.0, focal_px=1000.0, pixel_std=1.0, width_px=1000.0)
    assert cam.h(np.array([10.0, 0.0]))[0] == 0.0  # on the optical axis
    assert cam.h(np.array([10.0, 1.0]))[0] == pytest.approx(100.0)  # left of the axis is positive
    assert cam.h(np.array([10.0, -1.0]))[0] == pytest.approx(-100.0)
    assert cam.visible(np.array([10.0, 4.0]))  # |u| = 400 <= 500
    assert not cam.visible(np.array([10.0, 6.0]))  # outside the image
    assert not cam.visible(np.array([-10.0, 0.0]))  # behind the camera
    assert cam.half_fov == pytest.approx(np.arctan(0.5))

    # Rotating the camera and the target together leaves u unchanged.
    rot = PinholeCamera([0.0, 0.0], yaw=np.radians(70), focal_px=1000.0, pixel_std=1.0, width_px=1000.0)
    c, s = np.cos(np.radians(70)), np.sin(np.radians(70))
    np.testing.assert_allclose(rot.h(np.array([10 * c - s, 10 * s + c])), [100.0])


def test_every_must_be_positive():
    with pytest.raises(ValueError, match="every"):
        GnssPosition(1.0, every=0)
