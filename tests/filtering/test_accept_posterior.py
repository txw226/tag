"""The accept_posterior hook (rejected updates) and the shared lock-out counter."""

import numpy as np
import pytest

from tag.filtering.kalman import ExtendedKalmanFilter, LinearKalmanFilter, LockOutError, UpdateOutcome


def reject_calls(*indices: int):
    """A hook that rejects its calls with the given 0-based indices and accepts the rest."""
    calls = iter(range(10**9))
    chosen = set(indices)

    def accept(x, P):
        return next(calls) not in chosen

    return accept


def constant_velocity_lkf(**kwargs) -> LinearKalmanFilter:
    dt = 0.5
    return LinearKalmanFilter(
        F=[[1.0, dt], [0.0, 1.0]],
        H=[[1.0, 0.0]],
        Q=0.1 * np.eye(2),
        R=[[0.5]],
        x0=[0.0, 1.0],
        P0=np.eye(2),
        **kwargs,
    )


def constant_velocity_ekf(**kwargs) -> ExtendedKalmanFilter:
    lkf = constant_velocity_lkf()
    return ExtendedKalmanFilter(
        f=lambda x, u: lkf.F @ x,
        h=lambda x: lkf.H @ x,
        Q=lkf.Q,
        R=lkf.R,
        x0=lkf.x,
        P0=lkf.P,
        **kwargs,
    )


FILTERS = {"lkf": constant_velocity_lkf, "ekf": constant_velocity_ekf}


def measurements(N: int = 8) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (0.5 * np.arange(1, N + 1) + 0.7 * rng.standard_normal(N)).reshape(-1, 1)


@pytest.mark.parametrize("make", FILTERS.values(), ids=FILTERS.keys())
def test_rejected_steps_keep_the_prior_and_are_marked(make):
    zs = measurements()
    res = make(accept_posterior=reject_calls(2, 5)).filter(zs)

    np.testing.assert_array_equal(np.flatnonzero(res.rejected), [2, 5])
    for k in (2, 5):
        np.testing.assert_array_equal(res.x_post[k], res.x_prior[k])
        np.testing.assert_array_equal(res.P_post[k], res.P_prior[k])
    for k in (0, 1, 3, 4, 6, 7):
        assert not np.array_equal(res.x_post[k], res.x_prior[k])


@pytest.mark.parametrize("make", FILTERS.values(), ids=FILTERS.keys())
def test_no_hook_and_an_accepting_hook_match_bit_for_bit(make):
    zs = measurements()
    zs[4] = np.nan
    ref = make().filter(zs)
    res = make(accept_posterior=lambda x, P: True).filter(zs)
    for name in ("x_prior", "P_prior", "x_post", "P_post", "skipped", "rejected", "unabsorbed_run"):
        np.testing.assert_array_equal(getattr(res, name), getattr(ref, name))
    assert not ref.rejected.any()
    assert not ref.unabsorbed_run.any()


# Ten steps with a dropout at step 3. The hook is called only for available measurements
# (steps 0-2, 4-9 are calls 0-8), so rejecting calls 1, 2, 3 and 6 rejects steps 1, 2, 4 and 7.
DROPOUT_STEP = 3
LOCKOUT_CALLS = (1, 2, 3, 6)


def dropout_measurements() -> np.ndarray:
    zs = measurements(10)
    zs[DROPOUT_STEP] = np.nan
    return zs


@pytest.mark.parametrize("make", FILTERS.values(), ids=FILTERS.keys())
def test_rejected_and_skipped_steps_are_distinguished(make):
    res = make(accept_posterior=reject_calls(*LOCKOUT_CALLS)).filter(dropout_measurements())

    assert res.skipped.shape == res.rejected.shape == (10, 1)
    np.testing.assert_array_equal(np.flatnonzero(res.skipped), [3])
    np.testing.assert_array_equal(np.flatnonzero(res.rejected), [1, 2, 4, 7])


@pytest.mark.parametrize("make", FILTERS.values(), ids=FILTERS.keys())
def test_counter_reports_unabsorbed_runs_and_a_dropout_does_not_break_them(make):
    kf = make(accept_posterior=reject_calls(*LOCKOUT_CALLS))
    res = kf.filter(dropout_measurements())

    np.testing.assert_array_equal(res.unabsorbed_run, [0, 1, 2, 2, 3, 0, 0, 1, 0, 0])
    assert kf.consecutive_unabsorbed == 0


@pytest.mark.parametrize("make", FILTERS.values(), ids=FILTERS.keys())
def test_opt_in_limit_raises_once_exceeded(make):
    kf = make(accept_posterior=reject_calls(*LOCKOUT_CALLS), max_consecutive_unabsorbed=2)
    with pytest.raises(LockOutError, match=r"lock-out at step 4: 3 consecutive .* max_consecutive_unabsorbed=2"):
        kf.filter(dropout_measurements())

    # A limit equal to the longest run is not exceeded.
    make(accept_posterior=reject_calls(*LOCKOUT_CALLS), max_consecutive_unabsorbed=3).filter(dropout_measurements())


@pytest.mark.parametrize("make", FILTERS.values(), ids=FILTERS.keys())
def test_default_never_raises(make):
    res = make(accept_posterior=lambda x, P: False).filter(measurements(50))
    np.testing.assert_array_equal(res.unabsorbed_run, np.arange(1, 51))
    np.testing.assert_array_equal(res.x_post, res.x_prior)


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="max_consecutive_unabsorbed must be >= 0"):
        constant_velocity_lkf(max_consecutive_unabsorbed=-1)


def test_step_reports_the_outcome_and_counts_towards_lock_out():
    kf = constant_velocity_lkf(accept_posterior=reject_calls(0), max_consecutive_unabsorbed=1)
    assert kf.step([0.4]) is UpdateOutcome.REJECTED
    assert kf.consecutive_unabsorbed == 1
    assert kf.step(None) is UpdateOutcome.SKIPPED
    assert kf.consecutive_unabsorbed == 1
    assert kf.step([0.9]) is UpdateOutcome.ABSORBED
    assert kf.consecutive_unabsorbed == 0


class PositionSensor:
    """Measures state index 0 with noise variance ``var``."""

    residual = None

    def __init__(self, var: float) -> None:
        self.R = np.array([[var]])

    def h(self, x):
        return x[:1]

    def H_jac(self, x):
        return np.array([[1.0, 0.0]])


def fusion_streams() -> list[tuple[PositionSensor, np.ndarray]]:
    rng = np.random.default_rng(8)
    return [(PositionSensor(var), measurements(6) + rng.normal(0, 0.3, (6, 1))) for var in (0.5, 2.0)]


def test_fusion_rejecting_one_sensor_keeps_the_others():
    # Two streams, all rows present: the hook sees calls 2k (sensor A) and 2k + 1 (sensor B) at step k.
    # Reject B at step 2 (call 5) and both at step 4 (calls 8, 9).
    streams = fusion_streams()
    ekf = constant_velocity_ekf(accept_posterior=reject_calls(5, 8, 9))
    res = ekf.filter_sensors(streams)

    # The rejections must act exactly like the same measurements being missing.
    (a, zs_a), (b, zs_b) = streams
    zs_a, zs_b = zs_a.copy(), zs_b.copy()
    zs_b[2] = zs_a[4] = zs_b[4] = np.nan
    ref = constant_velocity_ekf().filter_sensors([(a, zs_a), (b, zs_b)])
    for name in ("x_prior", "P_prior", "x_post", "P_post"):
        np.testing.assert_array_equal(getattr(res, name), getattr(ref, name))

    assert res.rejected.shape == (6, 2)
    np.testing.assert_array_equal(np.argwhere(res.rejected), [[2, 1], [4, 0], [4, 1]])
    assert not res.skipped.any()
    np.testing.assert_array_equal(ref.skipped, res.rejected)
    # Sensor A was absorbed at step 2, so only step 4 counts towards lock-out.
    np.testing.assert_array_equal(res.unabsorbed_run, [0, 0, 0, 0, 1, 0])


def test_fusion_limit_raises():
    ekf = constant_velocity_ekf(accept_posterior=lambda x, P: False, max_consecutive_unabsorbed=3)
    with pytest.raises(LockOutError, match="lock-out at step 3"):
        ekf.filter_sensors(fusion_streams())
