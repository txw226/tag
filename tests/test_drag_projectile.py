import numpy as np
import pytest

from tooth.kalman import numerical_jacobian
from tooth.sims.drag_projectile import (
    G_EARTH,
    launch_state,
    make_filter,
    make_model,
    simulate,
)

DT = 0.05
ACCEL_STD = 0.5
RANGE_STD = 2.0
BEARING_STD = 0.01
VEL_STD = 5.0
SPEED, ANGLE = 50.0, 45.0


def rmse(err):
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def noise_free(drag):
    return simulate(
        launch_state(SPEED, ANGLE),
        DT,
        0.0,
        0.0,
        0.0,
        np.random.default_rng(0),
        drag=drag,
    )


def test_zero_drag_matches_vacuum_closed_form():
    run = noise_free(drag=0.0)
    x0, t = launch_state(SPEED, ANGLE), run.t
    np.testing.assert_allclose(run.truth[:, 0], x0[2] * t, atol=1e-9)
    np.testing.assert_allclose(run.truth[:, 1], x0[3] * t - 0.5 * G_EARTH * t**2, atol=1e-9)
    assert abs(t[-1] - 2 * x0[3] / G_EARTH) < DT


def test_drag_shortens_flight_and_dissipates_energy():
    vacuum, dragged = noise_free(0.0), noise_free(0.008)
    assert dragged.truth[-1, 0] < 0.7 * vacuum.truth[-1, 0]
    assert dragged.truth[:, 1].max() < vacuum.truth[:, 1].max()

    s = dragged.truth
    energy = 0.5 * np.sum(s[:, 2:] ** 2, axis=1) + G_EARTH * s[:, 1]
    assert np.all(np.diff(energy) < 0)


def test_noise_free_measurements_invert_to_truth():
    run = noise_free(0.008)
    pos = run.model.radar.to_position(run.measurements)
    np.testing.assert_allclose(pos, run.truth[:, :2], atol=1e-9)


@pytest.mark.parametrize("state", [[10.0, 20.0, 25.0, 15.0], [80.0, 5.0, -3.0, -30.0], [0.0, 0.0, 0.0, 0.0]])
def test_analytic_jacobians_match_numerical(state):
    m = noise_free(0.008).model
    x = np.array(state)
    np.testing.assert_allclose(m.F_jac(x), numerical_jacobian(m.f, x), atol=1e-8)
    np.testing.assert_allclose(m.radar.H_jac(x), numerical_jacobian(m.radar.h, x), atol=1e-8)


def test_ekf_beats_raw_measurements():
    rng = np.random.default_rng(42)
    x0 = launch_state(SPEED, ANGLE)
    run = simulate(x0, DT, ACCEL_STD, RANGE_STD, BEARING_STD, rng)

    ekf = make_filter(run, velocity_guess=x0[2:], velocity_std=VEL_STD)
    res = ekf.filter(run.measurements[1:])

    truth = run.truth[1:]
    meas_pos = run.model.radar.to_position(run.measurements[1:])
    assert rmse(res.x_post[:, :2] - truth[:, :2]) < 0.5 * rmse(meas_pos - truth[:, :2])

    settled = run.t[1:] > 1.0
    assert rmse(res.x_post[settled, 2:] - truth[settled, 2:]) < 1.0


def pooled_nees(q_scale: float = 1.0, runs: int = 50, seed: int = 7) -> float:
    """Mean NEES over a Monte Carlo set; ≈ state dimension (4) when consistent.

    Strong gusts (2 m/s^2), as in test_projectile.py, so mis-tuned Q is visible.
    """
    accel_std = 2.0
    rng = np.random.default_rng(seed)
    v_nominal = launch_state(SPEED, ANGLE)[2:]

    nees = []
    for _ in range(runs):
        x0 = np.concatenate([[0.0, 0.0], v_nominal + VEL_STD * rng.standard_normal(2)])
        run = simulate(x0, DT, accel_std, RANGE_STD, BEARING_STD, rng)
        ekf = make_filter(run, velocity_guess=v_nominal, velocity_std=VEL_STD)
        ekf.Q = q_scale * ekf.Q
        res = ekf.filter(run.measurements[1:])
        err = res.x_post - run.truth[1:]
        nees.append(np.einsum("ki,kij,kj->k", err, np.linalg.inv(res.P_post), err))
    return float(np.concatenate(nees).mean())


# Bounds chosen from a 20-seed sweep: correct Q gave 3.68-4.25, Q/4 gave
# 6.85-8.44 and 4Q gave 2.60-2.94.
NEES_BOUNDS = (3.45, 4.55)


def test_nees_is_consistent():
    lo, hi = NEES_BOUNDS
    assert lo < pooled_nees() < hi


@pytest.mark.parametrize("q_scale", [0.25, 4.0])
def test_nees_detects_mistuned_q(q_scale):
    lo, hi = NEES_BOUNDS
    assert not lo < pooled_nees(q_scale) < hi


# --- Regression ("golden") tests -------------------------------------------
# These answer "did the output change?", not "is it right?" (the tests above
# cover correctness). They pin a short fixed-seed run so that small unintended
# drift, e.g. a slightly wrong Q scaling or RK4 coefficient that still passes the
# physical tests, gets caught. After an *intended* change to the sim or filter
# numerics, regenerate the values with the same setup and say why in the commit.
# The sim and filter are pinned separately so a failure shows which one moved.
# (NumPy does not promise identical Generator streams across feature releases;
# a NumPy upgrade can also be a legitimate reason to regenerate.)

GOLDEN_RTOL = 1e-9


def golden_run():
    rng = np.random.default_rng(2024)
    x0 = launch_state(SPEED, ANGLE)
    run = simulate(x0, DT, ACCEL_STD, RANGE_STD, BEARING_STD, rng, t_max=1.0)
    return run, x0


def test_golden_simulation():
    run, _ = golden_run()
    assert len(run.t) == 21
    np.testing.assert_allclose(
        run.truth[-1],
        [29.96096601906115, 25.557234332497217, 25.777821594534863, 17.34026736444289],
        rtol=GOLDEN_RTOL,
    )
    np.testing.assert_allclose(
        run.measurements[-1],
        [131.92844584987535, 0.20208214786021106],
        rtol=GOLDEN_RTOL,
    )


def test_golden_ekf():
    run, x0 = golden_run()
    ekf = make_filter(run, velocity_guess=x0[2:] + [1.0, -1.0], velocity_std=VEL_STD)
    res = ekf.filter(run.measurements[1:])
    np.testing.assert_allclose(
        res.x_post[-1],
        [30.47189397723538, 25.90217710522656, 26.919173249536467, 17.508367219458894],
        rtol=GOLDEN_RTOL,
    )
    P_expected = np.array(
        [
            [
                5.6170465173478290e-01,
                3.2235492956264544e-02,
                6.3168510064340022e-01,
                -1.3483554490996419e-02,
            ],
            [
                3.2235492956264544e-02,
                2.4743189516305064e-01,
                5.4024346394836512e-04,
                2.8088658530473037e-01,
            ],
            [
                6.3168510064340022e-01,
                5.4024346394836512e-04,
                1.0751977658370901e00,
                -7.3390987982936348e-02,
            ],
            [
                -1.3483554490996419e-02,
                2.8088658530473037e-01,
                -7.3390987982936348e-02,
                4.3944589502397136e-01,
            ],
        ]
    )
    np.testing.assert_allclose(res.P_post[-1], P_expected, rtol=GOLDEN_RTOL)


def test_model_without_radar_raises_clearly():
    with pytest.raises(ValueError, match="model has no radar"):
        make_model(DT, ACCEL_STD).require_radar()
