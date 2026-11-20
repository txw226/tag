import numpy as np
import pytest

from tag.filtering.sims.projectile import G_EARTH, launch_state, make_filter, simulate

DT = 0.05
ACCEL_STD = 0.5
MEAS_STD = 2.0
VEL_STD = 5.0


def rmse(err):
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def test_noise_free_trajectory_matches_closed_form():
    x0 = launch_state(30.0, 45.0)
    run = simulate(x0, DT, accel_noise_std=0.0, meas_noise_std=0.0, rng=np.random.default_rng(0))
    t = run.t
    np.testing.assert_allclose(run.truth[:, 0], x0[2] * t)
    np.testing.assert_allclose(run.truth[:, 1], x0[3] * t - 0.5 * G_EARTH * t**2, atol=1e-9)
    assert np.all(run.truth[:, 1] >= 0)
    # Lands at the analytic flight time 2 vy / g.
    assert abs(t[-1] - 2 * x0[3] / G_EARTH) < DT


def test_filter_beats_raw_measurements():
    rng = np.random.default_rng(42)
    x0 = launch_state(30.0, 45.0)
    run = simulate(x0, DT, ACCEL_STD, MEAS_STD, rng)

    kf = make_filter(run, velocity_guess=x0[2:], velocity_std=VEL_STD)
    res = kf.filter(run.measurements[1:], run.model.u)

    truth = run.truth[1:]
    meas_rmse = rmse(run.measurements[1:] - truth[:, :2])
    kf_rmse = rmse(res.x_post[:, :2] - truth[:, :2])
    assert kf_rmse < 0.6 * meas_rmse

    # Velocity is never measured but should be inferred well after ~1 s.
    settled = run.t[1:] > 1.0
    assert rmse(res.x_post[settled, 2:] - truth[settled, 2:]) < 1.5


def pooled_nees(q_scale: float = 1.0, runs: int = 50, seed: int = 7) -> float:
    """Mean NEES over a Monte Carlo set; equals the state dimension (4) when consistent.

    Uses strong gusts (2 m/s^2) so the process noise actually matters over a
    ~4 s flight. `q_scale` mis-tunes the filter's Q to check the test's power.
    """
    accel_std = 2.0
    rng = np.random.default_rng(seed)
    v_nominal = launch_state(30.0, 45.0)[2:]

    nees = []
    for _ in range(runs):
        # Draw the true launch velocity from the filter's own prior.
        x0 = np.concatenate([[0.0, 0.0], v_nominal + VEL_STD * rng.standard_normal(2)])
        run = simulate(x0, DT, accel_std, MEAS_STD, rng)
        kf = make_filter(run, velocity_guess=v_nominal, velocity_std=VEL_STD)
        kf.Q = q_scale * kf.Q
        res = kf.filter(run.measurements[1:], run.model.u)
        err = res.x_post - run.truth[1:]
        # NEES per step k: e_kᵀ P_k⁻¹ e_k ("ki,kij,kj->k" = row vector · matrix · column vector, for every k).
        nees.append(np.einsum("ki,kij,kj->k", err, np.linalg.inv(res.P_post), err))
    return float(np.concatenate(nees).mean())


# Bounds chosen from a 20-seed sweep: correct Q gave 3.59-4.42, Q/4 gave
# 5.72-6.89 and 4Q gave 2.69-3.30.
NEES_BOUNDS = (3.45, 4.55)


def test_nees_is_consistent():
    lo, hi = NEES_BOUNDS
    assert lo < pooled_nees() < hi


@pytest.mark.parametrize("q_scale", [0.25, 4.0])
def test_nees_detects_mistuned_q(q_scale):
    lo, hi = NEES_BOUNDS
    assert not lo < pooled_nees(q_scale) < hi
