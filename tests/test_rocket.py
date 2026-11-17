import numpy as np
import pytest

from tooth.kalman import numerical_jacobian
from tooth.sensors import GnssPosition, RangeBearingRadar
from tooth.sims.projectile import G_EARTH
from tooth.sims.rocket import Motor, air_density, make_filter, rail_exit_state, simulate

DT = 0.1
ACCEL_STD = 0.5
POS_STD = 1.0  # m, rail-exit position uncertainty
VEL_STD = 1.0  # m/s, rail-exit velocity uncertainty
CD_NOMINAL = 0.45
CD_STD = 0.10


def sensors():
    return RangeBearingRadar([-2000.0, 0.0], range_std=5.0, bearing_std=0.001), GnssPosition(std=3.0, every=2)


def track(run, rng, cd_guess=CD_NOMINAL, cd_std=CD_STD, q_scale=1.0):
    """Measure ``run`` with radar + GNSS and filter it; returns (result, truth with Cd column)."""
    streams = [(s, s.measure_series(run.truth, rng)[1:]) for s in sensors()]
    ekf = make_filter(run.model, rail_exit_state(), POS_STD, VEL_STD, cd_guess, cd_std)
    ekf.Q = q_scale * ekf.Q
    res = ekf.filter_sensors(streams, us=run.controls)
    truth5 = np.column_stack([run.truth[1:], np.full(len(run.truth) - 1, run.cd)])
    return res, truth5


def random_run(rng, accel_std):
    """Rail-exit state and Cd drawn from the filter's prior."""
    x0 = rail_exit_state() + np.array([POS_STD, POS_STD, VEL_STD, VEL_STD]) * rng.standard_normal(4)
    cd = CD_NOMINAL + CD_STD * rng.standard_normal()
    return simulate(x0, cd, DT, accel_std, rng)


# --- physics -----------------------------------------------------------------


def test_rocket_equation():
    """No gravity or drag, thrust along v: speed gain = v_e ln(m0 / m_burnout)."""
    motor = Motor()
    x0 = rail_exit_state()
    run = simulate(x0, 0.0, DT, 0.0, np.random.default_rng(0), motor=motor, g=0.0, t_max=6.0)
    gain = np.linalg.norm(run.truth[-1, 2:]) - np.linalg.norm(x0[2:])
    assert gain == pytest.approx(motor.exhaust_velocity * np.log(motor.mass0 / motor.mass_burnout), rel=1e-10)


def test_unpowered_dragless_flight_is_ballistic():
    x0 = rail_exit_state()
    run = simulate(x0, 0.0, DT, 0.0, np.random.default_rng(0), motor=Motor(thrust=0.0))
    t = run.t
    np.testing.assert_allclose(run.truth[:, 0], x0[0] + x0[2] * t, atol=1e-9)
    np.testing.assert_allclose(run.truth[:, 1], x0[1] + x0[3] * t - 0.5 * G_EARTH * t**2, atol=1e-9)
    assert abs(t[-1] - x0[3] / G_EARTH) < DT  # stops at apogee


def test_drag_lowers_apogee():
    x0 = rail_exit_state()
    apogee = {cd: simulate(x0, cd, DT, 0.0, np.random.default_rng(0)).truth[-1, 1] for cd in (0.0, 0.3, 0.6)}
    assert apogee[0.0] > apogee[0.3] > apogee[0.6]


def test_air_density():
    assert air_density(0.0) == pytest.approx(1.225)
    assert air_density(8500.0) == pytest.approx(1.225 / np.e)


def test_motor_controls_conserve_impulse_across_burnout():
    motor = Motor(burn_time=4.03)
    assert motor.mean_thrust(4.0, 4.1) == pytest.approx(0.3 * motor.thrust)
    run = simulate(rail_exit_state(), CD_NOMINAL, DT, 0.0, np.random.default_rng(0), motor=motor)
    assert run.controls.shape == (len(run.t) - 1, 2)
    impulse = np.sum(run.controls[:, 0]) * DT
    assert impulse == pytest.approx(motor.thrust * motor.burn_time)
    np.testing.assert_allclose(run.controls[-1], [0.0, motor.mass_burnout])


@pytest.mark.parametrize("step", [5, 39, 40, 150])  # burning, last burning step, burnout step, coasting
def test_transition_jacobian_matches_numerical(step):
    run = simulate(rail_exit_state(), CD_NOMINAL, DT, 0.0, np.random.default_rng(0))
    m, u = run.model, run.controls[step]
    x = np.array([*run.truth[step], CD_NOMINAL])
    F = m.F_jac(x, u)
    F_num = numerical_jacobian(lambda s: m.f(s, u), x)
    np.testing.assert_allclose(F, F_num, rtol=1e-6, atol=1e-7)
    assert np.any(F[2:4, 4] != 0)  # Cd affects velocity...
    np.testing.assert_array_equal(F[4], [0, 0, 0, 0, 1])  # ...and stays constant


# --- drag estimation -----------------------------------------------------------


def test_cd_estimate_converges_and_beats_a_fixed_guess():
    rng = np.random.default_rng(42)
    run = simulate(rail_exit_state(), CD_NOMINAL, DT, ACCEL_STD, rng)
    meas_rng = np.random.default_rng(43)
    guess = CD_NOMINAL + 0.08  # starts wrong

    res, truth5 = track(run, meas_rng, cd_guess=guess)
    sigma_cd = np.sqrt(res.P_post[:, 4, 4])
    assert abs(res.x_post[-1, 4] - run.cd) < 2 * sigma_cd[-1]
    assert sigma_cd[-1] < CD_STD / 3

    fixed, _ = track(run, np.random.default_rng(43), cd_guess=guess, cd_std=0.0)

    def pos_rmse(r):
        return np.sqrt(np.mean(np.sum((r.x_post[:, :2] - truth5[:, :2]) ** 2, axis=1)))

    assert pos_rmse(res) < 0.5 * pos_rmse(fixed)


def test_cd_estimate_is_unbiased():
    """Final Cd error in units of its own sigma: mean ≈ 0, spread ≈ 1 over 20 rockets."""
    rng = np.random.default_rng(11)
    z = []
    for _ in range(20):
        run = random_run(rng, ACCEL_STD)
        res, _truth5 = track(run, rng)
        z.append((res.x_post[-1, 4] - run.cd) / np.sqrt(res.P_post[-1, 4, 4]))
    z = np.array(z)
    assert abs(z.mean()) < 3 / np.sqrt(len(z))  # 3 standard errors
    assert 0.6 < z.std() < 1.4


def pooled_nees(q_scale: float = 1.0, runs: int = 50, seed: int = 7) -> float:
    """Mean NEES over [x, y, vx, vy, Cd]; ≈ 5 when consistent."""
    rng = np.random.default_rng(seed)
    nees = []
    for _ in range(runs):
        run = random_run(rng, accel_std=2.0)
        res, truth5 = track(run, rng, q_scale=q_scale)
        err = res.x_post - truth5
        nees.append(np.einsum("ki,kij,kj->k", err, np.linalg.inv(res.P_post), err))
    return float(np.concatenate(nees).mean())


# Bounds chosen from a 20-seed sweep: correct Q gave 4.71-5.37, Q/4 gave
# 11.48-13.83 and 4Q gave 2.89-3.18.
NEES_BOUNDS = (4.4, 5.6)


def test_nees_is_consistent():
    lo, hi = NEES_BOUNDS
    assert lo < pooled_nees() < hi


@pytest.mark.parametrize("q_scale", [0.25, 4.0])
def test_nees_detects_mistuned_q(q_scale):
    lo, hi = NEES_BOUNDS
    assert not lo < pooled_nees(q_scale) < hi


# Regression ("golden") test: answers "did the output change?", not "is it
# right?". Regenerate deliberately after an intended numerical change and say
# why in the commit (see the note in test_drag_projectile.py).
GOLDEN_STEPS = 295
GOLDEN_X = [
    1433.0100373422215,
    4354.8691526913235,
    44.38450901275835,
    0.48759285771646788,
    0.31090161996001842,
]
GOLDEN_P = [
    [
        5.6879557562718208e-01,
        -1.7096341664686269e-02,
        1.5404836093303337e-01,
        -3.5508817135923593e-03,
        -1.4088977678313346e-05,
    ],
    [
        -1.7096341664686269e-02,
        5.6304724860711697e-01,
        -3.4856790663019095e-03,
        1.5422286676772023e-01,
        -9.2520434660399567e-06,
    ],
    [
        1.5404836093303337e-01,
        -3.4856790663019095e-03,
        8.6461621557417684e-02,
        -1.0147884250844516e-03,
        -6.4190050586786066e-06,
    ],
    [
        -3.5508817135923593e-03,
        1.5422286676772023e-01,
        -1.0147884250844516e-03,
        8.7086852655946734e-02,
        -1.0310792051491765e-06,
    ],
    [
        -1.4088977678313346e-05,
        -9.2520434660399567e-06,
        -6.4190050586786066e-06,
        -1.0310792051491765e-06,
        3.0690191069234597e-06,
    ],
]


def test_golden_run():
    rng = np.random.default_rng(2024)
    run = random_run(rng, ACCEL_STD)
    res, _ = track(run, rng)
    assert len(run.t) == GOLDEN_STEPS
    np.testing.assert_allclose(res.x_post[-1], GOLDEN_X, rtol=1e-9)
    np.testing.assert_allclose(res.P_post[-1], GOLDEN_P, rtol=1e-9)
