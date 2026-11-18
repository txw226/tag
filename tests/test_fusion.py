"""Multi-sensor fusion: GNSS + a pair of narrow-view cameras tracking a drag projectile.

Each camera sees only part of the flight (angle-only while it is alone), and
GNSS is slow with a 1 s outage, so neither source is good on its own.
"""

import numpy as np
import pytest

from tooth.sensors import GnssPosition, PinholeCamera
from tooth.sims.drag_projectile import (
    launch_state,
    make_fusion_filter,
    make_model,
    simulate_trajectory,
)

DT = 0.05
POS_STD = 0.5  # m, launch-point uncertainty
VEL_STD = 5.0  # m/s, launch-velocity uncertainty
GNSS_OUTAGE = (2.0, 3.0)  # s


def sensors():
    gnss = GnssPosition(std=2.0, every=10)
    cam_l = PinholeCamera(
        [-50.0, 1.0],
        np.radians(18),
        focal_px=1000.0,
        pixel_std=3.0,
        width_px=300.0,
        name="cam L",
    )
    cam_r = PinholeCamera(
        [190.0, 1.0],
        np.radians(162),
        focal_px=1000.0,
        pixel_std=3.0,
        width_px=300.0,
        name="cam R",
    )
    return gnss, cam_l, cam_r


def prior():
    v_nominal = launch_state(50.0, 45.0)[2:]
    x0 = np.array([0.0, 2.0, *v_nominal])
    P0 = np.diag([POS_STD**2, POS_STD**2, VEL_STD**2, VEL_STD**2])
    return x0, P0


def scenario(rng, accel_std):
    """True launch drawn from the filter's prior; returns (t, truth, {name: (sensor, zs)})."""
    x0, P0 = prior()
    x_true = x0 + np.sqrt(np.diag(P0)) * rng.standard_normal(4)
    t, truth = simulate_trajectory(x_true, DT, accel_std, rng)
    streams = {}
    for s in sensors():
        zs = s.measure_series(truth, rng)
        if isinstance(s, GnssPosition):
            zs[(t >= GNSS_OUTAGE[0]) & (t < GNSS_OUTAGE[1])] = np.nan
        streams[s.name] = (s, zs)
    return t, truth, streams


def run_filter(streams, names, accel_std, q_scale=1.0):
    x0, P0 = prior()
    ekf = make_fusion_filter(make_model(DT, accel_std), x0, P0)
    ekf.Q = q_scale * ekf.Q
    return ekf.filter_sensors([(streams[n][0], streams[n][1][1:]) for n in names])


CONFIGS = {
    "GNSS": ["GNSS"],
    "stereo": ["cam L", "cam R"],
    "fused": ["GNSS", "cam L", "cam R"],
}


def test_fusion_beats_each_source_alone():
    # Over 100 seeds the pooled RMSE was GNSS 2.39 m, stereo 1.72 m, fused 0.88 m.
    # Stereo-only errors are heavy-tailed (angle-only gaps), so across 20-seed
    # blocks fused/stereo ranged 0.34-0.85 and fused/GNSS stayed <= 0.40; the
    # thresholds below hold for every block.
    sq_err = {name: [] for name in CONFIGS}
    for seed in range(20):
        _t, truth, streams = scenario(np.random.default_rng(seed), accel_std=0.5)
        for name, sensor_names in CONFIGS.items():
            err = run_filter(streams, sensor_names, 0.5).x_post[:, :2] - truth[1:, :2]
            sq_err[name].append(np.sum(err**2, axis=1))
    rmse = {name: float(np.sqrt(np.concatenate(v).mean())) for name, v in sq_err.items()}
    assert rmse["fused"] < 0.5 * rmse["GNSS"]
    assert rmse["fused"] < rmse["stereo"]


def pooled_nees(q_scale: float = 1.0, runs: int = 50, seed: int = 7) -> float:
    """Mean NEES of the fused filter over a Monte Carlo set; ≈ 4 when consistent."""
    accel_std = 2.0
    rng = np.random.default_rng(seed)
    nees = []
    for _ in range(runs):
        _t, truth, streams = scenario(rng, accel_std)
        res = run_filter(streams, CONFIGS["fused"], accel_std, q_scale)
        err = res.x_post - truth[1:]
        # NEES per step k: e_kᵀ P_k⁻¹ e_k ("ki,kij,kj->k" = row vector · matrix · column vector, for every k).
        nees.append(np.einsum("ki,kij,kj->k", err, np.linalg.inv(res.P_post), err))
    return float(np.concatenate(nees).mean())


# Bounds chosen from a 20-seed sweep: correct Q gave 3.66-4.23, Q/4 gave
# 8.13-9.34 and 4Q gave 2.39-2.76. Keep the cameras well away from the flight:
# a target passing within a few metres of a camera makes its projection so
# nonlinear that the EKF update becomes overconfident (NEES spikes > 100).
NEES_BOUNDS = (3.45, 4.55)


def test_fused_nees_is_consistent():
    lo, hi = NEES_BOUNDS
    assert lo < pooled_nees() < hi


@pytest.mark.parametrize("q_scale", [0.25, 4.0])
def test_fused_nees_detects_mistuned_q(q_scale):
    lo, hi = NEES_BOUNDS
    assert not lo < pooled_nees(q_scale) < hi


# Regression ("golden") test: answers "did the output change?", not "is it
# right?". Regenerate deliberately after an intended numerical change and say
# why in the commit (see the note in test_drag_projectile.py).
GOLDEN_STEPS = 100
GOLDEN_X = [
    116.10448852984813,
    0.3124615620545579,
    13.965153053294346,
    -21.13281551980201,
]
GOLDEN_P = [
    [
        0.11731592104571772,
        -0.014635588836973756,
        0.031675498358460744,
        -0.00082966927602072,
    ],
    [
        -0.014635588836973756,
        0.024998149978732585,
        -0.00044184770602905525,
        0.014141739340515611,
    ],
    [
        0.03167549835846073,
        -0.0004418477060290552,
        0.01981626999541621,
        0.001788701453533504,
    ],
    [
        -0.0008296692760207196,
        0.014141739340515611,
        0.001788701453533504,
        0.014650042926720555,
    ],
]


def test_golden_fused_run():
    t, _truth, streams = scenario(np.random.default_rng(2024), accel_std=0.5)
    res = run_filter(streams, CONFIGS["fused"], 0.5)
    assert len(t) == GOLDEN_STEPS
    np.testing.assert_allclose(res.x_post[-1], GOLDEN_X, rtol=1e-9)
    np.testing.assert_allclose(res.P_post[-1], GOLDEN_P, rtol=1e-9)
