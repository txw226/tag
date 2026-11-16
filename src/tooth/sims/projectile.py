"""2D projectile under gravity with random acceleration disturbances.

State: [x, y, vx, vy] (m, m, m/s, m/s). Gravity enters as a known control
input u = [-g]. Random accelerations (e.g. wind gusts) are held constant over
each timestep, so the discrete model below is exact, not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tooth.kalman import LinearKalmanFilter

G_EARTH = 9.81  # m/s^2


@dataclass
class ProjectileModel:
    F: NDArray  # (4, 4)
    B: NDArray  # (4, 1)
    H: NDArray  # (2, 4)
    Q: NDArray  # (4, 4)
    R: NDArray  # (2, 2)
    Gamma: NDArray  # (4, 2) maps acceleration noise into the state
    u: NDArray  # (1,) gravity control input


@dataclass
class ProjectileRun:
    t: NDArray  # (N,)
    truth: NDArray  # (N, 4)
    measurements: NDArray  # (N, 2) noisy [x, y]
    model: ProjectileModel


def model_matrices(
    dt: float,
    accel_noise_std: float,
    meas_noise_std: float,
    g: float = G_EARTH,
) -> ProjectileModel:
    """Discrete-time linear model for a projectile observed by position."""
    h = 0.5 * dt**2
    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    B = np.array([[0.0], [h], [0.0], [dt]])
    Gamma = np.array([[h, 0], [0, h], [dt, 0], [0, dt]])
    Q = accel_noise_std**2 * Gamma @ Gamma.T
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    R = meas_noise_std**2 * np.eye(2)
    return ProjectileModel(F=F, B=B, H=H, Q=Q, R=R, Gamma=Gamma, u=np.array([-g]))


def simulate(
    x0: ArrayLike,
    dt: float,
    accel_noise_std: float,
    meas_noise_std: float,
    rng: np.random.Generator,
    t_max: float | None = None,
    g: float = G_EARTH,
) -> ProjectileRun:
    """Simulate the projectile until it hits the ground (y < 0) or t_max.

    Returns the true states and noisy position measurements at every step,
    starting at t = 0.
    """
    model = model_matrices(dt, accel_noise_std, meas_noise_std, g)
    x = np.asarray(x0, dtype=float).reshape(4)
    if x[1] < 0:
        raise ValueError("projectile must start at or above the ground (y >= 0)")

    max_steps = int(np.ceil(t_max / dt)) + 1 if t_max is not None else 1_000_000
    states = [x]
    for _ in range(max_steps - 1):
        a = rng.normal(0.0, accel_noise_std, size=2)
        x = model.F @ x + model.B @ model.u + model.Gamma @ a
        if x[1] < 0:
            break
        states.append(x)

    truth = np.array(states)
    noise = rng.normal(0.0, meas_noise_std, size=(len(truth), 2))
    return ProjectileRun(
        t=dt * np.arange(len(truth)),
        truth=truth,
        measurements=truth[:, :2] + noise,
        model=model,
    )


def make_filter(
    run: ProjectileRun,
    velocity_guess: ArrayLike,
    velocity_std: float,
) -> LinearKalmanFilter:
    """Kalman filter initialized from the run's first measurement.

    Position starts at the first measurement (covariance R); velocity starts at
    `velocity_guess` with standard deviation `velocity_std` per axis. Feed it
    ``run.measurements[1:]`` afterwards.
    """
    m = run.model
    x0 = np.concatenate([run.measurements[0], np.asarray(velocity_guess, dtype=float)])
    P0 = np.zeros((4, 4))
    P0[:2, :2] = m.R
    P0[2:, 2:] = velocity_std**2 * np.eye(2)
    return LinearKalmanFilter(F=m.F, H=m.H, Q=m.Q, R=m.R, x0=x0, P0=P0, B=m.B)


def launch_state(speed: float, angle_deg: float, height: float = 0.0) -> NDArray:
    """Initial state for a launch at `speed` m/s and `angle_deg` above horizontal."""
    th = np.radians(angle_deg)
    return np.array([0.0, height, speed * np.cos(th), speed * np.sin(th)])
