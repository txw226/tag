"""2D projectile with quadratic air drag, tracked by a range/bearing radar.

State: [x, y, vx, vy] (m, m, m/s, m/s). Continuous dynamics:

    dv/dt = -k |v| v + [0, -g] + a_gust

with drag constant k = rho * Cd * A / (2 * mass) (1/m). Gusts are random
accelerations held constant over each timestep, as in :mod:`projectile`.

:func:`simulate` adds a :class:`~tooth.sensors.RangeBearingRadar` measuring
z = [range, bearing]. Both the dynamics and the measurement are nonlinear,
so this sim pairs with :class:`~tooth.kalman.ExtendedKalmanFilter`. For other
or several sensors, use :func:`simulate_trajectory` for the truth and each
sensor's ``measure_series`` for its readings, then :func:`make_fusion_filter`.

The truth is integrated with RK4 over several substeps per timestep. The
filter model's ``f`` is a single RK4 step with no gust, and ``F_jac`` is the
exact Jacobian of that step. ``Q`` maps gust noise into the state with the
same ``Gamma`` as the drag-free model; under drag this is an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tooth.kalman import ExtendedKalmanFilter
from tooth.sensors import RangeBearingRadar
from tooth.sims.projectile import G_EARTH, launch_state, model_matrices

__all__ = [
    "DEFAULT_DRAG",
    "DragProjectileModel",
    "DragProjectileRun",
    "G_EARTH",
    "launch_state",
    "make_filter",
    "make_fusion_filter",
    "make_model",
    "simulate",
    "simulate_trajectory",
]

DEFAULT_DRAG = 0.008  # 1/m; terminal speed sqrt(g/k) ≈ 35 m/s
DEFAULT_RADAR = (-100.0, 0.0)  # m; off to the side so range never reaches 0


@dataclass
class DragProjectileModel:
    dt: float
    k: float  # drag constant (1/m)
    g: float
    Q: NDArray  # (4, 4)
    Gamma: NDArray  # (4, 2) maps acceleration noise into the state
    radar: RangeBearingRadar | None = None

    # --- dynamics -----------------------------------------------------------

    def deriv(self, x: NDArray, accel: NDArray | None = None) -> NDArray:
        """Continuous-time state derivative, with optional extra acceleration."""
        v = x[2:]
        a = -self.k * np.linalg.norm(v) * v
        a[1] -= self.g
        if accel is not None:
            a = a + accel
        return np.concatenate([v, a])

    def deriv_jac(self, x: NDArray) -> NDArray:
        """Jacobian of :meth:`deriv` with respect to the state."""
        v = x[2:]
        speed = np.linalg.norm(v)
        A = np.zeros((4, 4))
        A[0, 2] = A[1, 3] = 1.0
        if speed > 0:
            A[2:, 2:] = -self.k * (speed * np.eye(2) + np.outer(v, v) / speed)
        return A

    def rk4_step(
        self, x: NDArray, dt: float, accel: NDArray | None = None, jacobian: bool = False
    ) -> NDArray | tuple[NDArray, NDArray]:
        """One RK4 step; with ``jacobian=True`` also return d(x_next)/dx.

        The Jacobian is chained through the four stages, so it is the exact
        derivative of this discrete map (not an approximation of the flow's).
        """
        I = np.eye(4)
        k1 = self.deriv(x, accel)
        x2 = x + 0.5 * dt * k1
        k2 = self.deriv(x2, accel)
        x3 = x + 0.5 * dt * k2
        k3 = self.deriv(x3, accel)
        x4 = x + dt * k3
        k4 = self.deriv(x4, accel)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not jacobian:
            return x_next

        J1 = self.deriv_jac(x)
        J2 = self.deriv_jac(x2) @ (I + 0.5 * dt * J1)
        J3 = self.deriv_jac(x3) @ (I + 0.5 * dt * J2)
        J4 = self.deriv_jac(x4) @ (I + dt * J3)
        return x_next, I + dt / 6 * (J1 + 2 * J2 + 2 * J3 + J4)

    def f(self, x: NDArray, u: NDArray | None = None) -> NDArray:
        """Filter transition: one gust-free RK4 step over ``dt``. ``u`` is unused."""
        return self.rk4_step(np.asarray(x, dtype=float), self.dt)

    def F_jac(self, x: NDArray, u: NDArray | None = None) -> NDArray:
        """Exact Jacobian of :meth:`f`."""
        return self.rk4_step(np.asarray(x, dtype=float), self.dt, jacobian=True)[1]


@dataclass
class DragProjectileRun:
    t: NDArray  # (N,)
    truth: NDArray  # (N, 4)
    measurements: NDArray  # (N, 2) noisy [range, bearing]
    model: DragProjectileModel


def make_model(
    dt: float,
    accel_noise_std: float,
    range_noise_std: float | None = None,
    bearing_noise_std: float | None = None,
    drag: float = DEFAULT_DRAG,
    radar_pos: ArrayLike = DEFAULT_RADAR,
    g: float = G_EARTH,
) -> DragProjectileModel:
    """Filter model for a drag projectile, with a radar if its noise levels are given."""
    Gamma = model_matrices(dt, accel_noise_std, 0.0, g).Gamma
    radar = None
    if range_noise_std is not None or bearing_noise_std is not None:
        if range_noise_std is None or bearing_noise_std is None:
            raise ValueError("give both range_noise_std and bearing_noise_std, or neither")
        radar = RangeBearingRadar(radar_pos, range_noise_std, bearing_noise_std)
    return DragProjectileModel(
        dt=dt,
        k=drag,
        g=g,
        Q=accel_noise_std**2 * Gamma @ Gamma.T,
        Gamma=Gamma,
        radar=radar,
    )


def simulate_trajectory(
    x0: ArrayLike,
    dt: float,
    accel_noise_std: float,
    rng: np.random.Generator,
    drag: float = DEFAULT_DRAG,
    t_max: float | None = None,
    substeps: int = 10,
    g: float = G_EARTH,
) -> tuple[NDArray, NDArray]:
    """True trajectory until the projectile hits the ground (y < 0) or t_max.

    Returns ``(t, truth)`` with shapes (N,) and (N, 4), starting at t = 0.
    """
    m = make_model(dt, accel_noise_std, drag=drag, g=g)
    x = np.asarray(x0, dtype=float).reshape(4)
    if x[1] < 0:
        raise ValueError("projectile must start at or above the ground (y >= 0)")

    h = dt / substeps
    max_steps = int(np.ceil(t_max / dt)) + 1 if t_max is not None else 1_000_000
    states = [x]
    for _ in range(max_steps - 1):
        gust = rng.normal(0.0, accel_noise_std, size=2)
        for _ in range(substeps):
            x = m.rk4_step(x, h, gust)
        if x[1] < 0:
            break
        states.append(x)

    truth = np.array(states)
    return dt * np.arange(len(truth)), truth


def simulate(
    x0: ArrayLike,
    dt: float,
    accel_noise_std: float,
    range_noise_std: float,
    bearing_noise_std: float,
    rng: np.random.Generator,
    drag: float = DEFAULT_DRAG,
    radar_pos: ArrayLike = DEFAULT_RADAR,
    t_max: float | None = None,
    substeps: int = 10,
    g: float = G_EARTH,
) -> DragProjectileRun:
    """Simulate until the projectile hits the ground (y < 0) or t_max.

    Returns the true states and noisy radar measurements (``run.model.radar``)
    at every step, starting at t = 0.
    """
    m = make_model(dt, accel_noise_std, range_noise_std, bearing_noise_std, drag, radar_pos, g)
    t, truth = simulate_trajectory(x0, dt, accel_noise_std, rng, drag, t_max, substeps, g)
    return DragProjectileRun(t=t, truth=truth, measurements=m.radar.measure_series(truth, rng), model=m)


def make_filter(
    run: DragProjectileRun,
    velocity_guess: ArrayLike,
    velocity_std: float,
) -> ExtendedKalmanFilter:
    """Radar-only EKF initialized from the run's first measurement.

    Position starts at the first measurement converted to Cartesian, with
    covariance J R J^T (J the polar-to-Cartesian Jacobian); velocity starts at
    `velocity_guess` with standard deviation `velocity_std` per axis. Feed it
    ``run.measurements[1:]`` afterwards.
    """
    m = run.model
    radar = m.radar
    z0 = run.measurements[0]
    x0 = np.concatenate([radar.to_position(z0), np.asarray(velocity_guess, dtype=float)])
    P0 = np.zeros((4, 4))
    P0[:2, :2] = radar.position_cov(z0)
    P0[2:, 2:] = velocity_std**2 * np.eye(2)
    return ExtendedKalmanFilter(
        f=m.f, h=radar.h, Q=m.Q, R=radar.R, x0=x0, P0=P0, F_jac=m.F_jac, H_jac=radar.H_jac,
        residual=radar.residual,
    )


def make_fusion_filter(model: DragProjectileModel, x0: ArrayLike, P0: ArrayLike) -> ExtendedKalmanFilter:
    """EKF with this model's dynamics and no default sensor.

    Run it with ``filter_sensors([(sensor, zs), ...])`` or ``update(z, sensor)``.
    """
    return ExtendedKalmanFilter(f=model.f, h=None, Q=model.Q, R=None, x0=x0, P0=P0, F_jac=model.F_jac)
