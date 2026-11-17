"""Sounding rocket from rail exit to apogee, with an unknown drag coefficient.

Truth state: [x, y, vx, vy] (m, m/s). Continuous dynamics:

    dv/dt = (T(t) / m(t)) v̂  -  (rho(y) Cd A / (2 m(t))) |v| v  -  g ŷ  +  a_gust

* Thrust acts along the velocity (a gravity turn, so no attitude state). The
  motor is known: constant thrust ``thrust`` for ``burn_time`` seconds after
  rail exit, with mass falling linearly from ``mass0`` to ``mass_burnout``.
* Air density follows an exponential atmosphere, rho = rho0 exp(-y / H).
* Gusts are random accelerations held constant over each timestep.
* ``Cd`` is constant in the truth but unknown to the filter.

The filter estimates Cd by **state augmentation**: its state is
[x, y, vx, vy, Cd] with dCd/dt = 0. Thrust and mass enter the filter as the
control input u_k = [mean thrust over the step, mass at mid-step]
(``run.controls``); the truth integrates them continuously, so the filter
model is a close approximation, not exact.

The run starts at rail exit (thrust along v̂ is undefined at zero speed) and
ends at apogee (vy turning negative) or ``t_max``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tooth.kalman import ExtendedKalmanFilter
from tooth.sims._integrate import rk4_step
from tooth.sims.projectile import G_EARTH, model_matrices

__all__ = [
    "Motor",
    "RocketModel",
    "RocketRun",
    "air_density",
    "make_filter",
    "make_model",
    "rail_exit_state",
    "simulate",
]

RHO0 = 1.225  # kg/m^3, sea-level density
SCALE_HEIGHT = 8500.0  # m
DEFAULT_DIAMETER = 0.11  # m


def air_density(y: float | NDArray, rho0: float = RHO0, scale_height: float = SCALE_HEIGHT) -> float | NDArray:
    """Exponential-atmosphere density at altitude ``y`` (m)."""
    return rho0 * np.exp(-np.asarray(y) / scale_height)


@dataclass(frozen=True)
class Motor:
    """Constant-thrust motor; time is measured from rail exit."""

    thrust: float = 1500.0  # N
    burn_time: float = 4.0  # s
    mass0: float = 20.0  # kg at rail exit
    mass_burnout: float = 12.0  # kg

    @property
    def mass_flow(self) -> float:
        return (self.mass0 - self.mass_burnout) / self.burn_time

    @property
    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity T / mdot (= Isp * g0)."""
        return self.thrust / self.mass_flow

    def thrust_at(self, t: float) -> float:
        return self.thrust if t < self.burn_time else 0.0

    def mass_at(self, t: float) -> float:
        return self.mass0 - self.mass_flow * min(max(t, 0.0), self.burn_time)

    def mean_thrust(self, t0: float, t1: float) -> float:
        """Thrust averaged over [t0, t1] (exact across burnout)."""
        burning = max(0.0, min(t1, self.burn_time) - max(t0, 0.0))
        return self.thrust * burning / (t1 - t0)

    def control(self, t0: float, t1: float) -> NDArray:
        """Filter control input for the step [t0, t1]: [mean thrust, mid-step mass]."""
        return np.array([self.mean_thrust(t0, t1), self.mass_at(0.5 * (t0 + t1))])


@dataclass
class RocketModel:
    """Filter-side model for the augmented state [x, y, vx, vy, Cd]."""

    dt: float
    area: float  # m^2, reference area for drag
    g: float
    Q: NDArray  # (5, 5); no process noise on Cd
    rho0: float = RHO0
    scale_height: float = SCALE_HEIGHT

    def accel(self, x: NDArray, cd: float, thrust: float, mass: float) -> NDArray:
        """Acceleration (2,) at state ``x`` (position and velocity first)."""
        v = x[2:4]
        speed = np.linalg.norm(v)
        rho = air_density(x[1], self.rho0, self.scale_height)
        a = -(0.5 * rho * cd * self.area / mass) * speed * v
        if speed > 0:
            a = a + (thrust / mass) * v / speed
        a[1] -= self.g
        return a

    def deriv(self, x: NDArray, u: NDArray) -> NDArray:
        """d/dt of the augmented state; Cd is constant."""
        return np.concatenate([x[2:4], self.accel(x, x[4], u[0], u[1]), [0.0]])

    def deriv_jac(self, x: NDArray, u: NDArray) -> NDArray:
        """Jacobian of :meth:`deriv` w.r.t. the augmented state."""
        thrust, mass = u
        v, cd = x[2:4], x[4]
        speed = np.linalg.norm(v)
        rho = air_density(x[1], self.rho0, self.scale_height)
        kappa = 0.5 * self.area / mass
        A = np.zeros((5, 5))
        A[0, 2] = A[1, 3] = 1.0
        if speed > 0:
            vv = np.outer(v, v) / speed**2
            # thrust (T/m) v/|v|  and  drag -kappa rho Cd |v| v
            A[2:4, 2:4] = (thrust / mass) * (np.eye(2) - vv) / speed - kappa * rho * cd * speed * (np.eye(2) + vv)
            A[2:4, 1] = kappa * rho * cd * speed * v / self.scale_height  # d rho/dy = -rho/H
            A[2:4, 4] = -kappa * rho * speed * v
        return A

    def f(self, x: NDArray, u: NDArray) -> NDArray:
        """One RK4 step over ``dt`` with constant control u = [thrust, mass] and no gust."""
        x = np.asarray(x, dtype=float)
        return rk4_step(lambda t, s: self.deriv(s, u), x, self.dt)

    def F_jac(self, x: NDArray, u: NDArray) -> NDArray:
        """Exact Jacobian of :meth:`f`."""
        x = np.asarray(x, dtype=float)
        return rk4_step(lambda t, s: self.deriv(s, u), x, self.dt, lambda t, s: self.deriv_jac(s, u))[1]


@dataclass
class RocketRun:
    t: NDArray  # (N,)
    truth: NDArray  # (N, 4)
    cd: float  # true drag coefficient
    controls: NDArray  # (N - 1, 2) [mean thrust, mass] for the steps ending at t[1:]
    motor: Motor
    model: RocketModel


def make_model(
    dt: float,
    accel_noise_std: float,
    diameter: float = DEFAULT_DIAMETER,
    g: float = G_EARTH,
) -> RocketModel:
    Gamma = model_matrices(dt, accel_noise_std, 0.0, g).Gamma
    Q = np.zeros((5, 5))
    Q[:4, :4] = accel_noise_std**2 * Gamma @ Gamma.T
    return RocketModel(dt=dt, area=np.pi * (0.5 * diameter) ** 2, g=g, Q=Q)


def rail_exit_state(speed: float = 25.0, angle_deg: float = 85.0, height: float = 5.0) -> NDArray:
    """[x, y, vx, vy] leaving a launch rail at ``angle_deg`` above horizontal."""
    th = np.radians(angle_deg)
    return np.array([0.0, height, speed * np.cos(th), speed * np.sin(th)])


def simulate(
    x0: ArrayLike,
    cd: float,
    dt: float,
    accel_noise_std: float,
    rng: np.random.Generator,
    motor: Motor = Motor(),
    diameter: float = DEFAULT_DIAMETER,
    t_max: float | None = None,
    substeps: int = 10,
    g: float = G_EARTH,
) -> RocketRun:
    """Simulate from rail exit until apogee (vy < 0) or ``t_max``.

    The truth integrates thrust and mass continuously (RK4 with ``substeps``
    per ``dt``). Attach sensors with their ``measure_series(run.truth, rng)``.
    """
    model = make_model(dt, accel_noise_std, diameter, g)
    x = np.asarray(x0, dtype=float).reshape(4)
    if x[3] <= 0:
        raise ValueError("rocket must be climbing at rail exit (vy > 0)")

    def deriv(t: float, s: NDArray, thrust: float, gust: NDArray) -> NDArray:
        a = model.accel(s, cd, thrust, motor.mass_at(t)) + gust
        return np.concatenate([s[2:4], a])

    h = dt / substeps
    max_steps = int(np.ceil(t_max / dt)) + 1 if t_max is not None else 1_000_000
    states = [x]
    for k in range(max_steps - 1):
        gust = rng.normal(0.0, accel_noise_std, size=2)
        for j in range(substeps):
            t0 = k * dt + j * h
            # Thrust held at its exact mean over the substep: evaluating the step
            # function at RK4's stage times would lose impulse at burnout.
            thrust = motor.mean_thrust(t0, t0 + h)
            x = rk4_step(
                lambda t, s, thrust=thrust, gust=gust: deriv(t, s, thrust, gust),
                x,
                h,
                t=t0,
            )
        if x[3] < 0:
            break
        states.append(x)

    truth = np.array(states)
    t = dt * np.arange(len(truth))
    controls = np.array([motor.control(t0, t1) for t0, t1 in itertools.pairwise(t)]).reshape(-1, 2)
    return RocketRun(t=t, truth=truth, cd=cd, controls=controls, motor=motor, model=model)


def make_filter(
    model: RocketModel,
    x0_guess: ArrayLike,
    pos_std: float,
    vel_std: float,
    cd_guess: float,
    cd_std: float,
) -> ExtendedKalmanFilter:
    """EKF over [x, y, vx, vy, Cd] with no default sensor.

    ``cd_std = 0`` freezes Cd at ``cd_guess`` (a filter that trusts a nominal
    value). Run it with ``filter_sensors(streams, us=run.controls)`` on
    measurement rows ``[1:]``.
    """
    x0 = np.concatenate([np.asarray(x0_guess, dtype=float).reshape(4), [cd_guess]])
    P0 = np.diag([pos_std**2, pos_std**2, vel_std**2, vel_std**2, cd_std**2])
    return ExtendedKalmanFilter(f=model.f, h=None, Q=model.Q, R=None, x0=x0, P0=P0, F_jac=model.F_jac)
