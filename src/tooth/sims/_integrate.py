"""Fixed-step integration shared by the simulations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

DerivFn = Callable[[float, NDArray], NDArray]
DerivJacFn = Callable[[float, NDArray], NDArray]


def rk4_step(
    deriv: DerivFn,
    x: NDArray,
    dt: float,
    deriv_jac: DerivJacFn | None = None,
    t: float = 0.0,
) -> NDArray | tuple[NDArray, NDArray]:
    """One classic RK4 step of ``dx/dt = deriv(t, x)`` from time ``t``.

    With ``deriv_jac(t, x)`` (the Jacobian of ``deriv`` w.r.t. ``x``) also
    returns d(x_next)/dx, chained through the four stages, so it is the exact
    derivative of this discrete map rather than an approximation of the flow's.
    """
    h2 = 0.5 * dt
    k1 = deriv(t, x)
    x2 = x + h2 * k1
    k2 = deriv(t + h2, x2)
    x3 = x + h2 * k2
    k3 = deriv(t + h2, x3)
    x4 = x + dt * k3
    k4 = deriv(t + dt, x4)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if deriv_jac is None:
        return x_next

    I = np.eye(len(x))
    J1 = deriv_jac(t, x)
    J2 = deriv_jac(t + h2, x2) @ (I + h2 * J1)
    J3 = deriv_jac(t + h2, x3) @ (I + h2 * J2)
    J4 = deriv_jac(t + dt, x4) @ (I + dt * J3)
    return x_next, I + dt / 6 * (J1 + 2 * J2 + 2 * J3 + J4)
