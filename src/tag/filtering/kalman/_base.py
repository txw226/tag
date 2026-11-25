"""Pieces shared by the Kalman filter variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

AcceptPosterior = Callable[[NDArray, NDArray], bool]


class UpdateOutcome(Enum):
    """What one measurement update did."""

    ABSORBED = "absorbed"  # the posterior replaced the prior
    SKIPPED = "skipped"  # no measurement (None or a NaN row)
    REJECTED = "rejected"  # computed, then discarded by ``accept_posterior``; the prior is kept


# Outcomes that count towards lock-out: a measurement was available but not absorbed.
# Gating (#12) will add its outcome here.
_UNABSORBED = frozenset({UpdateOutcome.REJECTED})


@dataclass
class FilterResult:
    """Output of a filter's ``filter`` or ``filter_sensors`` method for N steps, n states
    and S measurement streams (S = 1 for ``filter``)."""

    x_prior: NDArray  # (N, n) predicted states
    P_prior: NDArray  # (N, n, n) predicted covariances
    x_post: NDArray  # (N, n) updated states
    P_post: NDArray  # (N, n, n) updated covariances
    skipped: NDArray  # (N, S) bool: no measurement in that stream at that step
    rejected: NDArray  # (N, S) bool: a rejected update (accept_posterior returned False)
    unabsorbed_run: NDArray  # (N,) int: consecutive-unabsorbed (lock-out) counter after each step

    @classmethod
    def empty(cls, N: int, n: int, S: int) -> FilterResult:
        return cls(
            x_prior=np.empty((N, n)),
            P_prior=np.empty((N, n, n)),
            x_post=np.empty((N, n)),
            P_post=np.empty((N, n, n)),
            skipped=np.zeros((N, S), dtype=bool),
            rejected=np.zeros((N, S), dtype=bool),
            unabsorbed_run=np.zeros(N, dtype=int),
        )


class LockOutError(RuntimeError):
    """Raised when more consecutive steps than ``max_consecutive_unabsorbed`` absorbed no
    available measurement."""


def _as_matrix(name: str, value: ArrayLike, shape: tuple[int, int]) -> NDArray:
    arr = np.atleast_2d(np.array(value, dtype=float))
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _controls(us: ArrayLike | None, N: int) -> NDArray | None:
    """(N, p) control inputs from an (N, p) array or a single (p,) input."""
    if us is None:
        return None
    us = np.asarray(us, dtype=float)
    if us.ndim <= 1:
        us = np.broadcast_to(us.reshape(1, -1), (N, us.size))
    return us


def joseph_update(
    x: NDArray, P: NDArray, y: NDArray, H: NDArray, R: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Kalman measurement update given innovation ``y`` and measurement Jacobian ``H``.

    Returns the updated ``(x, P)`` and the innovation covariance ``S`` and gain ``K``.
    """
    S = H @ P @ H.T + R
    # K = P H^T S^-1, computed as a solve against symmetric S.
    K = np.linalg.solve(S, H @ P).T

    x = x + K @ y
    # Joseph form: stays symmetric positive semi-definite despite round-off.
    IKH = np.eye(len(x)) - K @ H
    P = IKH @ P @ IKH.T + K @ R @ K.T
    return x, 0.5 * (P + P.T), S, K


class KalmanFilterBase:
    """Predict/update driver loop. Subclasses set ``x``, ``P``, ``n``, call
    ``_init_acceptance`` and define ``predict(u)`` and ``update(z)``; ``update`` computes a
    candidate posterior and hands it to ``_accept``.

    Rejected updates and lock-out:
        ``accept_posterior(x, P) -> bool``, when given, sees every candidate posterior.
        False discards it and keeps the prior (a rejected update). ``consecutive_unabsorbed``
        counts consecutive steps in which every available measurement went unabsorbed: a step
        that absorbs any measurement resets it to 0, a step with no measurement leaves it
        unchanged. It is report-only unless ``max_consecutive_unabsorbed`` is set, in which case
        ``step``/``filter``/``filter_sensors`` raise ``LockOutError`` once it exceeds that limit.
    """

    x: NDArray
    P: NDArray
    n: int
    accept_posterior: AcceptPosterior | None = None
    max_consecutive_unabsorbed: int | None = None
    consecutive_unabsorbed: int = 0

    def _init_acceptance(
        self, accept_posterior: AcceptPosterior | None, max_consecutive_unabsorbed: int | None
    ) -> None:
        if max_consecutive_unabsorbed is not None and max_consecutive_unabsorbed < 0:
            raise ValueError(f"max_consecutive_unabsorbed must be >= 0, got {max_consecutive_unabsorbed}")
        self.accept_posterior = accept_posterior
        self.max_consecutive_unabsorbed = max_consecutive_unabsorbed
        self.consecutive_unabsorbed = 0

    def _accept(self, x: NDArray, P: NDArray) -> UpdateOutcome:
        """Adopt the candidate posterior ``(x, P)`` unless ``accept_posterior`` rejects it."""
        if self.accept_posterior is not None and not self.accept_posterior(x, P):
            return UpdateOutcome.REJECTED
        self.x, self.P = x, P
        return UpdateOutcome.ABSORBED

    def _end_step(self, outcomes: list[UpdateOutcome], k: int | None = None) -> None:
        """Advance the lock-out counter with one step's update outcomes; raise past the limit."""
        if UpdateOutcome.ABSORBED in outcomes:
            self.consecutive_unabsorbed = 0
        elif any(o in _UNABSORBED for o in outcomes):
            self.consecutive_unabsorbed += 1
        limit = self.max_consecutive_unabsorbed
        if limit is not None and self.consecutive_unabsorbed > limit:
            at = "" if k is None else f" at step {k}"
            raise LockOutError(
                f"lock-out{at}: {self.consecutive_unabsorbed} consecutive steps absorbed no available measurement "
                f"(rejected updates), more than max_consecutive_unabsorbed={limit}"
            )

    def predict(self, u: ArrayLike | None = None) -> None:
        raise NotImplementedError

    def update(self, z: ArrayLike | None) -> UpdateOutcome:
        raise NotImplementedError

    def step(self, z: ArrayLike | None, u: ArrayLike | None = None) -> UpdateOutcome:
        """One predict + update cycle, counted towards lock-out."""
        self.predict(u)
        outcome = self.update(z)
        self._end_step([outcome])
        return outcome

    def filter(self, zs: ArrayLike, us: ArrayLike | None = None) -> FilterResult:
        """Run predict/update over a sequence of measurements.

        Args:
            zs: (N, m) measurements; rows may contain NaN to mark a dropout.
            us: optional (N, p) control inputs, or a single (p,) input used every step.

        Raises:
            LockOutError: if ``max_consecutive_unabsorbed`` is set and exceeded.
        """
        zs = np.asarray(zs, dtype=float)
        if zs.ndim == 1:
            zs = zs.reshape(-1, 1)
        N = zs.shape[0]

        us = _controls(us, N)

        out = FilterResult.empty(N, self.n, 1)
        for k in range(N):
            self.predict(None if us is None else us[k])
            out.x_prior[k], out.P_prior[k] = self.x, self.P
            z = zs[k]
            outcome = self.update(None if np.isnan(z).any() else z)
            out.x_post[k], out.P_post[k] = self.x, self.P
            out.skipped[k, 0] = outcome is UpdateOutcome.SKIPPED
            out.rejected[k, 0] = outcome is UpdateOutcome.REJECTED
            self._end_step([outcome], k)
            out.unabsorbed_run[k] = self.consecutive_unabsorbed
        return out
