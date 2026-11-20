"""Orbit propagation, backed by a compiled C++ core."""

from tag.propagation._propagation import eccentric_anomaly

__all__ = ["eccentric_anomaly"]
