"""Compiled core of tag.propagation."""

def eccentric_anomaly(mean_anomaly: float, e: float) -> float:
    """
    Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (radians).

    E is in the same revolution as M. Raises ValueError unless 0 <= e < 1 and M is finite.
    """
