"""Two-body propagation of three Earth orbits over one day.

A near-circular LEO, an eccentric Molniya-like orbit and GEO, each built as a TwoBodyOrbit from a
Cartesian state at the epoch and propagated to every 30 s over a day. The figure shows the orbits in
3D around the Earth, altitude against time, and how well specific energy and angular momentum are
conserved (relative error, against the epoch values). It also prints the cost of one Python call.

Run:  uv run python examples/propagation/two_body_day.py [--no-show]
Writes examples/propagation/out/two_body_day.html.
"""

from __future__ import annotations

import argparse
import timeit
from functools import partial
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tag.propagation import EARTH_MU, TwoBodyOrbit

EARTH_RADIUS = 6378.137  # km, equatorial
DAY = 86400.0  # s
STEP = 30.0  # s

# Osculating states at the epoch, km and km/s.
ORBITS = {
    "LEO (e≈0.001, i≈52°)": ([6778.0, 0.0, 0.0, 0.0, 4.72, 6.04], "#2563eb"),
    "Molniya-like (e≈0.68)": ([-1500.0, -6300.0, 2200.0, 8.2, -2.4, 4.9], "#dc2626"),
    "GEO (e=0, i=0)": ([42164.1696, 0.0, 0.0, 0.0, 3.0746761, 0.0], "#16a34a"),
}


def energy(rv: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(rv[..., 3:] ** 2, axis=-1) - EARTH_MU / np.linalg.norm(rv[..., :3], axis=-1)


def angular_momentum(rv: np.ndarray) -> np.ndarray:
    return np.cross(rv[..., :3], rv[..., 3:])


def earth_surface() -> go.Surface:
    lon, lat = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(-np.pi / 2, np.pi / 2, 30))
    return go.Surface(
        x=EARTH_RADIUS * np.cos(lat) * np.cos(lon),
        y=EARTH_RADIUS * np.cos(lat) * np.sin(lon),
        z=EARTH_RADIUS * np.sin(lat),
        colorscale=[[0, "#cbd5e1"], [1, "#94a3b8"]],
        showscale=False,
        opacity=0.6,
        hoverinfo="skip",
        name="Earth",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="write the HTML without opening a browser")
    args = parser.parse_args()

    times = np.arange(0.0, DAY + STEP, STEP)
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"type": "scene", "rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.55, 0.45],
        subplot_titles=("Orbits (km)", "Altitude", "Specific energy, relative error", "|h|, relative error"),
    )
    fig.add_trace(earth_surface(), row=1, col=1)

    for name, (rv0, color) in ORBITS.items():
        orbit = TwoBodyOrbit(rv0)
        states = np.array([orbit.propagate(t) for t in times])
        e_err = energy(states) / energy(np.array(rv0)) - 1
        h0 = np.linalg.norm(angular_momentum(np.array(rv0)))
        h_err = np.linalg.norm(angular_momentum(states), axis=1) / h0 - 1
        per_call = min(timeit.repeat(partial(orbit.propagate, 1234.5), number=100_000, repeat=5)) / 100_000
        print(
            f"{name:24s} a = {orbit.semi_major_axis:9.1f} km  e = {orbit.eccentricity:.4f}  "
            f"period = {orbit.period / 3600:5.2f} h  max |energy err| = {np.abs(e_err).max():.1e}  "
            f"max |h err| = {np.abs(h_err).max():.1e}  propagate: {per_call * 1e9:.0f} ns/call"
        )

        line = {"color": color, "width": 2}
        fig.add_trace(
            go.Scatter3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], mode="lines", line=line, name=name),
            row=1,
            col=1,
        )
        hours = times / 3600
        common = {"mode": "lines", "line": line, "legendgroup": name, "showlegend": False, "name": name}
        altitude = np.linalg.norm(states[:, :3], axis=1) - EARTH_RADIUS
        fig.add_trace(go.Scatter(x=hours, y=altitude, **common), row=1, col=2)
        fig.add_trace(go.Scatter(x=hours, y=e_err, **common), row=2, col=2)
        fig.add_trace(go.Scatter(x=hours, y=h_err, **common), row=3, col=2)

    fig.update_scenes(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z")
    fig.update_yaxes(title_text="km", type="log", row=1, col=2)
    fig.update_yaxes(exponentformat="e", row=2, col=2)
    fig.update_yaxes(exponentformat="e", row=3, col=2)
    fig.update_xaxes(title_text="t (h)", row=3, col=2)
    fig.update_layout(
        title="Two-body propagation over one day (conserved quantities stay at float64 rounding)",
        template="plotly_white",
        height=900,
        legend={"x": 0.0, "y": 1.0},
    )

    out = Path(__file__).parent / "out" / "two_body_day.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
