"""Track a projectile with air drag from radar range/bearing using an extended Kalman filter.

Run:  uv run python examples/drag_projectile_ekf.py [--seed N] [--no-show]
Writes examples/out/drag_projectile_ekf.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from _plotting import band, rmse
from plotly.subplots import make_subplots

from tooth.sims.drag_projectile import launch_state, make_filter, simulate

DT = 0.05  # s, radar rate 20 Hz
ACCEL_STD = 0.5  # m/s^2, gusts
RANGE_STD = 2.0  # m
BEARING_STD = 0.01  # rad (~0.57°)
VEL_STD = 5.0  # m/s, initial velocity uncertainty
DRAG = 0.008  # 1/m, terminal speed ≈ 35 m/s

COLORS = {
    "truth": "#1f2937",
    "meas": "#9ca3af",
    "kf": "#2563eb",
    "vacuum": "#9ca3af",
    "radar": "#16a34a",
    "x": "#2563eb",
    "y": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x0_true = launch_state(speed=50.0, angle_deg=45.0)
    run = simulate(x0_true, DT, ACCEL_STD, RANGE_STD, BEARING_STD, rng, drag=DRAG)
    vacuum = simulate(x0_true, DT, 0.0, 0.0, 0.0, rng, drag=0.0)

    # The filter knows the launch velocity only roughly.
    v_guess = x0_true[2:] + VEL_STD * rng.standard_normal(2)
    ekf = make_filter(run, velocity_guess=v_guess, velocity_std=VEL_STD)
    x_init, P_init = ekf.x.copy(), ekf.P.copy()
    res = ekf.filter(run.measurements[1:])

    t = run.t
    est = np.vstack([x_init, res.x_post])
    cov = np.concatenate([P_init[None], res.P_post])
    sig = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    truth = run.truth
    radar = run.model.radar
    meas = radar.to_position(run.measurements)

    meas_rmse = rmse(meas - truth[:, :2])
    ekf_rmse = rmse(est[:, :2] - truth[:, :2])
    print(
        f"steps: {len(t)}  flight time: {t[-1]:.2f} s  range: {truth[-1, 0]:.1f} m "
        f"(vacuum: {vacuum.truth[-1, 0]:.1f} m)"
    )
    print(
        f"position RMSE  radar (converted): {meas_rmse:.2f} m   EKF: {ekf_rmse:.2f} m"
    )
    print(
        f"final velocity error: {np.linalg.norm(est[-1, 2:] - truth[-1, 2:]):.2f} m/s"
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=("Trajectory", "Position error (shaded: ±2σ)", "Velocity"),
        vertical_spacing=0.12,
    )

    # 1. Trajectory
    fig.add_trace(
        go.Scatter(
            x=vacuum.truth[:, 0],
            y=vacuum.truth[:, 1],
            mode="lines",
            name="no-drag trajectory",
            line={"color": COLORS["vacuum"], "width": 1, "dash": "dot"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=meas[:, 0],
            y=meas[:, 1],
            mode="markers",
            name="radar (converted to x/y)",
            marker={"color": COLORS["meas"], "size": 5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=truth[:, 0],
            y=truth[:, 1],
            mode="lines",
            name="truth",
            line={"color": COLORS["truth"], "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=est[:, 0],
            y=est[:, 1],
            mode="lines",
            name="EKF estimate",
            line={"color": COLORS["kf"], "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[radar.position[0]],
            y=[radar.position[1]],
            mode="markers",
            name="radar",
            marker={"color": COLORS["radar"], "size": 12, "symbol": "triangle-up"},
        ),
        row=1,
        col=1,
    )

    # 2. Position error with ±2σ bands
    for i, axis in enumerate("xy"):
        err = est[:, i] - truth[:, i]
        band(fig, t, np.zeros_like(t), sig[:, i], COLORS[axis], f"±2σ {axis}", 2, 1)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=err,
                mode="lines",
                name=f"{axis} error",
                line={"color": COLORS[axis], "width": 1.5},
            ),
            row=2,
            col=1,
        )

    # 3. Velocity: estimate (with ±2σ) vs truth; never measured directly
    for i, axis in enumerate("xy", start=2):
        color = COLORS[axis]
        band(fig, t, est[:, i], sig[:, i], color, f"±2σ v{axis}", 2, 2)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=truth[:, i],
                mode="lines",
                name=f"v{axis} truth",
                line={"color": COLORS["truth"], "width": 1.5},
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=est[:, i],
                mode="lines",
                name=f"v{axis} estimate",
                line={"color": color, "width": 1.5, "dash": "dash"},
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="x (m)", row=1, col=1)
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=1)
    fig.update_yaxes(title_text="error (m)", row=2, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=2)
    fig.update_yaxes(title_text="velocity (m/s)", row=2, col=2)
    fig.update_layout(
        title=f"Extended Kalman filter: drag projectile, radar range/bearing. "
        f"Position RMSE {meas_rmse:.2f} m → {ekf_rmse:.2f} m",
        template="plotly_white",
        height=850,
        hovermode="x unified",
    )

    out = Path(__file__).parent / "out" / "drag_projectile_ekf.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
