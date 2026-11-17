"""Fuse GNSS and two narrow-view cameras to track a projectile with air drag.

Each camera sees only part of the flight and measures only the direction to
the target. GNSS gives positions at 2 Hz and drops out for 1 s. The EKF runs
three times on the same measurements: GNSS only, cameras only (stereo where
both see the target), and all three fused.

Run:  uv run python examples/drag_projectile_fusion.py [--seed N] [--no-show]
Writes examples/out/drag_projectile_fusion.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from _plotting import band, rmse
from plotly.subplots import make_subplots

from tooth.sensors import GnssPosition, PinholeCamera
from tooth.sims.drag_projectile import (
    launch_state,
    make_fusion_filter,
    make_model,
    simulate_trajectory,
)

DT = 0.05  # s, filter / camera rate 20 Hz
ACCEL_STD = 0.5  # m/s^2, gusts
POS_STD = 0.5  # m, launch-point uncertainty
VEL_STD = 5.0  # m/s, launch-velocity uncertainty
GNSS_OUTAGE = (2.0, 3.0)  # s

COLORS = {
    "truth": "#1f2937",
    "GNSS": "#d97706",
    "stereo": "#7c3aed",
    "fused": "#2563eb",
    "cam L": "#16a34a",
    "cam R": "#db2777",
    "x": "#2563eb",
    "y": "#dc2626",
}


def make_sensors() -> list:
    # Cameras sit well away from the flight: a target passing within a few
    # metres of a camera makes its projection too nonlinear for the EKF.
    return [
        GnssPosition(std=2.0, every=10),
        PinholeCamera([-50.0, 1.0], np.radians(18), focal_px=1000.0, pixel_std=3.0, width_px=300.0, name="cam L"),
        PinholeCamera([190.0, 1.0], np.radians(162), focal_px=1000.0, pixel_std=3.0, width_px=300.0, name="cam R"),
    ]


def fov_wedge(cam: PinholeCamera, length: float) -> tuple[list[float], list[float]]:
    px, py = cam.position
    a, b = cam.yaw - cam.half_fov, cam.yaw + cam.half_fov
    return (
        [px, px + length * np.cos(a), px + length * np.cos(b), px],
        [py, py + length * np.sin(a), py + length * np.sin(b), py],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    v_nominal = launch_state(speed=50.0, angle_deg=45.0)[2:]
    x0_prior = np.array([0.0, 2.0, *v_nominal])
    P0 = np.diag([POS_STD**2, POS_STD**2, VEL_STD**2, VEL_STD**2])

    # The true launch is drawn from the filter's prior.
    x0_true = x0_prior + np.sqrt(np.diag(P0)) * rng.standard_normal(4)
    t, truth = simulate_trajectory(x0_true, DT, ACCEL_STD, rng)

    sensors = make_sensors()
    zs = {}
    for s in sensors:
        z = s.measure_series(truth, rng)
        if isinstance(s, GnssPosition):
            z[(t >= GNSS_OUTAGE[0]) & (t < GNSS_OUTAGE[1])] = np.nan
        zs[s.name] = z
    by_name = {s.name: s for s in sensors}

    configs = {"GNSS": ["GNSS"], "stereo": ["cam L", "cam R"], "fused": ["GNSS", "cam L", "cam R"]}
    model = make_model(DT, ACCEL_STD)
    est, sig = {}, {}
    for name, names in configs.items():
        ekf = make_fusion_filter(model, x0_prior, P0)
        res = ekf.filter_sensors([(by_name[n], zs[n][1:]) for n in names])
        est[name] = np.vstack([x0_prior, res.x_post])
        cov = np.concatenate([P0[None], res.P_post])
        sig[name] = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

    print(f"steps: {len(t)}  flight time: {t[-1]:.2f} s  range: {truth[-1, 0]:.1f} m")
    for n in ("cam L", "cam R", "GNSS"):
        print(f"  {n:6s} measurements: {int((~np.isnan(zs[n][:, 0])).sum())}")
    errs = {name: e[:, :2] - truth[:, :2] for name, e in est.items()}
    print("position RMSE  " + "   ".join(f"{name}: {rmse(e):.2f} m" for name, e in errs.items()))

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
        row_heights=[0.48, 0.36, 0.16],
        subplot_titles=(
            "Trajectory and sensor layout",
            "Position error (shaded: GNSS outage)",
            "Fused position error (shaded: ±2σ)",
            "Measurements received",
        ),
        vertical_spacing=0.09,
    )

    # 1. Trajectory, camera fields of view, GNSS fixes
    for cam in sensors[1:]:
        wx, wy = fov_wedge(cam, 260.0)
        fig.add_trace(
            go.Scatter(x=wx, y=wy, fill="toself", fillcolor=COLORS[cam.name], opacity=0.1, mode="lines",
                       line={"width": 0}, name=f"{cam.name} field of view", hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[cam.position[0]], y=[cam.position[1]], mode="markers", name=cam.name,
                       marker={"color": COLORS[cam.name], "size": 12, "symbol": "square"}),
            row=1, col=1,
        )
    g = zs["GNSS"]
    fig.add_trace(
        go.Scatter(x=g[:, 0], y=g[:, 1], mode="markers", name="GNSS fixes",
                   marker={"color": COLORS["GNSS"], "size": 7, "symbol": "x"}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=truth[:, 0], y=truth[:, 1], mode="lines", name="truth",
                   line={"color": COLORS["truth"], "width": 2}),
        row=1, col=1,
    )
    for name, e in est.items():
        fig.add_trace(
            go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", name=f"{name} estimate",
                       line={"color": COLORS[name], "width": 1.5, "dash": "dash"}),
            row=1, col=1,
        )

    # 2. Position error norm for each configuration
    for name, e in errs.items():
        fig.add_trace(
            go.Scatter(x=t, y=np.linalg.norm(e, axis=1), mode="lines", name=f"{name} error",
                       line={"color": COLORS[name], "width": 1.5}),
            row=2, col=1,
        )
    fig.add_vrect(x0=GNSS_OUTAGE[0], x1=GNSS_OUTAGE[1], fillcolor=COLORS["GNSS"], opacity=0.12,
                  line_width=0, row=2, col=1)

    # 3. Fused x / y error with ±2σ bands
    for i, axis in enumerate("xy"):
        band(fig, t, np.zeros_like(t), sig["fused"][:, i], COLORS[axis], f"±2σ {axis}", 2, 2)
        fig.add_trace(
            go.Scatter(x=t, y=errs["fused"][:, i], mode="lines", name=f"fused {axis} error",
                       line={"color": COLORS[axis], "width": 1.5}),
            row=2, col=2,
        )

    # 4. When each sensor delivered a measurement
    for row, name in enumerate(["GNSS", "cam L", "cam R"]):
        have = ~np.isnan(zs[name][:, 0])
        fig.add_trace(
            go.Scatter(x=t[have], y=np.full(have.sum(), name), mode="markers", showlegend=False,
                       marker={"color": COLORS[name], "size": 5, "symbol": "line-ns-open"}),
            row=3, col=1,
        )

    fig.update_xaxes(title_text="x (m)", range=[-60, 200], row=1, col=1)
    fig.update_yaxes(title_text="y (m)", range=[-5, 60], scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=1)
    fig.update_yaxes(title_text="|error| (m)", row=2, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=2)
    fig.update_yaxes(title_text="error (m)", row=2, col=2)
    fig.update_xaxes(title_text="t (s)", range=[0, t[-1]], row=3, col=1)
    fig.update_layout(
        title="Sensor fusion, drag projectile. Position RMSE: "
              + ", ".join(f"{name} {rmse(e):.2f} m" for name, e in errs.items()),
        template="plotly_white",
        height=1000,
        hovermode="closest",
    )

    out = Path(__file__).parent / "out" / "drag_projectile_fusion.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
