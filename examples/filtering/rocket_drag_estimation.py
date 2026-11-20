"""Track a sounding rocket and estimate its unknown drag coefficient with an EKF.

The filter's state is [x, y, vx, vy, Cd] (state augmentation). A ground radar
and onboard GNSS measure position only; Cd is learned from how the trajectory
bends. For comparison, a second filter trusts the nominal Cd and never
corrects it.

Run:  uv run python examples/rocket_drag_estimation.py [--seed N] [--no-show]
Writes examples/out/rocket_drag_estimation.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from _plotting import band, rmse
from plotly.subplots import make_subplots

from tag.filtering.sensors import GnssPosition, RangeBearingRadar
from tag.filtering.sims.rocket import Motor, make_filter, rail_exit_state, simulate

DT = 0.1  # s, filter / radar rate 10 Hz
ACCEL_STD = 0.5  # m/s^2, gusts
POS_STD = 1.0  # m, rail-exit position uncertainty
VEL_STD = 1.0  # m/s, rail-exit velocity uncertainty
CD_NOMINAL = 0.45  # filter's prior mean for Cd
CD_STD = 0.10  # filter's prior std for Cd

COLORS = {
    "truth": "#1f2937",
    "meas": "#9ca3af",
    "est": "#2563eb",
    "fixed": "#d97706",
    "radar": "#16a34a",
    "burn": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    motor = Motor()
    x_nominal = rail_exit_state()
    # The true rail-exit state and Cd are drawn from the filter's prior.
    x0_true = x_nominal + np.array([POS_STD, POS_STD, VEL_STD, VEL_STD]) * rng.standard_normal(4)
    cd_true = CD_NOMINAL + CD_STD * rng.standard_normal()
    run = simulate(x0_true, cd_true, DT, ACCEL_STD, rng, motor=motor)

    radar = RangeBearingRadar([-2000.0, 0.0], range_std=5.0, bearing_std=0.001)
    gnss = GnssPosition(std=3.0, every=2)
    z_radar, z_gnss = (
        radar.measure_series(run.truth, rng),
        gnss.measure_series(run.truth, rng),
    )
    streams = [(radar, z_radar[1:]), (gnss, z_gnss[1:])]

    results = {}
    for name, cd_std in (("est", CD_STD), ("fixed", 0.0)):
        ekf = make_filter(run.model, x_nominal, POS_STD, VEL_STD, CD_NOMINAL, cd_std)
        x_init, P_init = ekf.x.copy(), ekf.P.copy()
        res = ekf.filter_sensors(streams, us=run.controls)
        est = np.vstack([x_init, res.x_post])
        cov = np.concatenate([P_init[None], res.P_post])
        results[name] = (est, np.sqrt(np.diagonal(cov, axis1=1, axis2=2)))

    t, truth = run.t, run.truth
    est, sig = results["est"]
    fixed, _ = results["fixed"]
    err_est = est[:, :2] - truth[:, :2]
    err_fixed = fixed[:, :2] - truth[:, :2]
    speed = np.linalg.norm(truth[:, 2:], axis=1)

    print(f"steps: {len(t)}  apogee: {truth[-1, 1]:.0f} m at t = {t[-1]:.1f} s  max speed: {speed.max():.0f} m/s")
    print(f"Cd  true: {cd_true:.4f}   estimate: {est[-1, 4]:.4f} ± {sig[-1, 4]:.4f}   (prior {CD_NOMINAL} ± {CD_STD})")
    print(f"position RMSE  estimating Cd: {rmse(err_est):.2f} m   fixed Cd = {CD_NOMINAL}: {rmse(err_fixed):.2f} m")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Trajectory",
            "Speed (shaded: motor burning)",
            "Drag coefficient estimate (shaded: ±2σ)",
            "Position error",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Trajectory with GNSS fixes and the radar
    g = z_gnss
    fig.add_trace(
        go.Scatter(
            x=g[:, 0],
            y=g[:, 1],
            mode="markers",
            name="GNSS fixes",
            marker={"color": COLORS["meas"], "size": 4},
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
            name="EKF estimating Cd",
            line={"color": COLORS["est"], "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fixed[:, 0],
            y=fixed[:, 1],
            mode="lines",
            name=f"EKF with fixed Cd = {CD_NOMINAL}",
            line={"color": COLORS["fixed"], "width": 1.5, "dash": "dot"},
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

    # 2. Speed, with the burn phase shaded
    fig.add_trace(
        go.Scatter(
            x=t,
            y=speed,
            mode="lines",
            name="true speed",
            showlegend=False,
            line={"color": COLORS["truth"], "width": 2},
        ),
        row=1,
        col=2,
    )
    fig.add_vrect(
        x0=0,
        x1=motor.burn_time,
        fillcolor=COLORS["burn"],
        opacity=0.1,
        line_width=0,
        row=1,
        col=2,
    )

    # 3. Cd estimate with ±2σ and the truth
    band(fig, t, est[:, 4], sig[:, 4], COLORS["est"], "±2σ Cd", 2, 1)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.full_like(t, cd_true),
            mode="lines",
            name="true Cd",
            line={"color": COLORS["truth"], "width": 1.5},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=est[:, 4],
            mode="lines",
            name="Cd estimate",
            line={"color": COLORS["est"], "width": 2},
        ),
        row=2,
        col=1,
    )
    fig.add_vrect(
        x0=0,
        x1=motor.burn_time,
        fillcolor=COLORS["burn"],
        opacity=0.1,
        line_width=0,
        row=2,
        col=1,
    )

    # 4. Position error norms
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.linalg.norm(err_est, axis=1),
            mode="lines",
            name="error, estimating Cd",
            line={"color": COLORS["est"], "width": 1.5},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.linalg.norm(err_fixed, axis=1),
            mode="lines",
            name="error, fixed Cd",
            line={"color": COLORS["fixed"], "width": 1.5},
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="downrange (m)", row=1, col=1)
    fig.update_yaxes(title_text="altitude (m)", row=1, col=1)
    fig.update_xaxes(title_text="t (s)", row=1, col=2)
    fig.update_yaxes(title_text="speed (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="t (s)", row=2, col=1)
    fig.update_yaxes(title_text="Cd", row=2, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=2)
    fig.update_yaxes(title_text="|error| (m)", row=2, col=2)
    fig.update_layout(
        title=f"Sounding rocket: Cd {cd_true:.3f} estimated as {est[-1, 4]:.3f} ± {sig[-1, 4]:.3f}. "
        f"Position RMSE {rmse(err_est):.2f} m (fixed Cd: {rmse(err_fixed):.2f} m)",
        template="plotly_white",
        height=850,
        hovermode="x unified",
    )

    out = Path(__file__).parent / "out" / "rocket_drag_estimation.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
