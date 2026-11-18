"""Common ways a Kalman filter goes wrong, and how they show up in the metrics.

The same projectile flight and position sensor, filtered five times by an
otherwise correctly tuned linear KF:

* baseline: everything matches the truth.
* overconfident start: the launch-velocity guess is 8 m/s off per axis but
  P0 claims 0.3 m/s, so the filter barely listens to the measurements at first.
* dropout: no measurements for 1.5 s; the filter coasts on its model and σ grows.
* outliers: three wild measurements (+20 m); NIS flags them instantly.
* sensor bias: every x reading is 3 m high. The filter can't tell bias from
  truth, so it follows it: errors sit outside ±2σ (NEES is huge), yet NIS
  looks perfect. Only the truth, or a second independent sensor, reveals it.

Unmodelled dynamics (the other big one) is in drag_lkf_vs_ekf.py.

Run:  uv run python examples/pitfalls.py [--seed N] [--no-show]
Writes examples/out/pitfalls.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from _plotting import band, chi2_bounds, nees, nis, rmse
from plotly.subplots import make_subplots

from tooth.kalman import LinearKalmanFilter
from tooth.sims.projectile import launch_state, simulate

DT = 0.05  # s
ACCEL_STD = 2.0  # m/s^2, gusts
MEAS_STD = 2.0  # m
VEL_STD = 5.0  # m/s, honest initial velocity uncertainty
DROPOUT = (1.5, 3.0)  # s
OUTLIER_STEPS = (20, 45, 70)
OUTLIER_SIZE = 20.0  # m, added to x
BIAS = 3.0  # m, added to every x reading

SCENARIOS = ("baseline", "overconfident start", "dropout", "outliers", "sensor bias")
COLORS = {"x": "#2563eb", "y": "#dc2626", "nis": "#16a34a", "bound": "#6b7280", "mark": "#d97706"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x0_true = launch_state(speed=30.0, angle_deg=45.0)
    run = simulate(x0_true, DT, ACCEL_STD, MEAS_STD, rng)
    m, t, truth = run.model, run.t, run.truth
    v_noise = VEL_STD * rng.standard_normal(2)
    nis_lo, nis_hi = chi2_bounds(2)

    fig = make_subplots(
        rows=len(SCENARIOS),
        cols=2,
        subplot_titles=[f"{name}: {col}" for name in SCENARIOS for col in ("position error, ±2σ", "NIS")],
        vertical_spacing=0.05,
        horizontal_spacing=0.08,
    )

    print(f"{'scenario':21s} {'pos RMSE':>9s} {'ANEES (4)':>10s} {'ANIS (2)':>9s} {'NIS > 95% bound':>16s}")
    for row, name in enumerate(SCENARIOS, start=1):
        zs = run.measurements.copy()
        v_guess, v_std = x0_true[2:] + v_noise, VEL_STD
        if name == "overconfident start":
            v_guess, v_std = x0_true[2:] + np.array([8.0, -8.0]), 0.3
        elif name == "dropout":
            zs[(t >= DROPOUT[0]) & (t < DROPOUT[1])] = np.nan
        elif name == "outliers":
            zs[list(OUTLIER_STEPS), 0] += OUTLIER_SIZE
        elif name == "sensor bias":
            zs[:, 0] += BIAS

        x0 = np.concatenate([zs[0], v_guess])
        P0 = np.diag([MEAS_STD**2, MEAS_STD**2, v_std**2, v_std**2])
        kf = LinearKalmanFilter(F=m.F, H=m.H, Q=m.Q, R=m.R, x0=x0, P0=P0, B=m.B)
        res = kf.filter(zs[1:], m.u)
        est = np.vstack([x0, res.x_post])
        sig = np.sqrt(np.diagonal(np.concatenate([P0[None], res.P_post]), axis1=1, axis2=2))
        e_nees = nees(truth[1:], res.x_post, res.P_post)
        e_nis = nis(zs[1:], res.x_prior, res.P_prior, m.H, m.R)
        frac = np.nanmean(e_nis > nis_hi)
        print(
            f"{name:21s} {rmse(est[:, :2] - truth[:, :2]):8.2f}m {e_nees.mean():10.1f} "
            f"{np.nanmean(e_nis):9.1f} {frac:16.0%}"
        )

        first = row == 1
        for i, axis in enumerate("xy"):
            band(fig, t, np.zeros_like(t), sig[:, i], COLORS[axis], f"±2σ {axis}", row, 1)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=est[:, i] - truth[:, i],
                    mode="lines",
                    name=f"{axis} error",
                    legendgroup=axis,
                    showlegend=first,
                    line={"color": COLORS[axis], "width": 1.5},
                ),
                row=row,
                col=1,
            )
        have = ~np.isnan(e_nis)
        fig.add_trace(
            go.Scatter(
                x=t[1:][have],
                y=e_nis[have],
                mode="lines",
                name="NIS",
                showlegend=first,
                line={"color": COLORS["nis"], "width": 1.2},
            ),
            row=row,
            col=2,
        )
        for y in (nis_lo, nis_hi):
            fig.add_trace(
                go.Scatter(
                    x=[t[1], t[-1]],
                    y=[y, y],
                    mode="lines",
                    name="95% χ²₂ bounds",
                    showlegend=first and y == nis_hi,
                    line={"color": COLORS["bound"], "dash": "dot", "width": 1},
                ),
                row=row,
                col=2,
            )
        if name == "dropout":
            for col in (1, 2):
                fig.add_vrect(
                    x0=DROPOUT[0], x1=DROPOUT[1], fillcolor=COLORS["mark"], opacity=0.12, line_width=0, row=row, col=col
                )
        fig.update_yaxes(title_text="m", row=row, col=1)
        fig.update_yaxes(type="log", row=row, col=2)

    for col in (1, 2):
        fig.update_xaxes(title_text="t (s)", row=len(SCENARIOS), col=col)
    fig.update_layout(
        title="Pitfalls: the same flight, five ways (shaded in dropout row: no measurements)",
        template="plotly_white",
        height=1500,
        hovermode="closest",
    )

    out = Path(__file__).parent / "out" / "pitfalls.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
