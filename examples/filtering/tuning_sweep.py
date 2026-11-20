"""Tune a Kalman filter: sweep the process noise Q and measurement noise R and watch the metrics.

The truth is the projectile with gusts of ACCEL_STD and a position sensor
with noise MEAS_STD. The filter's Q (or R) is set to scale × the true value,
for scales from 1/100 to 100, and every setting runs on the same Monte Carlo
flights. For each scale the figure shows:

* RMSE: accuracy, needs the truth.
* ANEES / dof and ANIS / dof: consistency (does P match the actual error?),
  ideal 1. NEES needs the truth; NIS only needs the measurements, so it is
  the one you can compute on real data.
* Lag-1 autocorrelation of the normalized innovations, ideal 0. Positive: the
  filter lags the truth (Q too small / R too big). Negative: it chases the
  measurement noise (Q too big / R too small).

The bottom row shows one flight with Q too small, right, and too big.

Run:  uv run python examples/tuning_sweep.py [--runs N] [--seed N] [--no-show]
Writes examples/out/tuning_sweep.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from _plotting import band, nees, nis, rmse
from plotly.subplots import make_subplots

from tag.filtering.kalman import LinearKalmanFilter
from tag.filtering.sims.projectile import ProjectileRun, launch_state, make_filter, simulate

DT = 0.05  # s
ACCEL_STD = 8.0  # m/s^2, true random acceleration: big enough that mis-tuned Q shows in a ~6 s flight
MEAS_STD = 0.5  # m, true sensor noise: precise, so Q (not R) limits accuracy at the right tuning
VEL_STD = 5.0  # m/s, initial velocity uncertainty
SCALES = np.logspace(-2, 2, 17)
EXAMPLE_SCALES = (0.01, 1.0, 100.0)

COLORS = {"Q": "#2563eb", "R": "#dc2626", "x": "#2563eb", "y": "#dc2626", "ref": "#6b7280"}


def run_filter(run: ProjectileRun, v_guess: np.ndarray, q_scale: float, r_scale: float):
    """Filter one flight with Q and R scaled; returns (estimates, covariances, result)."""
    kf = make_filter(run, velocity_guess=v_guess, velocity_std=VEL_STD)
    m = run.model
    # Same initial estimate and P0 for every scale; only Q and R change.
    kf = LinearKalmanFilter(F=m.F, H=m.H, Q=q_scale * m.Q, R=r_scale * m.R, x0=kf.x, P0=kf.P, B=m.B)
    x0, P0 = kf.x.copy(), kf.P.copy()
    res = kf.filter(run.measurements[1:], m.u)
    return np.vstack([x0, res.x_post]), np.concatenate([P0[None], res.P_post]), res


def metrics(run: ProjectileRun, v_guess: np.ndarray, q_scale: float, r_scale: float) -> dict:
    est, _, res = run_filter(run, v_guess, q_scale, r_scale)
    m, zs = run.model, run.measurements[1:]
    R = r_scale * m.R
    y = zs - res.x_prior @ m.H.T
    S = m.H @ res.P_prior @ m.H.T + R
    nu = y / np.sqrt(np.diagonal(S, axis1=1, axis2=2))  # normalized innovations per axis
    return {
        "sq_err": np.sum((est[:, :2] - run.truth[:, :2]) ** 2, axis=1),
        "nees": nees(run.truth[1:], res.x_post, res.P_post),
        "nis": nis(zs, res.x_prior, res.P_prior, m.H, R),
        "lag1": (nu[1:] * nu[:-1]).sum(axis=0),  # summed; normalized after pooling
        "nu2": (nu**2).sum(axis=0),
    }


def sweep(flights: list, which: str) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {"rmse": [], "anees": [], "anis": [], "lag1": []}
    for s in SCALES:
        q, r = (s, 1.0) if which == "Q" else (1.0, s)
        ms = [metrics(run, v, q, r) for run, v in flights]
        out["rmse"].append(float(np.sqrt(np.mean(np.concatenate([m["sq_err"] for m in ms])))))
        out["anees"].append(float(np.mean(np.concatenate([m["nees"] for m in ms]))))
        out["anis"].append(float(np.mean(np.concatenate([m["nis"] for m in ms]))))
        out["lag1"].append(float(np.mean(sum(m["lag1"] for m in ms) / sum(m["nu2"] for m in ms))))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo flights per setting")
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    x0_true = launch_state(speed=40.0, angle_deg=45.0)
    flights = []
    for i in range(args.runs):
        rng = np.random.default_rng(args.seed + i)
        run = simulate(x0_true, DT, ACCEL_STD, MEAS_STD, rng)
        flights.append((run, x0_true[2:] + VEL_STD * rng.standard_normal(2)))

    res = {w: sweep(flights, w) for w in ("Q", "R")}
    meas_rmse = rmse(np.concatenate([r.measurements - r.truth[:, :2] for r, _ in flights]))
    print(f"{args.runs} flights; raw measurement RMSE {meas_rmse:.2f} m")
    print("scale     " + "".join(f"{s:>8.3g}" for s in SCALES[::2]))
    for w in ("Q", "R"):
        for key in ("rmse", "anees", "anis", "lag1"):
            print(f"{w} {key:6s}  " + "".join(f"{v:8.2f}" for v in res[w][key][::2]))

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Accuracy: position RMSE",
            "Consistency: ANEES/4 (solid), ANIS/2 (dotted); ideal 1",
            "Whiteness: lag-1 innovation autocorrelation; ideal 0",
            *(f"One flight, Q × {s:g}: x error, ±2σ" for s in EXAMPLE_SCALES),
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.07,
    )

    for w in ("Q", "R"):
        c = COLORS[w]
        label = f"{w} × scale"
        fig.add_trace(
            go.Scatter(x=SCALES, y=res[w]["rmse"], mode="lines+markers", name=label, line={"color": c}),
            row=1,
            col=1,
        )
        for key, dof, dash in (("anees", 4, "solid"), ("anis", 2, "dot")):
            fig.add_trace(
                go.Scatter(
                    x=SCALES,
                    y=res[w][key] / dof,
                    mode="lines+markers",
                    name=f"{w}: {key.upper()}/{dof}",
                    showlegend=False,
                    line={"color": c, "dash": dash},
                ),
                row=1,
                col=2,
            )
        fig.add_trace(
            go.Scatter(x=SCALES, y=res[w]["lag1"], mode="lines+markers", showlegend=False, line={"color": c}),
            row=1,
            col=3,
        )
    for col, y in ((1, meas_rmse), (2, 1.0), (3, 0.0)):
        fig.add_trace(
            go.Scatter(
                x=[SCALES[0], SCALES[-1]],
                y=[y, y],
                mode="lines",
                name="raw measurements" if col == 1 else "ideal",
                showlegend=col < 3,
                line={"color": COLORS["ref"], "dash": "dash", "width": 1},
            ),
            row=1,
            col=col,
        )

    run, v = flights[0]
    examples = [run_filter(run, v, s, 1.0) for s in EXAMPLE_SCALES]
    lim = 1.2 * max(np.abs(est[10:, 0] - run.truth[10:, 0]).max() for est, _, _ in examples)
    for col, (est, cov, _) in enumerate(examples, start=1):
        err = est[:, 0] - run.truth[:, 0]
        band(fig, run.t, np.zeros_like(run.t), np.sqrt(cov[:, 0, 0]), COLORS["x"], "±2σ", 2, col)
        fig.add_trace(
            go.Scatter(
                x=run.t,
                y=err,
                mode="lines",
                showlegend=False,
                name="x error",
                line={"color": COLORS["x"], "width": 1.5},
            ),
            row=2,
            col=col,
        )
        fig.update_yaxes(range=[-lim, lim], title_text="error (m)" if col == 1 else None, row=2, col=col)
        fig.update_xaxes(title_text="t (s)", row=2, col=col)

    for col in (1, 2, 3):
        fig.update_xaxes(type="log", title_text="scale (filter value / true value)", row=1, col=col)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(
        title=f"Tuning: scale the filter's Q or R away from the truth ({args.runs} Monte Carlo flights per point)",
        template="plotly_white",
        height=850,
        hovermode="closest",
    )

    out = Path(__file__).parent / "out" / "tuning_sweep.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
