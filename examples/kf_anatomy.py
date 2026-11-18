"""Anatomy of a Kalman filter: one predict/update cycle drawn as covariance ellipses.

A ball thrown straight up, state [height y, vertical velocity vy], measured by a
noisy altimeter (height only). The figure shows, for one cycle:

* Predict: F moves the estimate and shears its uncertainty (position error
  grows with velocity error), then Q inflates it (random gusts).
* Update: H says only y is measured, so the measurement is a vertical strip
  of width set by R. The gain K pulls the estimate toward it; vy is corrected
  too, through the y-vy correlation that predict created.

Then, over the whole flight: the sawtooth of σ (predict grows it, update
shrinks it), the gain settling to a steady state, and the innovations with
their predicted ±2√S band.

Run:  uv run python examples/kf_anatomy.py [--seed N] [--step K] [--no-show]
Writes examples/out/kf_anatomy.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tooth.kalman import LinearKalmanFilter
from tooth.sims.projectile import simulate

DT = 0.25  # s, coarse so one step's shear is visible
ACCEL_STD = 2.0  # m/s^2, gusts
MEAS_STD = 2.0  # m, altimeter noise
VEL_STD = 5.0  # m/s, initial velocity uncertainty
IDX = [1, 3]  # [y, vy] out of the projectile's [x, y, vx, vy]

COLORS = {
    "post_prev": "#6b7280",
    "FPF": "#d97706",
    "prior": "#dc2626",
    "meas": "#16a34a",
    "post": "#2563eb",
    "truth": "#1f2937",
    "y": "#2563eb",
    "vy": "#dc2626",
}


def ellipse(mean: np.ndarray, cov: np.ndarray, n_sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Points on the n_sigma contour of a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, 100)
    pts = vecs @ (n_sigma * np.sqrt(vals)[:, None] * np.vstack([np.cos(a), np.sin(a)]))
    return mean[0] + pts[0], mean[1] + pts[1]


def add_ellipse(fig, mean, cov, color, name, row, col, dash="solid", fill=False) -> None:
    ex, ey = ellipse(mean, cov)
    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            name=name,
            fill="toself" if fill else None,
            fillcolor=color if fill else None,
            opacity=0.9,
            line={"color": color, "width": 2, "dash": dash},
            hoverinfo="name",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=[mean[0]],
            y=[mean[1]],
            mode="markers",
            showlegend=False,
            marker={"color": color, "size": 8},
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=int, default=1, help="which cycle to draw (1 = first)")
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    run = simulate([0.0, 0.0, 0.0, 25.0], DT, ACCEL_STD, MEAS_STD, rng)
    m = run.model
    F, B, Q = m.F[np.ix_(IDX, IDX)], m.B[IDX], m.Q[np.ix_(IDX, IDX)]
    H, R = np.array([[1.0, 0.0]]), np.array([[MEAS_STD**2]])
    truth, zs = run.truth[:, IDX], run.measurements[:, 1:2]

    x0 = np.array([zs[0, 0], 25.0 + VEL_STD * rng.standard_normal()])
    P0 = np.diag([MEAS_STD**2, VEL_STD**2])
    kf = LinearKalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, B=B)

    # Step through by hand, keeping each stage of every cycle.
    log: dict[str, list] = {k: [] for k in ("x_prev", "P_prev", "FPF", "x_prior", "P_prior", "x", "P", "K", "y", "S")}
    for z in zs[1:]:
        log["x_prev"].append(kf.x.copy())
        log["P_prev"].append(kf.P.copy())
        log["FPF"].append(F @ kf.P @ F.T)
        kf.predict(m.u)
        log["x_prior"].append(kf.x.copy())
        log["P_prior"].append(kf.P.copy())
        kf.update(z)
        assert kf.K is not None and kf.y is not None and kf.S is not None
        for k, v in (("x", kf.x), ("P", kf.P), ("K", kf.K[:, 0]), ("y", kf.y[0]), ("S", kf.S[0, 0])):
            log[k].append(np.copy(v))
    L = {k: np.array(v) for k, v in log.items()}
    t = run.t[1:]

    k = args.step - 1
    corr = L["P_prior"][k, 0, 1] / np.sqrt(L["P_prior"][k, 0, 0] * L["P_prior"][k, 1, 1])
    print(f"cycle {args.step} (t = {t[k]:.2f} s)")
    print(
        f"  prior   y = {L['x_prior'][k, 0]:6.2f} m  vy = {L['x_prior'][k, 1]:6.2f} m/s   y-vy correlation {corr:+.2f}"
    )
    print(f"  measured y = {zs[k + 1, 0]:6.2f} m   innovation {L['y'][k]:+.2f} m   K = {L['K'][k].round(3)}")
    print(f"  post    y = {L['x'][k, 0]:6.2f} m  vy = {L['x'][k, 1]:6.2f} m/s")
    print(f"  truth   y = {truth[k + 1, 0]:6.2f} m  vy = {truth[k + 1, 1]:6.2f} m/s")
    print(f"steady-state gain K ≈ {L['K'][-1].round(3)}")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"1. Predict (cycle {args.step}): x⁻ = F x + B u,  P⁻ = F P Fᵀ + Q",
            "2. Update: x = x⁻ + K (z − H x⁻),  P = (I − K H) P⁻",
            "σ of y: grows in predict, shrinks in update",
            "σ of vy (never measured, still shrinks)",
            "Kalman gain K settles to a steady state",
            "Innovation z − H x⁻ with predicted ±2√S",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.09,
    )

    # 1. Predict
    add_ellipse(fig, L["x_prev"][k], L["P_prev"][k], COLORS["post_prev"], "P (last posterior)", 1, 1)
    add_ellipse(fig, L["x_prior"][k], L["FPF"][k], COLORS["FPF"], "F P Fᵀ (sheared)", 1, 1, dash="dash")
    add_ellipse(fig, L["x_prior"][k], L["P_prior"][k], COLORS["prior"], "P⁻ = F P Fᵀ + Q (prior)", 1, 1)
    fig.add_annotation(
        x=L["x_prior"][k, 0],
        y=L["x_prior"][k, 1],
        ax=L["x_prev"][k, 0],
        ay=L["x_prev"][k, 1],
        axref="x",
        ayref="y",
        xref="x",
        yref="y",
        arrowhead=3,
        arrowwidth=1.5,
        arrowcolor=COLORS["truth"],
        text="",
        row=1,
        col=1,
    )

    # 2. Update
    z = zs[k + 1, 0]
    span = 2.6 * np.sqrt(L["P_prior"][k, 1, 1])
    vy_lo, vy_hi = L["x_prior"][k, 1] - span, L["x_prior"][k, 1] + span
    fig.add_trace(
        go.Scatter(
            x=[z - 2 * MEAS_STD, z + 2 * MEAS_STD, z + 2 * MEAS_STD, z - 2 * MEAS_STD],
            y=[vy_lo, vy_lo, vy_hi, vy_hi],
            fill="toself",
            fillcolor=COLORS["meas"],
            opacity=0.15,
            mode="lines",
            line={"width": 0},
            name="measurement z ± 2√R (y only: H = [1 0])",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[z, z],
            y=[vy_lo, vy_hi],
            mode="lines",
            showlegend=False,
            line={"color": COLORS["meas"], "width": 1.5},
        ),
        row=1,
        col=2,
    )
    add_ellipse(fig, L["x_prior"][k], L["P_prior"][k], COLORS["prior"], "prior", 1, 2)
    add_ellipse(fig, L["x"][k], L["P"][k], COLORS["post"], "P (posterior)", 1, 2)
    fig.add_trace(
        go.Scatter(
            x=[truth[k + 1, 0]],
            y=[truth[k + 1, 1]],
            mode="markers",
            name="truth",
            marker={"color": COLORS["truth"], "size": 12, "symbol": "star"},
        ),
        row=1,
        col=2,
    )

    # 3-4. Sawtooth: interleave prior and posterior σ at the same time stamp.
    tt = np.repeat(t, 2)
    for col, (i, name) in enumerate(((0, "y"), (1, "vy")), start=1):
        sig = np.ravel(np.column_stack([np.sqrt(L["P_prior"][:, i, i]), np.sqrt(L["P"][:, i, i])]))
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([[0.0], tt]),
                y=np.concatenate([[np.sqrt(P0[i, i])], sig]),
                mode="lines+markers",
                name=f"σ {name}",
                marker={"size": 4},
                line={"color": COLORS[name], "width": 1.5},
            ),
            row=2,
            col=col,
        )

    # 5. Gain
    for i, name in enumerate(("y", "vy")):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=L["K"][:, i],
                mode="lines+markers",
                name=f"K for {name}",
                marker={"size": 4},
                line={"color": COLORS[name]},
            ),
            row=3,
            col=1,
        )

    # 6. Innovation and its predicted spread
    s2 = 2 * np.sqrt(L["S"])
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([s2, -s2[::-1]]),
            fill="toself",
            fillcolor=COLORS["meas"],
            opacity=0.15,
            line={"width": 0},
            name="±2√S",
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=L["y"],
            mode="lines+markers",
            name="innovation",
            marker={"size": 4},
            line={"color": COLORS["meas"]},
        ),
        row=3,
        col=2,
    )

    for col in (1, 2):
        fig.update_xaxes(title_text="height y (m)", row=1, col=col)
        fig.update_yaxes(title_text="velocity vy (m/s)", row=1, col=col)
        fig.update_xaxes(title_text="t (s)", row=3, col=col)
    fig.update_yaxes(title_text="σ (m)", row=2, col=1)
    fig.update_yaxes(title_text="σ (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="gain", row=3, col=1)
    fig.update_yaxes(title_text="m", row=3, col=2)
    fig.update_layout(
        title="Anatomy of a Kalman filter: ball thrown upward, altimeter measures height only (ellipses: 2σ)",
        template="plotly_white",
        height=1150,
        hovermode="closest",
    )

    out = Path(__file__).parent / "out" / "kf_anatomy.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
