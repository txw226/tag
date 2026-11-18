"""Compare a linear KF with an extended KF on a projectile with quadratic air drag.

All filters get the same GNSS position fixes, the same initial estimate and
the same gust process noise. Only the dynamics model differs:

* LKF: drag-free model (F, B, gravity as control) with the true gust Q.
  The unmodelled drag makes it overconfident.
* LKF inflated Q: the same model, with Q inflated to absorb drag as if it
  were white noise. It is roughly consistent but less accurate.
* EKF: the drag model (one RK4 step with exact Jacobian).

GNSS is linear (H = [I 0]), so the same sensor serves every filter. NEES and
NIS are shown for one run and averaged over a Monte Carlo set (ANEES / ANIS),
with 95% χ² bounds.

Run:  uv run python examples/drag_lkf_vs_ekf.py [--seed N] [--runs M] [--no-show]
Writes examples/out/drag_lkf_vs_ekf.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from _plotting import band, chi2_bounds, nees, nis, rmse
from plotly.subplots import make_subplots

from tooth.kalman import FilterResult, LinearKalmanFilter
from tooth.sensors import GnssPosition
from tooth.sims.drag_projectile import (
    launch_state,
    make_fusion_filter,
    make_model,
    simulate_trajectory,
)
from tooth.sims.projectile import model_matrices

DT = 0.05  # s, filter rate 20 Hz
ACCEL_STD = 0.5  # m/s^2, gusts (the truth's process noise)
INFLATED_ACCEL_STD = 15.0  # m/s^2, from a sweep: pooled ANEES ≈ 4 (but see the time series)
GNSS_STD = 1.5  # m per axis
GNSS_EVERY = 2  # 10 Hz
POS_STD = 0.5  # m, launch-point uncertainty
VEL_STD = 5.0  # m/s, launch-velocity uncertainty
SPEED, ANGLE = 50.0, 45.0  # nominal launch

FILTERS = ("LKF", "LKF inflated Q", "EKF")
COLORS = {
    "truth": "#1f2937",
    "GNSS": "#9ca3af",
    "LKF": "#dc2626",
    "LKF inflated Q": "#d97706",
    "EKF": "#2563eb",
    "bound": "#6b7280",
}


def prior() -> tuple[np.ndarray, np.ndarray]:
    x0 = np.array([0.0, 2.0, *launch_state(SPEED, ANGLE)[2:]])
    P0 = np.diag([POS_STD**2, POS_STD**2, VEL_STD**2, VEL_STD**2])
    return x0, P0


def run_filters(gnss: GnssPosition, zs: np.ndarray, x0: np.ndarray, P0: np.ndarray) -> dict[str, FilterResult]:
    """Run every filter on measurement rows ``zs`` (steps 1..N-1)."""
    lin = model_matrices(DT, ACCEL_STD, 0.0)
    H = gnss.H_jac(x0)  # constant, since GNSS is linear
    out = {}
    for name, q_std in (("LKF", ACCEL_STD), ("LKF inflated Q", INFLATED_ACCEL_STD)):
        Q = q_std**2 * lin.Gamma @ lin.Gamma.T
        kf = LinearKalmanFilter(F=lin.F, H=H, Q=Q, R=gnss.R, x0=x0, P0=P0, B=lin.B)
        out[name] = kf.filter(zs, us=lin.u)
    ekf = make_fusion_filter(make_model(DT, ACCEL_STD), x0, P0)
    out["EKF"] = ekf.filter_sensors([(gnss, zs)])
    return out


def simulate_once(rng: np.random.Generator, gnss: GnssPosition) -> dict:
    """One flight: truth drawn from the prior, GNSS readings, every filter's results and metrics."""
    x0, P0 = prior()
    x0_true = x0 + np.sqrt(np.diag(P0)) * rng.standard_normal(4)
    t, truth = simulate_trajectory(x0_true, DT, ACCEL_STD, rng)
    zs = gnss.measure_series(truth, rng)
    results = run_filters(gnss, zs[1:], x0, P0)
    est, cov, m_nees, m_nis = {}, {}, {}, {}
    for name, res in results.items():
        est[name] = np.vstack([x0, res.x_post])
        cov[name] = np.concatenate([P0[None], res.P_post])
        m_nees[name] = nees(truth[1:], res.x_post, res.P_post)
        m_nis[name] = nis(zs[1:], res.x_prior, res.P_prior, gnss.H_jac(x0), gnss.R)
    return {
        "t": t,
        "truth": truth,
        "zs": zs,
        "est": est,
        "cov": cov,
        "nees": m_nees,
        "nis": m_nis,
    }


def monte_carlo(runs: int, seed: int, gnss: GnssPosition) -> dict:
    """ANEES / ANIS vs time (over the steps every run reaches) and pooled summary metrics."""
    sims = [simulate_once(np.random.default_rng(seed + i), gnss) for i in range(runs)]
    n = min(len(s["t"]) for s in sims) - 1  # metric rows start at step 1
    summary = {}
    anees_t, anis_t = {}, {}
    nees_lo, nees_hi = chi2_bounds(4, runs)
    nis_lo, nis_hi = chi2_bounds(2, runs)
    for name in FILTERS:
        nees_all = np.array([s["nees"][name][:n] for s in sims])
        nis_all = np.array([s["nis"][name][:n] for s in sims])
        anees_t[name] = nees_all.mean(axis=0)
        anis_t[name] = nis_all.mean(axis=0)  # NaN on steps without GNSS (same steps in every run)
        a_nis = anis_t[name][~np.isnan(anis_t[name])]
        pos_err = np.concatenate([s["est"][name][:, :2] - s["truth"][:, :2] for s in sims])
        vel_err = np.concatenate([s["est"][name][:, 2:] - s["truth"][:, 2:] for s in sims])
        summary[name] = {
            "pos RMSE (m)": rmse(pos_err),
            "vel RMSE (m/s)": rmse(vel_err),
            "ANEES (expect 4)": float(np.concatenate([s["nees"][name] for s in sims]).mean()),
            "ANEES in bounds": float(np.mean((anees_t[name] > nees_lo) & (anees_t[name] < nees_hi))),
            "ANIS (expect 2)": float(np.nanmean(np.concatenate([s["nis"][name] for s in sims]))),
            "ANIS in bounds": float(np.mean((a_nis > nis_lo) & (a_nis < nis_hi))),
        }
    t = DT * np.arange(1, n + 1)
    return {
        "t": t,
        "anees": anees_t,
        "anis": anis_t,
        "summary": summary,
        "nees_bounds": (nees_lo, nees_hi),
        "nis_bounds": (nis_lo, nis_hi),
    }


def bounds(fig, t: np.ndarray, lo_hi: tuple[float, float], row: int, col: int) -> None:
    """Dotted horizontal lines at the χ² bounds, spanning ``t``.

    Drawn as traces rather than with add_hline, which silently skips subplots
    that have no traces yet.
    """
    for y in lo_hi:
        fig.add_trace(
            go.Scatter(
                x=[t[0], t[-1]],
                y=[y, y],
                mode="lines",
                showlegend=False,
                name="95% bound",
                hovertemplate=f"95% bound: {y:.3g}<extra></extra>",
                line={"color": COLORS["bound"], "width": 1, "dash": "dot"},
            ),
            row=row,
            col=col,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo runs for ANEES / ANIS")
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    gnss = GnssPosition(std=GNSS_STD, every=GNSS_EVERY)
    one = simulate_once(np.random.default_rng(args.seed), gnss)
    mc = monte_carlo(args.runs, 1000 + args.seed, gnss)
    t, truth, zs = one["t"], one["truth"], one["zs"]

    print(f"steps: {len(t)}  flight time: {t[-1]:.2f} s  range: {truth[-1, 0]:.1f} m")
    print(f"Monte Carlo over {args.runs} runs:")
    for name, row in mc["summary"].items():
        print(f"  {name:15s} " + "  ".join(f"{k} {v:.0%}" if "bounds" in k else f"{k} {v:.2f}" for k, v in row.items()))

    fig = make_subplots(
        rows=6,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{"type": "domain", "colspan": 2}, None],
        ],
        row_heights=[0.26, 0.13, 0.13, 0.15, 0.15, 0.18],
        subplot_titles=(
            f"Trajectory (seed {args.seed})",
            "x error (shaded: ±2σ)",
            "y error",
            "vx error",
            "vy error",
            "NEES, single run (dotted: 95% χ²₄ bounds)",
            "NIS, single run (dotted: 95% χ²₂ bounds)",
            f"ANEES over {args.runs} runs",
            f"ANIS over {args.runs} runs",
            f"Monte Carlo summary ({args.runs} runs)",
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.08,
    )

    # 1. Trajectory
    fig.add_trace(
        go.Scatter(
            x=zs[:, 0],
            y=zs[:, 1],
            mode="markers",
            name="GNSS fixes",
            marker={"color": COLORS["GNSS"], "size": 5, "symbol": "x"},
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
            line={"color": COLORS["truth"], "width": 2.5},
        ),
        row=1,
        col=1,
    )
    for name in FILTERS:
        e = one["est"][name]
        fig.add_trace(
            go.Scatter(
                x=e[:, 0],
                y=e[:, 1],
                mode="lines",
                name=name,
                legendgroup=name,
                line={"color": COLORS[name], "width": 1.5, "dash": "dash"},
            ),
            row=1,
            col=1,
        )

    # 2. State errors with ±2σ bands
    cells = [(2, 1), (2, 2), (3, 1), (3, 2)]
    for i, (row, col) in enumerate(cells):
        for name in FILTERS:
            err = one["est"][name][:, i] - truth[:, i]
            sig = np.sqrt(one["cov"][name][:, i, i])
            band(fig, t, np.zeros_like(t), sig, COLORS[name], f"±2σ {name}", row, col)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=err,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    line={"color": COLORS[name], "width": 1.3},
                ),
                row=row,
                col=col,
            )

    # 3. Single-run NEES / NIS
    bounds(fig, t[1:], chi2_bounds(4), 4, 1)
    bounds(fig, t[1:], chi2_bounds(2), 4, 2)
    for name in FILTERS:
        fig.add_trace(
            go.Scatter(
                x=t[1:],
                y=one["nees"][name],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": COLORS[name], "width": 1.3},
            ),
            row=4,
            col=1,
        )
        have = ~np.isnan(one["nis"][name])
        fig.add_trace(
            go.Scatter(
                x=t[1:][have],
                y=one["nis"][name][have],
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=False,
                marker={"size": 3},
                line={"color": COLORS[name], "width": 1},
            ),
            row=4,
            col=2,
        )

    # 4. Monte Carlo ANEES / ANIS
    bounds(fig, mc["t"], mc["nees_bounds"], 5, 1)
    bounds(fig, mc["t"], mc["nis_bounds"], 5, 2)
    for name in FILTERS:
        fig.add_trace(
            go.Scatter(
                x=mc["t"],
                y=mc["anees"][name],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": COLORS[name], "width": 1.5},
            ),
            row=5,
            col=1,
        )
        have = ~np.isnan(mc["anis"][name])
        fig.add_trace(
            go.Scatter(
                x=mc["t"][have],
                y=mc["anis"][name][have],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": COLORS[name], "width": 1.5},
            ),
            row=5,
            col=2,
        )

    # 5. Summary table
    cols = list(next(iter(mc["summary"].values())))
    fig.add_trace(
        go.Table(
            header={
                "values": ["filter", *cols],
                "fill_color": "#e5e7eb",
                "align": "left",
            },
            cells={
                "values": [list(FILTERS)]
                + [
                    [f"{mc['summary'][n][c]:.0%}" if "bounds" in c else f"{mc['summary'][n][c]:.2f}" for n in FILTERS]
                    for c in cols
                ],
                "font": {"color": [[COLORS[n] for n in FILTERS]] + [["#111827"] * 3] * len(cols)},
                "align": "left",
                "height": 28,
            },
        ),
        row=6,
        col=1,
    )

    fig.update_xaxes(title_text="x (m)", row=1, col=1)
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    for (row, col), unit in zip(cells, ["m", "m", "m/s", "m/s"]):
        fig.update_yaxes(title_text=f"error ({unit})", row=row, col=col)
    for row in (4, 5):
        for col in (1, 2):
            fig.update_yaxes(type="log", row=row, col=col)
    for row in (3, 5):
        for col in (1, 2):
            fig.update_xaxes(title_text="t (s)", row=row, col=col)
    s = mc["summary"]
    fig.update_layout(
        title="LKF vs EKF, drag projectile, same GNSS. Position RMSE: "
        + ", ".join(f"{n} {s[n]['pos RMSE (m)']:.2f} m" for n in FILTERS),
        template="plotly_white",
        height=2000,
        hovermode="closest",
    )

    out = Path(__file__).parent / "out" / "drag_lkf_vs_ekf.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
