# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`tooth` is a Python Kalman filtering library (linear KF + EKF) (Python >= 3.14, pinned to 3.14 in `.python-version`) managed with **uv**, using the `uv_build` backend and a `src/` layout. Dependencies: numpy, plotly (plots); pytest (dev). It also holds a standalone scratch C++ file.

## Commands

```sh
uv sync                                    # create/update .venv from pyproject.toml + uv.lock
uv run pytest -q                           # run all tests
uv run pytest tests/test_linear_kf.py::test_scalar_constant_matches_running_mean   # single test
uv run python examples/projectile_kf.py    # sim + filter + plotly HTML (examples/out/, gitignored); --no-show, --seed N
uv run python examples/drag_projectile_ekf.py   # same for the EKF on the drag/radar sim
uv add <pkg>   /   uv add --dev <pkg>      # add a dependency (keeps uv.lock in sync; don't hand-edit)

g++ src/tooth/try.cpp -o build/try && ./build/try   # scratch C++ (not part of the Python build)
```

The user's shell setup (`~/tooth.zshrc`) activates `.venv` automatically and defines `$ToothRoot` and the `cdtooth` alias.

## Architecture

- `src/tooth/kalman/`: filters. `_base.py` holds `FilterResult`, `joseph_update` (the gain is computed with a `solve`, then the Joseph-form covariance update) and `KalmanFilterBase`, which provides `step`/`filter` on top of a subclass's `predict(u)`/`update(z)`. `update(None)`, or a NaN row in `filter`, skips a measurement. `filter` returns prior and posterior states/covariances for every step.
  - `LinearKalmanFilter` (`linear.py`): fixed `F, H, Q, R, (B)` matrices.
  - `ExtendedKalmanFilter` (`extended.py`): callables `f(x, u)`, `h(x)`, with optional analytic `F_jac(x, u)`/`H_jac(x)` (central-difference `numerical_jacobian` otherwise) and an optional `residual(z, z_pred)` for angle wrapping. With linear f/h it reproduces `LinearKalmanFilter` exactly (tested).
- `src/tooth/sims/`: physics simulations that produce truth plus noisy measurements and the matching model matrices. `projectile.py`: 2D projectile, state `[x, y, vx, vy]`, gravity as a known control input `u = [-g]`, random gusts as piecewise-constant acceleration, so the discrete model is exact and the KF's Q matches the truth's noise. `make_filter(run, ...)` initializes a KF from the first measurement; feed it `run.measurements[1:]`.
- `src/tooth/sims/drag_projectile.py`: projectile with quadratic drag `-k|v|v` (default k = 0.008/m), observed by a radar at `(-100, 0)` measuring `[range, bearing]`. Truth uses RK4 with substeps. The model's `f` is a single RK4 step, and `F_jac` is that step's exact chained Jacobian. Its `Q` reuses the drag-free `Gamma`, which is approximate under drag. `make_filter` returns an EKF initialized from the first measurement.
- `examples/`: runnable demos with plotly figures; `_plotting.py` holds the shared `band`/`rmse` helpers.
- `tests/test_projectile.py` holds a Monte Carlo **NEES consistency test** (pooled NEES ≈ state dim 4) plus tests showing it rejects a Q mis-scaled by 4×. Its bounds come from a 20-seed sweep; if you change the sim parameters it uses, re-derive them. Gust strength there is deliberately 2 m/s² because at 0.5 m/s² the process noise is too small to detect mis-tuning over a ~4 s flight.
- `tests/test_drag_projectile.py` holds the same kind of NEES test for the EKF (gusts 2 m/s², 50 m/s launch). Its bounds also come from a 20-seed sweep, so re-derive them if those sim parameters change.
- `src/tooth/try.py` / `try.cpp` are scratch files.
