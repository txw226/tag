# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`tooth` is a Python Kalman filtering library (Python >= 3.14, pinned to 3.14 in `.python-version`) managed with **uv**, using the `uv_build` backend and a `src/` layout. Dependencies: numpy, plotly (plots); pytest (dev). It also holds a standalone scratch C++ file.

## Commands

```sh
uv sync                                    # create/update .venv from pyproject.toml + uv.lock
uv run pytest -q                           # run all tests
uv run pytest tests/test_linear_kf.py::test_scalar_constant_matches_running_mean   # single test
uv run python examples/projectile_kf.py    # sim + filter + plotly HTML (examples/out/, gitignored); --no-show, --seed N
uv add <pkg>   /   uv add --dev <pkg>      # add a dependency (keeps uv.lock in sync; don't hand-edit)

g++ src/tooth/try.cpp -o build/try && ./build/try   # scratch C++ (not part of the Python build)
```

The user's shell setup (`~/tooth.zshrc`) activates `.venv` automatically and defines `$ToothRoot` and the `cdtooth` alias.

## Architecture

- `src/tooth/kalman/`: filters. `LinearKalmanFilter` (`linear.py`) has fixed `F, H, Q, R, (B)` matrices; `predict(u)` / `update(z)` / `filter(zs, us)`. The update uses a `solve` for the gain and the Joseph-form covariance update. `update(None)`, or a NaN row in `filter`, skips a measurement. `filter` returns a `FilterResult` with prior and posterior states/covariances for every step.
- `src/tooth/sims/`: physics simulations that produce truth plus noisy measurements and the matching model matrices. `projectile.py`: 2D projectile, state `[x, y, vx, vy]`, gravity as a known control input `u = [-g]`, random gusts as piecewise-constant acceleration, so the discrete model is exact and the KF's Q matches the truth's noise. `make_filter(run, ...)` initializes a KF from the first measurement; feed it `run.measurements[1:]`.
- `examples/`: runnable demos with plotly figures.
- `tests/test_projectile.py` holds a Monte Carlo **NEES consistency test** (pooled NEES ≈ state dim 4) plus tests showing it rejects a Q mis-scaled by 4×. Its bounds come from a 20-seed sweep; if you change the sim parameters it uses, re-derive them. Gust strength there is deliberately 2 m/s² because at 0.5 m/s² the process noise is too small to detect mis-tuning over a ~4 s flight.
- `src/tooth/try.py` / `try.cpp` are scratch files.
