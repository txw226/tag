# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`tag` is a Python Kalman filtering library (linear KF + EKF), with a C++ orbit propagator planned (see `docs/adr/0001`). It needs Python >= 3.14 (pinned to 3.14 in `.python-version`) and is managed with **uv**, using the `uv_build` backend and a `src/` layout. Dependencies: numpy, plotly (plots); pytest, ruff, mypy, pre-commit (dev). All code is kept ruff-clean (check + format) and mypy-clean; the pre-commit hook enforces both, and also blocks staged private keys (`detect-private-key`). Commit messages must reference an issue as `#N` (convention: `#N summary`); a commit-msg hook rejects ones that don't, except git-generated Merge/Revert/fixup! messages, and if a hook modifies files the commit fails so the fixes can be reviewed and re-staged.

## Commands

```sh
uv sync                                    # create/update .venv from pyproject.toml + uv.lock
uv run pytest -q                           # run all tests
uv run pytest tests/filtering/test_linear_kf.py::test_scalar_constant_matches_running_mean   # single test
uv run python examples/filtering/projectile_kf.py    # sim + filter + plotly HTML (examples/filtering/out/, gitignored); --no-show, --seed N
uv run python examples/filtering/drag_projectile_ekf.py   # same for the EKF on the drag/radar sim
uv run python examples/filtering/drag_projectile_fusion.py  # EKF fusing GNSS + two cameras, vs each alone
uv run python examples/filtering/rocket_drag_estimation.py  # EKF estimating a rocket's unknown Cd (state augmentation)
uv run python examples/filtering/drag_lkf_vs_ekf.py      # drag-free LKF (true / inflated Q) vs EKF on the same GNSS; NEES/NIS, Monte Carlo (--runs N)
uv run python examples/filtering/kf_anatomy.py           # tutorial: one predict/update cycle as covariance ellipses (--step K)
uv run python examples/filtering/tuning_sweep.py         # tutorial: sweep Q / R scale; RMSE, ANEES, ANIS, innovation whiteness (--runs N)
uv run python examples/filtering/pitfalls.py             # tutorial: bad P0, dropout, outliers, sensor bias vs the metrics
uv add <pkg>   /   uv add --dev <pkg>      # add a dependency (keeps uv.lock in sync; don't hand-edit)
uv run ruff check --fix && uv run ruff format   # lint + format (config in pyproject.toml, line length 120)
uv run mypy                                # type-check src, tests, examples (config in pyproject.toml)
uv run pre-commit install                  # once per clone: installs the pre-commit + commit-msg hooks (see .pre-commit-config.yaml)
uv run pre-commit run --all-files          # run the hooks over the whole repo
```

The user's shell setup (`~/tag.zshrc`) activates `.venv` automatically and defines `$TagRoot` and the `cdtag` alias.

## Architecture

One package `tag` (ADR 0001): `tag.filtering` holds everything below, and a future `tag.propagation` the orbit propagator; filtering may import propagation, never the reverse. No namespace package. Tests and examples follow the same split: `tests/filtering/`, `examples/filtering/`, and the root `tests/`/`examples/` for code using both halves (pytest runs in `importlib` mode, so same-named test files in different folders don't clash). `tag:main` is the `tag` console script.

- `src/tag/filtering/kalman/`: filters. `_base.py` holds `FilterResult`, `joseph_update` (the gain is computed with a `solve`, then the Joseph-form covariance update) and `KalmanFilterBase`, which provides `step`/`filter` on top of a subclass's `predict(u)`/`update(z)`. `update(None)`, or a NaN row in `filter`, skips a measurement. `filter` returns prior and posterior states/covariances for every step.
  - `LinearKalmanFilter` (`linear.py`): fixed `F, H, Q, R, (B)` matrices.
  - `ExtendedKalmanFilter` (`extended.py`): callables `f(x, u)`, `h(x)`, with optional analytic `F_jac(x, u)`/`H_jac(x)` (central-difference `numerical_jacobian` otherwise) and an optional `residual(z, z_pred)` for angle wrapping. With linear f/h it reproduces `LinearKalmanFilter` exactly (tested). For fusion, `h`/`R` may be omitted. `update(z, sensor)` then uses any object with `h`, `H_jac`, `R` and `residual`, and `filter_sensors([(sensor, zs), ...])` does one predict per step followed by a sequential update for each stream whose row isn't NaN.
- `src/tag/filtering/sims/`: physics simulations that produce truth plus noisy measurements and the matching model matrices. `projectile.py`: 2D projectile, state `[x, y, vx, vy]`, gravity as a known control input `u = [-g]`, random gusts as piecewise-constant acceleration, so the discrete model is exact and the KF's Q matches the truth's noise. `make_filter(run, ...)` initializes a KF from the first measurement; feed it `run.measurements[1:]`.
- `src/tag/filtering/sensors/`: reusable measurement models (`Sensor` base class): `RangeBearingRadar`, `PinholeCamera` (2D world, so a 1D image `u = f·lateral/depth`; angle-only, and two cameras give stereo) and `GnssPosition`. Conventions: position is at state indices 0 and 1, Jacobians are zero-padded to `len(x)`, `R = diag(noise_std²)`, and `measure_series(truth, rng)` returns NaN rows when the step is off-rate (`every`) or the target isn't `visible`. Noise is drawn for every row, so masking doesn't shift the RNG stream.
- `src/tag/filtering/sims/drag_projectile.py`: projectile with quadratic drag `-k|v|v` (default k = 0.008/m). `simulate_trajectory` gives the truth only, using RK4 with substeps. `simulate` adds a `RangeBearingRadar` at `(-100, 0)` (stored as `run.model.radar`; `model.require_radar()` returns it, or raises if the model was built without radar noise levels). The model's `f` is a single RK4 step, and `F_jac` is that step's exact chained Jacobian. Its `Q` reuses the drag-free `Gamma`, which is approximate under drag. `make_filter` returns a radar EKF initialized from the first measurement. `make_fusion_filter(model, x0, P0)` returns an EKF with no default sensor.
- `src/tag/filtering/sims/rocket.py`: sounding rocket from rail exit to apogee. Thrust acts along v̂ (a gravity turn), and a known `Motor` gives constant thrust with linearly falling mass. It uses an exponential atmosphere. The filter state is augmented to `[x, y, vx, vy, Cd]`, with no process noise on Cd (the truth is constant, and the normalized Cd error checks out without a random walk). Thrust and mass reach the filter as control input `run.controls` = [step-mean thrust, mid-step mass], aligned with measurement rows `[1:]`, so pass `us=run.controls`. The truth holds thrust at its exact mean over each substep, which keeps impulse right at burnout (the rocket equation is tested to 1e-10). `make_filter(..., cd_std=0)` freezes Cd at the guess.
- `src/tag/filtering/sims/_integrate.py`: generic `rk4_step(deriv(t, x), x, dt, deriv_jac)` with the exact chained Jacobian, shared by the drag and rocket sims.
- `docs/kalman_tutorial.md`: concept tutorial (how a KF works, not the API), built around the examples; its numbers come from the examples' default seeds, so update them if an example's parameters change. `docs/img/*.svg` are static diagrams referenced by it.
- `examples/filtering/`: runnable demos with plotly figures; `_plotting.py` holds the shared `band`/`rmse`/`nees`/`nis` helpers and exact χ² quantiles (`chi2_ppf`/`chi2_bounds`, incomplete gamma + bisection, no scipy).
- `tests/filtering/test_projectile.py` holds a Monte Carlo **NEES consistency test** (pooled NEES ≈ state dim 4) plus tests showing it rejects a Q mis-scaled by 4×. Its bounds come from a 20-seed sweep; if you change the sim parameters it uses, re-derive them. Gust strength there is deliberately 2 m/s² because at 0.5 m/s² the process noise is too small to detect mis-tuning over a ~4 s flight.
- `tests/filtering/test_drag_projectile.py` holds the same kind of NEES test for the EKF (gusts 2 m/s², 50 m/s launch). Its bounds also come from a 20-seed sweep, so re-derive them if those sim parameters change.
- `tests/filtering/test_fusion.py` holds a NEES test for GNSS + two cameras, with bounds also taken from a 20-seed sweep. Keep the cameras well away from the flight path: a target within a few metres of a camera makes the projection too nonlinear, and NEES spikes above 100. Stereo-only errors are heavy-tailed, so compare RMSE over at least 20 seeds.
- `tests/filtering/test_rocket.py` holds a NEES test over the 5-element augmented state (≈ 5), with bounds from a 20-seed sweep. It's the slowest test file (about 13 s), because each NEES call simulates 50 flights of about 270 steps.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for txw226/tag, using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels are used as-is (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root, created only when a term or decision first needs recording. See `docs/agents/domain.md`.
