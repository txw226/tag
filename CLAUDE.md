# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`tooth` is an early-stage Python package (Python >= 3.14, pinned to 3.14 in `.python-version`) managed with **uv**, using the `uv_build` backend and a `src/` layout. It also holds standalone C++ source alongside the Python code. There is no test suite, linter, or formatter configured yet.

## Commands

```sh
uv sync                          # create/update .venv from pyproject.toml + uv.lock
uv run tooth                     # run the console entry point (tooth:main in src/tooth/__init__.py)
uv run python src/tooth/try.py   # run the scratch Python script
uv add <pkg>                     # add a dependency (updates pyproject.toml and uv.lock)

g++ src/tooth/try.cpp -o build/try && ./build/try   # build/run the C++ file (build/ is gitignored)
```

The user's shell setup (`~/tooth.zshrc`) activates `.venv` automatically and defines `$ToothRoot` and the `cdtooth` alias.

## Structure notes

- `src/tooth/__init__.py`: package root; `main()` is the `tooth` console script declared in `[project.scripts]`.
- `src/tooth/try.py` and `src/tooth/try.cpp` are scratch/experiment files. The C++ is **not** wired into the Python build (there is no extension module, CMake, or pybind11). Compile it separately.
- `uv.lock` is committed; keep it in sync by changing dependencies through `uv add` / `uv remove` rather than editing `pyproject.toml` by hand.
