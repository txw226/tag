# tag
Satellite orbit propagation and Kalman filtering. The filtering half is `tag.filtering`; the propagation half, `tag.propagation`, is a compiled C++ extension (so far a Kepler-equation solver and two-body propagation).

New to Kalman filters? Start with the [tutorial](docs/kalman_tutorial.md).

## Developer Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), git and a C++20 compiler (g++ or clang++). CMake, Ninja and clang-format come from PyPI via uv. uv installs Python 3.14 itself (pinned in `.python-version`).

```sh
git clone git@github.com:txw226/tag.git && cd tag
uv sync                          # create .venv with the package (compiling its C++ extension) and dev tools
uv run pre-commit install        # enable the pre-commit and commit-msg hooks
uv run pytest -q                 # check the setup: all tests should pass
```

On each commit the hooks run `ruff check --fix`, `ruff format`, `mypy` and `clang-format`, and block private keys. A hook that changes files stops the commit; review the changes, `git add` them and commit again. Commit messages must reference an issue, e.g. `#12 add X`. Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`) so `uv.lock` stays in sync. C++ edits are recompiled by the next `uv sync`/`uv run`; `cpp/bench/run.sh` runs the C++ microbenchmarks.
