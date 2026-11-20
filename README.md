# tag
Satellite orbit propagation and Kalman filtering. Today the package holds the filtering half, `tag.filtering`.

New to Kalman filters? Start with the [tutorial](docs/kalman_tutorial.md).

## Developer Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and git. uv installs Python 3.14 itself (pinned in `.python-version`).

```sh
git clone git@github.com:txw226/tooth.git && cd tooth
uv sync                          # create .venv with the package and dev tools (pytest, ruff, mypy, pre-commit)
uv run pre-commit install        # enable the pre-commit and commit-msg hooks
uv run pytest -q                 # check the setup: all tests should pass
```

On each commit the hooks run `ruff check --fix`, `ruff format` and `mypy`, and block private keys. A hook that changes files stops the commit; review the changes, `git add` them and commit again. Commit messages must reference an issue, e.g. `#12 add X`. Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`) so `uv.lock` stays in sync.
