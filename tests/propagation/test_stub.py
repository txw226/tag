"""The checked-in type stub for the compiled extension must match what nanobind generates from it,
after the same ruff fixes the pre-commit hook applies. -P keeps underscore-prefixed bindings (the
internal test oracles) in the stub, so tests importing them type-check."""

import subprocess
import sys
from pathlib import Path

import tag.propagation

STUB = Path(tag.propagation.__file__).parent / "_propagation.pyi"
PYPROJECT = STUB.parents[3] / "pyproject.toml"  # src/tag/propagation/ -> repo root
REGENERATE = (
    "uv run python -m nanobind.stubgen -P -m tag.propagation._propagation -o src/tag/propagation/_propagation.pyi"
    " && uv run ruff check -q --fix src/tag/propagation/_propagation.pyi"
    " && uv run ruff format src/tag/propagation/_propagation.pyi"
)


def test_extension_stub_is_up_to_date(tmp_path: Path) -> None:
    generated = tmp_path / "_propagation.pyi"
    run = [sys.executable, "-m"]
    subprocess.run(
        [*run, "nanobind.stubgen", "-q", "-P", "-m", "tag.propagation._propagation", "-o", str(generated)], check=True
    )
    subprocess.run([*run, "ruff", "check", "-q", "--fix", "--config", str(PYPROJECT), str(generated)], check=True)
    subprocess.run([*run, "ruff", "format", "-q", "--config", str(PYPROJECT), str(generated)], check=True)
    assert generated.read_text() == STUB.read_text(), f"stub is stale; regenerate with: {REGENERATE}"
