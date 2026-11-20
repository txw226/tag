# Tag is one compiled package, restructured in place from tooth

The C++ orbit propagator and the Python Kalman filtering code live in one repo, `Tag` (the GitHub repo txw226/tooth renamed, so history, issues and the `#N` commit convention carry over), as a **single** installable package `tag` built with scikit-build-core (CMake) and bound with nanobind. Filtering code lives under `tag.filtering` (kalman, sensors, sims, and future filtering-only tools such as metrics and plotting), and the propagator under `tag.propagation`. We considered a uv workspace with two packages (`uv_build` for the pure-Python filtering half and scikit-build-core for the C++ half), sharing a PEP 420 `tag` namespace. We rejected it because filtering depends on propagation (orbit process models use the propagator and its analytic STM), so installing the filtering half would compile the C++ anyway, and the split would only add machinery.

## Consequences

- Every `uv sync` builds the C++ extension. uv `cache-keys` on the C++ sources make it rebuild only when they change; Python edits apply live through scikit-build-core's editable mode.
- Builds are for personal use only: `-O3 -march=native` locally, portable flags in CI, no PyPI wheels.
