#!/usr/bin/env sh
# Build and run the C++ microbenchmarks (Release, native flags), outside the Python build.
# Usage: cpp/bench/run.sh   (from anywhere; needs uv, which provides cmake and ninja)
set -eu
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="$root/build/bench"
uv run --project "$root" cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTAG_BUILD_BENCHMARKS=ON
uv run --project "$root" cmake --build "$build" --target tag_bench
"$build/tag_bench"
