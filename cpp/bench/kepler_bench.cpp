#include <cstddef>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <nanobench.h>

#include "tag/propagation/kepler.hpp"

void bench_kepler() {
  // Pre-drawn inputs over the full M range, for low and high eccentricity. Cycling through a table
  // keeps the inputs unpredictable while costing only a load per call.
  constexpr std::size_t n = 4096;
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> mean_anomaly(-std::numbers::pi, std::numbers::pi);
  std::vector<double> ms(n);
  for (double &m : ms) {
    m = mean_anomaly(rng);
  }

  ankerl::nanobench::Bench bench;
  bench.title("Kepler solve").unit("solve").warmup(10000).minEpochIterations(200000).relative(true);
  for (const double e : {0.01, 0.5, 0.99}) {
    std::size_t i = 0;
    bench.run("eccentric_anomaly e=" + std::to_string(e).substr(0, 4),
              [&] { ankerl::nanobench::doNotOptimizeAway(tag::propagation::eccentric_anomaly(ms[i++ % n], e)); });
  }
}
