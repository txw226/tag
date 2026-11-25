#include <cstddef>
#include <random>
#include <vector>

#include <nanobench.h>

#include "tag/propagation/two_body.hpp"

namespace {

// Pre-drawn times over several periods, so each call lands at an unpredictable mean anomaly.
constexpr std::size_t n = 4096;

std::vector<double> random_times() {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> time(-86400.0, 86400.0);
  std::vector<double> dts(n);
  for (double &dt : dts) {
    dt = time(rng);
  }
  return dts;
}

struct Case {
  const char *name;
  tag::propagation::State rv;
};
const Case cases[] = {
    {"propagate LEO (e~0.001)", {6778.0, 0.0, 0.0, 0.0, 4.72, 6.04}},
    {"propagate GEO (e=0, i=0)", {42164.1696, 0.0, 0.0, 0.0, 3.0746761, 0.0}},
    {"propagate Molniya (e~0.68)", {-1500.0, -6300.0, 2200.0, 8.2, -2.4, 4.9}},
};

} // namespace

void bench_two_body() {
  const std::vector<double> dts = random_times();

  ankerl::nanobench::Bench bench;
  bench.title("Two-body scalar propagate").unit("call").warmup(10000).minEpochIterations(200000).relative(true);
  for (const Case &c : cases) {
    const tag::propagation::TwoBodyOrbit orbit(c.rv, tag::propagation::earth_mu);
    std::size_t i = 0;
    bench.run(c.name, [&] { ankerl::nanobench::doNotOptimizeAway(orbit.propagate(dts[i++ % n])); });
  }
}

void bench_two_body_batched() {
  const std::vector<double> dts = random_times();

  // The same times, written into a preallocated buffer, reported per propagated point.
  std::vector<double> out(6 * n);
  ankerl::nanobench::Bench batched;
  batched.title("Two-body batched propagate (4096 times)").unit("point").batch(n).warmup(10).minEpochIterations(50);
  for (const Case &c : cases) {
    const tag::propagation::TwoBodyOrbit orbit(c.rv, tag::propagation::earth_mu);
    batched.run(c.name, [&] {
      orbit.propagate(dts, out);
      ankerl::nanobench::doNotOptimizeAway(out.data());
    });
  }
  std::vector<tag::propagation::TwoBodyOrbit> orbits;
  for (const Case &c : cases) {
    orbits.emplace_back(c.rv, tag::propagation::earth_mu);
  }
  std::vector<double> grid(6 * orbits.size() * n);
  batched.batch(orbits.size() * n).run("propagate_grid (3 orbits x 4096 times)", [&] {
    tag::propagation::propagate_grid(orbits, dts, grid);
    ankerl::nanobench::doNotOptimizeAway(grid.data());
  });
}
