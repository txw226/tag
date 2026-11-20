// The only translation unit that includes nanobind: adapts Python arguments to the C++ core.
#include <cmath>
#include <format>
#include <string>

#include <nanobind/nanobind.h>

#include "tag/propagation/kepler.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_propagation, m) {
  m.doc() = "Compiled core of tag.propagation.";

  m.def(
      "eccentric_anomaly",
      [](double mean_anomaly, double e) {
        if (!(e >= 0.0 && e < 1.0)) {
          throw nb::value_error(std::format("eccentricity must satisfy 0 <= e < 1, got {}", e).c_str());
        }
        if (!std::isfinite(mean_anomaly)) {
          throw nb::value_error(std::format("mean anomaly must be finite, got {}", mean_anomaly).c_str());
        }
        return tag::propagation::eccentric_anomaly(mean_anomaly, e);
      },
      "mean_anomaly"_a, "e"_a,
      "Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (radians).\n\n"
      "E is in the same revolution as M. Raises ValueError unless 0 <= e < 1 and M is finite.");
}
