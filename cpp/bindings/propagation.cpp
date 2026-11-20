// The only translation unit that includes nanobind: adapts Python arguments to the C++ core.
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>

#include "tag/propagation/cartesian.hpp"
#include "tag/propagation/kepler.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace tp = tag::propagation;

namespace {

// Raised as tag.propagation.HyperbolicOrbitError (a ValueError subclass).
class HyperbolicOrbitError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

void require_positive_mu(double mu) {
  if (!(std::isfinite(mu) && mu > 0.0)) {
    throw nb::value_error(std::format("mu must be finite and positive, got {}", mu).c_str());
  }
}

// Validates a Cartesian state as the start of an elliptic two-body orbit: finite, r != 0, and
// e < 1 with nonzero angular momentum (a rectilinear orbit has e = 1). Establishes the
// preconditions of the core's orbit and element functions.
void require_elliptic(const tp::State &rv, double mu) {
  require_positive_mu(mu);
  for (const double x : rv) {
    if (!std::isfinite(x)) {
      throw nb::value_error("state must be finite");
    }
  }
  const tp::Vec3 r = tp::position(rv);
  const tp::Vec3 v = tp::velocity(rv);
  const double r_norm = tp::norm(r);
  if (r_norm == 0.0) {
    throw nb::value_error("position must be nonzero");
  }
  const double energy = 0.5 * tp::dot(v, v) - mu / r_norm;
  const double h = tp::norm(tp::cross(r, v));
  const double e = tp::norm(tp::eccentricity_vector(rv, mu));
  if (!(energy < 0.0 && h > 0.0 && e < 1.0)) {
    throw HyperbolicOrbitError(
        std::format("orbit is not elliptic: eccentricity {} (e >= 1 is a hyperbolic, parabolic or rectilinear orbit)",
                    h > 0.0 ? e : 1.0));
  }
}

} // namespace

NB_MODULE(_propagation, m) {
  m.doc() = "Compiled core of tag.propagation.";

  nb::exception<HyperbolicOrbitError>(m, "HyperbolicOrbitError", PyExc_ValueError);
  m.attr("EARTH_MU") = tp::earth_mu;

  m.def(
      "eccentric_anomaly",
      [](double mean_anomaly, double e) {
        if (!(e >= 0.0 && e < 1.0)) {
          throw nb::value_error(std::format("eccentricity must satisfy 0 <= e < 1, got {}", e).c_str());
        }
        if (!std::isfinite(mean_anomaly)) {
          throw nb::value_error(std::format("mean anomaly must be finite, got {}", mean_anomaly).c_str());
        }
        return tp::eccentric_anomaly(mean_anomaly, e);
      },
      "mean_anomaly"_a, "e"_a,
      "Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (radians).\n\n"
      "E is in the same revolution as M. Raises ValueError unless 0 <= e < 1 and M is finite.");
}
