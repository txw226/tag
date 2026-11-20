// The only translation unit that includes nanobind: adapts Python arguments to the C++ core.
#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include "tag/propagation/cartesian.hpp"
#include "tag/propagation/kepler.hpp"
#include "tag/propagation/two_body.hpp"

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

using Vector6 = nb::ndarray<nb::numpy, double, nb::shape<6>>;

// A new numpy array owning a copy of the state.
Vector6 to_numpy(const tp::State &rv) {
  auto *data = new double[6];
  std::copy(rv.begin(), rv.end(), data);
  nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return Vector6(data, {6}, owner);
}

void require_finite_time(double dt) {
  if (!std::isfinite(dt)) {
    throw nb::value_error(std::format("dt must be finite, got {}", dt).c_str());
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

  nb::class_<tp::TwoBodyOrbit>(
      m, "TwoBodyOrbit",
      "Two-body propagation of an elliptic orbit from an osculating Cartesian state.\n\n"
      "``rv = [x, y, z, vx, vy, vz]`` is the state at the epoch, in units consistent with ``mu``\n"
      "(default EARTH_MU: km, km/s, s). Near-circular, near-equatorial and GEO orbits are\n"
      "accurate. Raises HyperbolicOrbitError if e >= 1 (hyperbolic, parabolic or rectilinear),\n"
      "and ValueError for a non-finite state, zero position or mu <= 0.")
      .def(
          "__init__",
          [](tp::TwoBodyOrbit *self, const tp::State &rv, double mu) {
            require_elliptic(rv, mu);
            new (self) tp::TwoBodyOrbit(rv, mu);
          },
          nb::sig("def __init__(self, rv: numpy.typing.ArrayLike, mu: float = EARTH_MU) -> None"), "rv"_a,
          "mu"_a = tp::earth_mu)
      .def(
          "propagate",
          [](const tp::TwoBodyOrbit &orbit, double dt) {
            require_finite_time(dt);
            return to_numpy(orbit.propagate(dt));
          },
          nb::sig("def propagate(self, dt: float) -> numpy.typing.NDArray[numpy.float64]"), "dt"_a,
          "The Cartesian state ``dt`` seconds (any sign) after the epoch, as a float64 array of shape (6,).\n\n"
          "Precision doesn't decay with long ``dt``: the mean anomaly is reduced modulo 2 pi.")
      .def_prop_ro("mu", &tp::TwoBodyOrbit::mu, "Gravitational parameter.")
      .def_prop_ro("semi_major_axis", &tp::TwoBodyOrbit::semi_major_axis, "Semi-major axis a.")
      .def_prop_ro("eccentricity", &tp::TwoBodyOrbit::eccentricity, "Eccentricity e, in [0, 1).")
      .def_prop_ro(
          "period", [](const tp::TwoBodyOrbit &orbit) { return 2.0 * std::numbers::pi / orbit.mean_motion(); },
          "Orbital period, 2 pi / n.");
}
