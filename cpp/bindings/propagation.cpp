// The only translation unit that includes nanobind: adapts Python arguments to the C++ core.
#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "tag/propagation/cartesian.hpp"
#include "tag/propagation/elements.hpp"
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

// Stub annotations for the signatures below (nanobind's defaults, Sequence[float] and
// Annotated[..., dict(shape=...)], reject numpy input under mypy and fail ruff's C408).
#define ARRAY_LIKE "numpy.typing.ArrayLike"
#define FLOAT_ARRAY "numpy.typing.NDArray[numpy.float64]"
#define ANOMALY "typing.Literal['mean', 'true', 'eccentric']"

// A float64 numpy array of shape (6,) owning a copy of the values.
using Vector6 = nb::ndarray<nb::numpy, double, nb::shape<6>>;

Vector6 to_numpy(const tp::State &rv) {
  auto *data = new double[6];
  std::copy(rv.begin(), rv.end(), data);
  nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return Vector6(data, {6}, owner);
}

tp::Anomaly parse_anomaly(const std::string &anomaly) {
  if (anomaly == "mean") {
    return tp::Anomaly::mean;
  }
  if (anomaly == "true") {
    return tp::Anomaly::true_anomaly;
  }
  if (anomaly == "eccentric") {
    return tp::Anomaly::eccentric;
  }
  throw nb::value_error(std::format("anomaly must be 'mean', 'true' or 'eccentric', got '{}'", anomaly).c_str());
}

void require_finite(const std::array<double, 6> &values, const char *what) {
  for (const double x : values) {
    if (!std::isfinite(x)) {
      throw nb::value_error(std::format("{} must be finite", what).c_str());
    }
  }
}

// Validates a semi-major axis and an eccentricity as an elliptic orbit.
void require_elliptic_shape(double a, double e) {
  if (!(a > 0.0)) {
    throw nb::value_error(std::format("semi-major axis must be positive, got {}", a).c_str());
  }
  if (e < 0.0) {
    throw nb::value_error(std::format("eccentricity must be non-negative, got {}", e).c_str());
  }
  if (e >= 1.0) {
    throw HyperbolicOrbitError(
        std::format("orbit is not elliptic: eccentricity {} (e >= 1 is a hyperbolic or parabolic orbit)", e));
  }
}

} // namespace

NB_MODULE(_propagation, m) {
  m.doc() = "Compiled core of tag.propagation.";

  nb::exception<HyperbolicOrbitError>(m, "HyperbolicOrbitError", PyExc_ValueError);
  m.attr("EARTH_MU") = tp::earth_mu;

  // Indices into Keplerian [a, e, i, raan, argp, anomaly] and equinoctial [a, h, k, p, q, lambda] arrays.
  m.attr("KEP_A") = tp::kep::a;
  m.attr("KEP_E") = tp::kep::e;
  m.attr("KEP_I") = tp::kep::i;
  m.attr("KEP_RAAN") = tp::kep::raan;
  m.attr("KEP_ARGP") = tp::kep::argp;
  m.attr("KEP_ANOMALY") = tp::kep::anomaly;
  m.attr("EQ_A") = tp::eq::a;
  m.attr("EQ_H") = tp::eq::h;
  m.attr("EQ_K") = tp::eq::k;
  m.attr("EQ_P") = tp::eq::p;
  m.attr("EQ_Q") = tp::eq::q;
  m.attr("EQ_LAMBDA") = tp::eq::lambda;

  m.def(
      "cartesian_to_keplerian",
      [](const tp::State &rv, double mu, const std::string &anomaly) {
        const tp::Anomaly kind = parse_anomaly(anomaly);
        require_elliptic(rv, mu);
        return to_numpy(tp::cartesian_to_keplerian(rv, mu, kind));
      },
      "rv"_a, "mu"_a = tp::earth_mu, "anomaly"_a = "mean",
      nb::sig("def cartesian_to_keplerian(rv: " ARRAY_LIKE ", mu: float = EARTH_MU, anomaly: " ANOMALY
              " = 'mean') -> " FLOAT_ARRAY),
      "Convert a Cartesian state [r, v] to Keplerian elements [a, e, i, raan, argp, anomaly].\n\n"
      "Angles are in radians, in [0, 2pi) (i in [0, pi]); index with the KEP_* constants. anomaly selects\n"
      "the last element: 'mean', 'true' or 'eccentric'.\n\n"
      "Singular cases (e or sin(i) below 1e-11) follow the usual conventions:\n"
      "- circular: argp = 0 and the anomaly is measured from the ascending node;\n"
      "- equatorial: raan = 0 and argp is measured from the x axis;\n"
      "- circular and equatorial: raan = argp = 0 and the anomaly is the true longitude.\n"
      "Near these cases raan, argp and the anomaly are ill-conditioned individually; use\n"
      "cartesian_to_equinoctial instead.\n\n"
      "Raises HyperbolicOrbitError if e >= 1, ValueError for invalid input.");

  m.def(
      "keplerian_to_cartesian",
      [](const tp::Elements &kep, double mu, const std::string &anomaly) {
        const tp::Anomaly kind = parse_anomaly(anomaly);
        require_positive_mu(mu);
        require_finite(kep, "elements");
        require_elliptic_shape(kep[tp::kep::a], kep[tp::kep::e]);
        return to_numpy(tp::keplerian_to_cartesian(kep, mu, kind));
      },
      "elements"_a, "mu"_a = tp::earth_mu, "anomaly"_a = "mean",
      nb::sig("def keplerian_to_cartesian(elements: " ARRAY_LIKE ", mu: float = EARTH_MU, anomaly: " ANOMALY
              " = 'mean') -> " FLOAT_ARRAY),
      "Convert Keplerian elements [a, e, i, raan, argp, anomaly] to a Cartesian state [r, v].\n\n"
      "Angles are in radians (any value); index with the KEP_* constants. anomaly says what the last\n"
      "element is: 'mean', 'true' or 'eccentric'. Raises HyperbolicOrbitError if e >= 1, ValueError for\n"
      "invalid input.");

  m.def(
      "cartesian_to_equinoctial",
      [](const tp::State &rv, double mu) {
        require_elliptic(rv, mu);
        return to_numpy(tp::cartesian_to_equinoctial(rv, mu));
      },
      "rv"_a, "mu"_a = tp::earth_mu,
      nb::sig("def cartesian_to_equinoctial(rv: " ARRAY_LIKE ", mu: float = EARTH_MU) -> " FLOAT_ARRAY),
      "Convert a Cartesian state [r, v] to equinoctial elements [a, h, k, p, q, lambda].\n\n"
      "Direct set: h = e sin(argp + raan), k = e cos(argp + raan), p = tan(i/2) sin(raan),\n"
      "q = tan(i/2) cos(raan), lambda = M + argp + raan (mean longitude, radians, in [0, 2pi)); index with\n"
      "the EQ_* constants. Well defined for circular and equatorial orbits; singular only for retrograde\n"
      "equatorial orbits (i = 180 deg). Raises HyperbolicOrbitError if e >= 1, ValueError for invalid input.");

  m.def(
      "equinoctial_to_cartesian",
      [](const tp::Elements &eq, double mu) {
        require_positive_mu(mu);
        require_finite(eq, "elements");
        require_elliptic_shape(eq[tp::eq::a], std::hypot(eq[tp::eq::h], eq[tp::eq::k]));
        return to_numpy(tp::equinoctial_to_cartesian(eq, mu));
      },
      "elements"_a, "mu"_a = tp::earth_mu,
      nb::sig("def equinoctial_to_cartesian(elements: " ARRAY_LIKE ", mu: float = EARTH_MU) -> " FLOAT_ARRAY),
      "Convert equinoctial elements [a, h, k, p, q, lambda] to a Cartesian state [r, v].\n\n"
      "See cartesian_to_equinoctial for the definitions; index with the EQ_* constants. Raises\n"
      "HyperbolicOrbitError if e = sqrt(h^2 + k^2) >= 1, ValueError for invalid input.");

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
