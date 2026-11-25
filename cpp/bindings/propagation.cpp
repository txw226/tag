// The only translation unit that includes nanobind: adapts Python arguments to the C++ core.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "tag/propagation/cartesian.hpp"
#include "tag/propagation/central_difference.hpp"
#include "tag/propagation/elements.hpp"
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

// A float64 numpy array of shape (6, 6) owning a copy of the row-major values.
using Matrix66 = nb::ndarray<nb::numpy, double, nb::shape<6, 6>>;

Matrix66 to_numpy(const tp::Matrix6 &m) {
  auto *data = new double[36];
  std::copy(m.begin(), m.end(), data);
  nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return Matrix66(data, {6, 6}, owner);
}

void require_finite_time(double dt) {
  if (!std::isfinite(dt)) {
    throw nb::value_error(std::format("dt must be finite, got {}", dt).c_str());
  }
}

// A batched input as a C-contiguous float64 array. Shapes are checked by hand for clearer errors
// than nanobind's.
using ConvertedInput = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
// A caller's out buffer: written in place, so never converted.
using OutputBuffer = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;

std::string shape_string(std::span<const std::size_t> shape) {
  std::string s = "(";
  for (std::size_t d = 0; d < shape.size(); ++d) {
    s += std::format("{}{}", d ? ", " : "", shape[d]);
  }
  return s + (shape.size() == 1 ? ",)" : ")");
}

std::string array_shape_string(const auto &array) {
  std::vector<std::size_t> shape(array.ndim());
  for (std::size_t d = 0; d < shape.size(); ++d) {
    shape[d] = array.shape(d);
  }
  return shape_string(shape);
}

// Any real numeric array-like (lists, scalars, integer dtypes, any layout), converted by numpy
// unless it already is a C-contiguous float64 array; other dtypes (bool, complex, str, object)
// raise TypeError rather than being coerced. The result keeps the converted array alive.
ConvertedInput as_doubles(nb::handle x, const char *name) {
  ConvertedInput array;
  if (nb::try_cast(x, array, false)) {
    return array;
  }
  const nb::module_ numpy = nb::module_::import_("numpy");
  nb::object converted = numpy.attr("asarray")(x);
  const auto kind = nb::cast<std::string>(converted.attr("dtype").attr("kind"));
  if (kind != "f" && kind != "i" && kind != "u") {
    const auto dtype = nb::cast<std::string>(converted.attr("dtype").attr("name"));
    throw nb::type_error(std::format("{} must be real numbers, got dtype {}", name, dtype).c_str());
  }
  converted = numpy.attr("ascontiguousarray")(converted, "dtype"_a = "float64");
  return nb::cast<ConvertedInput>(converted, false);
}

// The validated times, as an owned copy: the loop runs without the GIL and may write into memory
// the caller's dts shares (out could alias it), so it must not read the caller's array.
std::vector<double> checked_times(nb::handle dts_like) {
  const ConvertedInput dts = as_doubles(dts_like, "dts");
  if (dts.ndim() != 1) {
    throw nb::value_error(std::format("dts must be 1-D, got shape {}", array_shape_string(dts)).c_str());
  }
  std::vector<double> times(dts.data(), dts.data() + dts.shape(0));
  for (std::size_t i = 0; i < times.size(); ++i) {
    if (!std::isfinite(times[i])) {
      throw nb::value_error(std::format("dts[{}] must be finite, got {}", i, times[i]).c_str());
    }
  }
  return times;
}

// The output array and its elements: out itself when given (checked, never copied), else a new
// array of the given shape. Allocates before any computation, so the loop doesn't.
std::pair<nb::object, std::span<double>> output_array(nb::handle out, std::initializer_list<std::size_t> shape) {
  std::size_t size = 1;
  for (const std::size_t n : shape) {
    size *= n;
  }
  if (out.is_none()) {
    auto *data = new double[size];
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return {nb::cast(OutputBuffer(data, shape.size(), shape.begin(), owner)), {data, size}};
  }
  OutputBuffer array;
  if (!nb::try_cast(out, array, false)) {
    throw nb::type_error("out must be a writable, C-contiguous numpy array of dtype float64");
  }
  bool match = array.ndim() == shape.size();
  for (std::size_t d = 0; match && d < shape.size(); ++d) {
    match = array.shape(d) == shape.begin()[d];
  }
  if (!match) {
    throw nb::value_error(std::format("out must have shape {}, got {}", shape_string({shape.begin(), shape.size()}),
                                      array_shape_string(array))
                              .c_str());
  }
  return {nb::borrow(out), {array.data(), size}};
}

void require_disjoint(std::span<const double> a, std::span<const double> b, const char *what) {
  if (a.data() < b.data() + b.size() && b.data() < a.data() + a.size()) {
    throw nb::value_error(std::format("{} must not overlap", what).c_str());
  }
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
        const tp::Elements eq = tp::cartesian_to_equinoctial(rv, mu);
        // Only an exactly retrograde equatorial orbit (i = pi) divides by zero in p and q.
        if (!std::isfinite(eq[tp::eq::p]) || !std::isfinite(eq[tp::eq::q])) {
          throw nb::value_error("equinoctial elements are singular for a retrograde equatorial orbit (i = 180 deg)");
        }
        return to_numpy(eq);
      },
      "rv"_a, "mu"_a = tp::earth_mu,
      nb::sig("def cartesian_to_equinoctial(rv: " ARRAY_LIKE ", mu: float = EARTH_MU) -> " FLOAT_ARRAY),
      "Convert a Cartesian state [r, v] to equinoctial elements [a, h, k, p, q, lambda].\n\n"
      "Direct set: h = e sin(argp + raan), k = e cos(argp + raan), p = tan(i/2) sin(raan),\n"
      "q = tan(i/2) cos(raan), lambda = M + argp + raan (mean longitude, radians, in [0, 2pi)); index with\n"
      "the EQ_* constants. Well defined for circular and equatorial orbits; singular only for retrograde\n"
      "equatorial orbits (i = 180 deg), which raise ValueError. Raises HyperbolicOrbitError if e >= 1, ValueError\n"
      "for invalid input.");

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

  m.def(
      "propagate_many",
      [](nb::handle rvs_like, nb::handle dts_like, double mu, nb::handle out) {
        const ConvertedInput rvs = as_doubles(rvs_like, "rvs");
        if (rvs.ndim() != 2 || rvs.shape(1) != 6) {
          throw nb::value_error(std::format("rvs must have shape (M, 6), got {}", array_shape_string(rvs)).c_str());
        }
        require_positive_mu(mu);
        const std::vector<double> times = checked_times(dts_like);
        // Validate and build every orbit before writing anything.
        std::vector<tp::TwoBodyOrbit> orbits;
        orbits.reserve(rvs.shape(0));
        for (std::size_t i = 0; i < rvs.shape(0); ++i) {
          tp::State rv;
          std::copy_n(rvs.data() + 6 * i, 6, rv.begin());
          try {
            require_elliptic(rv, mu);
          } catch (const HyperbolicOrbitError &err) {
            throw HyperbolicOrbitError(std::format("rvs[{}]: {}", i, err.what()));
          } catch (const nb::builtin_exception &err) {
            throw nb::value_error(std::format("rvs[{}]: {}", i, err.what()).c_str());
          }
          orbits.emplace_back(rv, mu);
        }
        auto [result, states] = output_array(out, {orbits.size(), times.size(), 6});
        nb::gil_scoped_release release;
        tp::propagate_grid(orbits, times, states);
        return result;
      },
      "rvs"_a, "dts"_a, "mu"_a = tp::earth_mu, "out"_a = nb::none(),
      nb::sig("def propagate_many(rvs: " ARRAY_LIKE ", dts: " ARRAY_LIKE ", mu: float = EARTH_MU, out: " FLOAT_ARRAY
              " | None = None) -> " FLOAT_ARRAY),
      "Propagate many orbits to many times: shape (M, N, 6) for rvs (M, 6) and dts (N,).\n\n"
      "``out[m, j]`` equals ``TwoBodyOrbit(rvs[m], mu).propagate(dts[j])`` exactly. Pass ``out`` (a\n"
      "writable, C-contiguous float64 array of shape (M, N, 6)) to fill it in place; it is returned.\n"
      "Every state and time is validated before anything is written (HyperbolicOrbitError or\n"
      "ValueError, naming the bad row or time), and the GIL is released while propagating.");

  m.def(
      "_two_body_central_difference_stm",
      [](const tp::State &rv, double dt, double mu) {
        require_elliptic(rv, mu);
        require_finite_time(dt);
        // Shrink the step with what sets the truncation error: the phase swept, n |dt|, and the
        // eccentricity (nonlinearity near perigee). The exponent 1.5 on (1 - e) / (1 + e) was tuned by
        // sweep: higher lets rounding dominate. Good to ~1e-10 of the largest entry at low e, ~1e-7 at
        // e = 0.9, but only ~1e-5 at e = 0.99, where central differences are badly conditioned.
        const tp::TwoBodyOrbit orbit(rv, mu);
        const double phase = orbit.mean_motion() * std::abs(dt);
        const double e = orbit.eccentricity();
        const double relative_step =
            std::cbrt(std::numeric_limits<double>::epsilon()) * std::pow((1.0 - e) / (1.0 + e), 1.5) / (1.0 + phase);
        return to_numpy(tp::central_difference_stm(
            rv, [&](const tp::State &x) { return tp::TwoBodyOrbit(x, mu).propagate(dt); }, relative_step));
      },
      "rv"_a, "dt"_a, "mu"_a = tp::earth_mu,
      nb::sig("def _two_body_central_difference_stm(rv: " ARRAY_LIKE
              ", dt: float, mu: float = EARTH_MU) -> " FLOAT_ARRAY),
      "Internal test oracle: the two-body state transition matrix by central differences in C++.\n\n"
      "Not public API (not exported from tag.propagation); tests compare analytic state transition\n"
      "matrices against it. Entries are good to about 1e-10 of the largest at low e, 1e-7 at e = 0.9\n"
      "and only about 1e-5 at e = 0.99.");

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
          nb::sig("def __init__(self, rv: " ARRAY_LIKE ", mu: float = EARTH_MU) -> None"), "rv"_a,
          "mu"_a = tp::earth_mu)
      .def(
          "propagate",
          [](const tp::TwoBodyOrbit &orbit, double dt) {
            require_finite_time(dt);
            return to_numpy(orbit.propagate(dt));
          },
          nb::sig("def propagate(self, dt: float) -> " FLOAT_ARRAY), "dt"_a,
          "The Cartesian state ``dt`` seconds (any sign) after the epoch, as a float64 array of shape (6,).\n\n"
          "Precision doesn't decay with long ``dt``: the mean anomaly is reduced modulo 2 pi.")
      .def(
          "propagate_many",
          [](const tp::TwoBodyOrbit &orbit, nb::handle dts_like, nb::handle out) {
            const std::vector<double> times = checked_times(dts_like);
            auto [result, states] = output_array(out, {times.size(), 6});
            nb::gil_scoped_release release;
            orbit.propagate(times, states);
            return result;
          },
          nb::sig("def propagate_many(self, dts: " ARRAY_LIKE ", out: " FLOAT_ARRAY " | None = None) -> " FLOAT_ARRAY),
          "dts"_a, "out"_a = nb::none(),
          "The Cartesian states at each of ``dts`` (a 1-D array of times after the epoch), shape (N, 6).\n\n"
          "Row i equals ``propagate(dts[i])`` exactly. Pass ``out`` (a writable, C-contiguous float64 array\n"
          "of shape (N, 6)) to fill it in place; it is returned. Every dt is checked before anything is\n"
          "written, and the GIL is released while propagating.")
      .def(
          "propagate_with_stm",
          [](const tp::TwoBodyOrbit &orbit, double dt) {
            require_finite_time(dt);
            const tp::StateWithStm x = orbit.propagate_with_stm(dt);
            return nb::make_tuple(to_numpy(x.rv), to_numpy(x.stm));
          },
          nb::sig("def propagate_with_stm(self, dt: float) -> tuple[" FLOAT_ARRAY ", " FLOAT_ARRAY "]"), "dt"_a,
          "The state ``dt`` seconds after the epoch, shape (6,), and the state transition matrix\n"
          "d x(t) / d x(epoch), shape (6, 6).\n\n"
          "The state equals ``propagate(dt)`` exactly. The matrix is in closed form and reuses its Kepler\n"
          "solution, costing about as much again as ``propagate``, which is unaffected.")
      .def(
          "propagate_many_with_stm",
          [](const tp::TwoBodyOrbit &orbit, nb::handle dts_like, nb::handle out, nb::handle stm_out) {
            const std::vector<double> times = checked_times(dts_like);
            auto [states, state_values] = output_array(out, {times.size(), 6});
            auto [stms, stm_values] = output_array(stm_out, {times.size(), 6, 6});
            require_disjoint(state_values, stm_values, "out and stm_out");
            {
              nb::gil_scoped_release release;
              orbit.propagate_with_stm(times, state_values, stm_values);
            }
            return nb::make_tuple(states, stms);
          },
          nb::sig("def propagate_many_with_stm(self, dts: " ARRAY_LIKE ", out: " FLOAT_ARRAY
                  " | None = None, stm_out: " FLOAT_ARRAY " | None = None) -> tuple[" FLOAT_ARRAY ", " FLOAT_ARRAY "]"),
          "dts"_a, "out"_a = nb::none(), "stm_out"_a = nb::none(),
          "The states at each of ``dts``, shape (N, 6), and their state transition matrices, shape (N, 6, 6).\n\n"
          "Row i equals ``propagate_with_stm(dts[i])`` exactly. ``out`` and ``stm_out`` (writable,\n"
          "C-contiguous float64 arrays of those shapes, not overlapping) are filled in place and returned\n"
          "when given. Every dt is checked before anything is written, and the GIL is released while\n"
          "propagating.")
      .def_prop_ro("mu", &tp::TwoBodyOrbit::mu, "Gravitational parameter.")
      .def_prop_ro("semi_major_axis", &tp::TwoBodyOrbit::semi_major_axis, "Semi-major axis a.")
      .def_prop_ro("eccentricity", &tp::TwoBodyOrbit::eccentricity, "Eccentricity e, in [0, 1).")
      .def_prop_ro(
          "period", [](const tp::TwoBodyOrbit &orbit) { return 2.0 * std::numbers::pi / orbit.mean_motion(); },
          "Orbital period, 2 pi / n.");
}
