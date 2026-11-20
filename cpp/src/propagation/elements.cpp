#include "tag/propagation/elements.hpp"

#include <cmath>
#include <numbers>

#include "tag/propagation/kepler.hpp"

namespace tag::propagation {

namespace {

using std::numbers::pi;

// Below this, e counts as zero (circular) and sin(i) as zero (equatorial).
constexpr double singular_tol = 1e-11;

// Wraps an angle to [0, 2pi).
double wrap_two_pi(double angle) noexcept {
  const double wrapped = std::fmod(angle, 2.0 * pi);
  return wrapped < 0.0 ? wrapped + 2.0 * pi : wrapped;
}

// Signed angle from direction d to direction x, both in the plane with normal h_hat.
double angle_in_plane(const Vec3 &d, const Vec3 &x, const Vec3 &h_hat) noexcept {
  return std::atan2(dot(cross(d, x), h_hat), dot(d, x));
}

} // namespace

Elements cartesian_to_keplerian(const State &rv, double mu, Anomaly anomaly) noexcept {
  const Vec3 r = position(rv);
  const Vec3 v = velocity(rv);
  const Vec3 h = cross(r, v);
  const double h_norm = norm(h);
  const Vec3 h_hat = {h[0] / h_norm, h[1] / h_norm, h[2] / h_norm};
  const Vec3 e_vec = eccentricity_vector(rv, mu);
  const double e = norm(e_vec);
  const double a = 1.0 / (2.0 / norm(r) - dot(v, v) / mu);

  // Node vector z x h, and i from atan2 (accurate at every inclination, unlike acos).
  const Vec3 node = {-h[1], h[0], 0.0};
  const double node_norm = norm(node);
  const double i = std::atan2(node_norm, h[2]);
  const bool equatorial = node_norm < singular_tol * h_norm;
  const bool circular = e < singular_tol;

  // Reference direction for argp: the node, or x for an equatorial orbit.
  const Vec3 ref = equatorial ? Vec3{1.0, 0.0, 0.0} : node;
  const double raan = equatorial ? 0.0 : std::atan2(node[1], node[0]);
  const double argp = circular ? 0.0 : angle_in_plane(ref, e_vec, h_hat);
  // The true anomaly is measured from periapsis, or from the reference direction when circular.
  const double nu = angle_in_plane(circular ? ref : e_vec, r, h_hat);

  double last = nu;
  if (anomaly != Anomaly::true_anomaly) {
    const double E = std::atan2(std::sqrt(1.0 - e * e) * std::sin(nu), e + std::cos(nu));
    last = anomaly == Anomaly::eccentric ? E : E - e * std::sin(E);
  }
  return {a, e, i, wrap_two_pi(raan), wrap_two_pi(argp), wrap_two_pi(last)};
}

State keplerian_to_cartesian(const Elements &kep, double mu, Anomaly anomaly) noexcept {
  const double a = kep[kep::a];
  const double e = kep[kep::e];
  const double s_e = std::sqrt(1.0 - e * e);
  const double last = kep[kep::anomaly];
  double E = last;
  if (anomaly == Anomaly::mean) {
    E = eccentric_anomaly(last, e);
  } else if (anomaly == Anomaly::true_anomaly) {
    E = std::atan2(s_e * std::sin(last), e + std::cos(last));
  }
  const double sin_E = std::sin(E);
  const double cos_E = std::cos(E);

  // Position and velocity in the perifocal frame (P towards periapsis, Q 90 degrees ahead).
  const double x_pf = a * (cos_E - e);
  const double y_pf = a * s_e * sin_E;
  const double vel = std::sqrt(mu / a) / (1.0 - e * cos_E);
  const double vx_pf = -vel * sin_E;
  const double vy_pf = vel * s_e * cos_E;

  const double cO = std::cos(kep[kep::raan]), sO = std::sin(kep[kep::raan]);
  const double cw = std::cos(kep[kep::argp]), sw = std::sin(kep[kep::argp]);
  const double ci = std::cos(kep[kep::i]), si = std::sin(kep[kep::i]);
  const Vec3 P = {cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si};
  const Vec3 Q = {-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si};
  State rv;
  for (int j = 0; j < 3; ++j) {
    rv[j] = x_pf * P[j] + y_pf * Q[j];
    rv[j + 3] = vx_pf * P[j] + vy_pf * Q[j];
  }
  return rv;
}

Elements cartesian_to_equinoctial(const State &rv, double mu) noexcept { return {}; }

State equinoctial_to_cartesian(const Elements &eq, double mu) noexcept { return {}; }

} // namespace tag::propagation
