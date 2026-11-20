#pragma once

#include <array>
#include <cmath>

namespace tag::propagation {

using Vec3 = std::array<double, 3>;
// A Cartesian state [x, y, z, vx, vy, vz]. Units are the caller's, consistent with mu (default km, s).
using State = std::array<double, 6>;

// Earth's gravitational parameter, km^3/s^2 (EGM-96 / WGS-84 value, as used by Vallado).
inline constexpr double earth_mu = 398600.4418;

[[nodiscard]] constexpr double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] constexpr Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] inline double norm(const Vec3 &a) noexcept { return std::sqrt(dot(a, a)); }

[[nodiscard]] constexpr Vec3 position(const State &rv) noexcept { return {rv[0], rv[1], rv[2]}; }
[[nodiscard]] constexpr Vec3 velocity(const State &rv) noexcept { return {rv[3], rv[4], rv[5]}; }

// The eccentricity vector (pointing at periapsis, length e) of the two-body orbit through rv:
// e = ((v^2 - mu/r) r - (r.v) v) / mu. Precondition: r != 0, mu > 0 (unchecked).
[[nodiscard]] inline Vec3 eccentricity_vector(const State &rv, double mu) noexcept {
  const Vec3 r = position(rv);
  const Vec3 v = velocity(rv);
  const double cr = dot(v, v) - mu / norm(r);
  const double cv = dot(r, v);
  return {(cr * r[0] - cv * v[0]) / mu, (cr * r[1] - cv * v[1]) / mu, (cr * r[2] - cv * v[2]) / mu};
}

} // namespace tag::propagation
