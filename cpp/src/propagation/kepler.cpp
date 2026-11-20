#include "tag/propagation/kepler.hpp"

#include <cmath>
#include <numbers>

namespace tag::propagation {

namespace {

using std::numbers::pi;

// 2*pi as the nearest double plus its remainder. With fma each product k*part is exact, so
// M - k*2*pi stays accurate for large k.
constexpr double two_pi_hi = 6.28318530717958623200e+00;
constexpr double two_pi_lo = 2.44929359829470635445e-16;

// Halley steps after Markley's starter. The first correction is at most ~5e-4 rad over 0 <= e < 1,
// and each Halley step triples the number of correct digits, so two steps reach float64 precision.
constexpr int halley_steps = 2;

// Markley (1995), "Kepler equation solver", Celest. Mech. 63, 101-111: a cubic-based starter for
// M in [0, pi].
double markley_starter(double m, double e) noexcept {
  constexpr double pi2 = pi * pi;
  const double alpha = (3.0 * pi2 + 1.6 * pi * (pi - m) / (1.0 + e)) / (pi2 - 6.0);
  const double d = 3.0 * (1.0 - e) + alpha * e;
  const double q = 2.0 * alpha * d * (1.0 - e) - m * m;
  const double r = 3.0 * alpha * d * (d - 1.0 + e) * m + m * m * m;
  const double w_cbrt = std::cbrt(std::abs(r) + std::sqrt(q * q * q + r * r));
  const double w = w_cbrt * w_cbrt;
  return (2.0 * r * w / (w * w + w * q + q * q) + m) / d;
}

} // namespace

double eccentric_anomaly(double mean_anomaly, double e) noexcept {
  // Reduce to m in [-pi, pi], then solve for |m| and restore the sign (E is odd in M).
  const double k = std::nearbyint(mean_anomaly / two_pi_hi);
  const double m = std::fma(-k, two_pi_lo, std::fma(-k, two_pi_hi, mean_anomaly));
  const double a = std::abs(m);

  double E = markley_starter(a, e);
  // One sin/cos evaluation; later steps update them by angle addition. Corrections are below
  // ~5e-4 rad, where these Taylor series of sin and cos are exact to ~1e-27.
  double s = std::sin(E);
  double c = std::cos(E);
  for (int i = 0; i < halley_steps; ++i) {
    const double f = E - e * s - a;
    const double df = 1.0 - e * c;
    const double delta = -2.0 * f * df / (2.0 * df * df - f * e * s);
    E += delta;
    const double d2 = delta * delta;
    const double sin_d = delta * (1.0 - d2 / 6.0 * (1.0 - d2 / 20.0));
    const double cos_d = 1.0 - d2 / 2.0 * (1.0 - d2 / 12.0 * (1.0 - d2 / 30.0));
    const double s_next = s * cos_d + c * sin_d;
    c = c * cos_d - s * sin_d;
    s = s_next;
  }
  return std::copysign(E, m) + k * two_pi_hi + k * two_pi_lo;
}

} // namespace tag::propagation
