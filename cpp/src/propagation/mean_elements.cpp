#include "tag/propagation/mean_elements.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "tag/propagation/kepler.hpp"

namespace tag::propagation {

namespace {

using std::numbers::pi;

double wrap_two_pi(double angle) noexcept {
  const double wrapped = std::fmod(angle, 2.0 * pi);
  return wrapped < 0.0 ? wrapped + 2.0 * pi : wrapped;
}

} // namespace

// Brouwer's first-order J2 theory (Brouwer 1959, AJ 64, 378) with Lyddane's (1963, AJ 68, 555)
// non-singular recombination, in the single-step form of Schaub & Junkins, Analytical Mechanics of
// Space Systems, Appendix F (short-period plus long-period terms, evaluated on the input elements).
// Classical (e, i, raan, argp, M) appear only inside first-order corrections, whose combinations are
// regular at e = 0 and i = 0, so the conventions used for the undefined angles there don't matter.
Elements mean_to_osculating_elements(const Elements &mean, double re, double j2) noexcept {
  const double a = mean[eq::a];
  const double eh = mean[eq::h];
  const double ek = mean[eq::k];
  const double p = mean[eq::p];
  const double q = mean[eq::q];
  const double lambda = mean[eq::lambda];

  // Classical angles, with argp = 0 when circular and raan = 0 when equatorial.
  const double e = std::hypot(eh, ek);
  const double t = std::hypot(p, q); // tan(i / 2)
  const double varpi = e > 0.0 ? std::atan2(eh, ek) : 0.0;
  const double sin_raan = t > 0.0 ? p / t : 0.0;
  const double cos_raan = t > 0.0 ? q / t : 1.0;
  const double raan = std::atan2(sin_raan, cos_raan);
  const double argp = varpi - raan;
  const double M = lambda - varpi;

  const double t2 = t * t;
  const double c = (1.0 - t2) / (1.0 + t2);      // cos i
  const double s = 2.0 * t / (1.0 + t2);         // sin i
  const double s_half = t / std::sqrt(1.0 + t2); // sin(i / 2)
  const double c_half = 1.0 / std::sqrt(1.0 + t2);
  const double c2 = c * c;
  const double s2 = s * s; // 1 - cos^2 i, exact near i = 0
  const double den = 1.0 - 5.0 * c2;

  // True anomaly from the eccentric anomaly; f - M as the equation of centre, without wrapping.
  const double eta = std::sqrt(1.0 - e * e);
  const double E = eccentric_anomaly(M, e);
  const double sin_E = std::sin(E);
  const double cos_E = std::cos(E);
  const double one_m_ecosE = 1.0 - e * cos_E;
  const double sin_f = eta * sin_E / one_m_ecosE;
  const double cos_f = (cos_E - e) / one_m_ecosE;
  const double beta = e / (1.0 + eta);
  const double f_minus_M = e * sin_E + 2.0 * std::atan2(beta * sin_E, 1.0 - beta * cos_E);
  const double f = M + f_minus_M;

  const double gamma2 = 0.5 * j2 * (re / a) * (re / a);
  const double eta2 = eta * eta;
  const double eta3 = eta2 * eta;
  const double eta6 = eta3 * eta3;
  const double gp = gamma2 / (eta2 * eta2); // gamma2'
  const double a_r = (1.0 + e * cos_f) / eta2;
  const double a_r3 = a_r * a_r * a_r;

  const double cos_2w = std::cos(2.0 * argp), sin_2w = std::sin(2.0 * argp);
  const double cos_2w_f = std::cos(2.0 * argp + f), sin_2w_f = std::sin(2.0 * argp + f);
  const double cos_2w_2f = std::cos(2.0 * argp + 2.0 * f), sin_2w_2f = std::sin(2.0 * argp + 2.0 * f);
  const double cos_2w_3f = std::cos(2.0 * argp + 3.0 * f), sin_2w_3f = std::sin(2.0 * argp + 3.0 * f);

  // 1 - 11 cos^2 i - 40 cos^4 i / (1 - 5 cos^2 i), factored so its sin^2 i zero at i = 0 is exact.
  const double lp = s2 * (1.0 - 15.0 * c2) / den;
  const double lp_omega = 11.0 + 80.0 * c2 / den + 200.0 * c2 * c2 / (den * den);
  const double cos_f2 = cos_f * cos_f;
  const double cos_f3 = cos_f2 * cos_f;
  const double centre = f_minus_M + e * sin_f; // f - M + e sin f
  const double s_terms = 3.0 * sin_2w_2f + 3.0 * e * sin_2w_f + e * sin_2w_3f;
  const double c_terms = 3.0 * cos_2w_2f + 3.0 * e * cos_2w_f + e * cos_2w_3f;

  // Brouwer's delta l and delta g share the bracket X / e, with factors -eta^3 and +eta^2. In e delta M it
  // keeps -eta^3; in delta lambda = delta l + delta g + delta h the two leave eta^2 (1 - eta) / e =
  // eta^2 e / (1 + eta), regular at e = 0. (Dropping that O(e) term leaves mean lambda and varpi with a
  // once-per-revolution wobble: tests/propagation/test_mean_elements.py checks the angles' steadiness.)
  const double ar_eta2 = a_r * a_r * eta2;
  const double X = 2.0 * (3.0 * c2 - 1.0) * (ar_eta2 + a_r + 1.0) * sin_f +
                   3.0 * s2 * ((-ar_eta2 - a_r + 1.0) * sin_2w_f + (ar_eta2 + a_r + 1.0 / 3.0) * sin_2w_3f);

  const double da = a * gamma2 * ((3.0 * c2 - 1.0) * (a_r3 - 1.0 / eta3) + 3.0 * s2 * a_r3 * cos_2w_2f);

  const double de1 = gp / 8.0 * e * eta2 * lp * cos_2w;
  const double de =
      de1 + eta2 / 2.0 *
                (gamma2 * ((3.0 * c2 - 1.0) / eta6 *
                               (e * eta + e / (1.0 + eta) + 3.0 * cos_f + 3.0 * e * cos_f2 + e * e * cos_f3) +
                           3.0 * s2 / eta6 * (e + 3.0 * cos_f + 3.0 * e * cos_f2 + e * e * cos_f3) * cos_2w_2f) -
                 gp * s2 * (3.0 * cos_2w_f + cos_2w_3f));

  // -e de1 / (eta^2 tan i), with the sin^2 i in de1 cancelled against tan i.
  const double di = -gp / 8.0 * e * e * c * s * (1.0 - 15.0 * c2) / den * cos_2w + gp / 2.0 * c * s * c_terms;

  const double draan = -gp / 8.0 * e * e * c * lp_omega * sin_2w - gp / 2.0 * c * (6.0 * centre - s_terms);

  const double dlambda = gp / 8.0 * eta3 * lp * sin_2w -
                         gp / 16.0 *
                             (2.0 + e * e - 11.0 * (2.0 + 3.0 * e * e) * c2 -
                              40.0 * (2.0 + 5.0 * e * e) * c2 * c2 / den - 400.0 * e * e * c2 * c2 * c2 / (den * den)) *
                             sin_2w +
                         gp / 4.0 * (-6.0 * den * centre + (3.0 - 5.0 * c2) * s_terms) +
                         gp / 4.0 * eta2 * e / (1.0 + eta) * X + draan;

  const double edM = gp / 8.0 * e * eta3 * lp * sin_2w - gp / 4.0 * eta3 * X;

  // Lyddane: perturb (e cos M, e sin M) and sin(i/2) (cos raan, sin raan), then read off the
  // equinoctial elements, with varpi' = lambda' - M'.
  const double sin_M = std::sin(M), cos_M = std::cos(M);
  const double d1 = (e + de) * sin_M + edM * cos_M; // e' sin M'
  const double d2 = (e + de) * cos_M - edM * sin_M; // e' cos M'
  const double lambda_osc = lambda + dlambda;
  const double sin_l = std::sin(lambda_osc), cos_l = std::cos(lambda_osc);

  const double sin_half_osc = s_half + c_half * di / 2.0;
  const double d3 = sin_half_osc * sin_raan + s_half * draan * cos_raan; // sin(i'/2) sin raan'
  const double d4 = sin_half_osc * cos_raan - s_half * draan * sin_raan; // sin(i'/2) cos raan'
  const double cos_half_osc = std::sqrt(1.0 - d3 * d3 - d4 * d4);

  return {a + da,
          d2 * sin_l - d1 * cos_l,
          d2 * cos_l + d1 * sin_l,
          d3 / cos_half_osc,
          d4 / cos_half_osc,
          wrap_two_pi(lambda_osc)};
}

MeanElements osculating_to_mean_elements(const Elements &osculating, double re, double j2) noexcept {
  // Fixed point of mean = mean + (osculating - mean_to_osculating(mean)), starting from the
  // osculating elements; the map is the identity plus O(J2), so each iteration gains about a factor
  // J2, except near the critical inclination, where the long-period terms blow up.
  Elements mean = osculating;
  for (int iteration = 1; iteration <= mean_elements_max_iterations; ++iteration) {
    const Elements guess = mean_to_osculating_elements(mean, re, j2);
    double change = 0.0;
    bool finite = true;
    for (int j = 0; j < 6; ++j) {
      double diff = osculating[j] - guess[j];
      if (j == eq::lambda) {
        diff = std::remainder(diff, 2.0 * pi);
      }
      mean[j] += diff;
      // Checked separately: std::max would silently drop a NaN.
      finite = finite && std::isfinite(mean[j]);
      change = std::max(change, std::abs(j == eq::a ? diff / mean[eq::a] : diff));
    }
    if (!finite) {
      return {mean, iteration, false};
    }
    if (change < mean_elements_tolerance) {
      mean[eq::lambda] = wrap_two_pi(mean[eq::lambda]);
      return {mean, iteration, true};
    }
  }
  return {mean, mean_elements_max_iterations, false};
}

} // namespace tag::propagation
