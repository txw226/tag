#pragma once

namespace tag::propagation {

// An angle reduced to [-pi, pi] (Cody-Waite split of 2*pi, so accurate for large angles).
// Precondition: angle finite (unchecked).
[[nodiscard]] double reduce_angle(double angle) noexcept;

// Eccentric anomaly E with its sine and cosine, which the solver gets for free.
struct KeplerSolution {
  double E;
  double sin_E;
  double cos_E;
};

// Solves Kepler's equation E - e sin E = M for a mean anomaly already reduced to [-pi, pi]
// (see reduce_angle); E is then in [-pi, pi] too. sin_E and cos_E come from the solver's
// angle-addition updates, accurate to float64 rounding, so callers need no extra sincos.
//
// Precondition: 0 <= e < 1 and -pi <= M <= pi (unchecked). Converged to float64 machine
// precision with a fixed, branch-free amount of work: Markley's starter, then Halley steps.
[[nodiscard]] KeplerSolution solve_kepler_reduced(double mean_anomaly, double e) noexcept;

// Solves Kepler's equation E - e sin E = M for the eccentric anomaly E, given the mean anomaly
// M (radians, any value) and the eccentricity e. E is returned in the same revolution as M, so
// E - e sin E = M holds for M outside [-pi, pi] too.
//
// Precondition: 0 <= e < 1 and M finite (unchecked; the bindings validate).
[[nodiscard]] double eccentric_anomaly(double mean_anomaly, double e) noexcept;

} // namespace tag::propagation
