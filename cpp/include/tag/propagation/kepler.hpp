#pragma once

namespace tag::propagation {

// Solves Kepler's equation E - e sin E = M for the eccentric anomaly E, given the mean anomaly
// M (radians, any value) and the eccentricity e. E is returned in the same revolution as M, so
// E - e sin E = M holds for M outside [-pi, pi] too.
//
// Precondition: 0 <= e < 1 and M finite (unchecked; the bindings validate). Converged to float64
// machine precision with a fixed, branch-free amount of work: Markley's starter, then Halley steps.
[[nodiscard]] double eccentric_anomaly(double mean_anomaly, double e) noexcept;

} // namespace tag::propagation
