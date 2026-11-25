#pragma once

#include "tag/propagation/elements.hpp"

namespace tag::propagation {

// Earth's equatorial radius, km (WGS-84).
inline constexpr double earth_re = 6378.137;
// Earth's unnormalized second zonal harmonic J2 (EGM-96 / JGM-3 value, as used by Vallado).
inline constexpr double earth_j2 = 1.08262668e-3;

// Osculating <-> mean elements under first-order J2, per Brouwer-Lyddane: the short-period terms
// and Brouwer's long-period terms, in Lyddane's form, which stays regular at e = 0 and i = 0. Both
// element sets are equinoctial [a, h, k, p, q, lambda] (see elements.hpp); the mapping depends on
// J2 and re / a only, not on mu.
//
// The long-period terms divide by 1 - 5 cos^2 i, so the theory is singular at the critical
// inclination (about 63.43 deg, or 116.57 deg retrograde): close to it the corrections grow without
// bound and are meaningless (Molniya orbits sit right there by design). This is a known limitation
// of the theory, not fixed here.

// Mean -> osculating equinoctial elements, closed form. Precondition: a > 0, h^2 + k^2 < 1, finite
// values; re and j2 finite. The result may be non-finite at the critical inclination.
[[nodiscard]] Elements mean_to_osculating_elements(const Elements &mean, double re, double j2) noexcept;

// The fixed-point iteration's cap and convergence tolerance (on |da| / a and on the change in each
// dimensionless element, per iteration).
inline constexpr int mean_elements_max_iterations = 30;
inline constexpr double mean_elements_tolerance = 1e-12;

struct MeanElements {
  Elements mean;
  int iterations;
  // False if the iteration hit the cap or went non-finite (near the critical inclination); mean is
  // then meaningless.
  bool converged;
};

// Osculating -> mean equinoctial elements: fixed-point iteration of the inverse of
// mean_to_osculating_elements, about 3-5 iterations to 1e-12. Mean lambda is returned in [0, 2pi).
// Precondition: as for mean_to_osculating_elements.
[[nodiscard]] MeanElements osculating_to_mean_elements(const Elements &osculating, double re, double j2) noexcept;

} // namespace tag::propagation
