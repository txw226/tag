#pragma once

#include <functional>

#include "tag/propagation/cartesian.hpp"

namespace tag::propagation {

// The state transition matrix of any propagation x0 -> x(t), by central differences: the test oracle
// for analytic state transition matrices, never used in production paths.
//
// Each component is stepped by relative_step times the norm of the epoch position or velocity (the
// step taken is the difference actually represented in floating point). The truncation error grows
// with the square of the step times the propagation's nonlinearity, which for an orbit is the phase
// swept and, near a high-e perigee, the eccentricity: the two-body oracle binding scales cbrt(eps)
// down by both. Precondition: every state propagate_from is given stays inside the
// propagation's domain (unchecked).
[[nodiscard]] Matrix6 central_difference_stm(const State &x0, const std::function<State(const State &)> &propagate_from,
                                             double relative_step);

} // namespace tag::propagation
