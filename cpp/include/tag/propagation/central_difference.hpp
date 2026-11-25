#pragma once

#include <cstddef>

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
template <class Propagate>
[[nodiscard]] Matrix6 central_difference_stm(const State &x0, Propagate &&propagate_from, double relative_step) {
  const double steps[2] = {relative_step * norm(position(x0)), relative_step * norm(velocity(x0))};
  Matrix6 stm{};
  for (std::size_t j = 0; j < 6; ++j) {
    State plus = x0;
    State minus = x0;
    plus[j] += steps[j / 3];
    minus[j] -= steps[j / 3];
    const State y_plus = propagate_from(plus);
    const State y_minus = propagate_from(minus);
    const double step = plus[j] - minus[j];
    for (std::size_t i = 0; i < 6; ++i) {
      stm[6 * i + j] = (y_plus[i] - y_minus[i]) / step;
    }
  }
  return stm;
}

} // namespace tag::propagation
