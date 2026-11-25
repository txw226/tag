#include "tag/propagation/central_difference.hpp"

#include <cstddef>

namespace tag::propagation {

Matrix6 central_difference_stm(const State &x0, const std::function<State(const State &)> &propagate_from,
                               double relative_step) {
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
