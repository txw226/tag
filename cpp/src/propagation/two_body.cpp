#include "tag/propagation/two_body.hpp"

#include <cmath>

#include "tag/propagation/kepler.hpp"

namespace tag::propagation {

TwoBodyOrbit::TwoBodyOrbit(const State &rv, double mu) noexcept : mu_(mu) {
  const Vec3 r = position(rv);
  const Vec3 v = velocity(rv);
  const double r_norm = norm(r);
  a_ = 1.0 / (2.0 / r_norm - dot(v, v) / mu);
  const Vec3 e_vec = eccentricity_vector(rv, mu);
  e_ = norm(e_vec);
  n_ = std::sqrt(mu / a_) / a_;
  const double b = a_ * std::sqrt((1.0 - e_) * (1.0 + e_));

  const Vec3 h = cross(r, v);
  const double h_norm = norm(h);
  const Vec3 w = {h[0] / h_norm, h[1] / h_norm, h[2] / h_norm};
  const Vec3 p =
      e_ > 0.0 ? Vec3{e_vec[0] / e_, e_vec[1] / e_, e_vec[2] / e_} : Vec3{r[0] / r_norm, r[1] / r_norm, r[2] / r_norm};
  const Vec3 q = cross(w, p);

  // r = a (cos E - e) P + b sin E Q at the epoch.
  const double E0 = std::atan2(dot(r, q) / b, dot(r, p) / a_ + e_);
  m0_ = E0 - e_ * std::sin(E0);
  for (int i = 0; i < 3; ++i) {
    A_[i] = a_ * p[i];
    B_[i] = b * q[i];
  }
}

State TwoBodyOrbit::propagate(double dt) const noexcept {
  const KeplerSolution k = solve_kepler_reduced(reduce_angle(std::fma(n_, dt, m0_)), e_);
  const double x = k.cos_E - e_;
  const double y = k.sin_E;
  const double rate = n_ / (1.0 - e_ * k.cos_E);
  const double vx = -rate * k.sin_E;
  const double vy = rate * k.cos_E;
  return {A_[0] * x + B_[0] * y,   A_[1] * x + B_[1] * y,   A_[2] * x + B_[2] * y,
          A_[0] * vx + B_[0] * vy, A_[1] * vx + B_[1] * vy, A_[2] * vx + B_[2] * vy};
}

} // namespace tag::propagation
