#include "tag/propagation/two_body.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

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

  r0_ = r;
  v0_ = v;
  r0_norm_ = r_norm;
  sigma0_ = dot(r, v);
  alpha_ = 2.0 / r_norm - dot(v, v) / mu;
  sin_E0_ = std::sin(E0);
  cos_E0_ = std::cos(E0);
}

State TwoBodyOrbit::state_at(double cos_E, double sin_E) const noexcept {
  const double x = cos_E - e_;
  const double y = sin_E;
  const double rate = n_ / (1.0 - e_ * cos_E);
  const double vx = -rate * sin_E;
  const double vy = rate * cos_E;
  return {A_[0] * x + B_[0] * y,   A_[1] * x + B_[1] * y,   A_[2] * x + B_[2] * y,
          A_[0] * vx + B_[0] * vy, A_[1] * vx + B_[1] * vy, A_[2] * vx + B_[2] * vy};
}

KeplerSolution TwoBodyOrbit::kepler_at(double dt) const noexcept {
  return solve_kepler_reduced(reduce_angle(std::fma(n_, dt, m0_)), e_);
}

State TwoBodyOrbit::propagate(double dt) const noexcept {
  const KeplerSolution k = kepler_at(dt);
  return state_at(k.cos_E, k.sin_E);
}

namespace {

// d/d(r0, sigma0, alpha) of a scalar that depends on them directly and through theta.
struct Partials {
  double r0;
  double sigma;
  double alpha;
};

Partials total(double d_r0, double d_sigma, double d_alpha, double d_theta, const Partials &theta) noexcept {
  return {d_r0 + d_theta * theta.r0, d_sigma + d_theta * theta.sigma, d_alpha + d_theta * theta.alpha};
}

} // namespace

StateWithStm TwoBodyOrbit::propagate_with_stm(double dt) const noexcept {
  const KeplerSolution k = kepler_at(dt);
  StateWithStm out{state_at(k.cos_E, k.sin_E), {}};

  // r = f r0 + g v0 and v = fd r0 + gd v0, with f, g, fd, gd functions of the epoch scalars
  // r0 = |r0|, sigma = r0 . v0 and alpha = 1/a, and of theta = E - E0, which Kepler's equation in
  // difference form ties to them: n dt = theta - (1 - r0 alpha) sin(theta) + sigma sqrt(alpha/mu)
  // (1 - cos(theta)). Only sin and cos of theta appear (from the Kepler solution and E0, so no
  // revolution count), apart from the secular 3/2 n dt / alpha in d theta / d alpha.
  const double mu = mu_;
  const double r0 = r0_norm_;
  const double sg = sigma0_;
  const double al = alpha_;
  const double S = k.sin_E * cos_E0_ - k.cos_E * sin_E0_;
  const double c = k.cos_E * cos_E0_ + k.sin_E * sin_E0_;
  const double C = c > 0.0 ? S * S / (1.0 + c) : 1.0 - c; // 1 - cos(theta), without cancellation
  const double q = 1.0 / std::sqrt(mu * al);
  const double q_al = -0.5 * q / al;

  // r = C / alpha + r0 cos(theta) + sigma q sin(theta), the radius at dt.
  const double r = C / al + r0 * c + sg * S * q;
  const double r_r0 = c;
  const double r_sg = S * q;
  const double r_al = -C / (al * al) + sg * S * q_al;
  const double r_th = S / al - r0 * S + sg * q * c;

  // theta, implicitly: dF = d(n dt) with F_theta = alpha r.
  const double F_th = al * r;
  const Partials th = {-al * S / F_th, -al * q * C / F_th, (1.5 * n_ * dt / al - (r0 * S + 0.5 * sg * C * q)) / F_th};

  // f = 1 - C / (alpha r0)
  const double f = 1.0 - C / (al * r0);
  const Partials df = total(C / (al * r0 * r0), 0.0, C / (al * al * r0), -S / (al * r0), th);
  // g = r0 q sin(theta) + sigma C / (mu alpha)
  const double g = r0 * S * q + sg * C / (mu * al);
  const Partials dg =
      total(S * q, C / (mu * al), r0 * S * q_al - sg * C / (mu * al * al), r0 * q * c + sg * S / (mu * al), th);
  // fd = -mu q sin(theta) / (r r0)
  const double fd = -mu * q * S / (r * r0);
  const double fd_r = mu * q * S / (r * r * r0); // -d fd / d r
  const Partials dfd = total(fd_r * r_r0 + mu * q * S / (r * r0 * r0), fd_r * r_sg,
                             -mu * S * q_al / (r * r0) + fd_r * r_al, -mu * q * c / (r * r0) + fd_r * r_th, th);
  // gd = 1 - C / (alpha r)
  const double gd = 1.0 - C / (al * r);
  const double gd_r = C / (al * r * r); // d gd / d r
  const Partials dgd =
      total(gd_r * r_r0, gd_r * r_sg, C / (al * al * r) + gd_r * r_al, -S / (al * r) + gd_r * r_th, th);

  // The gradient of a coefficient with respect to the epoch state, as multiples of r0 and v0:
  // d/d r0_vec = (D_r0 / r0 - 2 D_alpha / r0^3) r0 + D_sigma v0, d/d v0_vec = D_sigma r0 - 2 D_alpha / mu v0.
  struct Gradient {
    // d/d r0_vec = by_r0.along_r0 r0 + by_r0.along_v0 v0, and likewise for d/d v0_vec.
    double by_r0_along_r0, by_r0_along_v0, by_v0_along_r0, by_v0_along_v0;
  };
  const auto gradient = [&](const Partials &d) {
    return Gradient{d.r0 / r0 - 2.0 * d.alpha / (r0 * r0 * r0), d.sigma, d.sigma, -2.0 * d.alpha / mu};
  };
  const Gradient Gf = gradient(df), Gg = gradient(dg), Gfd = gradient(dfd), Ggd = gradient(dgd);

  // Block (rows 3 br.., columns 3 bc..) = k I + r0 (grad of the r0 coefficient)^T + v0 (grad of the v0 coefficient)^T.
  const auto block = [&](int br, int bc, double diagonal, double cr, double cv, double dr, double dv) {
    // Column direction: cr r0 + cv v0 from the r0 coefficient's gradient, dr r0 + dv v0 from the v0 one's.
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        out.stm[6 * (3 * br + i) + 3 * bc + j] =
            (i == j ? diagonal : 0.0) + r0_[i] * (cr * r0_[j] + cv * v0_[j]) + v0_[i] * (dr * r0_[j] + dv * v0_[j]);
      }
    }
  };
  block(0, 0, f, Gf.by_r0_along_r0, Gf.by_r0_along_v0, Gg.by_r0_along_r0, Gg.by_r0_along_v0);
  block(0, 1, g, Gf.by_v0_along_r0, Gf.by_v0_along_v0, Gg.by_v0_along_r0, Gg.by_v0_along_v0);
  block(1, 0, fd, Gfd.by_r0_along_r0, Gfd.by_r0_along_v0, Ggd.by_r0_along_r0, Ggd.by_r0_along_v0);
  block(1, 1, gd, Gfd.by_v0_along_r0, Gfd.by_v0_along_v0, Ggd.by_v0_along_r0, Ggd.by_v0_along_v0);
  return out;
}

void TwoBodyOrbit::propagate(std::span<const double> dts, std::span<double> out) const noexcept {
  for (std::size_t i = 0; i < dts.size(); ++i) {
    const State rv = propagate(dts[i]);
    std::copy(rv.begin(), rv.end(), out.begin() + static_cast<std::ptrdiff_t>(6 * i));
  }
}

void TwoBodyOrbit::propagate_with_stm(std::span<const double> dts, std::span<double> out,
                                      std::span<double> stm_out) const noexcept {
  for (std::size_t i = 0; i < dts.size(); ++i) {
    const StateWithStm x = propagate_with_stm(dts[i]);
    std::copy(x.rv.begin(), x.rv.end(), out.begin() + static_cast<std::ptrdiff_t>(6 * i));
    std::copy(x.stm.begin(), x.stm.end(), stm_out.begin() + static_cast<std::ptrdiff_t>(36 * i));
  }
}

void propagate_grid(std::span<const TwoBodyOrbit> orbits, std::span<const double> dts, std::span<double> out) noexcept {
  const std::size_t stride = 6 * dts.size();
  for (std::size_t m = 0; m < orbits.size(); ++m) {
    orbits[m].propagate(dts, out.subspan(m * stride, stride));
  }
}

} // namespace tag::propagation
