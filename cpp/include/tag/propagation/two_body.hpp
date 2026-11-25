#pragma once

#include <span>

#include "tag/propagation/cartesian.hpp"
#include "tag/propagation/kepler.hpp"

namespace tag::propagation {

// A propagated state and its state transition matrix d x(t) / d x(epoch).
struct StateWithStm {
  State rv;
  Matrix6 stm;
};

// A two-body (Keplerian) elliptic orbit, built from an osculating Cartesian state at its epoch.
//
// Construction precomputes the orbit's constants, so propagate() is one Kepler solve plus a
// 2x3 rotation: r = A (cos E - e) + B sin E and v = n (B cos E - A sin E) / (1 - e cos E), with
// A = a P and B = b Q along the perifocal axes. No true anomaly, no atan2. The perifocal axes are
// ill-defined as e -> 0 and only the node is singular as i -> 0, but every term along the noisy
// periapsis direction is scaled by e, so the propagated states stay accurate for near-circular,
// near-equatorial and GEO orbits (at e == 0 exactly, P is taken along r).
//
// Precondition: rv finite with r != 0, mu > 0, and the orbit elliptic (e < 1, nonzero angular
// momentum); unchecked, the bindings validate.
class TwoBodyOrbit {
public:
  TwoBodyOrbit(const State &rv, double mu) noexcept;

  // The Cartesian state dt (any sign) after the epoch. The mean anomaly is advanced with one
  // rounding and reduced modulo 2*pi, so precision doesn't decay with |dt|.
  [[nodiscard]] State propagate(double dt) const noexcept;

  // Batched propagate: writes propagate(dts[i]) to out[6 i .. 6 i + 5], with no allocation.
  // Precondition: out.size() == 6 * dts.size() and every dt finite.
  void propagate(std::span<const double> dts, std::span<double> out) const noexcept;

  // The state dt after the epoch and the state transition matrix d x(t) / d x(epoch), in closed form
  // (Lagrange f and g, differentiated through Kepler's equation in difference form, Battin style).
  // It reuses the propagation's Kepler solution; propagate() pays nothing for it.
  [[nodiscard]] StateWithStm propagate_with_stm(double dt) const noexcept;

  // Batched propagate_with_stm: states to out[6 i ..] and matrices to stm_out[36 i ..], with no
  // allocation. Precondition: out.size() == 6 * dts.size(), stm_out.size() == 36 * dts.size(), every
  // dt finite, and out and stm_out not overlapping.
  void propagate_with_stm(std::span<const double> dts, std::span<double> out, std::span<double> stm_out) const noexcept;

  [[nodiscard]] double mu() const noexcept { return mu_; }
  [[nodiscard]] double semi_major_axis() const noexcept { return a_; }
  [[nodiscard]] double eccentricity() const noexcept { return e_; }
  [[nodiscard]] double mean_motion() const noexcept { return n_; }

private:
  // The Kepler solution dt after the epoch, shared by propagate and propagate_with_stm.
  [[nodiscard]] KeplerSolution kepler_at(double dt) const noexcept;
  [[nodiscard]] State state_at(double cos_E, double sin_E) const noexcept;

  double mu_;
  double a_;
  double e_;
  double n_;  // mean motion, rad per time unit
  double m0_; // mean anomaly at the epoch, in [-pi, pi]
  Vec3 A_;    // a P
  Vec3 B_;    // b Q
  // Epoch quantities for the state transition matrix.
  Vec3 r0_;
  Vec3 v0_;
  double r0_norm_;
  double sigma0_; // r0 . v0
  double alpha_;  // 1 / a = 2 / r0 - v0^2 / mu
  double sin_E0_;
  double cos_E0_;
};

// Propagates every orbit to every time: writes orbits[m].propagate(dts[j]) to
// out[6 (m N + j) .. 6 (m N + j) + 5], N = dts.size(), with no allocation.
// Precondition: out.size() == 6 * orbits.size() * dts.size() and every dt finite.
void propagate_grid(std::span<const TwoBodyOrbit> orbits, std::span<const double> dts, std::span<double> out) noexcept;

} // namespace tag::propagation
