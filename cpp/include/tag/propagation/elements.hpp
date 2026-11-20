#pragma once

#include <array>

#include "tag/propagation/cartesian.hpp"

namespace tag::propagation {

// Orbital elements as plain arrays of six doubles, angles in radians, lengths in the units of mu.
using Elements = std::array<double, 6>;

// Keplerian elements [a, e, i, raan, argp, anomaly].
namespace kep {
inline constexpr int a = 0, e = 1, i = 2, raan = 3, argp = 4, anomaly = 5;
} // namespace kep

// Equinoctial elements [a, h, k, p, q, lambda] (direct set): h = e sin(argp + raan),
// k = e cos(argp + raan), p = tan(i/2) sin(raan), q = tan(i/2) cos(raan), and lambda the mean
// longitude M + argp + raan.
namespace eq {
inline constexpr int a = 0, h = 1, k = 2, p = 3, q = 4, lambda = 5;
} // namespace eq

// Which anomaly the last Keplerian element holds.
enum class Anomaly { mean, true_anomaly, eccentric };

// Cartesian state -> Keplerian elements. Angles are returned in [0, 2pi) (i in [0, pi]).
// Singular cases follow the usual conventions, below a tolerance of 1e-11 on e and on sin(i):
// - circular (e = 0): argp = 0 and the anomaly is measured from the ascending node (argument of
//   latitude);
// - equatorial (i = 0 or pi): raan = 0 and argp is measured from the x axis (longitude of periapsis);
// - circular and equatorial: raan = argp = 0 and the anomaly is the true longitude from x.
// Near these cases the split between raan, argp and the anomaly is ill-conditioned (their sum is
// not); use equinoctial elements there.
// Precondition: an elliptic orbit (finite rv, r != 0, nonzero angular momentum, e < 1) and mu > 0.
[[nodiscard]] Elements cartesian_to_keplerian(const State &rv, double mu, Anomaly anomaly) noexcept;

// Keplerian elements -> Cartesian state. Precondition: a > 0, 0 <= e < 1, finite angles, mu > 0.
[[nodiscard]] State keplerian_to_cartesian(const Elements &kep, double mu, Anomaly anomaly) noexcept;

// Cartesian state -> equinoctial elements; lambda is returned in [0, 2pi). Non-singular for
// circular and equatorial orbits; singular only for retrograde equatorial orbits (i = pi).
// Precondition: as for cartesian_to_keplerian.
[[nodiscard]] Elements cartesian_to_equinoctial(const State &rv, double mu) noexcept;

// Equinoctial elements -> Cartesian state. Precondition: a > 0, h^2 + k^2 < 1, finite values, mu > 0.
[[nodiscard]] State equinoctial_to_cartesian(const Elements &eq, double mu) noexcept;

} // namespace tag::propagation
