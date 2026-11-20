# Tag

Satellite orbit propagation and Kalman filtering: predicting where a satellite will be, and estimating where it is from noisy measurements.

## Language

### Propagation

**Propagation**:
Computing a satellite's state at another time from its state at an epoch, using a gravity model.
_Avoid_: Integration (propagation here is analytical, not numerical), prediction (reserved for the filter's predict step)

**Epoch**:
The reference time an orbit's state is given at. Propagation times are seconds relative to it.

**Two-body**:
The gravity model in which the satellite feels only a point-mass central body, so the orbit is a fixed Keplerian ellipse.
_Avoid_: Keplerian propagation, unperturbed

**Secular J2**:
The gravity model that adds the Earth's oblateness (J2) as steady drifts in the node, argument of perigee and mean anomaly of the mean elements.
_Avoid_: Analytical J2, J2 propagation (ambiguous with numerically integrated J2)

**Short-period terms**:
The oscillations, on the timescale of one orbit, that J2 adds on top of the secular drift. They are the difference between osculating and mean elements.

**Osculating elements**:
The orbital elements of the instantaneous two-body orbit through the actual position and velocity. Measurements and Cartesian states are osculating.
_Avoid_: True elements, instantaneous elements

**Mean elements**:
Orbital elements with the short-period terms averaged out, as defined by the Brouwer–Lyddane theory. Secular J2 drifts are applied to these.

**Keplerian elements**:
The classical element set (a, e, i, Ω, ω, anomaly). Undefined for circular or equatorial orbits.
_Avoid_: Classical elements, COEs

**Equinoctial elements**:
A non-singular element set that stays well defined for circular and equatorial orbits.

**State transition matrix**:
The 6×6 sensitivity of a propagated Cartesian state to the initial Cartesian state.
_Avoid_: STM in prose, Jacobian (reserved for the filter's `F_jac`/`H_jac`)

**Hyperbolic orbit**:
An orbit with eccentricity ≥ 1, i.e. one that escapes. Outside the propagator's domain.
_Avoid_: Open orbit, escape trajectory

### Filtering

**Rejected update**:
A measurement update the filter discarded after computing it, keeping the prior, because the posterior was physically invalid (e.g. a hyperbolic orbit).
_Avoid_: Skipped update (a skip is a missing measurement), failed update

**Gating**:
Discarding a measurement before the update because its innovation is statistically implausible (NIS above a chi-squared threshold).
_Avoid_: Outlier rejection (ambiguous with rejected update)

**Divergence**:
The estimate's actual error growing well beyond what its covariance claims, i.e. the filter is wrong and overconfident.

**Lock-out**:
Divergence that sustains itself because the filter keeps refusing measurements (by gating or rejected updates) as judged against its own wrong estimate.
_Avoid_: Divergence (lock-out is a specific mechanism of it)
