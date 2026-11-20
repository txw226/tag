# J2 is analytical: secular drift plus first-order Brouwer–Lyddane short-period terms

J2 propagation applies secular drift rates to Brouwer–Lyddane **mean** elements and converts back to **osculating** at the output (first-order in J2), rather than numerically integrating the J2 acceleration. Integration would be ~100× slower and would break the closed-form fast path that two-body uses, and 95% of use is two-body anyway. Osculating→mean has no closed form, so it is done by fixed-point iteration of the inverse. The theory is singular at the critical inclination (63.4°); that limitation is documented, not fixed. `mean=True` skips the final conversion for speed. The J2 state transition matrix is the analytic derivative of this first-order theory, verified against central differences computed in C++.

## Consequences

- Accuracy is first-order: errors grow at O(J2²). Tests compare against an in-repo high-accuracy RK4 integration of the true J2 acceleration, with thresholds taken from a sweep.
