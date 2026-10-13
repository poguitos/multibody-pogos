# Phase 3: linearisation

- **Dates:** 13 October 2026
- **Plan:** task 3.7; decision D27
- **Commits:** see the index
- **Written:** at the time

## Goal

State matrices about an operating point, natural frequencies, damping and
mode shapes. *Done when quarter-car frequencies match the analytic ones.*

## Starting point

Statics (task 3.5) gives the operating point and already differentiates the
accelerations along the constraints by projected differences. The
quarter-car test measured its body mode from zero crossings of a simulation,
to 10 %.

## What was done

- `linearize(sim, options)` in `kernel/linearization.hpp`: reduced
  coordinates along independent columns of the weighted projector onto the
  allowed motions (picked by the pivots of `RankRevealingLDLT`, which gained
  a `pivot(k)` accessor); `A_q`, `A_v` by central differences, positions
  projected back onto the constraints; the reduced state matrix; modes from
  Eigen's general eigensolver, a complex pair listed once, with natural
  frequency, damping ratio and shape in velocity coordinates.
- `LinearModel`: operating point, residual there, basis, A, modes, warnings
  (K080 off the constraints, K081 at rest out of equilibrium), `summary`.
- `LeastChange::solve` now sizes its result (see below).

## Decisions

- **D27**: reduced coordinates from the projector's pivoted columns rather
  than an SVD or QR null space; differences through projections; the general
  eigenproblem, for damping.

## What went wrong, and how it was found

- **A segmentation fault on the first run.** Every linearisation test
  crashed. `LeastChange::solve` zeroed its output without sizing it; assembly
  and statics had always passed vectors of the right size, linearisation
  passed empty ones, and with Eigen's size checks off in release builds
  `v += dv` wrote past the end. Found by reading the code at the first call
  that could see an empty vector; fixed in `solve` itself, which now sizes
  its output, so no caller can repeat it.
- **A bound guessed rather than derived.** The damped quarter-car check
  (eigenvalues as roots of the characteristic determinant) was set at 1e-9
  and measured 3e-10 and 6e-10. The derivation (rounding of the differences
  against the damping terms, amplified by the determinant's sensitivity)
  gives about 1e-9, so the bound became 1e-8, with the derivation written
  next to it.

## Verification

Six new test cases, 413 tests in all:

- **Damped mass:** lambda = -2 + 19.8997i to 2.8e-10, zeta 0.1 to 1.4e-11.
- **Four-bar on a spring** (one degree of freedom in three coordinates,
  closed by five equations of rank 2): omega against sqrt(V'' / n^T M n)
  from the closed-form configurations, to 1.7e-8 of 9.83 rad/s; the shape is
  the mechanism's motion to 4e-16.
- **Off the constraints:** both warnings, and the model taken on the
  constraints.
- **Quarter car (done when):** both undamped frequencies to 1e-11 relative
  (1.356 and 11.811 Hz), the shape ratios k_s / (k_s - w^2 m_s) to 5e-10;
  with damping, both eigenvalues are roots of det(l^2 M + l C + K) to 6e-10
  relative.
- **Sedan:** ten degrees of freedom, exactly three zero eigenvalues (the
  neutral position and heading), no eigenvalue with a positive real part
  beyond 1.1e-10 (passivity).

## Measurements

In [data/2026-10-13_linearization.md](data/2026-10-13_linearization.md): the
margins, the sedan's fourteen modes, and the compile time (31 s for the file
that instantiates the general eigensolver).

## Lessons

- A helper that writes into a caller's vector should size it. The bug lay
  quiet through two tasks because their callers happened to pre-size.
- The sedan's modes at rest are distorted by the tyre's low-speed
  regularisation, which acts as a large damper. A linear model is only as
  meaningful as the forces at the operating point; vehicle modes belong at
  speed.

## Open ends

- Input matrices (B) for generalized forces or actuators, when a controller
  needs them (Phase 13), and output matrices beyond the basis.
- Linearisation about a state in motion is exact for joints whose
  coordinates add; for spherical and free joints the moving frame's terms
  are not modelled (noted in the header).
- A tyre model at standstill (or about a rolling state) so that vehicle
  modes at rest mean something; Phase 7 and 9.

## For the book

- Reduced coordinates on constraints without a null-space decomposition:
  the projector's pivoted columns.
- The quarter car as the textbook check: frequencies, shapes, and damped
  eigenvalues checked as roots of the characteristic determinant rather
  than against another eigensolver.
- How a linear model inherits the regularisations of the forces it is built
  from (the tyre at rest).
