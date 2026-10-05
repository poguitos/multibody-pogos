# Phase 2: the kinematics kernel

- **Dates:** 3 October 2026
- **Plan:** tasks 2.1 to 2.4; decisions D1, D2
- **Commits:** `cbab561`
- **Written:** reconstructed on 5 October 2026 from the commit, `docs/kernel.md`
  and the session record

## Goal

Replace the hand-written propagation loops, the cause of F1 to F4, with one
implementation of each joint's kinematics and one recursive algorithm per
quantity (D1), with quaternion coordinates for spherical and free joints (D2).

## What was done

- **Spatial algebra (2.1).** Motion and force vectors ordered
  `[angular; linear]`; frame changes as functions and 6 x 6 matrices; the two
  cross products; spatial inertia; the exponential and logarithm of rotations
  and rigid motions, with series expansions below 1e-2 to keep their small-angle
  limits accurate.
- **The joint interface (2.2).** Each joint describes itself once: its
  transform `X_J(q)`, motion subspace `S(q)`, velocity-product term `c(q, v)`,
  `q_dot = G(q) v`, and `integrate` and `difference` on its configuration
  space. Eight joints: revolute, prismatic, fixed, universal, cylindrical,
  planar, and quaternion spherical and free. Velocities are body-fixed, in the
  child-side joint frame.
- **Model and Data (2.3).** `Model` holds what does not change and can be
  shared by threads; `Data` holds what the algorithms compute and is sized
  once, so the algorithms do not allocate. The layout follows Pinocchio.
- **Algorithms (2.4).** Forward kinematics, RNEA, CRBA, ABA, body Jacobians,
  momentum, energies, centre of mass, all from Featherstone's *Rigid Body
  Dynamics Algorithms*.

## Decisions

- Quaternions stored `(x, y, z, w)`, Eigen's order; a Runge-Kutta step combines
  its stages' `q_dot` linearly and normalises once at the end, because
  normalising or retracting each stage would cost the scheme its order
  (`docs/kernel.md`, "integrate and difference").
- Gravity enters RNEA and ABA as an upward acceleration of the ground,
  Featherstone's trick, so no body needs a separate gravity force.
- The kernel's default gravity is `(0, 0, -9.81)`: Z up, as ISO 8855 (D3)
  will make the vehicle layer.

## Verification

`test_kernel`, 23 cases: the spatial identities; every joint as root and as
child of every other, against finite differences of the poses, d'Alembert's
principle, the agreement of RNEA, CRBA and ABA, gravity as the gradient of the
potential, energy and momentum conservation over a second of RK4; the bead on
a spinning rod in closed form; one `Model` on two threads with bit-identical
results; and agreement with the legacy algorithms on random trees to 1e-10
(largest difference 5e-13, in the accelerations). The legacy spherical and
free joints used a rotation vector, so the comparison maps velocities through
`w = E(r) r_dot` and compares `M = T^T M_legacy T`.

## Measurements

Forward dynamics 4 to 46 times faster than the legacy code on the same models,
the gap growing with size (the legacy mass matrix was built from dense
per-body Jacobians); `docs/performance.md`, 3 October.

## For the book

- The joint interface is the heart of the design: one description per joint,
  read by every algorithm. A figure of a joint's two frames and `X_J`, `S`, `c`.
- Why a Runge-Kutta step must not normalise its stages.
- The comparison with the legacy code needed a change of velocity
  coordinates: a clear small example of how to compare two formulations.
