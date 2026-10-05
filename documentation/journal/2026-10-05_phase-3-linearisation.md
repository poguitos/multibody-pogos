# Phase 3: linearisation

- **Dates:** 5 October 2026
- **Plan:** task 3.7; finding F14; decisions D24 (derivatives), D27, D28
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session. The session's worker process
  was restarted while this task was being written (after the shared basis
  of allowed motions had been moved into `src/kernel/motions.hpp`, before
  any linearisation code existed); the work continued from the files on disk,
  and nothing was lost because tasks 3.4 and 3.5 had been committed and
  pushed.

## Goal

State matrices about an operating point, natural frequencies, damping and
mode shapes. *Done when quarter-car frequencies match the analytic ones.*

## Starting point

Statics (task 3.5) had just given the operating points to linearise about,
the basis N of the allowed motions, and a finite-difference tangent
stiffness. The simulator's `acceleration()` evaluates everything applied at
any (q, v, t). The existing quarter-car test (from task 2.7) measured the
body-bounce frequency from zero crossings of a simulation, to 10 %.

## What was done

1. The basis of allowed motions moved from `statics.cpp` into
   `src/kernel/motions.hpp`, shared by statics and linearisation.
2. `kernel::linearize(sim, options)`: the reduced state `y = N^T (q (-) q0)`,
   `z = N^T v`; each of the 2d coordinates moved by +-1e-5, the point
   projected onto the constraints (positions, then velocities), its actual
   reduced coordinates and its derivatives `(y', z')` evaluated; A fitted as
   `dF dX^-1`. B from unit generalized forces (exact, the accelerations being
   affine in them). The reduced M, K, C; the modes of A and the undamped
   modes of `(K_s, M)`; shapes in the velocity coordinates with the
   coordinate of the largest entry named. The simulator's state, tau
   included, is restored.
3. Codes K080 (not an equilibrium), K081 (zero-frequency modes), K082
   (growing modes).
4. Seven tests (`tests/kernel/test_kernel_linearization.cpp`).

## Decisions

D28: reduce to the coordinates of the constraint surface before
differentiating, rather than linearise the full state and project out the
constraint directions; fit A against the points' actual coordinates; an
orthonormal basis rather than a set of independent coordinates picked from
q.

## What went wrong, and how it was found

Nothing failed in this task: the first build passed every test. Two things
were caught at the design stage, before code:

- **The chart of the rotations.** For a moving state, `d/dt (q (-) q0)` is not
  `v` for a spherical or free joint away from q0; using `N^T v` would put an
  error of the order of the angular velocity into A. At rest, or at q0, it
  is exact, and those are the only points an equilibrium linearisation
  visits; elsewhere a central difference in time is used.
- **Points that are not where they were asked to be.** Projecting a
  perturbed configuration back onto the constraints, or its velocities onto
  `J v = nu`, moves it slightly in the other reduced coordinates as well.
  Dividing by the intended step would bias A by that; fitting against the
  actual coordinates removes it.

## Verification

| Check | Measured | Bound, and why |
|---|---|---|
| Quarter car, undamped frequencies against the closed form | equal to 12 digits (1.356369445435, 11.811152065254 Hz) | 1e-9 relative; roundoff of the differences, about 1e-11 |
| Quarter car, damped eigenvalues against the state matrix written by hand | 2e-11 relative | 1e-9 |
| Quarter car, body-bounce shape against `(ks - w1^2 ms) / ks` (0.0921) | 2.4e-14 | 1e-9 |
| Damped oscillator: frequency, damping ratio, k, c, 1/m | 3e-16, 8e-12, 0, 8e-12, 2e-16 | 1e-9 (1e-12 for M and B) |
| Pendulum held only by a revolute closure, `m g d / I` | 1.4e-11 | 1e-8: the central difference's truncation |
| Four-bar against the energy method `V'' / M_eff` (2.592 Hz) | 2e-8 | 1e-6: the energy method's own differences |
| Upright balance: omega^2 and growth rate | 2.3e-11, 1.1e-11 | 1e-8 |
| Off-equilibrium point reported (K080); state, velocities and tau restored | exactly | |
| Sedan at rest: 10 degrees of freedom, 3 zero modes, none negative, no growing mode | as stated | |

411 tests pass with GCC 13 (404 before).

## Measurements

[data/2026-10-05_linearisation.md](data/2026-10-05_linearisation.md). The
double-wishbone sedan, settled by statics: undamped body modes at 1.15, 1.70
and 1.85 Hz and four wheel-hop modes at 17.6 to 17.8 Hz; three zero modes
(position and heading); in A the free directions show as zero or real
eigenvalues, damped by the tyres' slip at low speed. 8 ms in the container
(4 x 10 + 27 evaluations). The 17.7 Hz wheel hop is higher than the 10 to
15 Hz usual for a sedan; whether that is the template's tyre stiffness or
its light unsprung masses (finding F9: wheel masses are dropped) is for the
vehicle subsystems of Phase 7, and it is a number to check then.

## Lessons

- Settle the coordinates before differentiating. Choosing what y and z mean
  (the constraint surface, the chart of the rotations) took most of the
  time and left the code short.
- An independent method for every number: closed forms, a hand-built state
  matrix, the energy method along a loop. The pendulum held only by a
  closure is the case that would have exposed a missing geometric stiffness
  of the constraint forces.

## Open ends

- Linearisation about steady motions in a moving frame (cornering, Phase 9).
- The sedan's wheel-hop frequency against the template's parameters
  (Phase 7, with F9).
- The analytic tangent matrices (D24), when larger models make the
  differences costly; implicit integration (Phase 4) needs the same
  Jacobian.

## For the book

- The chapter on linearisation: why reduce first; the secant fit; the chart
  of the rotations; the geometric stiffness of constraint forces, with the
  pendulum held by a closure as the example. Figure: the quarter car's poles
  in the complex plane, against the hand-built ones; the sedan's mode
  shapes.
