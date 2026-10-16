# Phase 4: the fixed-cost step for real time

- **Dates:** 16 October 2026
- **Plan:** task 4.5; decision D35
- **Commits:** see the index
- **Written:** at the time

## Goal

A fixed-cost implicit step: linearly implicit Euler with a single linear
solve, the real-time integrator. *Done when its cost per step does not
vary.* Task 4.3 had deferred analytic Jacobians to here.

## Starting point

Three implicit methods (generalised-alpha, SDIRK4, Radau IIA), all with
iteration matrices from differences of the whole dynamics, kept across
steps and rebuilt when Newton slows: cheap on average, but a rebuild costs
2 nv + nx evaluations at once. On the sedan generalised-alpha's steps took
from 2 to 76 evaluations. The force elements expose only apply(); their
law() functions give derivatives in each element's own coordinates.

## What was done

- **Design.** Read what the kernel offers (body Jacobians with the rows the
  body states use, the mass matrix of each evaluation, the constraint
  solver's range-space factorizations) and chose a step that makes only the
  applied forces implicit, through their derivatives, as a correction to the
  explicit Euler step, solved in the constrained velocities; the derivatives
  built every step, element by element, by differences of each element
  alone.
- `ConstraintSolver::constrained_solve` (an SPD matrix in place of M, the
  correction confined to J delta = 0); the dense LU moved to
  `src/kernel/dense_lu.hpp` for all the implicit methods;
  `Integrator::LinearlyImplicitEuler` and `Simulator::force_jacobian` in
  `src/kernel/simulator_linear.cpp`, its work behind a pointer
  (`src/kernel/linear_work.hpp`) so that the public header does not carry it.
- Four kernel tests passed at the first build: backward Euler's recurrence,
  the stiff spring, the derivatives against differences of the generalized
  forces, the non-positive-definite case (code K045).
- On the vehicles the sedan accelerated a fifth too slowly; coupling the
  states (wheel spins) to the bodies mended that and made a coasting car gain
  energy at long steps; the position coupling Xq was found to be the cause
  and left out (below).
- Tests on the rolling car and on the sedan through a drive; the benchmark
  extended to time single steps of three methods over a drive, pinned.

## Decisions

- **D35**: linearly implicit Euler with force derivatives element by
  element, symmetric parts, the states coupled through their velocities, not
  their positions.

## What went wrong, and how it was found

- **A W-method is only as good as its stiff couplings.** The first version
  made the wheel spins implicit only in themselves. Every test of the step
  passed, the car rolled stably, and the sedan reached 1.17 m/s after a
  second of half throttle where RK4 and generalised-alpha reached 1.47. The
  tyre's force depends steeply on both the spin and the hub's velocity
  (slip); implicit in each separately, the hub was pulled every step towards
  the spin the step started with, a strong damper that braked the car. Such
  a method stays consistent with any approximate Jacobian, but its error is
  of order h times the stiffness it leaves out, which here is not small.
  Found by comparing the top speed with RK4's in a measuring run, not by
  any test written before: the vehicle test now checks it.
- **A hypothesis about symmetry, wrong.** With the coupling, a car coasting
  at walking pace gained speed at long steps (+0.5 % in 2 s at 5 ms, +5 % at
  10 ms), growing much faster than the step. Since the solve uses symmetric
  parts, the antisymmetric part of the spin-hub coupling seemed the obvious
  culprit: measured, that coupling is symmetric to 1e-7. Switching the
  coupling's parts off one at a time found the cause in Xq, the spins' rates'
  dependence on positions, and reading the tyre explained its size: its
  rolling radius depends on its deflection, so the wheel's height changes the
  slip. That loop is not stiff and not symmetric; it is left out.
- **An incomplete type.** Putting the work behind a `std::unique_ptr`
  compiled in its own file but not in the simulator's constructor, which
  must be able to destroy it if construction throws; its definition moved
  to an internal header both include.
- **Timing outliers.** The first pinned run gave every method a maximum
  near 1 ms: the first step, which sizes the work. The benchmark now leaves
  out the first ten steps. Two runs then still differed in their longest
  steps (258 against 2301 us for the same method), which marks them as the
  operating system's.

## Verification

Seven new test cases (five in the kernel, two on the vehicles); the method
added to the allocation and brake-to-stop tests; 454 tests, no compiler
warning. Numbers and commands in the data file
`2026-10-16_linearly-implicit-euler.md`.

- **The method is the one intended:** on a linear system each step equals
  backward Euler's to 1.5e-13; the stiff spring shrinks by exactly 1 /
  sqrt(1 + (w h)^2) = 0.0111796.
- **The derivatives:** element by element they equal differences of the
  generalized forces computed by the kernel's own functions, to 1.2e-9
  (damping) and 2.5e-8 (stiffness, at a load-free state, where the
  geometric terms they leave out vanish).
- **First order:** ratio 2.07 on a spatial pendulum.
- **The vehicles:** the rolling car keeps its speed (slightly falling, as
  backward Euler must) at every step from 1 to 10 ms, wheels within 1e-4 of
  free rolling; the sedan drives, brakes to a stop and stands as RK4 does
  (top speed 1.4684 against 1.4685 m/s).
- **The "done when":** one evaluation and one set of derivatives on every
  one of the 4000 steps of the drive; pinned, a step takes 106 us (median)
  with 99 % within 1.3 times that, where generalised-alpha's longest steps
  take 2.1 to 2.3 ms in every run.

## Open ends

- The projection after the step takes one Gauss-Newton iteration or none,
  as the drift requires: the step's one remaining variation. Task 4.7
  (constraints per integrator) is the place for an index-2 form.
- First order with damping of (w h)^2 / 2 per step: a second-order
  linearly implicit method (a trapezoid, or generalised-alpha with one
  Newton iteration on these derivatives) would suit real time better where
  the wheel hop matters.
- The element-by-element derivatives could build the other implicit
  methods' iteration matrices at a fraction of their cost; analytic
  versions for the commonest elements (springs, bushings) where timing asks.
- Non-symmetric parts (the tyres' forces turning with the wheels, a fifth of
  K on the sedan) stay explicit.

## For the book

- A real-time integrator: why "the same cost every step" matters more than
  the average, and generalised-alpha's two-millisecond steps as the
  counter-example.
- W-methods: consistency with any Jacobian, and why stability and accuracy
  still need the stiff couplings; the car that accelerated a fifth too
  slowly.
- Derivatives element by element: what is differenced, what is left out
  (the geometric stiffness of the tree), and the test that separates them.
- The tyre's rolling radius as a hidden coupling, found by switching parts
  off one at a time.
