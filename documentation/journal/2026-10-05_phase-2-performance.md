# Phase 2: performance, and no allocation per step

- **Dates:** 5 October 2026
- **Plan:** task 2.10; finding F15
- **Commits:** `9472d8d`
- **Written:** at the time. Raw data: `data/2026-10-05_bench-pinned.md`,
  `data/2026-10-05_sedan-ab.md`

## Goal

Benchmarks for each algorithm on chains of 2 to 64 bodies and on the detailed
sedan, recorded in `docs/performance.md`. *Target: the sedan steps in under
0.25 ms on average on this laptop, with zero allocations per step.*

## Starting point

On 4 October the sedan had stepped in 292 us (`24faade`), from 739 us on the
legacy code in the same run.

## What happened, in order

1. **A false alarm.** The benchmark was extended to break one evaluation of
   the accelerations into its parts, built, and run: the step took 703 us, 2.4
   times the figure of the day before, and the parts did not add up. Before
   blaming the morning's changes, the chain and tree rows were compared with
   their recorded values: they had barely moved, and they were erratic within
   the run (16 bodies cost 5.5 times 8 bodies).
2. **The cause was the machine.** The i7-13700H mixes performance and
   efficiency cores. A single-threaded benchmark goes wherever Windows puts
   it, and an efficiency core runs it at about half speed. Pinned to a
   performance core at high priority, two runs agreed within a few per cent.
3. **Was there a regression?** The commit of 4 October was extracted with
   `git archive`, built separately against the dependency sources already
   downloaded, and its benchmark run alternately with the current one, both
   pinned to the same core: 180.8 and 182.3 us against 178.4 and 178.7 us.
   No regression; the three tasks done that day (2.7b to 2.9) had cost nothing.
4. **But the machine itself also changes state.** An hour earlier, pinned runs
   of the same code had given 395 to 413 us. Same code, same core: the laptop
   was 2.2 times slower then, for reasons outside the code (temperature,
   power plan or memory pressure; not established). From then on, versions
   were compared only alternately within one session.
5. **Where the time went.** Per evaluation of the accelerations the simulator
   ran the kinematics for the forces, then the constraint solver ran them
   again for the constraint equations, again inside the mass matrix and again
   inside the bias forces: four passes. The constraint matrix `J M^-1 J^T` was
   formed through `M^-1 J^T`, two triangular solves with 17 right-hand sides.
   The projection after each step computed a fresh mass matrix and factorised
   it, and evaluated the constraints once more for the velocities.
6. **The changes.**
   - One pass, `forward_kinematics(q, v, 0)`, now serves the forces, the
     constraints, the mass matrix and the bias forces. Two new functions,
     `mass_matrix(model, data)` and `bias_forces(model, data)`, work from the
     data it leaves. The bias forces need RNEA's "acceleration minus gravity";
     because gravity is a uniform field, that is the physical acceleration
     from the zero-acceleration pass less gravity in the body's axes,
     `a_gf = a - [0; R^T g]`, with no further pass.
   - The constraint matrix is `Z^T Z` with `Z = L^-1 J^T`, one triangular
     solve; `M^-1 J^T` is never formed, and its products with vectors are
     applied as `L^-T (Z x)`.
   - The projection weighs by the mass matrix that the step's last stage
     factorised, close to the current state: any positive definite weight
     gives a valid projection, and only which nearby point of the constraint
     manifold is chosen changes, to second order. Its velocity part reuses
     the `J` and `nu` of its last position evaluation, since `nu` does not
     depend on v.
7. **Zero allocations, proven.** A new test executable, `test_alloc`, replaces
   the global `operator new` with a counting one and steps the sedan (tyres,
   drivetrain, steering) and a four-bar with both integrators for 100 steps
   each: no allocation. Eigen allocates with `malloc`, which this does not
   see, so the CI job `kernel-no-malloc` runs the same test in a Debug build
   with `EIGEN_RUNTIME_NO_MALLOC`, where any Eigen allocation fails an
   assertion. Both passed.

## Decisions

- Benchmark protocol: pinned to a performance core, high priority, versions
  compared alternately in one session (decision D21).
- One kinematics pass per evaluation; `Z = L^-1 J^T`; projection weighted by
  the last stage's mass matrix (decision D20).
- Callbacks may read the simulator's data but must not run its kinematics
  (documented on the callbacks), since the evaluation in progress uses them.

## Verification

All 366 tests pass; the mass matrix is bit-identical to before (the
placements do not depend on v), the bias forces differ by rounding only, and
the projection's change is second order in the drift, so no tolerance moved.

## Measurements

Pinned to logical processor 2, the old and new code alternately, two rounds:

| | 4 October | Now |
|---|---|---|
| constrained forward dynamics | 30.4, 30.5 us | 23.7, 23.6 us |
| projection (state already on the constraints) | 33.9, 34.2 us | 24.1, 23.8 us |
| one RK4 step of 1 ms | 180.8, 182.3 us | 124.4, 124.2 us |

The step takes 124 us, half the target, in the machine's normal state; in the
slow state seen that morning it would take about 280 us. That caveat is
written into the plan's result.

## Lessons

- Establish the measurement before believing a number: the first figure of the
  day was off by 2.4 for reasons that had nothing to do with the code.
- Keep the old version buildable for A/B runs; `git archive` plus the cached
  dependencies made it a five-minute job.
- "Four passes" was found by reading the call chain with the timings beside
  it, not by guessing.

## Open ends

Of the 25.8 us of one evaluation, the linear algebra takes about 11 and the
constraint equations about 6. The mass matrix of the sedan is block-sparse
(chassis, plus four independent corners), which a tree-structured
factorisation would exploit if more speed is needed (Phase 12).

## For the book

- "The benchmark that lied": hybrid cores, pinning, A/B in one session. A
  figure of the same benchmark unpinned, pinned, and pinned an hour apart.
- The gravity identity `a_gf = a - [0; R^T g]` as an example of letting one
  pass serve several algorithms.
- How to prove "no allocation": counting `operator new`, plus Eigen's own
  switch for its `malloc`.
