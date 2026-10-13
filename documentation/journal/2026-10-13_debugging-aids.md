# Debugging aids: describe, state dump, step trace

- **Dates:** 13 October 2026
- **Plan:** documentation track H, task H.4
- **Commits:** see the index
- **Written:** at the time

## Goal

`describe(system)`: bodies, joints, constraints, force elements and the
degree-of-freedom count, as text. A step trace switched on by an option,
written to CSV: projection iterations and residuals, constraint residual,
energy, redundancy, per step. A state dump. *Done when a test shows a
deliberately broken model diagnosed from its trace alone.*

## Starting point

`validate()` (task 2.9) reported problems with a model but not the model
itself; a simulation that went wrong left only the last projection's
numbers (`last_projection()`) and a warning at the first failure.

## What was done

- `describe(system)` and `describe(system, q, t)` (`kernel/describe.hpp`):
  counts with `validate()`'s rank, degrees of freedom and numbers of
  findings, gravity, each body with its joint, coordinate and velocity
  ranges, mass and centre of mass, each constraint, force element and joint
  force with its bodies.
- `dump_state(sim)`: time, q, v, each body's world placement and velocity,
  the constraint residuals.
- `Simulator::trace_steps`, `trace()`, `write_trace(path)`: one
  `StepRecord` per step (time, step, drift before the projection, residuals
  after it, iterations, converged, equations, rank, kinetic and
  gravitational energy, events), as CSV with 17 significant digits.
  `ProjectionInfo` gained the residual before the projection.
- Code K053 (the trace file cannot be written).

## Decisions

- The trace is a vector of records in the simulator, written on request,
  rather than a stream opened at the start: nothing touches the file system
  during a run, and a test (or a debugger) can read the records directly.
  It is off by default, so stepping stays allocation-free.
- `describe()` reuses `validate()` for its counts, so the rank and degrees of
  freedom it prints are the solver's own; like `validate()`, it can be asked
  at a given configuration, because the neutral one may be neither assembled
  nor of full rank (the test's four-bar is both, all links along X).

## What went wrong, and how it was found

- **The first broken model was not crisp.** The plan for the test was a
  pendulum whose tip is held out of reach. Working out what its trace would
  show, before writing the test, gave a residual that wanders: the closest
  approach is where the constraint's gradient vanishes, so the projection's
  Newton steps blow up near it and the step halving leaves the state
  anywhere close. Two drivers fighting over one coordinate give a trace that
  is exact and deterministic (rank 1 of 2, residual equal to the targets'
  difference), and they are the modelling error people actually make.
- **A closure listed its bodies three times.** Reading `describe()`'s output
  for the four-bar showed "body 0 (ground), body 3 (rocker)" three times on
  the closure's line: a constraint set returns its parts' bodies one after
  another. The test had passed because it matched only the start of the
  line. `describe()` now lists each body once, and the test checks the end
  of the line.
- **A slow machine.** The full rebuild of this task took 28.8 minutes for 96
  objects, against 5.7 minutes for 99 earlier the same day, with no other
  process busy and 4 GB free: the laptop's own state (power or temperature),
  as noted on 5 October, not the code.

## Verification

Three test cases, 423 tests in all:

- **describe():** the four-bar with a spring, at an assembled configuration:
  "3 bodies, 3 coordinates, 3 velocities; 1 constraints (5 equations, rank
  2)", one degree of freedom, validate()'s one note, each body with its
  joint and mass, the closure and the spring with their bodies, each once.
- **dump_state():** the state, each body's world placement, the residuals.
- **The broken model (done when):** a slider driven to 0.1 m and held at
  0.3 m, ten steps traced, written to CSV and read back: every row shows two
  equations of rank 1, a projection not converged after its 20 iterations,
  and a position residual of 0.2 to 3e-17, the distance between the two
  targets; K050 and K051 were each reported once. The same model with both
  targets at 0.1: rank 1 of 2 again, but converged and with a residual of 0.
  The residual column alone tells conflicting constraints from harmless
  redundancy.

## Lessons

- A diagnostic is only as good as the case it was tried on: choose the
  broken model for a crisp signature, then show the healthy twin beside it.

## Open ends

- The trace could also record the force elements' energies (needs a
  potential on the `ForceElement` interface) and per-constraint residuals.
- The troubleshooting guide (H.3) should start from these tools.

## For the book

- Debugging a simulation from its trace: the signature of conflicting
  constraints against harmless redundancy.
