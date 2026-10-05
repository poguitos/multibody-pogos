# Phase 3: output requests

- **Dates:** 5 October 2026
- **Plan:** task 3.3; working rule 4 (one place per formula)
- **Commits:** see the index
- **Written:** at the time

## Goal

Joint reaction forces and moments, constraint forces from the multipliers,
force-element loads, body accelerations, energies, and a recorder with named
channels. *Done when pendulum pivot reactions match the analytic values.*

## The design question

A joint's reaction in a tree without loops is recursive Newton-Euler with
the actual accelerations. With loops, it also needs each constraint's force
on each body, and the solver only had their sum in generalized coordinates,
`J^T lambda`, from which the per-body forces cannot be recovered. Every
Jacobian row, though, is built in one function, `add_wrench_row`, from a
world wrench on a body. Three ways to get the body wrenches:

1. Write each primitive's wrenches a second time for the forces: two copies of
   each formula, the very pattern behind findings F1 to F4.
2. Record the wrenches inside `add_wrench_row` through a thread-local
   pointer: no new code in the primitives, but the row index it sees is local
   to each nested block, so recovering the absolute row needs pointer
   arithmetic on the matrix.
3. Lift each primitive's row formulas into one function that hands every
   (body, wrench, row) to a sink: the Jacobian sink adds the row to J, the
   wrench sink adds the multiplier times the wrench to the body.

The third was chosen: one place per formula (working rule 4), visible in the
code, at the cost of a virtual call per row.

## What was done

- `ConstraintModel::add_wrenches(model, data, t, lambda, wrenches)`, with the
  row functions shared with `calc()`; a joint driver adds nothing (its
  multiplier is its drive force, on its joint's coordinate).
- `joint_reactions(model, data, external, reactions)` and
  `compute_loads(sim)`.
- `Curve::integral`, exact for the cubic pieces; `potential_energy()` for the
  spring-damper, the joint coordinate force and the rotational
  spring-damper.
- `Recorder` with named channels and CSV output at 17 significant digits, so
  values read back exactly.
- Codes K006, A020 to A023.

## What went wrong, and how it was found

- When reading the measured margins, one test printed nothing: its name has a
  comma, which Catch2 takes as a separator between test names. Escaped as
  `\,`. Worth knowing for anyone running single tests by name.

## Verification

| Check | Measured | Bound |
|---|---|---|
| hinge reaction against `m a - m g`, six states of a swing | 3.2e-14 N | 2e-8 N |
| hinge reaction moment about the pivot | 3.6e-15 N m | 1.8e-8 N m |
| revolute closure's force from its multipliers | 6.4e-13 N | 2e-7 N |
| its moment about the pivot | 3.3e-15 N m | 1.8e-7 N m |
| energy, pendulum with torsion spring, spring and preloads, 2 s | 1.1e-13 J | 1e-9 of E0 |

Potentials against forces (`-dV/dx` by central differences) for every spring
element, and `Curve::integral` against the curve. 390 tests pass; all the
constraint tests pass unchanged after the refactor of the rows.

## For the book

- How reactions are computed: recursive Newton-Euler with external forces,
  and why loops need the constraint forces per body.
- "One place per formula" applied: the row sink, and why the alternative of a
  second copy was the very cause of the original defects.
- Energy conservation as a test of potentials and forces together.
