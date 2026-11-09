# Ideas from external sources, second review: suspension kinematics, validation, target cascading

- **Dates:** 9 November 2026
- **Plan:** at the Phase 5 gate, on branch `phase-5`; decision D51
- **Commits:** see the index
- **Written:** at the time

## Goal

While CI ran on the Phase 5 gate, the author recommended three sources and
asked for more ideas: Nick McCleery's series on suspension kinematic
simulation, Suspension Explorer (its web application and its open-core
solver), and AVL's pages on vehicle engineering, "the highest standard in
automotive engineering solutions", not very technical but a guide to what
can be achieved. The question was the same as in the first review
(D42): what would make the program better, and where does it go in the
plan?

## What was done

- **The sources read.**
  - McCleery's three posts. Suspension Explorer's home page and three of its
    documentation pages (solver design, supported models, metrics).
  - The solver's README and its diagnostics and anti-geometry code, from a
    shallow clone that was read but not installed or run.
  - Twelve AVL pages: concept development, vehicle development, virtual and
    functional development, chassis simulation, lap time, VSM, EXCITE M, two
    blog posts, and AVL RACETECH.
  - Local copies of all of them in `C:\dev\references\`, with the index
    updated.
- **AVL's claim checked.** AVL claims "up to 97 % model correlation", with no
  definition. Asking what correlation should mean led to ISO 19364 and ISO
  19365, the standards for validating a vehicle-dynamics simulation against
  tests. A third standard on lateral transient response could not be
  confirmed and is not cited.
- **Checked against our own code first.**
  - `Kinematics::velocities()` already gives the exact tangent of a sweep,
    but returns the least-norm motion where freedom is left.
  - `camber_gain()` averages differences across a sweep.
  - A failed sweep step is kept and flagged.
  - Our distance residual is smooth where McCleery's needed a smoothed norm.
  - Our twist needs no fit to points.
- **The review written**
  (`documentation/review_2026-11-09/external_sources.md`): what each source
  is and what is useful, what the sources point to together, and fourteen
  proposals, each placed in a task.
- **The author adopted all of them** (D51), and the plan was amended:
  - 9.1: the steering axis from the motion, holds, responses in the null
    space, metric definitions, sweep diagnostics, an axle's road plane, and
    a second code;
  - 8.1 and 8.4: shims;
  - 9.5: a metric catalogue;
  - 9.6 and 9.7: the ISO procedures, coast-down and weave, and a model
    fitted to target figures;
  - 9.4: frequency responses;
  - 9.3: a self-tuning driver;
  - 13.3: targets cascaded to subsystems.

  The effort estimate rose by 50 to 75 hours.

## What went wrong, and how it was found

- **A licence that rules out more than code.** `suspension-explorer-core`
  is AGPL-3.0, and this repository has no licence yet. Its example hardpoint
  files are part of the licensed work too, so not even its data may be
  copied in. A comparison has to run it separately, as Chrono is run in
  `crosscheck/`. Recorded in D51 and in the references index.
- **A carriage return in a path.** A Windows path written through a Python
  string (`...\review_2026-11-09`) put a carriage return into the references
  index, because `\r` is an escape sequence. It was found by printing the
  line's characters, and fixed by a script working on bytes.
- **Vendor figures are not evidence.** AVL's numbers (40,000 laps per hour,
  300 validated models, 97 % correlation) are quoted in the review as
  claims only, and none is used to justify a proposal.

## Lessons

- A source that "is not very technical" still shows which workflows the
  industry sells: identification from a standard manoeuvre matrix, targets
  cascaded to subsystems, one model from the office to the simulator. Most
  of these were already in the plan, which is itself a useful finding.
- The two kinematics sources agree on what makes outputs trustworthy:
  definitions, holds, and refusing to report what is not defined. That is
  D44 and D50's attitude again, carried over to sensitivities.

## For the book

- Part on vehicle analyses: the virtual kingpin (a screw axis of a motion,
  decided by what is held), and the axodes as a figure.
- Part X: "report, don't guess".
- Points with distance residuals against joint coordinates with loop
  closures: two formulations of the same problem.
