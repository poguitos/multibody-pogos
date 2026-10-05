# Phase 2 retrospective, and how the record is kept

- **Dates:** 3 to 5 October 2026 (the phase); 5 October (this entry)
- **Plan:** Phase 2 gate (2.11); decisions D22, D23; documentation track H
- **Commits:** `cbab561` to `9472d8d`; tag `v0.3.0` on `main`
- **Written:** at the time

## What the phase achieved, against its goal

Goal: one kinematics kernel and a compiled library. All eleven tasks are done:

- The kernel describes each joint once and every algorithm reads that
  description, which removes the cause of F1 to F4 (D1). Quaternion joints
  (D2), Model and Data, the recursive algorithms, five constraint primitives
  and their closures, a constrained solve that handles redundancy, a
  simulator, drivers.
- Everything runs on it: the vehicle layer, suspension kinematics, the
  analyses. The legacy core is gone (D17).
- One compiled library; a full rebuild in 7.4 minutes at one compiler; one
  edited source file recompiles one file (D18).
- Checks in every build and a `validate()` (D19).
- The detailed sedan steps in 124 us and allocates nothing (D20), against 0.84
  to 1.43 ms at the review (F15), measured by a protocol that makes the number
  trustworthy (D21).
- 366 tests; CI green on `main` with sanitizers and the allocation check.

## What took longer than expected, and why

- Task 2.7 grew into 2.7b: porting the vehicle layer was a task of its own,
  not part of "remove the legacy path".
- Measurements misled twice: the build-time model (task 2.8) and the
  benchmark (task 2.10). Both times, measuring how the measurement works came
  first, and saved a wrong change.
- Each full build took minutes at one compiler, so every source was checked
  against the APIs it uses before building, and work that could not affect a
  running build (documents, staged copies, scripts) was done while it ran.

## What to do differently

- Keep the old version buildable for comparisons from the start of a phase,
  not only when a number looks wrong.
- Write the journal entry with the task, not at the end of the phase: the
  entries for 2.1 to 2.7 had to be reconstructed.

## How the record is kept from now on (D22)

The author plans a long account of the project, a thesis or a book, and asked
for the record it needs. Git keeps what changed and the plan keeps which tasks
are done, but neither keeps the reasoning, the alternatives, the dead ends or
how a number was measured; the conversations that hold them are temporary and
outside the repository. So the record now lives in the repository and is
written as the work happens:

- `documentation/journal/`: one entry per task, from a template, committed with
  the task; a retrospective per phase.
- `documentation/decisions.md`: each decision when it is taken.
- `documentation/journal/data/`: the raw data and commands behind the numbers.
- `documentation/book/`: the outline, mapped to its sources, and the
  bibliography.

The entries before this one were reconstructed on this day from git, the plan
and the session record, and say so. The period before the review (December
2025 to September 2026) is known mostly to the author; its entry lists the
questions to answer.

## Help documentation (D23)

The author also asked for the help documentation every program has, to use
the program and to debug it. Leaving it to Phase 15, as the plan did, would
mean documenting hundreds of messages and options after the fact. A
documentation track H now runs alongside the phases: a help folder, a code on
every message with a catalogue checked by a test, a troubleshooting guide, a
model description and step trace in the program, an API reference built in
CI, and guides that grow with each phase, their examples run as tests.
Message codes come first, before Phase 3 adds more messages.

## Open ends

- Phase 3 begins on branch `phase-3`. Tasks 3.2 (prescribed motion) and 3.6
  (kinematic analysis along a prescribed motion) are largely done by the
  drivers of task 2.7 (D15) and are to be checked against their "done when"
  conditions.
- The 250 us target holds in the machine's normal state only (performance
  entry).

## For the book

- The phase is the core of Part IV of the outline.
- The two misleading measurements are a theme for the engineering-practice
  part: "measure the measurement".
