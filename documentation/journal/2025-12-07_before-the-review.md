# Before the review: December 2025 to September 2026

- **Dates:** 7 December 2025 to 30 September 2026
- **Plan:** precedes `documentation/Master_plan.md`; the earlier plans are
  `documentation/Development_schedule.txt` and `documentation/Multibody plan.pdf`
- **Commits:** `11bfdd3` to `3a9673e`, then the snapshot `859243e`
- **Written:** reconstructed on 5 October 2026 from the git history and the
  review. Only the author knows most of this period: the sections marked
  *Author* are questions to answer.

## Goal

A multibody solver for vehicle dynamics, of the kind ADAMS provides, with a
vehicle builder, lap-time optimisation, aerodynamics from CFD, real-time
simulation and a Python and AI layer on top (`documentation/Master_plan.md`,
section 2, keeps these as the project's eight capabilities).

*Author:* where did the idea come from, and why build a solver instead of
using an existing one?

## What the history shows

| Date | Commit | Event |
|---|---|---|
| 7 Dec 2025 | `11bfdd3`, `84b4bbd`, `37138aa` | Project set up: CMake, Eigen, Catch2, pybind11; CI on Windows |
| 8 Dec 2025 | `6c572a9` to `e5c5186` | Core types, logging, rigid-body inertia and state, single-body dynamics with semi-implicit Euler |
| 10 to 11 Dec 2025 | `e44d575` to `e992499` | CI trimmed ("not run 900+ tests"); `Transform3` for rotations and translations |
| 14 Dec 2025 | `8870ffb` to `00e28d2` | Gyroscopic term in the rotational acceleration; a small-angle approximation corrected; a multibody system class, constraints, a constraint solver with a pendulum test, and constraint stabilisation |
| 2 Mar 2026 | `7eae101` | "Updated the entire code for the new Claude proposal": the code base was rewritten along a design proposed by an AI assistant |
| 6 Apr 2026 | `1a27413` | "Bicycle model working" |
| 13 May 2026 | `3a9673e` | "Error has been detected in the solver, and everything will be revised" |
| Jun to Jul 2026 | (uncommitted) | Solver fixes: rotation-vector canonicalisation for free and spherical joints, post-step projection onto the constraints, angular coupling in the child-side joint frame, constraint evaluation shared between dynamics and kinematics |
| 2 Oct 2026 | `859243e` | All work since 13 May committed as one snapshot: the restructured tree and the June to July fixes. 464 of 466 tests passed |

## What went wrong

On 13 May 2026 an error was found in the solver, and the project stopped to
revise everything. The June and July fixes treated three of its symptoms. The
review of 2 October showed why more kept appearing: the same joint
kinematics was written out by hand in several routines, and the copies
disagreed (see the review entry, findings F1 to F4).

*Author:* what was the error noticed in May, and how was it noticed? What was
tried in June and July, and what did each fix change?

Five months of work were not committed (finding F19), so this period's
intermediate states are lost; and two all-core builds restarted the PC in
mid-2026 (the reason the build now runs one compiler at a time, decision D9).

*Author:* when did the restarts happen, and what was being built?

## For the book

- Chapter 1 (motivation and goals) and chapter 2 (the first version) draw on
  this entry; the author's answers to the questions above are its main source.
- The rewrite of March 2026 and the error of May 2026 are the turning points
  of the early history: the first is the moment the design came from an AI
  proposal, the second the moment its reliability came into question, which
  the review and the whole present plan answer.
