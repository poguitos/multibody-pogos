# Phase 5: the validation report, version 1

- **Dates:** 9 November 2026
- **Plan:** task 5.6 (validation report); milestone M1
- **Commits:** see the index
- **Written:** at the time, on the author's laptop, by Claude working with
  the author

## Goal

The plan's task 5.6: "`docs/validation.md`, version 1." Phase 5's goal is
"evidence, independent of our own tests, that the solver is right"; the
report gathers that evidence for milestone M1, a trustworthy core.

## Starting point

The evidence existed, spread over the journal entries of Phases 1 to 5, the
plan's result notes, the integrators' report (task 4.9), the HMMWV benchmark
(5.4), the comparison scripts (5.3) and the regression harness (5.5). No
page said what is validated, against what, how well, and what is not.

## What was done

`docs/validation.md`: a summary table (area, reference, how independent the
reference is, the agreement, where), then a section per area: tree dynamics
against Pinocchio; closed mechanisms against Chrono; closed-form motions;
manufactured motions; the IFToMM benchmarks; invariants; the integrators;
statics, assembly, linearisation and kinematics; force elements, contact
and events; the HMMWV; change detection; platforms and memory safety. Then
what is not validated yet, and the commands that repeat each piece. Every
number was copied from the plan's result notes or the journal entries, not
from memory, and checked against them before the commit (three were
corrected: a range given as its lower end, a test's bound given as a
measured value, a result that needed D38's fix and did not say so).

## Decisions

None of design. The report separates the kinds of reference by their
independence (another code, mathematics, published results, the code's own
past), because that is what a reader of a validation report needs to weigh
the evidence.

## What went wrong, and how it was found

- Three statements in the first draft said more than their sources: the
  line of four masses agreed "to 4e-12" where the entry says 4e-12 to
  4e-11; the HMMWV's tyres "carry the weight to 1e-3 N" where 1e-3 N is the
  test's bound, not a measurement; the 1000 s drift result was given to
  task 4.7 alone where two methods needed D38. Found by checking each
  number against its source before the commit.

## Verification

Each figure in the report against the plan's result note or the journal
entry it comes from; the links between pages.

## Measurements

None new.

## Lessons

- A summary is where numbers drift: check each against its source, not
  against memory.

## Open ends

- Version 2 comes with the tyres (task 7.4) and the vehicle's validation
  (task 9.6), which the plan already points to this report.

## For the book

Part VII's opening: what "validated" means here, sorted by the reference's
independence; the summary table.
