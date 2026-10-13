# The troubleshooting guide

- **Dates:** 13 October 2026
- **Plan:** documentation track H, task H.3
- **Commits:** see the index
- **Written:** at the time

## Goal

From symptom to cause to fix: builds (compiler memory, code page,
precompiled headers), models (`validate()`), simulations (projection
failures, constraint drift, energy growth, stiff tyres at low speed, step
size) and slow runs (the timing protocol, D21); and how to use the debugging
tools. *Done when every problem met so far in the journal has its entry.*

## What was done

`docs/help/troubleshooting.md`, linked from the help index: a table of the
debugging tools and when to use each; entries for builds, models, initial
conditions and statics and linearisation, simulations, slow runs and
numerical traps, each naming the journal entry where the problem was met;
how to check a new joint, constraint or force; and a coverage table that
maps every journal entry's problems to the guide's entries.

## How it was done

Every journal entry was read for its "what went wrong" section (and, for the
entries without one, for problems mentioned elsewhere in them). Two entries
record none (the kernel, and prescribed motion), and the table says so. One
tip that no entry records (commit messages in Windows PowerShell 5.1) is in
the guide without a journal attribution, rather than credited to an entry
that does not mention it.

## Lessons

- A guide built from the journal is only as complete as the journal. The
  entries written at the time are much richer here than the reconstructed
  ones, which is the case for D22 in miniature.

## Open ends

- The guide grows with each phase (H.6): a task that meets a problem adds its
  entry and a row in the coverage table.
- The period before the review is the author's to complete; its problems
  (the error of May 2026, the restarts) are only outlined.
