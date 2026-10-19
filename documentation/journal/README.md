# The development record

This folder is the project's lab notebook. Its purpose is to keep, while the
work is fresh, everything a long account of the project will need: what was
done and in what order, why one way was chosen over the others, what went
wrong and how it was noticed, and the numbers, with how they were measured.
Git keeps *what* changed and the master plan keeps *which* tasks are done;
neither keeps the reasoning, the alternatives or the dead ends. A book or a
thesis about the project is written from this record.

## What goes where

| Record | Holds | Feeds |
|---|---|---|
| `documentation/journal/` (this folder) | One entry per task, group of tasks or session, in the order they happened | The story of the development: how, in what order, what went wrong |
| `documentation/decisions.md` | Every design decision: context, alternatives, choice, consequences | Why the program is built the way it is |
| `documentation/journal/data/` | The raw measurements behind the numbers in the entries, each with the command that produced it | Tables and figures, which can be redrawn |
| `documentation/book/` | The book's outline, kept current, and its bibliography | The structure of the final text |
| `documentation/Master_plan.md` | Tasks, their "done when" conditions and a short result | The plan and its progress |
| `docs/` | How the program works now | The technical chapters |

## Rules

1. **An entry closes every task**, in the commit that finishes the task, so the
   record always matches the code. Small tasks done together share an entry.
2. **Every phase ends with a retrospective entry**: what the phase achieved
   against its goal, what took longer than expected and why, what to do
   differently.
3. **A decision is recorded when it is taken**, in `documentation/decisions.md`,
   and the journal entry refers to it by number.
4. **A number comes with its measurement**: the command, the machine state
   that matters (for timings: which core, as `docs/performance.md`
   explains), and the raw output in `data/`.
5. **Write down what went wrong.** A defect found, a wrong hypothesis, a
   measurement that misled: these are the most useful parts of the record
   and the first to be forgotten.
6. **Say where the content comes from.** Entries written at the time say so;
   entries reconstructed later say from what (commits, the plan, session
   notes), and mark what only the author can complete.
7. **The book outline is updated** when an entry adds a topic worth a section.

Entries follow [TEMPLATE.md](TEMPLATE.md). File names start with the date the
work began, `YYYY-MM-DD_short-title.md`.

## Index

| Dates | Entry | Plan | Commits | Written |
|---|---|---|---|---|
| Dec 2025 to Sep 2026 | [Before the review](2025-12-07_before-the-review.md) | | `11bfdd3` to `859243e` | Reconstructed 5 Oct 2026; to be completed by the author |
| 2 Oct 2026 | [The review](2026-10-02_review.md) | Findings F1 to F21, decisions D1 to D8 | `859243e`, `37950f8` | Reconstructed 5 Oct 2026 |
| 2 to 3 Oct 2026 | [Phase 0: secure the work, make builds safe](2026-10-02_phase-0.md) | 0.1 to 0.7 | `d811521`, `38534c1` | Reconstructed 5 Oct 2026 |
| 3 Oct 2026 | [Phase 1: fix the core defects and prove it](2026-10-03_phase-1.md) | 1.1 to 1.9 | `d811521` to `4c59a64`, tag `v0.2.0` | Reconstructed 5 Oct 2026 |
| 3 Oct 2026 | [Phase 2: the kinematics kernel](2026-10-03_phase-2-kernel.md) | 2.1 to 2.4 | `cbab561` | Reconstructed 5 Oct 2026 |
| 4 Oct 2026 | [Phase 2: constraints and the constrained solve](2026-10-04_phase-2-constraints.md) | 2.5, 2.6 | `d33a128` | Reconstructed 5 Oct 2026 |
| 4 Oct 2026 | [Phase 2: the simulator, and the vehicles on the kernel](2026-10-04_phase-2-simulator-and-vehicles.md) | 2.7 | `24faade` | Reconstructed 5 Oct 2026 |
| 5 Oct 2026 | [Phase 2: the legacy core goes, one library, checks](2026-10-05_phase-2-library-and-checks.md) | 2.7b, 2.8, 2.9 | `860b264`, `32f2075`, `253e788` | At the time |
| 5 Oct 2026 | [Phase 2: performance and allocations](2026-10-05_phase-2-performance.md) | 2.10 | `9472d8d` | At the time |
| 5 Oct 2026 | [Phase 2 retrospective, and how the record is kept](2026-10-05_phase-2-retrospective.md) | Phase 2 gate, tag `v0.3.0` | `9472d8d`, `49b96bc` | At the time |
| 5 Oct 2026 | [Help folder and message codes](2026-10-05_help-and-message-codes.md) | H.1, H.2 | `47315ba` | At the time |
| 5 Oct 2026 | [Phase 3: prescribed motion and kinematic analysis](2026-10-05_phase-3-prescribed-motion-and-kinematics.md) | 3.2, 3.6 | `30e6f08` | At the time |
| 5 Oct 2026 | [Phase 3: the force library](2026-10-05_phase-3-force-library.md) | 3.1 | `e1e3863` | At the time |
| 5 Oct 2026 | [Phase 3: output requests](2026-10-05_phase-3-outputs.md) | 3.3 | `8243549` | At the time |
| 13 Oct 2026 | [Phase 3: initial-condition assembly](2026-10-13_phase-3-assembly.md) | 3.4, D25 | `961ca83` | At the time |
| 13 Oct 2026 | [Phase 3: static equilibrium](2026-10-13_phase-3-statics.md) | 3.5, D26 | `bbc110e` | At the time |
| 13 Oct 2026 | [Phase 3: linearisation](2026-10-13_phase-3-linearization.md) | 3.7, D27 | `bbb3864` | At the time |
| 13 Oct 2026 | [Phase 3: simple contact](2026-10-13_phase-3-contact.md) | 3.8, D28 | `5463d17` | At the time |
| 13 Oct 2026 | [Phase 3: events](2026-10-13_phase-3-events.md) | 3.9, D29 | `536cd2d` | At the time |
| 13 Oct 2026 | [Debugging aids: describe, state dump, step trace](2026-10-13_debugging-aids.md) | H.4 | `598e517` | At the time |
| 13 Oct 2026 | [The troubleshooting guide](2026-10-13_troubleshooting-guide.md) | H.3 | `ba4ece4` | At the time |
| 13 Oct 2026 | [The API reference in CI](2026-10-13_api-reference.md) | H.5 | `5a2e38f`, `9be974a`, `f811714` | At the time |
| 13 Oct 2026 | [Phase 3 retrospective](2026-10-13_phase-3-retrospective.md) | Phase 3 gate, tag `v0.4.0` | `49b96bc` to `4a3a757` | At the time |
| 14 Oct 2026 | [Phase 4: the integrator interface and state elements](2026-10-14_phase-4-state-elements.md) | 4.1, D30 | `aa8876c` | At the time |
| 14 Oct 2026 | [Phase 4: the Dormand-Prince pair](2026-10-14_phase-4-dormand-prince.md) | 4.2, D31 | `b70e614` | At the time |
| 14 Oct 2026 | [Phase 4: generalised-alpha](2026-10-14_phase-4-generalized-alpha.md) | 4.3, D32 | `d8a2c0c` | At the time |
| 14-16 Oct 2026 | [Phase 4: SDIRK4, its order on stiff bushings, and Radau IIA](2026-10-16_phase-4-sdirk4-radau.md) | 4.4, D33, D34 | `052f2d9` | At the time; 14 Oct reconstructed |
| 16 Oct 2026 | [Phase 4: the fixed-cost step for real time](2026-10-16_phase-4-linearly-implicit-euler.md) | 4.5, D35 | `b74a2c5` | At the time |
| 17 Oct 2026 | [Phase 4: multirate RK4 for the fast states](2026-10-17_phase-4-multirate.md) | 4.6, D36 | `08099dd` | At the time |
| 17 Oct 2026 | [Phase 4: constraints per integrator](2026-10-17_phase-4-constraints-per-integrator.md) | 4.7, D37 | `8ed4c6c` | At the time |
| 18 Oct 2026 | [Phase 4: the inertia's velocity products implicit in the Euler steps](2026-10-18_phase-4-implicit-inertia.md) | 4.7's open end, D38 | `576df64` | At the time |
| 18-19 Oct 2026 | [Phase 4: the mass matrix factorized along the tree](2026-10-19_phase-4-sparse-solve.md) | 4.8, D39 | `953f48b` | At the time |
| 19 Oct 2026 | [Phase 4: the integrators verified](2026-10-19_phase-4-integrators-verified.md) | 4.9, D40, D41 | see git log | At the time |
