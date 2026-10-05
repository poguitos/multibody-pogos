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

## Where the work happens

Until 5 October 2026 every session ran on the author's laptop (Windows,
MSVC), on the branches `core-revision`, `phase-2` and `phase-3`. From then on
work is also done in cloud sessions (decision D25): each starts from a fresh
clone, works on a branch of its own (`claude/...`) started from the latest
work branch, builds and tests with GCC on Linux, and pushes at the end of
every task; CI then builds it with MSVC as well. The rules a session needs are
in [`CLAUDE.md`](../../CLAUDE.md) at the repository root, which every session
reads, so none of them depends on a session's memory. Entries say which kind
of session wrote them, and timings which machine took them.

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
| 5 Oct 2026 | [Phase 3: output requests](2026-10-05_phase-3-outputs.md) | 3.3 | `8243549` | At the time, on the laptop |
