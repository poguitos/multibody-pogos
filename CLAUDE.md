# Working on this repository

Read by every Claude Code session, on the author's laptop or in the cloud.
The working rules live here and in the plan, not in any one session's memory:
a cloud session starts from a fresh clone and sees only what is committed
(decision D25).

## Where things are

- The plan and its progress: `documentation/Master_plan.md`. Tick a task's box
  and write its *Result* in the commit that finishes it.
- The development record (D22): `documentation/journal/` (one entry per task,
  from `TEMPLATE.md`; a retrospective per phase), `documentation/decisions.md`,
  raw measurements in `documentation/journal/data/`, the book's outline in
  `documentation/book/`.
- Help (D23): `docs/help/`. Every error and warning starts with a code
  (`MBD-K031`) documented in `docs/help/messages.md`; `test_message_catalogue`
  fails otherwise.
- How the engine works: `docs/kernel.md`, `docs/conventions.md`,
  `docs/performance.md`.

## Rules for every task

1. **Close the task in one commit** with: the code, its tests, the plan's
   result, the journal entry (and its line in the journal index), the help
   pages it affects, and any decision taken (`decisions.md`). The record
   always matches the code.
2. **Tests assert derived values**; a tolerance comes with its derivation in a
   comment (working rule 3). New joints, constraints and forces pass the
   invariant tests first (rule 2). One place per formula (rule 4).
3. **Write down what went wrong**, how it was noticed, and the numbers with
   the command that produced them.
4. **Say where an entry was written**: "at the time", and in which kind of
   session (laptop or cloud), because the machine matters for timings and for
   which compiler ran the tests.

## Branches

- `main` stays green and moves only at phase gates, by the author, with a tag.
- Work goes on a branch. On the laptop: `phase-N`. In a cloud session: the
  branch the session names (`claude/...`), started from the latest work
  branch, so it carries everything before it. Push at the end of every task.
- A journal entry's *Commits* field cannot name its own commit: write "this
  commit" and let the index give the hash in the next commit.

## Building

- Laptop (Windows, MSVC): `scripts\build.ps1 -Test`, one compiler at a time
  (D9). Never raise `MBD_COMPILE_JOBS` there without the author.
- Cloud or Linux (GCC): no pool limit is needed for safety, but keep it
  modest:

  ```sh
  cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DMBD_COMPILE_JOBS=3 -DMBD_BUILD_PYTHON=OFF
  cmake --build build-linux
  ctest --test-dir build-linux --output-on-failure -j3
  ```

- CI (`.github/workflows/ci.yml`) runs Windows/MSVC, Linux with sanitizers,
  a no-allocation check and the API reference (Doxygen; any comment error
  fails it; `--target docs` locally) on every push. A cloud session builds
  with GCC only, so CI is the first MSVC build of its work: check it after
  pushing.
- **Timings** are comparable only on one machine in one session (D21). Numbers
  from a cloud container are not compared with the laptop's; say which
  machine produced them.
