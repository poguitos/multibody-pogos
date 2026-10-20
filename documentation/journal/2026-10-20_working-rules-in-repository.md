# The working rules in the repository

- **Dates:** 20 October 2026
- **Plan:** task H.7 (D42)
- **Commits:** see the index
- **Written:** at the time

## Goal

H.7: put the working rules, until now kept only in the author's local agent
memory, into the repository, as `CLAUDE.md` with an `AGENTS.md` for other
tools. *Done when the rules are in the repository and the local memory points
to them instead of repeating them.*

## Starting point

The plan's section 7 holds eight working rules. Everything else an agent
needed to work safely lived in three memory files on the author's laptop:
one compiler at a time and why, the build script and the code-page trap, the
benchmark protocol, the record kept with every task, the commit format,
merges only with the author's go-ahead, and pitfalls met before. The author
then decided to continue the work with a cloud agent, which starts from the
repository alone, on a Linux machine where the Windows build script does not
run. That made the task urgent.

## What was done

- **`CLAUDE.md`** at the root: where things are (plan, decisions, journal,
  book, docs, help, the review of external sources); where the work stands;
  what is not in the repository (the PDFs git ignores, the local copies of
  the reviewed sources); building and testing on the laptop (the script, one
  compiler) and on Linux (the CI's commands, three compile jobs), the test
  executables, the checks that run in the suite, CI, and why timings compare
  only on one machine; ten working rules (the plan's, plus the record, the
  gates and the author's go-ahead for merges, the commit format, faithful
  reporting); how the documentation is written; pitfalls met before.
- **`AGENTS.md`** pointing to it, for tools that read that name.
- **The README's layout** lists both files.
- **The local memory** now points to `CLAUDE.md` for the rules and keeps only
  the progress notes.

## What went wrong, and how it was found

- **An untested claim avoided.** The first draft gave the Linux commands as if
  they built everything; the CI file shows its Linux jobs build only
  `test_core`, `test_model` and `test_kernel`. The other executables have only
  ever been compiled with MSVC, so `CLAUDE.md` now says that a GCC error there
  is a portability fix to make, not a broken build.

## Verification

Nothing compiled changes. The file's statements were checked against the CI
workflow, the presets, the build options and the plan; the Linux commands are
the CI's.

## Open ends

- The other executables' first build with GCC, by whoever works on Linux next.
- Keep the "where the work stands" line of `CLAUDE.md` current at each task.

## For the book

- Rules written down where the work happens: the move from an assistant's
  private memory to a file in the repository, prompted by handing the work to
  another agent.
