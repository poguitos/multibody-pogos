# Help: the API reference

- **Dates:** 5 October 2026
- **Plan:** task H.5; decision D23
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session

## Goal

Doxygen from the header comments, a CMake target `docs`, built in CI. *Done
when CI builds it with no warnings for the public headers.*

## What was done

1. `docs/api/Doxyfile.in` and `docs/api/mainpage.md`; CMake configures the
   Doxyfile into the build directory and defines the target `docs` when it
   finds Doxygen.
2. What "no warnings" means had to be decided: with Doxygen's defaults,
   every undocumented member is a warning, and many small accessors and
   private-looking members carry no comment. The reference lists every
   declaration (`EXTRACT_ALL`), and the warnings that count are errors in the
   comments that exist (`WARN_IF_DOC_ERROR`: a misnamed parameter, a broken
   command, bad markup), which fail the build (`FAIL_ON_WARNINGS`). Requiring
   a comment on every member would be a separate, larger task; the plan's
   15.3 can take it up.
3. A CI job, `api-docs`, installs Doxygen, builds the target and keeps the
   HTML as an artifact.
4. Doxygen was installed in the container (1.9.8, the version Ubuntu 24.04
   runners get), so the job was rehearsed before pushing.

## What went wrong, and how it was found

Nothing failed: the headers' comments, written as `///` prose with a few
`\param` commands, had no errors. A silent pass could also mean the check
checks nothing, so it was tested: a scratch header with a misnamed `\param`
and an unfinished `\retval`, run through the same configuration, gave two
errors and exit code 1.

## Verification

| Check | Result |
|---|---|
| `cmake --build build-linux --target docs` | no warnings; 702 HTML files, 154 class pages |
| The same configuration on a deliberately broken comment | 2 errors, exit code 1 |
| The CI job's commands in a fresh build directory (`BUILD_TESTING=OFF`, no Python) | no warnings |

## Open ends

- A comment on every public member, with warnings for the undocumented ones
  (task 15.3).
- Publishing the HTML (GitHub Pages), if wanted.

## For the book

- What "no warnings" should mean for generated documentation, and testing
  that a check can fail.
