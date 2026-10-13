# The API reference in CI

- **Dates:** 13 October 2026
- **Plan:** documentation track H, task H.5
- **Commits:** see the index
- **Written:** at the time

## Goal

Doxygen from the header comments, a CMake target `docs`, built in CI. *Done
when CI builds it with no warnings for the public headers.*

## What was done

- A `docs` target in the top-level `CMakeLists.txt` through CMake's
  `doxygen_add_docs`, defined only when Doxygen is installed: the headers
  under `include/mbd`, HTML into `<build>/docs/html`, undocumented members
  listed (`EXTRACT_ALL`) rather than warned about, and every warning an error
  (`WARN_AS_ERROR = FAIL_ON_WARNINGS`).
- A CI job `docs` on Ubuntu that installs Doxygen and builds the target.

## Decisions

- Undocumented members are not warnings. The headers document their
  classes and functions in prose, often in a block comment at the top of the
  file rather than on each member; demanding a comment on every member would
  produce boilerplate. What the reference must not contain is a malformed
  comment: a `\param` naming no parameter, a stray tag, a broken link. Those
  are warnings, and warnings fail the build.
- Doxygen is not installed on the development laptop, and installing
  software there is the author's decision, so the target is verified in CI.
  Before the first push the headers were searched for what Doxygen misreads
  (`<` followed by a letter, read as an HTML tag; backslash and `@` commands;
  `#name`, read as a link request) and every `\param` was checked against its
  function's parameters.

## What went wrong, and how it was found

Nothing failed, but the first success proved less than it seemed. The CI
step that builds the reference finished within a second, and its log needs
authentication to read. A missing target would have failed the step, so
Doxygen had run; but whether it wrote anything could not be seen. A second
step now checks that the HTML index and the class list exist and list
`ConstraintSolver`, so the job's public step results show the reference was
built (run 38449183633: every step green), and Doxygen's
`FAIL_ON_WARNINGS` shows that it was built without a warning.

## Verification

- CI run 38448929154 (`5a2e38f`): the `docs` job green; Doxygen ran with
  warnings as errors over every header under `include/mbd`.
- CI run 38449183633 (`9be974a`): the same, and the check of the written
  HTML green.
- Locally the target is absent (no Doxygen), and configuring is unchanged.

## Open ends

- How-to pages in the reference (`docs/help`) as Doxygen pages, and examples
  compiled as tests (H.6).
- Publishing the HTML (GitHub Pages) is the author's decision.
