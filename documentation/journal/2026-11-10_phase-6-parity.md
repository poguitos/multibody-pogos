# Phase 6: parity tests

- **Dates:** 10 November 2026
- **Plan:** task 6.3; D55
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 6.3: "Ten C++ tests reproduced through Python. *Done when the numbers
are identical.*" The package calls the same compiled library as the C++
tests, so what parity can find is what the bindings add: a conversion, a
default, a step taken another way, a callback that sees another state.

## Starting point

The package (6.1, D52), installable (6.2, D53) and tuned with the author
(D54), with 43 Python tests that check it against closed forms to
tolerances. D53 had already placed the parity tests in the suite's build,
where the C++ test executables and the module are built together with the
same flags.

## What was done

1. **How the numbers travel.** A C++ test case cannot be read from Python;
   its numbers can be written. `tests/support/parity.hpp` gives
   `parity(name, value)`, which appends "name value" lines to the file
   named by `MBD_PARITY_FILE`, the value in hexadecimal floating point (exact),
   a vector entry by entry, a matrix column by column, a complex number as
   `.re` and `.im`; without the variable it does nothing.
   `python/tests/parity.py` runs one case (checking it passed and ran
   alone), reads the file, keeps the same names from Python, and compares the
   doubles as hexadecimal. CTest gives the Python tests the folder of the
   test executables (`MBD_TEST_BIN_DIR`); without it the tests skip.
2. **The ten chosen** from the C++ cases by two conditions: everything the
   case uses is bound, and its setup can be written in Python to the same
   doubles. Out for the first reason: the four-bar's energy and the heavy
   top (they drive the kernel's solver and algorithms directly), the driven
   pendulum and the driven point (the solver's multipliers), the Radau
   oscillator (a joint force written in C++). Out for the second: the
   engine on three bushings (rotations multiplied in Eigen, which NumPy
   would multiply otherwise). The ten cover assembly, statics,
   linearisation, events, a state element and a force callback in Python,
   constraints and projection, loads, contact with friction, nonlinear
   springs, bushings, and five integrators.
3. **The C++ side:** a few `parity()` calls at the end of each case, after
   its checks, recording what the engine computed: states, report fields,
   loads, modes, statistics. Its own arithmetic on them (a speed along a
   slope, a spring's length) is left out, as Python would redo it in another
   order.
4. **The Python side** (`test_parity.py`): each case's model and steps,
   expression for expression, including the four-bar's closed-form
   configuration (cos, sin, atan2, and Eigen's norm as the root of the sum
   of squares in order) and the frames from an axis and angle
   (`Transform3.from_axis_angle` builds the quaternion as the C++ does).
5. **One binding added:** the constrained pendulum checks
   `last_projection()`, which was not bound: now `Simulator.last_projection()`
   and `ProjectionInfo`, with its repr.
6. **The first run: all 516 numbers identical**, in all ten. Nothing in a
   reproduction was changed afterwards.
7. **Checked that the comparison can fail:** one double off in the crank
   angle, or in the lag's time constant, a renamed number, a frame built from
   its rotation matrix: each fails, naming the numbers.
8. **The author's environment broke, and now says why.** The new class is
   named in the package's Python files, which an editable install reads from
   the repository, while `mbd-try` still had the module of the morning:
   `import multibody` failed with a bare `AttributeError`. The package now
   turns that into MBD-P011 ("its compiled module ... is older than its
   Python files; install the package again"), with a test, a catalogue entry
   and a troubleshooting entry. `mbd-try` reinstalled.

## Decisions

D55: parity as bit-for-bit identity with the C++ case's own numbers, run
then, in the same build; only the engine's numbers; setups copied
expression for expression; skipped against an installed package. Smaller:
the record keeps a name's numbers in order, so cases that run several times
compare run by run; the ten are chosen by what is bound, and what was
missing for one of them is bound.

## What went wrong, and how it was found

- **Nothing in the parity itself:** identical at the first run. That is the
  expected outcome for a correct binding layer; the mutations show the
  tests would have said otherwise.
- **The author's editable install stopped importing** after the binding was
  added (above). Found by trying the import in `mbd-try` before reinstalling;
  it would have met the author at the next `import multibody`.
- **The first build stopped at once.** Run as `& build.ps1 -Test *> log`,
  CMake's "-- GLOB mismatch!" (the new test file) on stderr became an error
  record in Windows PowerShell 5.1 and the script stopped. Started as its own
  process with `cmd` redirecting, it ran. A troubleshooting entry, and the
  assistant's notes, say so.
- **A slow run, not explained.** In the first suite run `test_parity.py`
  took 218 s; alone through CTest 8 s, by hand 15 s, in the next suite run
  14 s. The other Python files took their usual times in that run. The
  laptop was on battery; the cause was not found.
- **A timing misread, then corrected:** one run put the bushed pendulum's
  reference at 6.7 s from Python against 2.8 s for the whole C++ case, which
  looked like an overhead of the bindings. Alternated, twice, the two take
  the same time (3.4 against 3.5 s, 3.1 against 3.3 s).

## Verification

- `python/tests/test_parity.py`, ten tests, each against its C++ case:
  516 numbers identical (the data: names and numbers per test).
- The mutations (the data): each fails, with the differences named.
- `test_interface.py`: a compiled module that lacks a class the Python files
  name raises MBD-P011 on import, the classes left as they were.
- The suite: 534 tests pass (the new file is one CTest test); the message
  catalogue's tests include MBD-P011. In `mbd-try`, against the installed
  package: 44 passed, 10 skipped.

## Measurements

In `data/2026-11-10_phase-6-parity.md`: the ten with their counts, the
mutations, the frames' quaternions, the times (the slow run included), the
build stopped by stderr, the author's environment before and after.

## Lessons

- **Compare with the C++ test's own numbers, from the same build, run then.**
  Stored references would differ by platform and age; a tolerance would pass
  the differences parity is for, which begin at one ulp (a frame one ulp
  apart moved 48 of 54 numbers, half of them by 1024 ulps or more).
- **A setup is part of the reproduction.** The model is identical only if
  its inputs are the same doubles: copy the expression, not its meaning.
- **A change to the bindings reaches the author's environment.** Say so in
  the error, and reinstall it as part of the task.
- **Time on a laptop on battery twice before believing it.**

## Open ends

- `JointForce` written in Python, the solver's multipliers, and the kernel's
  algorithms are not bound (`docs/python.md`); C++ cases that need them
  cannot be reproduced yet.
- Linux CI builds without the interface, so the parity tests run on Windows
  only.
- The 218 s run.

## For the book

Chapter on the Python interface (6): parity as identity, not agreement; why
it is achievable (one library, the same inputs, IEEE arithmetic without
contraction) and what breaks it (an input one ulp apart); the C++ test
writing its numbers for its reproduction to read; mutations as the evidence
that a comparison can fail.
