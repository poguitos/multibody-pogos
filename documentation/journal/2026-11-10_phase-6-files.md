# Phase 6: files

- **Dates:** 10 November 2026
- **Plan:** task 6.4; D56
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 6.4: "Models to and from JSON. Results to pandas, Parquet and CSV.
*Done when a model survives a round trip unchanged.*"

## Starting point

D8 had chosen JSON for model files; task 8.2 will extend the format to the
vehicle with a schema and units, an unknown key an error. A model could be
built in C++ and Python but not read back out: most elements and every
constraint kept their settings private, and a curve its points (the open
end of D54). The crosscheck programs wrote models as JSON by hand, for their
own scripts. A run's results came out of Python as a dict of arrays.

## What was done

1. **Two choices put to the author.** Where files are read and written: in
   the C++ library, or in Python only; the author chose C++, one reader for
   every program. How a run's table names its columns: by vector and index,
   by body and coordinate, or both as two levels; the author chose the two
   levels.
2. **Read accessors** in the headers, for everything the format writes; a
   constraint set now owns its name.
3. **The format and its code**, `mbd/io/model_file.hpp` and
   `src/io/model_file.cpp`, with nlohmann/json 3.12.0 (pinned by its
   archive's hash, used in that one file). Writing: numbers in shortest
   round-trip form, "inf" for infinite values, closures as the calls that
   make them when exactly that, a printer that keeps short objects on a
   line. Reading: every object's keys checked, every error named by its
   place, values through the constructors, transforms set member by member.
   What a file cannot hold refused by name. Codes in a new area, I.
4. **Python:** the four functions bound; a curve's points readable and its
   repr the call that makes it; `dataframe`, `write_results`, `read_results`
   and `coordinate_names` in `python/multibody/_results.py`, pandas imported
   only there.
5. **Tests:** five in C++ (`tests/io/test_model_files.cpp`), nine in Python
   (`test_files.py`); the environments `mbd-python` and `mbd-try` gained
   pandas and pyarrow, CI installs them, and the package has the extra
   `results`.
6. **`docs/model-files.md`** describes version 1, its example the text the
   format's test expects.

## Decisions

D56. Smaller ones: written files are complete, hand-written ones may leave
out what has a default; joints are named by their kind, everything else by
its class or function; the CSV reader asks pandas for exact parsing only
before pandas 3, which parses exactly by default and deprecates the option.

## What went wrong, and how it was found

- **Two rules of the message catalogue's test, met by its own failure.** The
  reader built its codes as `"MBD-" + code`, so the test found codes I003
  to I007 documented and used nowhere; and two statements rethrew a
  constructor's error with its place in the file, keeping the error's code
  inside its message, which the test reads as throws without a code. The
  codes are now written in full where each error is raised; the test
  accepts, and its comment says so, a caught error's message rethrown
  (`.what()`) and a code a helper is given by its callers.
- **A test that did not compile:** a `Simulator` takes a `System&`, and the
  test's systems were const. Found by the first build (447 s, stopped there).
- **A deprecation:** pandas 3.1.1 warns that `float_precision` is deprecated.
  Found by reading the tests' two warnings; the CSV round trip stays exact
  without it.
- **Transform3 normalises its quaternion**, which could move a written
  quaternion's last bit on reading. Found when writing the reader, before
  any test: the reader sets the members.
- **A dropped connection** while conda downloaded a library of pyarrow's;
  the retry worked.

## Verification

- `tests/io/test_model_files.cpp`: a system with every kind of part the
  format holds writes the same text after a round trip, describes itself
  the same, and gives the same accelerations bit for bit at a generic
  state; a four-bar on a spring saved and loaded runs 1 s bit for bit as the
  original; a pendulum's text is the documented one, and the same model
  written by hand with the defaults left out writes the same text; a user
  force, a time function given as code and an unknown element are refused
  by name; sixteen errors in files each name their place (or the path, for
  a file that cannot be read), and a constructor's own error keeps its code.
- `python/tests/test_files.py`: a model built in Python, saved through a
  `pathlib.Path`, runs 0.5 s bit for bit after loading; errors and a
  refused user force with their codes; a curve's points; a run's table, its
  column names for one coordinate, a free joint, no name and a shared name,
  its values the run's to the bit; tables through CSV and Parquet to the
  bit; a recorder's table and its own CSV; MBD-P012 to P014.
- What the round trip's comparison sees (the data): a setting dropped by a
  reader, or a number written one ulp off, changes the text written again
  and 1,290 of 1,503 coordinates of a 0.5 s run.
- The suite: 540 tests pass (five model-file tests and one Python file
  added). In `mbd-try`: 53 passed, 10 skipped (parity).

## Measurements

In `data/2026-11-10_phase-6-files.md`: the dependency's hash, the
environments, the four builds, the round trip and its mutations, the times.
On mains power the parity tests took 5.9 s (8 to 15 s in energy-saving mode
in task 6.3).

## Lessons

- **A test of the record catches the record's own code.** The catalogue's
  test failed on the first version of the reader, rightly for one rule and,
  for the other, for a form of throw it did not know; both now say what
  they accept.
- **Ask where a format lives before writing it.** The choice between C++ and
  Python decided the dependency, the error handling and who can read files.
- **Look for a constructor that changes what it is given** (here, the
  normalised quaternion) before claiming a round trip exact.

## Open ends

- The vehicle layer's elements, time functions given as code and forces
  written as code are not in the format; task 8.2 extends it with a schema
  and units.
- `read_results` relies on pandas 3's exact default; the oldest pandas the
  extra allows (2.2.2) is not tested.
- The tables of a run name state elements' states by number only.

## For the book

Chapter on the Python interface (6), or on files: a format that is exact
and strict (shortest round-trip numbers, every key checked, every error
placed), and the round trip as its test, with mutations showing the test
sees a dropped setting and a one-ulp number.
