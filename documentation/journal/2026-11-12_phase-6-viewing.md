# Phase 6: viewing

- **Dates:** 12 November 2026
- **Plan:** task 6.5; D57
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 6.5: "Three-dimensional playback of recorded poses, and channel plots.
*Done when the four-bar and the car can be replayed.*"

## Starting point

D8 named Rerun or Meshcat; the author chose Rerun. The model has no shapes:
bodies have an inertia, joints frames, elements points. Runs come out of
Python as arrays or tables (6.4). The project's car in the kernel is the
HMMWV of task 5.4; its tyres are forces written in C++ against its road,
which a model file cannot hold.

## What was done

1. **Rerun 0.43.0** into the suite's environment and the author's, after
   dry runs that changed nothing installed; its interface read from the
   installed package rather than remembered, since it changes between
   versions (no `rerun.dataframe` in this one).
2. **Bindings** for what a view needs: the elements' anchors, markers and
   contact geometry, the constraints' markers, axes and time functions, a
   set's parts.
3. **`python/multibody/_viewer.py`:** `_scene` computes what is drawn (each
   body's pose at each frame from the engine's kinematics, the box of its
   inertia, lines to its joints, the elements' lines, the points, the up
   axis from gravity), and `replay` sends it to Rerun with a layout: the view
   on the left, q, v and given channels plotted on the right, one time line.
   `plot` sends a table's channels alone.
4. **The car:** `mbd_hmmwv record <folder>` runs it in C++ and writes its
   model file without the tyres and its coordinates every 10 ms;
   `python/examples/replay_hmmwv.py` plays them back. A C++ test checks
   that the car, without its tyres, survives a round trip through a model
   file. The four-bar: `python/examples/replay_four_bar.py`.
5. **Tests** (`test_viewer.py`, seven): poses against a pendulum's closed
   form, a spring's line, the inertia box against a box's sizes, frames at
   the rate asked, the motion's three forms giving the same scene, the
   four-bar and a toy car written to recordings, the errors.
6. **Both cars recorded and replayed** to recording files (the data).

## Decisions

D57. Smaller ones: 60 frames per second of motion by default, the plots at
every sample; a body without inertia drawn as a point; elements without
points left out of the view rather than refused.

## What went wrong, and how it was found

- **The author's environment had NumPy on Intel's MKL** since it was made
  for the interface's tuning, without the pin to OpenBLAS that the suite's
  `environment.yml` carries for this very reason (the MKL build stops
  Python on its first linear algebra on this laptop, code 0xc06d007f, as in
  the crosscheck environment in task 5.3). Nothing had called
  `numpy.linalg` there until the viewer's inertia box. Found by running the
  package's tests in `mbd-try` after the reinstall; switched to OpenBLAS,
  and the documented command for such an environment now carries the pin.
- **Entity paths half escaped:** the points' paths put an index and a space
  before the escaped label, and Rerun warned, on the process's stderr, for
  each. Found by reading the first replay's output; the tests passed, since
  the warnings bypass Python. The path is escaped whole, and the recordings'
  test now requires no warning on the process's stderr, shown to fail with
  the old path.
- **A command's output read too quickly:** the first run in `mbd-try` was
  read as a test run without its summary line; it was a crash.

## Verification

- `test_viewer.py`: a pendulum's bob at (L sin q, -L cos q, 0) and turned by
  q about Z, to 1e-15; a spring's line from (0, 0, -1) to (0, 0, q) exactly;
  a box's half sizes recovered from its moments to 1e-12; 51 frames at 50
  per second over 1 s of 1 ms samples; a run, its two-level table and a
  one-level CSV giving the same poses; the four-bar and a toy car (a free
  chassis, four wheels on sliders with springs and contact spheres) written
  to recordings, the car's bodies, lines and points counted, no warning from
  Rerun; MBD-P013 and P015.
- `test_model_files.cpp`: the HMMWV with joints and with bushings, its tyres
  refused by name, then without them, writes the same text after a round
  trip and has the same accelerations bit for bit.
- The suite: 542 tests pass. In `mbd-try`: 60 passed, 10 skipped (parity).
- Both cars and the four-bar replayed to recording files (the data). Not
  verified: how the recordings look; the assistant has no display.

## Measurements

In `data/2026-11-12_phase-6-viewing.md`: the installs, the suite's times, the
two cars' records and recordings, the warning, the author's environment
before and after.

## Lessons

- **Make every environment from the pinned file, or with its pins.** The one
  made by hand lacked the one pin that mattered.
- **Read a tool's warnings even when the tests pass**, and make them part of
  the test when they can be caught (here, at the process's stderr).
- **Separate what is computed from what is drawn,** so that the computation
  is tested against closed forms and the drawing library is not mocked.

## Open ends

- Shapes: a model file has none, so bodies are their inertia's boxes; the
  vehicle's geometry comes with Phases 7 and 8.
- Forces written in Python (`UserForce`) are not drawn; nor are contact
  planes.
- The recordings are not read back in the tests (no reader in Rerun 0.43's
  package was found).
- How the views look is for the author to see (`rerun <file>`, or the
  examples without a path).

## For the book

Chapter on the Python interface (6): showing a model that has no shapes (an
inertia's box, a joint's lines); a viewer tested through what it is given to
draw; an environment's hidden library as a defect found only when new code
first used it.
