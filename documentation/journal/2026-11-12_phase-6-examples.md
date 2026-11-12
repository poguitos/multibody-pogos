# Phase 6: examples

- **Dates:** 12 November 2026
- **Plan:** task 6.6; D58 (which amends D57)
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 6.6: "Pendulum, four-bar and quarter car as notebooks." The plan gave
no "done when"; D58 adds one: each notebook runs from top to bottom in the
suite and in CI, its numbers checked against closed forms.

## Starting point

The author had just looked at 6.5's recordings, liked them ("I like the
visuals a lot"), and asked to go on in that direction. The package had a
pendulum in `docs/python.md`, the four-bar in `python/examples/`, and the
quarter car only in C++ (`tests/test_quarter_car.cpp`), on the vehicle
layer's tyre, which is not bound. Both conda environments already had
Jupyter's kernel (it came with rerun-sdk on conda-forge); the author's had
matplotlib, the suite's had not.

## What was done

1. **Prototypes as scripts first**, each result against its closed form:
   the pendulum's period, speed and pull, the four-bar's angles, turning
   points, rest and frequency, the quarter car's statics, modes, one
   frequency of the shaker and the bump. The quarter car's road is a plate
   driven by a function written in Python: 6 s of RK4 at 1 ms took 0.10 s,
   so a sweep of twelve frequencies costs 2 s.
2. **A bound missed in the pendulum's prototype** (below): semi-implicit
   Euler's energy error was 0.073 J against the 0.060 J the modified energy
   predicts. The cause was the event that counted the passes. The notebook's
   energy runs have no event.
3. **Viewing changed for notebooks** (D58): each `replay` a recording of its
   own, under an application named after the model; the viewer started from
   the binary in Rerun's package. A test checks both; it failed on the old
   code (`'pendulum' != 'pendulum'`, the two recordings' ids).
4. **The three notebooks**, written through a generator script kept out of
   the repository (the `.ipynb` files are the source from now on), stored
   without outputs, each ending with a replay.
5. **`python/tests/test_notebooks.py`** runs their code cells in order in
   one namespace, the plots off screen, the replays to recordings in a
   temporary folder, no warning from Rerun allowed.
6. **Each notebook run once in a real Jupyter kernel** (a script in the
   session's scratch folder, through `jupyter_client`), to read every output
   and look at every figure as a PNG. Three legends were moved off the
   curves, an allowance printed to one digit (1e-05 for 9.8e-6) now prints
   two, and the word "closed form" in the printed checks became "expected",
   since one compares with a first-order term.
7. **matplotlib-base 3.11.2** into the suite's environment, after a dry run
   showed only new packages; `python/environment.yml`, the extra
   `notebooks` in `pyproject.toml`, CI.
8. **Documentation:** `docs/python.md` (the notebooks; recordings in the
   viewing section), the help's index and troubleshooting, README.

## Decisions

D58. Smaller ones: the five-line `check` is repeated in each notebook
rather than put in the package, since it belongs to the examples, not to the
engine. The quarter car keeps the C++ test's masses, springs and damper, with
a tyre radius of 0.3 m and a free length of 0.4 m. Z is up and gravity is
-Z, as ISO 8855 will have it. The body's centre of mass is raised 0.35 m only
so that its box is drawn above the wheel; with joints that allow no
rotation, no equation changes.

## What went wrong, and how it was found

- **Semi-implicit Euler above its bound.** The notebook's energy check was
  derived before it was written: a symplectic method keeps the modified
  energy E - (h/2) th' V'(th) + O(h^2) (Hairer, Lubich and Wanner, IX.1), so
  E's largest swing should be (h/2) max|th' V'| = 0.0605 J, plus terms near
  1e-3 J. The prototype measured 0.073 J. A hand-written v-first symplectic
  Euler matched the engine's steps to 2e-15, so the method was right; the
  difference was the prototype's event, which cuts the step at each pass
  through the bottom. Without it the error is 0.0604 J and the modified
  energy stays within 1.7e-4 J. With it, over 80 s, the modified energy
  falls steadily (by 0.054 J) and E's largest error grows from 0.073 J in
  the first 20 s to 0.120 J in the last. Now in troubleshooting.
- **Re-runs merged in the viewer.** Found by reading `_stream` while
  thinking about how notebooks are used (a cell is run again and again), not
  from a failure; Rerun's documentation says that data under one recording
  id is merged. The test now holds the fix.
- **An edit that did nothing.** A shell command joining two here-documents
  with `||` left `_viewer.py` unchanged; seen in `git diff` before going on.
- **A test run against the build's copy of the package.** The first run of
  the new viewer test used `build/python`, where the build had copied the
  package's files before the change. It failed, which served as the check
  that it fails on the old code. Later runs used `mbd-try`, whose editable
  install reads the source.
- **`pose_WB` is a method,** not a property: the four-bar's first run
  stopped at its energy cell.

## Verification

Each check with its result and allowance, as printed by the run in a kernel
of `mbd-try` (the suite's run, in `mbd-python`, passes the same checks; its
printed numbers were not read):

- **Pendulum.** Period 5.4e-13 s from the elliptic integral, the first pass
  1.5e-14 s from a quarter period (allowed 1e-9 s); the rate at the bottom
  1.6e-12 rad/s (1e-9) and the rod's pull 8.9e-12 N (1e-8); the periods of
  fifteen releases from 0.1 to 3.0 rad, 1.2e-12 s at most (1e-9);
  semi-implicit Euler's largest energy error 0.060439 J against (h/2)
  max|th' V'| = 0.060486 J (allowed 3e-3 J, the terms of order h^2) and its
  modified energy's change 1.7e-4 J (3e-3); the linearised frequency 2.9e-13
  rad/s from sqrt(g/L) (3.5e-8), damping 0.
- **Four-bar.** Angles over a full turn 8.9e-16 rad and rates 1.1e-15 from
  the loop's geometry (1e-9); at the two turning points the rocker's angle
  8.9e-16 and 4.5e-14 rad from the law of cosines, its rate 6.7e-16 and
  1.0e-13 (1e-9); the loop closed at rest to 3.1e-15 rad; dV/dth at rest
  2.1e-7 N m (1e-6), which is statics' stopping residual, 7.9e-7 rad/s^2,
  times the generalised mass, 0.27 kg m^2; the frequency of small swings
  9.833154624 rad/s against 9.83315498228 from Lagrange's equation, 3.6e-7
  apart (9.8e-6). The swing's total energy, 26.238656 J, changed by 4.0e-12
  J (printed, not asserted).
- **Quarter car.** Heights at rest 3.9e-14 and 5.9e-13 m (1e-7). Body
  bounce: frequency 6.4e-11 rad/s (8.7e-8), damping ratio 3.5e-11 (1e-8),
  shape 8.1e-12 (9.5e-10). Wheel hop: 7.2e-10 rad/s (7.3e-7), 2.3e-11,
  1.1e-9 (1.2e-7). Transmissibility at twelve frequencies from 0.5 to 20
  Hz: at most 2.0e-7 relative up to 10 Hz (allowed 2e-6), 1.6e-6 at 14.3 Hz
  (5.4e-6) and 5.2e-6 at 20 Hz (2.1e-5). RK4's estimate (h w)^4 / 120 fell
  short by a factor of 2.5 at 20 Hz, hence ten times it allowed. The tyre
  kept at least 10.2 mm of deflection in every shaker run. The momentum lost
  in the 55 ms in the air, 156.4695 kg m/s, matched (m_s + m_u) g dt to
  1.4e-13 (1e-10).
- **`test_viewer.py`:** two replays under one name are two recordings
  under one application; the viewer's binary is found in `rerun_cli`.
- **The suite:** 543 tests pass (542 before; the new file
  `test_notebooks.py` is one CTest test, 20.2 s), with the notebooks' four
  tests run and none skipped in `mbd-python`. In `mbd-try` (Python 3.14): 65 passed, 10
  skipped (the parity tests, as intended against an installed package).
- **Not verified:** the notebooks opened in VS Code, and the viewer started
  from a kernel there; the assistant has no display. The figures were seen
  as files.

## Measurements

In `data/2026-11-12_phase-6-examples.md`: the prototypes' timings, the
energy of semi-implicit Euler with and without the event, the shaker's
errors, the kernel runs' outputs, the environment change.

## Lessons

- **Derive the allowance before the first run.** The energy check found the
  event's effect only because the bound was written down first.
- **An example that runs as a test does not go stale,** and a notebook can be
  run as a test without Jupyter: its code cells in order.
- **Look at the figures.** Through a kernel, saved as PNG files; three
  legends covered curves.
- **Think through how a tool will be used,** not only whether it works:
  notebooks re-run cells, and the viewer's recording ids were not made for
  that.

## Open ends

- The notebooks in VS Code, and the viewer opened from their kernel, are for
  the author to try.
- Rerun's viewer inside the notebook (`rerun-notebook`, PyPI only) not
  tried.
- The C++ quarter car still uses the vehicle layer's tyre; the notebook's is
  the kernel's contact element.
- The Phase 6 gate: retrospective, how-to guides, CI green; the merge needs
  the author's go-ahead.

## For the book

Chapter on the Python interface (6): notebooks that check themselves. Worth
writing out: a symplectic integrator's modified energy used as a check, and
how an event broke it; a quarter car on a shaker, checked linearly against
its transmissibility and nonlinearly through the momentum of a car in the
air; a four-bar's frequency from Lagrange's equation with closed-form
geometry against the engine's linearisation. Figures: the period against the
amplitude, the three integrators' energy errors, the transmissibility with
the simulated points, the bump with the wheel in the air.
