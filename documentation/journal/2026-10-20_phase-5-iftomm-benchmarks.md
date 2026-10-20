# Phase 5: the IFToMM benchmarks, singular positions, and contradictory constraints named

- **Dates:** 20 October 2026
- **Plan:** task 5.2 (IFToMM benchmarks, with the D42 addition: conflicting
  equations named); decisions D42, D44
- **Commits:** see the index
- **Written:** at the time, by a cloud agent (Claude) working from the
  repository alone, on a Linux machine; the Windows build was left to CI.
  First committed with 5.2 open (`cc76a14`); completed the same day when the
  author supplied the library's results for A04 and A05 (the section
  "Closing the task" and the corrections marked there)

## Goal

The plan's task 5.2: "Simple pendulum, N-four-bar mechanism, Andrew's
mechanism, Bricard mechanism (redundant constraints), flyball governor
(stiff). When a mechanism cannot be closed because its constraints
contradict each other, the message names the conflicting equations, not only
the residual (D42). Done when each meets its published reference."

## Starting point

The kernel had passed its analytic cases and manufactured trajectories
(task 5.1). No IFToMM problem had been built. A projection that failed said
only its residual (MBD-K051), and so did a failed assembly (MBD-K062).

## What was done

1. **The problem statements.** The IFToMM library's site
   (www.iftomm-multibody.org) and its older address (lim.ii.udc.es) did not
   resolve from this machine; the container's proxy refused them too. The
   five problems' parameters were read from a reimplementation of the library
   for OpenSim (Tagliapietra et al. 2015, MBS-BOS, GPL, cloned from GitHub),
   whose result folders are empty, and Andrews' mechanism, with its reference
   solution at 0.03 s, from the Test Set for IVP Solvers as the R package
   deTestSet carries it (cloned from CRAN's mirror on GitHub). Both were read
   as data; nothing was copied: each mechanism is built in
   `tests/kernel/iftomm.hpp` from its geometry.
2. **References that do not need the library's files.** The pendulum has a
   closed form in Jacobi's functions. The N-four-bar of identical bars moves
   as one parallelogram, so it reduces to a pendulum started upside down with
   spin, which turns over and over: again Jacobi's amplitude. Jacobi's
   functions moved from the analytic tests to `tests/support/elliptic.hpp`,
   with the amplitude added (one place per formula).
3. **The pendulum** (A01) followed its closed form at once: 1.1e-9 m over 10 s
   on a revolute joint, 1.4e-8 m as a free body on a distance constraint.
4. **The N-four-bar** (A02) failed: see below. Traced to the constraint
   tolerance; with the constraints held to 1e-14 it passes 19 singular
   positions in 10 s on its closed form (2e-8 rad).
5. **Andrews' mechanism** (A03) was 68 to 118 rad off at first: a sign
   (below). Then within 1.4e-9 rad of the published reference, a pure time
   shift of 1.0e-12 s.
6. **Bricard's mechanism** (A04): the orthogonal six-revolute loop along the
   edges of a cube. It moves with one degree of freedom although Grubler's
   count gives none (one redundant equation, rank 4 of 5). Its joint angles
   showed a symmetry (joints 1, 3, 5 equal; 2, 4 equal); an independent
   closure solve in Python confirmed the simulated pairs, and sympy reduced
   the six closure equations, in half-angle tangents, to a factor common to
   all: u w^2 + (u + 1)^2 w + u = 0. From it, the dynamics reduced to one
   coordinate, written in the test from the geometry alone and integrated by
   its own RK4, is a reference for the whole motion: the simulation follows it
   within 2e-9 rad while swinging from 0 to pi and back.
7. **The flyball governor** (A05): spindle, sliding collar, two rods with 5
   kg balls, spring-dampers of 8e5 N/m and 4e4 N s/m. Its motion has no
   closed form and does not settle within a test's span: the collar swings
   between 0.50 and 0.744 m about once a second, and the dampers, which act
   on the springs' length alone, take only 3 mJ in 10 s. (This entry first
   said "a slow swing near 40 s": wrong, see "What went wrong".) Checked
   against what physics keeps: the angular momentum about the spindle (to
   1.1e-10 relative) and the energy balance with the dampers' work,
   integrated as a state element (1e-10 of the energy); Dormand-Prince within
   1.4e-9 of Radau IIA after 10 s; generalised-alpha converging to it at
   second order.
8. **Contradictory constraints named** (D44): `src/kernel/conflicts.cpp`, a
   deletion filter on the linearised constraints at the least-squares
   nearest positions; in MBD-K062 (and `AssemblyReport::conflicting_equations`)
   and MBD-K051.
9. **Four messages** (MBD-K047, K051, K054, K055) printed small numbers as
   "0.000000"; they now use scientific notation.
10. Documentation: the message catalogue (K051, K055, K062), the
    troubleshooting guide (singular positions, the projection's report, the
    header edited during a build, and two numerical traps from task 5.1 that
    its coverage table had missed), the kernel's documentation (constraints
    per integrator, assembly, tests).

## Decisions

D44: references from closed forms, the test set and physics where the
library's files could not be had; the solver left as it is, and singular
positions passed by holding the constraints near rounding; the conflicting
equations by a deletion filter at the least-squares point; small numbers in
messages in scientific notation.

## What went wrong, and how it was found

- **The N-four-bar stopped at a singular position, then the cause was not
  the obvious one.** Dormand-Prince stopped at t = 4.56 s (MBD-K055, at the
  fifth singular pass); the first pass had been clean (error 3e-11), the
  second added 1.4e-7 rad, which the slow passes over the unstable upright
  position then grew. The first hypothesis was the solver's rank cut: near
  the singular position the pivoted factorization drops nearly dependent
  equations, and a scan of the accelerations along the exact motion
  confirmed that it does (rank 8 to 4 within |theta| < 5e-6) and that the
  accelerations are then wrong (0.75 rad/s^2 for four loops, even exactly at
  the singular position, where the dropped equations' right-hand sides carry
  the curvature that keeps the bars parallel). A cut of 1e-13 narrowed that
  window fiftyfold and kept the whole kernel suite passing, but the runs got
  worse: the failure came at the second pass. Logging every evaluation inside
  the failing step showed the coupler, which should not turn, turning faster
  and faster as theta fell: 1.7e-4 rad/s at 7.5e-5 rad. That is what an
  imperfect parallelogram does, its coupler tilted by delta / (L theta); the
  step trace gave delta, a residual of 2e-13 m left by every step, below the
  projection's tolerance of 1e-10 and so never removed. With the constraints
  held to 1e-14 every run passed; the rank cut went back to 1e-10, which was
  then measured better than 1e-13 (2e-8 rad against 5e-4).
- **Andrews' mechanism 100 rad off, with the right accelerations.** The
  accelerations at the start matched the test set's to ten digits, yet every
  loop was open by the same (-0.031, 0.063) m. A common offset could only be
  the shared point, E on body 3: placed at (0, +ss) instead of (0, -ss). The
  accelerations had agreed because bodies 3 to 7 are at rest at the start.
- **Andrews' remaining 1.4e-9 rad.** Two methods (Radau IIA at 1e-10,
  Dormand-Prince at 1e-12) agreed with each other to 6e-13 and with the
  reference only to 1.4e-9, so the gap was not this side's integration
  error. Dividing each coordinate's error by its velocity gave -1.005e-12 s
  for all seven, to three digits: our motion is the published one reached
  1e-12 s later, as an integration's phase error would make it. This is the
  likely explanation, not a proven one.
- **The flyball governor's rods first pointed inwards** (a sign of the hinge
  axes); printing every body's position before the first run showed the
  balls crossing.
- **Bounds guessed before measuring failed three times**: the pendulum on a
  constraint (1.4e-8 m against a guessed 1e-8), generalised-alpha on the
  governor (2.8e-5 against 1e-5), and Bricard's closed form near phi = pi,
  where -(u + 1)^2 + sqrt(...) cancels catastrophically (2.2e-10 rad off the
  path, 2.1e-3 rad from the reduced model). The first two became measured
  bounds with a convergence check (the error must fall with the tolerance, or
  by 4 when generalised-alpha's step halves); the third a formula without
  cancellation (the roots' product is 1).
- **A balance check that could not fail.** The governor's dampers do only 3.1
  mJ of work in 10 s, and the first bound on the energy balance, 1e-6 of
  239 J, was 0.24 mJ: tight enough, but only by a factor of 13. Tightened to
  1e-7, so that leaving the work out would fail it by a factor of 130.
- **The first diagnosis of contradictory constraints found nothing.** On a
  bob held 1 m from two anchors 3 m apart, the deletion filter at the
  positions where Gauss-Newton stopped reported no conflict: it had reduced
  |phi| only from 1.20 to 0.96 (the least possible is 0.88), off the line
  between the anchors, where the two linearised distances still have a
  solution. Going first to the least-squares point by Levenberg-Marquardt
  made the filter return exactly the two distances.
- **Small numbers printed as zero.** The N-four-bar's MBD-K055 read "needs a
  step of 0.000000 s ... below step_control.min_step (0.000000 s)", and the
  40-loop run's MBD-K051 "residual 0.000000": `std::to_string` writes six
  decimals. Four messages fixed.
- **The library's own references could not be fetched.** WebFetch could
  not resolve the IFToMM hosts, the shell's proxy refused them, and the web
  archive was not reachable; GitHub was. So A04 and A05 were first checked
  against derived references only; the author then downloaded the library's
  results (see "Closing the task").
- **A motion of about 1 s read as one of 40 s.** The governor's first long
  run was printed every 2 s, and its collar seemed to rise over 20 s and
  fall back: this entry and the model's comment called it a slow swing with
  a period near 40 s, which the dampers hardly touch. The library's result
  showed the collar swinging about once a second (period 1.0247 s, measured
  afterwards on our own run sampled every 1 ms). 2 s is 1.952 such swings,
  so each sample falls 0.0482 of a swing earlier in it than the one before,
  and the samples trace one swing every 2 / 0.0482 = 41.5 s (aliasing). The
  collar's velocity printed in the same run, -0.748 m/s at 10 s, falling
  while the samples rose, did not fit a slow swing of 0.12 m, and was not
  read. The committed tests did not depend on it (they sample every 0.1 s);
  the claim was corrected in the model's comment, this entry and the data
  file.
- **A sanitizer run read as clean that was not** (D45). The run before the
  closing commit (`85de229`) was recorded, there and in its message, as
  "175 of 175 passed with no sanitizer report": a search of the ctest
  summary for reports found none. A second check of the closing, run after
  that push, read the run's full log
  (`build-asan/Testing/Temporary/LastTest.log`) and found two
  undefined-behaviour reports, `LLT.h:66: load of value 32697, which is not
  a valid value for type 'ComputationInfo'`, in a constraints-per-integrator
  case and a manufactured-trajectories case. The sanitizer recovers and the
  tests pass, and ctest shows a passing test's output nowhere, so neither
  this machine's summary nor CI's log had ever shown them. Stopped at the
  first report, the first case failed, with the stack at
  `src/kernel/simulator_linear.cpp:490`: an `Eigen::LLT` built with its
  size only, whose status Eigen leaves unset, moved into the work space.
  The same at two more places, there since task 4.7 (17 and 18 October).
  Fixed by computing the factorization of the identity instead, and CI's
  sanitizer job now stops at the first report. The earlier runs recorded as
  clean (173 of 173 before `cc76a14`, 166 of 166 in task 5.1) were read the
  same way; their full logs are gone.

## Verification

Tests added (506 registered with ctest on Linux, all passing; 499 before):

- `test_kernel_iftomm.cpp`, five cases and one hidden: A01 on a joint and on a
  constraint, against the closed form at two tolerances (seen 1.1e-9 / 8.6e-11
  and 1.4e-8 / 1.5e-10 m; allowed 5e-8 and 1e-9, and a fivefold fall); A02
  with four loops (2.1e-8 / 3.1e-9 rad; allowed 1e-7 and 1.5e-8, and a fall),
  forty in the hidden `[long]` case (3.4e-7; allowed 1e-6); A03, the initial
  accelerations against the test set's (1e-9 relative) and the positions at
  0.03 s against its reference (1.43e-9; allowed 5e-9, the two methods'
  agreement to 1e-10); A04, its counts (one degree of freedom, one redundant
  equation), the closed-form path (9e-15 rad; allowed 1e-12), energy
  (1.5e-10 J; 1e-9), the reduced dynamics (2.0e-9 rad; 1e-7); A05, the
  angular momentum and the energy balance (1e-10 relative; allowed 1e-6 and
  1e-7), Dormand-Prince against Radau IIA (1.4e-9; 1e-6) and
  generalised-alpha's second order (ratio 3.90; 3.5 to 4.5).
- `test_kernel_assembly.cpp`: contradictory constraints named, and only they
  (two distances, and nothing of a four-bar that closes); a loop out of
  iterations but not contradicted, reported as such.
- `test_kernel_debugging.cpp`: the two drivers fighting over a slider, whose
  MBD-K051 now names both.
- Added when closing the task: A04 against three results from the library
  (P2 within 2.5e-5, 3.5e-4 and 7.7e-3 m of Tagliapietra's, González's and
  Pankiewicz's; allowed 5e-5, 1e-3 and 2e-2, from how each result's error
  behaves) and A05 against two (the sleeve within 2.0e-6 and 6.8e-6 m;
  allowed 5e-6 and 2e-5, the reasoning for each bound in the test). Each
  comparison first checks that the file has the rows, the times and the
  columns it should, so that an empty, cut or re-encoded file fails instead
  of passing with nothing compared (a review found that a file in UTF-16
  read as no rows and passed; copies in UTF-16 and cut to six lines now
  fail at the row count). After that, 508 tests registered with ctest on
  Linux, all passing, and CI's sanitizer configuration (`test_core`,
  `test_model`, `test_kernel`) run again on this machine: 175 of 175 passed,
  in 556 s (the two new cases 4.4 and 6.7 s). The closing commit
  (`85de229`) recorded that run as having no sanitizer report; its full log
  held two undefined-behaviour reports ("What went wrong", D45). With the
  code fixed (`e7ea65e`), 508 of 508 passed again on Linux, and the
  sanitizer configuration rebuilt with CI's new flags, which stop at the
  first report: 175 of 175 passed in 348 s, its full log without an
  undefined-behaviour or address report. CI's Debug no-allocation job, run
  here the same way (`EIGEN_RUNTIME_NO_MALLOC`): `test_kernel "[alloc]"` (1
  case) and `test_alloc` (2 cases, 24 assertions) passed.

Before the first push (`cc76a14`), CI's sanitizer configuration was also run
on this machine: 173 of 173 tests passed, in 535 s, of which the five
benchmarks took 195 s (A05 80 s, A04 63 s; 3 s in a normal build). It was
recorded as having no sanitizer report, read from the ctest summary in the
same way; its full log was overwritten, and the code behind the reports was
there. Not run here: the Windows build (no MSVC on this machine) and the
Doxygen reference (Doxygen not installed); CI runs both. Until the fix, CI's
Debug no-allocation job had not been run here either.

## Measurements

No timings compared. The tables behind every number above (the failing runs,
the scans at the singular position, the tolerance grid, Andrews' time shift,
Bricard's closed form) are in
`documentation/journal/data/2026-10-20_iftomm-benchmarks.md`, with how each
was produced.

## Lessons

- A residual below the tolerance is not harmless everywhere: near a singular
  position it changes which mechanism is simulated. Tolerances are a model
  question, not only a numerical one.
- When the obvious cause is confirmed (the rank cut does drop equations), keep
  measuring the symptom: fixing the confirmed cause made the runs worse.
- An error that is one shift in time across every coordinate is a phase
  error, not a model error; dividing by the velocity shows it at once.
- A diagnosis must look at the point the theory is about (the least-squares
  minimum), not where an iteration happened to stop.

## Closing the task

On the same day the author downloaded from the library and gave the project
five results (`tests/kernel/data/iftomm/`, with a README naming their
contributors): three for Bricard's mechanism (point P2's position every 0.01
or 0.05 s for 10 s; E. L. Pankiewicz, Luca Tagliapietra and Francisco
González) and two for the flyball governor (the sleeve's height every 0.1 s;
González, at six and at nine digits).

- **The definitions are confirmed.** Pankiewicz's start energy, -29.43 J, is
  this model's (the bars' centres at heights summing to -3 m, times 9.81);
  every result starts at P2 = (1, -1, 0); and the governor built to the
  OpenSim version's description follows González's results within 7e-6 m.
- **Bricard's mechanism.** Ours, on the closed form of its path to 9e-15 rad
  and within 2e-9 rad of its reduced dynamics, is within 2.5e-5 m of
  Tagliapietra's result (a constant offset of
  about 4e-6 m in that result, with peaks at the turning points and one at
  0.05 s; 3.7e-8 m at 10 s), 3.5e-4 m of González's (an error growing with
  the energy drift that result itself reports) and 7.7e-3 m of Pankiewicz's
  (a phase error: small where P2 stops at the turning points, and between
  them peaks growing 8.1e-4 to 8.5e-4 m per second). The three differ among
  themselves by up to 8.1e-3 m. So the differences measure the results, and
  each test bound follows from how that result's error behaves.
- **The flyball governor.** Within 2.0e-6 m of the six-digit result and
  6.8e-6 m of the nine-digit one; the two differ from each other by 5.7e-6 m
  at 10 s; the nine-digit one drifts from ours at about 7e-7 m per second.
  Ours changes by at most 8.2e-10 m between tolerances 1e-8 and 1e-10. The
  bounds would catch a fault of the model: a period off by a fraction e
  moves the sleeve at 10 s by up to 7.5 e m (half the swing, 0.122 m, times
  2 pi times 9.76 swings). Both results lie above ours at 10 s, where the
  sleeve falls, and a longer period raises ours there, so the limits depend
  on the sign. Measured on our own motion run slower or faster by 1 + e: a
  period shorter by 4.3e-7 or longer by 8.1e-7 fails against the six-digit
  result, one shorter by 1.8e-6 or longer by 3.6e-6 against the nine-digit
  one. (The closing commit, `85de229`, said that e = 7e-7 fails either way:
  true only for a period too short, as a second check found.)
- **The mistake it exposed**: the 40 s "slow swing" (in "What went wrong").

With that, every problem meets a published reference (A03, A04, A05) or a
closed form that the library's references only approximate (A01, A02), and
5.2 is done.

## Open ends

- Closed: A04 and A05 against the library's results (see "Closing the
  task").
- Regularised multipliers or an augmented-Lagrangian formulation, the usual
  robust answer to singular positions, which would not need the constraints
  held near rounding.
- Whether the time shift of Andrews' reference is the reference's: compare
  with another published value at 0.03 s if one exists.

## For the book

- Part VII (validation): the N-four-bar as a story: the obvious cause
  confirmed and fixed, the runs worse, the real cause found in the
  coupler's rate; the imperfect parallelogram's coupler, delta / (L theta),
  as a figure. Bricard's mechanism: Grubler's count against the geometry, the
  closed form of its path derived by machine algebra and checked by Newton,
  and a reference for its whole motion from that form alone.
- Part X (testing a numerical code): references of our own while the
  published files were out of reach (closed forms, a reduced model from
  geometry, conserved quantities, convergence between methods); a balance
  check made tight enough to fail. Then published results judged by a model
  checked independently: the differences measure the results' errors (an
  offset, a drift with the result's own energy, a phase error), and each
  bound follows from how that error behaves. A test that could not fail: a
  file read as no rows compared nothing and passed. And aliasing: a swing of
  1 s printed every 2 s and read as one of 40 s, with the velocity in the
  same output that did not fit.
