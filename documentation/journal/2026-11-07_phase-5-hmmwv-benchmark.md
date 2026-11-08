# Phase 5: the HMMWV benchmark, and a static solver that lost soft directions

- **Dates:** 7 and 8 November 2026
- **Plan:** task 5.4 (the HMMWV benchmark from the reference paper); decisions
  D47 (statics), D48 (the benchmark)
- **Commits:** see the index
- **Written:** at the time, on the author's laptop (Windows, MSVC), by
  Claude working with the author

## Goal

The plan's task 5.4: "The HMMWV benchmark from your reference paper.
Fourteen bodies, four double-wishbone corners, tie rods as distance
constraints, driven at 10 mph over a 0.1 m half-cylinder. Then the bushed
variant (2e7 N/m, 2e6 Ns/m). Done when explicit and implicit runs agree and
their costs are tabulated."

The paper is Haug, Negrut, Serban and Solis, "Numerical methods for high
speed vehicle dynamic simulation" (1999), in
`documentation/Selección/Selección 2/` on the laptop. Its HMMWV-14 tests
implicit integrators on a stiff vehicle; its tables give CPU times on a
200 MHz SGI for tolerances from 1e-2 to 1e-5.

## Starting point

Tasks 5.1 to 5.3 had checked the kernel against closed forms, the IFToMM
problems and two independent codes, on mechanisms of up to about ten bodies
and without vehicle-sized stiffness ratios. No model of the HMMWV existed.
The paper describes its model in words only: no geometry, masses or
suspension laws.

## What was done

1. **The data.** The paper's figures (bodies, joints, the tyres' 296,325 N/m
   and 3,502 N s/m, the bushings' 2e7 N/m and 2e6 N s/m, the bump, the
   speed, the timings of the bumps and the settled chassis height of about
   0.71 m) are used as stated. The rest comes from Project Chrono's HMMWV
   (BSD-3-Clause), read as data from the conda package's
   `data/vehicle/hmmwv` and from `HMMWV_DoubleWishbone.cpp` on GitHub: the
   hardpoints, masses and inertias, the progressive spring table, the
   dampers, the rack. Chrono's model comes from the same research group,
   and its chassis's centre of mass sits 0.709 m above the ground at the
   design position: the paper's 0.71 m.
2. **The model** (`tests/kernel/hmmwv.hpp`): the chassis on a free joint, a
   steering rack on a prismatic joint held by a driver, and at each corner
   a lower and an upper arm on revolute joints and a spindle body (Chrono's
   upright, spindle, wheel and tyre in one: the paper's model has no wheel
   spin) on a spherical joint to the lower arm, closed by an upper ball
   joint and a tie rod. Each body's frame is placed so that the design
   position is the model's neutral configuration. The tyres are radial
   spring-dampers to the nearest point of the road; the road is flat ground
   with a half cylinder across it. The bushed variant puts each upper arm
   on a spherical joint at its front point, with two spring-dampers across
   its axis at its back point. 27 velocities and 17 equations leave 10
   degrees of freedom; with bushings, 35 and 18, as the paper counts them.
3. **Chrono's data checked before use.** Our `validate()` rejected the
   control arms' moments of inertia: Chrono's (0.4, 0.4, 0.8938) and (0.03,
   0.03, 0.06276) kg m^2 break the triangle inequality, so no body has them.
   Taken as flat arms, Izz = Ixx + Iyy (0.8 and 0.06). And with Chrono's
   spring free lengths the vehicle settled 0.085 m high (chassis at 0.794
   m), its springs at or past their rebound stops: the progressive springs
   give 45 kN at the design length, three times the static load. The
   paper's model rests at its design height, so the free lengths were found
   (`mbd_hmmwv calibrate`, Newton by differences) that rest the arms at
   their design angles: 0.294734288 and 0.332951200 m against Chrono's
   0.339 and 0.382, the table's shape kept. The chassis then rests at
   0.6875 m (the design height less the tyres' static deflection).
4. **The bushed model would not settle** (D47). Static equilibrium from the
   design position stalled at 2.37 m/s^2 with the chassis in the air, then
   its dynamic relaxation blew up. Fixed in `src/kernel/statics.cpp`: the
   Newton step by a rank-revealing QR, and relaxation that ends worse is
   undone. Then it settles in 4 iterations, 3e-5 m from the joint model.
5. **The modes** (`mbd_hmmwv modes`): with joints the fastest is 67 s^-1;
   with bushings, -2.06e6 s^-1 (a bushing's damping on about 1 kg of upper
   arm), eight times the paper's -2.6e5. RK4's step must stay below 2.785 /
   2.06e6 = 1.35e-6 s; at 5e-6 s it blew up within ten steps.
6. **The benchmark program** (`bench/hmmwv.cpp`, `mbd_hmmwv`): every method
   at several steps or tolerances on both variants for the paper's 5 s,
   every body's position every 10 ms against Radau IIA at 1e-10, and the
   cost: wall-clock time of the stepping, steps, evaluations, Newton
   iterations, Jacobians. The tables are in `docs/hmmwv.md`.
7. **The bump's discontinuities.** A first run showed irregular convergence
   (generalised-alpha's difference rising from 1 to 0.5 ms) and a 4 m
   difference at 4 ms steps. The tyre force jumps where it turns to the
   bump: its direction by 26 degrees, its damper rate by about 2 m/s, about
   7 kN. The model got events at those switches (`Hmmwv::add_events`), and
   a second table runs the methods with them.
8. **Tests** (`tests/kernel/test_kernel_hmmwv.cpp`, six cases).
9. **The results** (`docs/hmmwv.md`, one pinned run of 45 minutes on 8
   November). Explicit and implicit agree: with joints Dormand-Prince at
   1e-8 is within 2.2e-7 m of the reference and Radau IIA at 1e-8 within
   6.4e-10 m; with bushings RK4 at 1.25e-6 s and Dormand-Prince at 1e-6
   within 3.6e-6 and 2.0e-6 m, Radau IIA at 1e-6 within 2.4e-7 m. The
   costs: on the bushed model RK4 takes 1,060 s and Dormand-Prince 1,520 s,
   both held at the stability limit, where Radau IIA takes 0.78 s for a
   better result; on either model Radau IIA takes about 700 steps at 1e-6.
   The reference, run again with events and so with other steps, moves by
   1.6e-11 m. Across the bump, the fixed-step methods converge irregularly
   with or without events, SDIRK4 does not converge without events, and
   Dormand-Prince needs a tolerance of 1e-5 or tighter.

## Decisions

D47: the static solver's Newton step from a rank-revealing QR of the scaled
Jacobian's transpose, not the normal equations; a relaxation that ends worse
is undone. D48: the benchmark's model, data and method (below and in the
decision log).

## What went wrong, and how it was found

- **The static solver lost the soft directions of a stiff model.** Newton
  stopped at 2.37 m/s^2 on the bushed HMMWV. A temporary probe printed the
  singular values of the scaled Jacobian: the directions with stiffness run
  from 1 down to 4.8e-6 (the chassis's heave, pitch and roll on its tyres
  under bushings of 2e7 N/m), then the noise of the differences, at most
  2e-10. The step came from the normal equations, which square that range:
  2.3e-11, below their rank cut of 1e-10, so 12 of the 15 directions were
  seen and the chassis never came down onto its tyres. A test with two
  springs a millionfold apart failed the same way before the fix, and the
  relaxation that followed returned NaN.
- **Dynamic relaxation blew up beside the bushings.** Its explicit step of
  1 ms against modes of 1,860 rad/s: the chassis ended 17 m up and Newton
  carried on from there. Relaxation that ends with larger accelerations
  than it began with is now undone and reported.
- **A guard too eager.** The first version undid a relaxation unless the
  accelerations fell; a body with nothing to hold it falls with its
  acceleration exactly g, and an existing test caught the change. Now only
  a relaxation that ends larger is undone.
- **The first test of the undo could not fail.** Two independent blocks: the
  stiff one sat exactly balanced, nothing excited it, and relaxation
  succeeded. Put on top of each other, the landing block excites the stiff
  spring, as the test means.
- **A rounding bound guessed against the wrong entries.** The two-spring
  test's bound on the soft position (1e-13 m) compared the differences'
  noise with the stiff entries of G; against the soft ones it is 5e-10
  relative, about 5e-11 m (seen 2.3e-13).
- **Chrono's data had two faults** (the inertias, the free lengths; item 3).
- **A tyre that ignores the bump while the ground is nearer.** The bump was
  first placed where the unloaded tyre touches it; the single spring of
  this tyre model follows the nearest surface, so its force turned 10 ms
  later. The bump is now placed where the force turns, at 0.5 s exactly.
- **Events, measured, are no cure-all** (the second table): with them
  generalised-alpha fails (MBD-K058, at t = 0.606 s on a 0.36 ms remainder
  of a step cut by an event), Dormand-Prince is less accurate at loose
  tolerances (it locates events on its continuous extension, which assumes
  the dynamics are smooth across the crossing; here the crossing is the
  discontinuity), and Radau IIA's reference costs many times more. Without
  events every method runs. Recorded as open ends.
- **The machine's speed moved threefold during the day.** The full test
  suite took 29.5 s in the morning and 85.9 s in the evening; RK4 at 1 ms
  on the model took 0.44 s, then 0.80 s. The benchmark's timings come from
  one pinned run, and compare only within it.
- **A background command can be stopped after ten minutes** in this
  environment; the first pinned benchmark run was started inside one and
  was restarted detached, writing to files.
- **The laptop slept in the middle of the benchmark.** Started at 21:35, the
  run had used 23.5 min of processor time by 12:41 the next day: the system
  log shows sleep from about 22:00 to 12:40, half a minute into the second
  explicit run on the bushed model. That run's wall-clock time could not be
  trusted and the session was broken, so it was stopped and run again from
  the start, keeping the machine from idle sleep for the run's duration
  (`SetThreadExecutionState`), with one explicit run per method on the
  bushed model instead of two (their cost is set by stability, not by the
  step or tolerance). The joint model's tables of the first run, finished
  before the sleep, were superseded by the second run's.
- **Shell quoting.** Three scripted edits wrote `\n` inside C strings as real
  line breaks (Python under a quoted heredoc in Git Bash); fixed with the
  editor.

## Verification

- `test_kernel_statics.cpp`: a soft spring below a stiff one (stiffness
  ratio 1e-6) settles in one Newton step to the closed form (q1 within
  1e-15 m, q2 within 1e-11 m, seen 2.3e-13); a relaxation too long for a
  stiff element is undone, and one at a tenth of the step succeeds. Every
  earlier statics case passes unchanged.
- `test_kernel_hmmwv.cpp`: the counts (14 bodies; 10 and 18 degrees of
  freedom); at rest the tyres carry the weight to 1e-3 N and its moments to
  1e-3 N m, the arms rest at their design angles (seen 2.3e-9 rad; with
  bushings 7.9e-6 rad, bound 7e-4 from the bushings' compliance), the
  bushings move the chassis by 2.3e-5 m (bound 3e-4); the front tyres turn
  to the bump at 0.5 s (bound 1e-8 s); over both bumps Dormand-Prince
  converges to Radau IIA (1.3e-5 m at 1e-6, 4.3e-8 m at 1e-8) and RK4 does
  slowly (7.6e-4 m at 1 ms, 2.7e-4 m at 0.5 ms); with bushings RK4 at
  2.5e-6 s fails within 1 ms (NaN, 381 projection failures) and at 1.25e-6 s
  agrees with Radau IIA to 7.9e-8 m over the first 15 ms of a bump; Radau
  IIA and SDIRK4 at 1e-6 agree with Radau IIA at 1e-8 (6.7e-8 and 4.3e-6
  m), and the bushings move the chassis's height by 3.4e-5 m in motion.
- `scripts\build.ps1 -Test`: 516 of 516 pass; no compiler warning.

## Measurements

`documentation/journal/data/2026-11-07_hmmwv-benchmark.md`: the
calibration, the static solver's spectra and histories, the modes, the
exploratory runs, the probe of events, and the full benchmark (also in
`docs/hmmwv.md`).

## Lessons

- Check a published data set before building on it: our own validation
  caught inertias no body has, and a static solve caught springs that do
  not hold the vehicle where its data puts it.
- A model of a kind the code has not met finds its weak spots: the first
  vehicle with a stiffness ratio of a million found a squared ratio in the
  static solver, and its first discontinuous road found how events and
  integrators interact.
- Measure the spectrum before choosing a threshold: the singular values set
  the stiffness cut with margins both ways.

## Open ends

- Generalised-alpha with events fails on this model (MBD-K058 on the
  remainder of a cut step); Dormand-Prince's event location assumes smooth
  dynamics across the crossing; events make Radau IIA's tight runs many
  times dearer. To be looked at with the events (D29) and the integrators
  (D31, D32).
- RK4 converges irregularly over the bump even with the tyres' events
  (7.5e-4, 2.0e-3, 1.1e-3, 5.1e-4 m from 2 to 0.25 ms), so something else
  not smooth remains; a tyre grazing loss of contact near the crest is a
  candidate, not checked.
- The linearly implicit Euler step finds its matrix not positive definite
  where the tyres turn to the bump (MBD-K045 at t = 0.500 s, the instant
  the front tyres turn): the bump's geometry gives the tyre force a
  negative stiffness across its direction.
- Chrono's spring stops and the shocks' hydraulic stops are left out; the
  springs and shocks stay inside them over the runs (checked in the
  exploration).
- A comparison with Chrono's own HMMWV belongs to task 9.6; the model here
  becomes a reference car file in task 8.6.

## For the book

Part VII, a section on the HMMWV: a benchmark rebuilt from a paper that
gives no data, with a data set from a descendant code checked before use;
stiffness that spans a millionfold, what it does to an explicit method (the
step set by a bushing's damping on an arm of 1 kg) and to a static solver
(the normal equations' squared ratio); the bump's discontinuities and what
events can and cannot do. Figures: the chassis height through both bumps;
cost against accuracy for every method, with joints and with bushings.
