# Phase 7 retrospective

- **Dates:** 14 to 27 November 2026 (the phase); 27 November (this entry)
- **Plan:** Phase 7, tasks 7.1 to 7.12; documentation track H, task H.6
  for this phase; decisions D60 to D73
- **Commits:** `d9f9466` (task 7.1's scenarios recorded before the move) to
  the commit that adds this entry, on branch `phase-7`, which starts from
  `v0.7.0`. `main` was fast-forwarded to task 7.10 (`4d81e40`, not tagged)
  on 27 November with the author's go-ahead.
- **Written:** at the time

## What the phase achieved, against its goal

Goal: "each subsystem modelled properly, on the corrected core." Every
subsystem the plan names is built, on the kernel's joints, constraints,
force elements and state elements, and checked against a closed form
(the table in `docs/validation.md`, section 9).

- **Axes** (7.1, D60): the vehicle layer in ISO 8855's axes, up decided in
  one place; every scenario gave the same result before and after.
- **Road** (7.2, D61): the tyres' only source of the ground; shapes, height
  maps, ISO 8608 random roads, a track's banked surface.
- **Wheel** (7.3, D62): a body on a spin joint, its contact found from the
  wheel plane and the road; the drivetrain's torques in pairs.
- **Tyres** (7.4, D63, D64; 7.5, D65): the Magic Formula 6.1 and 5.2 from
  `.tir` files, against MFeval to 3.5e-12; its slips lagged to standstill;
  linear, Fiala and brush tyres behind one interface.
- **Suspensions** (7.6, D66, D67): five architectures from the kernel's
  joints with their parts' masses where their joints put them; springs set
  by virtual work to hold the design position; hysteresis as state
  elements (a damper on a top mount, Bouc-Wen bushings) with a virtual
  damper dyno.
- **Steering** (7.7, D68): a rack geared to a pinion, a compliant column,
  by angle or torque; Ackermann behaviour from the geometry.
- **Powertrain** (7.8, D69), **brakes** (7.9, D70): state elements beside
  the bodies, from the engine's speed to the half-shafts' twist, from the
  pedal to each wheel's line pressure, with lock and an anti-lock
  controller.
- **Aerodynamics** (7.10, D71): a map over ride heights, pitch, roll and
  yaw, in the body's axes, with wind; the constant model a one-node map.
- **Mass properties** (7.11, D72): the chassis's own centre of mass and
  inertia tensor, lumped masses, and a roll-up of the built car with sprung
  and unsprung mass by virtual work.
- **Subsystem tests** (7.12, D73): the table of every subsystem's analytic
  check, and the gaps it showed closed.
- **Help** (H.6 for this phase): [the vehicles how-to
  guide](../../docs/help/howto/vehicles.md), five examples run and checked
  against hand calculations in `tests/howto/test_howto_vehicles.cpp`.

Finding F9 (mass, inertia and equilibrium of the linkage cars) is closed:
the wheels' and the arms' masses are where they are, springs hold the
design position, the steering's fields are read, and the chassis's centre
of mass reaches the model. 660 tests at the end of 7.12, against 544 at the
end of Phase 6.

Not done in this phase, by design: the vehicle layer in model files and in
Python (Phase 8, the vehicle builder); analyses of a vehicle (Phase 9).

## What took longer than expected, and why

- **The tyre** (7.4, two parts): the book's printed forms are not those of
  the industry's codes, so the first version disagreed with MFeval until
  each difference was traced; and the tyre on a wheel had to hold still at
  rest, which the old model had done through an unphysical damper.
- **The suspensions** (7.6, two parts): a mixed car could not be placed
  level until the springs were set by virtual work from the design
  position, with the tyre's overturning moment at rest in the balance.
- **Cars that would not keep still** (7.3, 7.4): parked cars crept or
  rolled away once the contact and the tyre were physical; each needed the
  physics fixed (the lock, the low-speed tyre), not the test.

## What went well

- **Virtual work as one tool.** Motion ratios, the springs' design forces,
  anti-squat, and the unsprung mass all come from the same rates of a
  corner's mechanism; each new use was checked against the old ones.
- **A closed form for every subsystem**, written in the test before the
  first run, so that a failure meant a question about the physics.
- **The regression harness** made every change of a model's meaning (the
  tyre, the aerodynamic forces in the body's axes) visible and measured by
  phase before a reference was written again.
- **State elements** carried subsystems with internal dynamics (the tyre's
  lag, hysteresis, the powertrain, the brakes) without new kernel code.
- **An independent code for the hardest subsystem**: MFeval for the Magic
  Formula, agreeing to 3.5e-12 over 1812 points.

## What to do differently

- **Derive an allowance from the whole model.** Twice an estimate left out
  a term and a correct result failed: the locked differential's decay
  (7.8), and the driveline's ringing without the inertia reflected through
  the gear ratio (7.12). Write down every inertia and stiffness the
  estimate uses, from the parameters, before choosing the margin.
- **Edit text with backslashes in the editor.** Scripted edits through the
  shell mangled escapes in 7.1, 7.2, 7.5 and a data file in 7.9; the fix
  (scan added lines for control bytes and tabs) catches it after the fact.
- **Measure before writing.** A comment claimed a gyroscope result before
  it was run (7.3); a decision first gave a shift's direction backwards
  (7.11). Both were caught on reading; neither should have been written.
- **Rewrite only references that failed**, and only for a change
  understood (7.3 rewrote one that had not failed; 7.6 and 7.10 restored
  one rewritten by rounding).

## Open ends carried forward

- The vehicle layer is not in model files or Python (Phase 8).
- No corner weighting: a car whose centre of mass is off its centre line
  rests slightly rolled (D72). The presets' centres of mass sit at their
  wheel centres' height, lower than a real car's.
- The lap model reads the constant aerodynamic areas, not a map.
- Checked in direction only: the anti-lock controller, a car driving off
  once its brakes let go. The tyres are checked against MFeval, not
  against measured tyres.
- Roll centres, swing arms and the other kinematic outputs: task 9.1.

## The gate

The phase's work, this entry, the vehicles how-to guide and version 0.8.0
are on `phase-7`. CI on the gate's commit, and the author's go-ahead for
the merge into `main` and the tag `v0.8.0`, are to be recorded here.
