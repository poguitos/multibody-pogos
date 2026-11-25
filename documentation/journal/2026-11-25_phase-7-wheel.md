# Phase 7: the wheel and its contact

- **Dates:** 24 and 25 November 2026
- **Plan:** task 7.3; findings F6 and F8; D4, D62
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.3: "Wheel and contact (F6, F8). Wheel as a rotating body on a spin
joint. Contact point and frame from the wheel plane and the road. *Done when a
pitching upright no longer moves the contact point, and a spinning wheel shows
gyroscopic precession.*"

## Starting point

The wheels were not bodies. Each corner had one body under the chassis: the
slider of a simple corner (with the whole unsprung mass, a box of inertia) or
the upright of a linkage corner (the wheel's mass left out, F9). The tyre sat
on it with its contact point fixed in it at (0, 0, -R): when the body
pitched, as an upright does under braking, the point swung forward or back
by R sin(pitch) (F8). The wheels' spins were the drivetrain's four
first-order states (tasks 1.7 and 4.1), handed to the tyres at every
evaluation, each driven wheel's inertia carrying half the engine's seen
through the gears; a locked wheel's state was set to zero at the end of a
step. Multirate RK4 (task 4.6, D36) had been built and verified on these
spins as its fast states. D4 (October) had decided the spin joint.

## What was done

1. **The wheel** (`mbd/vehicle/wheel.hpp`): `add_wheel` puts a body on a
   revolute joint about the carrier's Y axis, at the wheel centre, its axes
   the carrier's at zero, so that a forward roll is positive; its inertia
   (`wheel_inertia`) a uniform thin disc of the wheel's mass and the tyre's
   free radius, m R^2 / 2 about the axle (the drivetrain's value since 1.7)
   and m R^2 / 4 about a diameter. `set_free_rolling` sets a wheel's joint
   so that its spin is its tyre's free rolling.
2. **The builders:** every corner now has an upright (`upright_body`; on a
   simple corner the slider, now massless) and the wheel on it
   (`wheel_body`, the tyre's body). The upright takes the spring, the
   anti-roll bar, the steering, and gives toe and camber. The simple
   vehicle's `VehicleModel` has `carrier_bodies`, `wheel_bodies` and
   `spin_v`. `roll_wheels` sets every wheel of a vehicle at free rolling.
   `wheel_half_extents` is gone.
3. **The contact** (`wheel_contact` in `mbd/forces/tire.hpp`, used by both
   tyres): from the wheel centre, its spin axis and the road. The road is
   asked under the centre for its normal; the wheel's lowest point is the
   point of its circle furthest against that normal; the road is asked again
   there for height, normal and friction; the penetration is the lowest
   point's depth below the tangent plane, (h - z) n_z; the contact point is
   the lowest point brought onto the plane; the heading is a x n. A simple
   corner's steering turns the wheel plane about the normal.
4. **The slips:** the spin is the wheel's angular velocity along its axis;
   the contact point's velocity is the wheel's without that spin. The forces
   act at the contact point; the tyre records their moment about the axle
   (`last_axle_moment`). `omega_wheel` and `auto_free_roll` are gone: a tyre
   on a body that does not spin sees a locked wheel.
5. **The drivetrain** lost its states. Its torques go through a force element
   that `connect` adds to the system: the drive torque between each driven
   wheel and the chassis, the brake's between the wheel and its upright. It
   reads each spin relative to the upright. Its `end_step` still locks,
   releases and shifts; a locked wheel's joint rate is set to zero there, and
   within the step its brake opposes the other torques with a damping
   (`lock_time`, 5 ms). The engine's inertia is left out (D62).
6. **Tests:** the contact under pitch, camber and on an incline; the wheel's
   inertia; a gyroscope on a pivot; free rolling at the start; the
   drivetrain's torque pairs; and every test that counted bodies, degrees of
   freedom or neutral directions, set a moving car's wheels, or read toe and
   camber from the old wheel body. Multirate's vehicle test was removed.
7. **Regression:** sedan-drive, sedan-slalom, sports-car-turn,
   hatchback-brake, sedan-rough-road and sedan-modes written again (below).
8. **Found on the way:** task 7.2's checkbox in the plan had been left
   unticked although its result was recorded; ticked.

## Decisions

D62. Smaller ones: one function for the contact (`wheel_contact`), used by
both tyres; the drivetrain's missing-corner error is a force element's
(MBD-F052); the settling tests run RK4 at 0.5 ms rather than an implicit
method, to keep testing what they tested.

## What went wrong, and how it was found

- **Parked cars that would not keep still.** Three tests that let a car
  settle at rest (RK4 at 1 ms) failed: a velocity of 0.037 rad/s left after
  5 s, and a chassis that drifted 2.1e-5 m in a second from its static
  equilibrium. Before, no drivetrain meant no spin dynamics at all (the
  tyres assumed free rolling); now every wheel spins on its joint, and at
  rest its tyre's slip, clamped below V_low, damps it at about 3000 per
  second, just beyond RK4's 2785 at 1 ms. The spins chatter, by hundredths
  of a rad/s, and the car creeps. The cars with a drivetrain had had this on
  their undriven wheels since task 4.1 (troubleshooting); the driven ones had
  been held by the engine's inertia. The tests now run at 0.5 ms and say why.
- **A lock that let wheels turn back.** The first drivetrain held a locked
  wheel by forces only: the opposite of the other torques, and a damping.
  The braking-to-a-stop test failed for every method: wheels turned back by
  up to 0.008 rad/s (RK4) and 0.11 rad/s (Radau IIA), and at rest stayed at
  -1.7e-6 rad/s instead of zero. The holding torque fixes the wheel's own
  spin, not its spin relative to an upright that pitches with the car; and a
  step that crosses zero before the lock leaves the overshoot to be damped,
  where the old drivetrain had set it to zero. A locked wheel's joint rate is
  now set to zero at the end of each step, as the old spin state was; the
  test passes unchanged. This needs the simulator, which `end_step` gets as
  const: the drivetrain keeps the one it was connected to (D62).
- **Tests that had become trivial.** With the spins started at exact free
  rolling, the low-speed tests for generalised-alpha and the linearly
  implicit step had no transient left (rolling within 1.8e-15): they still
  passed, but no longer showed the methods settling a stiff spin. The
  wheels are now started 3 % fast, as the drivetrain started them before.
- **A claim written before it was measured.** The gyroscope test's comment
  first said that an inertia off by 1e-6 would move the axle by 6e-9 rad; a
  hidden probe, then removed, measured what a precession 1e-6 too fast does
  (3.0e-9 rad high, 5.2e-9 rad behind after 3 s), and the comment says that.
- **References written by a stale executable.** After the lock was changed
  (the reset at the end of the step) only `test_vehicle` was rebuilt; the
  references were then written by a `test_regression.exe` still linked to
  the force-only lock. The suite rebuilt everything and the hatchback, the
  one scenario that brakes to a stop, failed by up to 1.7e6 times its
  tolerance; run alone or in its whole executable, the same, so not a
  dependence on what ran before. Its reference was written again from the
  full build, and the comparison in the data recomputed. Build everything
  before writing references.
- **Two road tests that described the old contact.** The tyre on a bump's
  flank expected the force of the point below the centre (970 N of depth
  0.004849 m); the new geometry gives 0.011921 m, and the exact depth of the
  wheel's circle in that bump is 0.013302 m (a scratch computation of the
  nearest point of the road to the centre). On the ramp's plane the new
  depth is the exact one. The friction test's wheel did not spin, so it was
  a locked wheel, whose lateral force the combined slip takes away; it now
  spins at free rolling. Both tests were run only by the suite: the tyre's
  own tests had been run by their tag, `[full_tire]`, which missed them.
  Writing the bump's expected value, a lambda returning an Eigen expression
  of a temporary dangled (a depth of 1e-323); it returns a `Vec3` now.
- **A scenario that lost its sensitivity.** The hidden sensitivity probe,
  run by hand after the suite, failed on `sedan-modes`: a change of 1e-6 in
  gravity moved it by only 8.5 times its tolerance (26 before; the probe
  asks for 10). Its four new modes, the wheels' spins decaying at 2759 to
  2920 per second, had become its frequency channel's largest value, and
  the tolerance follows that: the body's modes, 8 to 260 rad/s, were held
  to a tolerance set by the spins. The spins now have a channel of their
  own: 29.9 times. The rough road's lesson again, in another form: a
  channel's values need one scale.
- **A reference rewritten that had not failed.** Writing the references
  again rewrote `wishbone-sweep` too, which had passed: its values moved by
  5e-14 of their scale (rounding: its builder's file was recompiled). Its
  old reference was restored.
- **A here-document that bash would not read** (a quote inside it); the
  edit scripts were written as files.
- **A record file emptied by its own edit.** The script that appended D62
  opened `decisions.md` for writing and then encoded the text as ASCII; the
  file holds four characters that are not (a name), the encoding failed,
  and the file was left empty. It was restored from git (it had no change of
  this task yet), and the edit repeated with the text encoded before the
  file is opened, in UTF-8; the edit scripts' helper now does the same.

## Verification

- **The done when, first part** (`test_full_tire.cpp`): a wheel of radius
  0.35 m, centre at (1, 0.5, R - 0.02), moving at 20 m/s and spinning at
  15 / R_eff, turned about its axle by 0, 0.3, -1.0 and 2.5 rad: the contact
  point is (1, 0.5, 0), the depth 0.02, the normal force 4000 N, the slip
  ratio -0.25 and the slip angle 0, all within 1e-12 (rounding in turning the
  axis), Fx the same to 1e-9, and the moment about the axle -(R - 0.02) Fx.
  A point fixed in the wheel would have moved by R sin(beta), 0.10 m at
  0.3 rad. A wheel leaning by 0.1 rad has its contact R sin(0.1) to the side,
  as the closed form of the circle's lowest point gives; on a plane inclined
  by 0.2 rad the contact point is c - (R - 0.02) n and the force k_z 0.02
  along n, within 1e-12 and 1e-9.
- **The done when, second part** (`test_wheel.cpp`): a 10 kg wheel of
  radius 0.3 m, its centre 0.2 m along the axle from a spherical pivot,
  spinning at 100 rad/s with the axle horizontal: m g l = I_s omega_s
  Omega_p exactly at a horizontal axle (the heavy symmetric top), so
  Omega_p = 0.43585 rad/s. After 3 s the axle has turned 1.3076 rad about Z
  to 9.1e-15 rad (1.8e-14 at half the step), its elevation 3.3e-16, the spin
  unchanged; spun the other way, it precesses the other way. Allowed
  1e-12: steady precession has zero joint accelerations, so RK4 is exact but
  for rounding, and a precession 1e-6 too fast moves the axle by 3e-9.
- **Free rolling:** with the chassis pitching at 0.3 rad/s, roll_wheels
  leaves every slip ratio zero to 1e-12; the drivetrain's initialize gives
  20 / (R - d / 2) to 1e-9.
- **The torque pairs:** on the sports car at 15 m/s with throttle and brake,
  the drivetrain's torques add up to 1.1e-13 N m against 1102 N m: drive
  between wheel and chassis, brake between wheel and upright, each to 1e-12.
- **The drivetrain's tests**, with the spins on joints: half-pedal braking
  5.53912 m/s^2 against 5.52811 by hand (the hand calculation now with the
  force's arm to the road, r_l = R - d, beside the rolling radius; the 0.2 %
  left is the slip it leaves out; allowed 0.5 %, was 2 %); Dormand-Prince
  equal to RK4 to ten digits; braking to a stop with six methods, the wheels
  locked at zero and the car below 1e-4 m/s; generalised-alpha at 5 ms
  rolling within 3.8e-5 (RK4 at 0.5 ms: 4.1e-5; at 5 ms 14.9 %); the
  linearly implicit step within 3.9e-5 at one evaluation a step; on the
  double-wishbone sedan its top speed 1.55137 m/s against RK4's 1.55138.
- **A forward force** of 5 kN on the simple vehicle: 3.03599 m/s after 1 s
  against 3.03582 from F / (m + 4 I / (R_eff r_l)), the wheels' spin inertia
  87 kg of it; allowed 1e-3 (was 5 % without the wheels' inertia).
- **The suite:** 569 tests pass (563 before: seven new, the multirate
  vehicle test removed), the build without a warning, the laptop on mains
  power, 55.12 s (the last run, after the modes scenario's channel). The regression sensitivity probe passes for every
  scenario (its table in `tests/regression/README.md`).
- **The spins' modes, independently:** the linearisation of the sedan at
  rest finds the four wheels' spins decaying at 2759, 2759, 2853 and 2920
  per second, where task 4.1 measured 2800 to 3100 per second by RK4's
  stability limit on the drivetrain's spin states (at 0.5 m/s rather than
  at rest).

## Measurements

In `data/2026-11-25_phase-7-wheel.md`: the regression scenarios before and
after, the one-dimensional estimate of sedan-drive, the gyroscope, the
lock's first version, the sensitivity probe.

## Lessons

- **A change of state variables changes what the integrators see.** Moving
  the spins from first-order states into the bodies' state was meant as a
  change of model; it also took away multirate's subject and put the spins'
  stiffness into every vehicle, parked or not.
- **Make a test's initial state as disturbed as the method it tests needs.**
  An exact initial state can make a stability test pass for the wrong reason.
- **A friction lock is a state change, not only a force.** The old
  drivetrain's reset at the end of the step was doing work no force did.

## Open ends

- The engine's inertia is out of the model until task 7.8 gives the engine a
  speed of its own (plan amended). Driven acceleration from rest is about
  12 % quicker in first gear than with it.
- At rest the spins need RK4 below 0.9 ms (the tyre's clamped slip below
  V_low); task 7.4's lag to standstill should take that stiffness into a
  first-order state, where multirate can step it again.
- The integrator report's sedan (task 4.9) was measured with the spins as
  states; not run again here.
- A simple corner steers its tyre, not its wheel: the wheel body spins about
  the unsteered axle (task 7.7's steering).
- `TireContactForce` and `FullTireForce` take the contact from the wheel's
  circle, the road asked twice: on a bump's flank 0.011921 m deep where the
  wheel's circle is 0.013302 m in the road. Iterating the normal, or the
  nearest point of the road, would close the gap; a single point on a
  strongly curved road is an approximation either way (enveloping models,
  later).
- The template's `total_mass()` still leaves out arms and uprights (F9).

## For the book

Part on the vehicle (Phase 7): the wheel as a body and what it changes; the
contact point as geometry (the lowest point of a circle against a normal)
rather than a point of a body; a gyroscope as a test with an exact answer at
one angle; a brake's stick as a state change; and how moving a state from one
part of a model to another moved the stiffness, and took away an
integrator's purpose.
