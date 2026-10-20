# How to get outputs, add contact, act on events and check a model

The loads a simulation is asked for and channels recorded over a run;
contact of spheres with a plane; actions when a function of the state
crosses zero; and the checks and records that find what went wrong (Phase
3, tasks 3.3, 3.8 and 3.9, and the debugging aids of H.4). Every example is
compiled and run, its result checked, in
`tests/howto/test_howto_outputs.cpp`, and this page quotes that code exactly
(task H.6).

The examples use `using namespace mbd; using namespace mbd::kernel;`, and
the models of the other guides: the rod hanging from a hinge, a 2 kg ball
on a vertical slider (its coordinate its height), the four-bar `fb` of
[analyses](analyses.md).

## The loads a joint carries

<!-- example: loads -->
```cpp
Simulator sim(sys);                          // the rod hanging at rest from its hinge
sim.initialize();
const Loads loads = compute_loads(sim);      // at the simulator's state
const Vec6& hinge = loads.joint_reaction[1]; // [moment; force] the hinge passes to the rod, in its joint frame
// loads.v_dot: the accelerations; loads.lambda and loads.constraint_W: the constraints' forces.
```

At rest the hinge carries the rod's weight, 9.81 N upward, and no moment.

- `joint_reaction[i]` is for the joint of body i, in the child-side joint
  frame, about its origin; it includes the joint's own actuation, whose
  component along the joint's motion is the generalized force there.
- `constraint_W[i]` is the constraints' wrench on body i, in world axes,
  about the world origin; `lambda` the multipliers in the order of the
  constraints' equations (a driver's is its force or torque).
- Energies: `kinetic_energy(model, data)` and `potential_energy(model,
  data)` (gravity's, `mbd/kernel/algorithms.hpp`), each force element's
  own `potential_energy`.

## Record channels over a run

<!-- example: recorder -->
```cpp
Recorder rec;
rec.add("angle", [&] { return sim.q(0); });  // any function of anything: read at every sample
rec.add("rate", [&] { return sim.v(0); });
for (int k = 0; k < 100; ++k) {
    sim.step(0.01);
    rec.sample(sim.time);
}
const std::vector<Real>& angle = rec.column("angle");   // one value per sample
rec.write_csv(path);                         // a header row, then time and the channels, 17 digits
```

- A channel is any function: a body's position (`sim.states()[b].p_WB`), a
  joint reaction from `compute_loads`, a spring's force, an energy. Add the
  channels before the first sample; names are unique.
- `add_drift_channels(rec, sim)` adds the constraint drift channels
  ([time integration](time-integration.md#a-closed-mechanism)).
- `Recorder` is in `mbd/analysis/recorder.hpp`, namespace `mbd`.

## Contact with a plane

A 2 kg ball dropped onto the ground:

<!-- example: contact -->
```cpp
ContactParams contact;
contact.stiffness = 1e6;                     // k [N/m^1.5]: force k delta^e at penetration delta
contact.exponent = 1.5;                      // e: 1.5 for a sphere (Hertz)
contact.damping = 2000.0;                    // [N s/m], reached at damping_depth
contact.damping_depth = 1e-4;                // [m]
contact.mu_static = 0.8;                     // friction, for sliding (none here)
contact.mu_dynamic = 0.6;
sys.force_elements.push_back(std::make_shared<PlaneContact>(
    ball, std::vector<Vec3>{Vec3::Zero()}, 0.1,   // a sphere of 0.1 m at the ball's origin
    0, Transform3(), contact));              // against the ground's XY plane, solid below
Simulator sim(sys);
sim.q(0) = 0.15;                             // the ball's centre 5 cm above touching
sim.initialize();
sim.run(2.0, 0.0005);
```

It bounces, settles, and rests (m g / k)^(2/3) = 0.73 mm into the ground,
where the contact carries its weight.

- The normal force is ADAMS's IMPACT law: k delta^e plus a damping that
  grows from zero to `damping` over the first `damping_depth` of
  penetration, so the force has no jump at first touch; it never pulls.
- Friction rises to `mu_static` up to the speed `v_static`, then falls to
  `mu_dynamic` beyond `v_dynamic`: a regularised law, so a body on a slope
  creeps slowly instead of sticking
  ([troubleshooting](../troubleshooting.md)).
- Several spheres per body (`centres_B`), against any body's plane
  (`plane_body`, `plane_X`, its Z axis pointing out of the solid side).
- A stiff contact needs short steps with an explicit method; the implicit
  ones carry it ([time integration](time-integration.md#a-stiff-model)).

## Act when something happens

An event is a function of the state whose zero crossing matters; the step is
shortened to the crossing, the action runs there, and the step goes on. The
ball, without contact, dropped from 1 m:

<!-- example: event -->
```cpp
Event half_way;
half_way.name = "half way";
half_way.function = [](const Simulator& s) { return s.q(0) - 0.5; };   // zero at 0.5 m
half_way.direction = -1;                     // falling crossings only
half_way.action = [&](Simulator& s) { speed_at_crossing = -s.v(0); };  // may change q and v
sim.events.push_back(half_way);
sim.run(1.0, 0.01);
// sim.event_log(): every event that fired, and when.
```

It fires once, at sqrt(2 x 0.5 / g) = 0.319 s, found to
`sim.event_tolerance` (1e-10 s).

- `direction`: -1 falling crossings only, +1 rising, 0 both. An action may
  change `q` and `v` (a bounce reverses the velocity), and the state just
  past the crossing is where it acts.
- Only the signs at the ends of each stretch of a step are compared (with
  Dormand-Prince, of each of its own steps, the crossing then found on
  its continuous extension): a function that crosses and comes back
  within one step is missed; keep the step shorter than the time it stays
  across.
- More than `max_events_per_step` within one step (an action that sends its
  function back across zero) stops the chain with MBD-K052.

## Check a model, and find what went wrong

<!-- example: checks -->
```cpp
const VecX q0 = mbd_test::FourBar::configuration(1.0);   // a configuration with the loop closed
const ValidationReport check = validate(fb.sys, q0); // before simulating: errors, warnings, counts
// check.ok(), check.degrees_of_freedom, check.redundant_equations; check.summary() as text.
const std::string text = describe(fb.sys);           // bodies, joints, constraints, forces
Simulator sim(fb.sys);
sim.q = q0;
sim.trace_steps = true;                              // a record of every step
sim.initialize();
sim.run(0.05, 0.001);
sim.write_trace(trace_path);                         // projection, residuals, energy... per step, as CSV
const std::string state = dump_state(sim);           // the state now, every coordinate named
```

The four-bar closed in 3D has 3 velocities and 5 constraint equations, 2
of them independent: 1 degree of freedom, 3 redundant equations.

- **Validate at a configuration where the constraints hold.** `validate(sys)`
  alone checks at the neutral configuration (every coordinate zero), where
  a loop may be wide open: it warns (MBD-M040), and counts the rank there,
  which can differ.
- Every message carries a code; [the catalogue](../messages.md) says what
  each means and what to do.
- The trace records, for every step, the projection's iterations and
  residuals, the drift, the constraints' rank and the energies: where a run
  went wrong, the step where they changed shows it. `describe(sys, q)`
  gives validate()'s counts at q instead of the neutral configuration;
  `dump_state` gives each body's placement and velocity and the
  constraint residuals.
- [Troubleshooting](../troubleshooting.md) goes from symptom to cause.
