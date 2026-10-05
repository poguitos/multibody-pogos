# Messages

Every error and warning the program reports starts with a code: `MBD-`, a
letter for the area, and three digits. Search this page for the code.

| Letter | Area | How it reaches you |
|---|---|---|
| K | The kernel: building models, the algorithms, constraints, the solver, the simulator | Errors: an exception `mbd::MbdError`, whose `what()` is the message. Warnings: the diagnostic sink, see below |
| M | `kernel::validate()`: findings in a model | Lines of the report: `errors`, `warnings` and `notes`, and `summary()` |
| F | Force elements | Errors, when the element is constructed |
| A | Analysis: tracks and lap simulation, recorders and traces | Errors |

**Warnings** go through one replaceable sink, `mbd::diagnostic_sink()`. By
default it writes `[mbd warning] <message>` to standard error;
`mbd::init_logging()` (`mbd/core/logging.hpp`) sends them to the logger
instead, and a program may install its own sink.

**Errors are thrown where they are detected**, before anything is computed
with the bad input, so the stack trace points at the call that passed it.
The checks behind them stay on in release builds.

A test (`tests/core/test_message_catalogue.cpp`) checks that every code in the
sources has an entry here and every entry here is used, and that every error
and warning in the sources carries a code.

## K: the kernel

### MBD-K001

`<function>: <argument> has <n> entries, but the model has <nq or nv> = <m>.`

- **Means:** a vector passed to a kernel function has the wrong size for the
  model: coordinates must have `nq` entries; velocities, accelerations and
  forces `nv`.
- **Usual causes:** a velocity vector passed as coordinates or the reverse;
  they differ in size when the model has spherical or free joints (a
  quaternion is four coordinates for three velocities). A vector made for
  another model, or before the last bodies were added. `Simulator::q` or `v`
  resized by user code.
- **What to do:** start coordinates from `model.neutral_configuration()`,
  velocities and forces from `VecX::Zero(model.nv)`, after the model is
  complete.

### MBD-K002

`<function>: the Data was made for another model (<b> bodies, nv = <n>; this model has <b'> bodies, nv = <n'>).`

- **Means:** the working storage `kernel::Data` was sized for a different
  model.
- **Usual causes:** `Data` constructed before more bodies were added; one
  `Data` shared between two models.
- **What to do:** construct `Data data(model)` once the model is complete; one
  `Data` per model, and per thread.

### MBD-K003

`<function>: there is no body <i>; the bodies are 0 to <n>.`

- **Means:** a body index out of range. Body 0 is the ground.
- **What to do:** use the indices returned by `Model::add_body`.

### MBD-K004

`kernel::q_dot: qd must not be q itself`

- **Means:** the output vector of `q_dot` is the input vector.
- **What to do:** pass a separate vector for the result.

### MBD-K005

`kernel::generalized_forces: <n> forces given, but the model has <m> bodies.`

- **Means:** the force list needs one `RigidBodyForces` per body, the ground
  (index 0) included.

### MBD-K006

`kernel::joint_reactions: <n> external wrenches given, but the model has <m> bodies.`

- **Means:** the external wrenches need one entry per body, the ground (index
  0) included. `compute_loads(sim)` assembles them for a simulator.

### MBD-K010

`kernel::Model::add_body: parent body does not exist` or `no joint model` or
`joint has an invalid number of coordinates`

- **Means:** the body cannot be added as asked.
- **Usual causes:** a parent index from another model or not yet added (a
  parent must come before its children); a null joint model; a custom joint
  model with more than six velocities, or fewer coordinates than velocities.

### MBD-K011

`kernel::Model::add_body: negative mass`

- **What to do:** masses are zero or positive. A massless body is allowed if
  what it carries gives every joint inertia; `validate()` checks that
  (MBD-M021).

### MBD-K012

`RigidBodyInertia::from_solid_box: mass must be > 0` or `half extents must be > 0`

- **Means:** a box needs a positive mass and positive half extents. The
  arguments are half the box's dimensions, not the dimensions.

### MBD-K020

`kernel constraint: axis must be 0 (X), 1 (Y) or 2 (Z)`

- **Means:** a dot-1 or dot-2 constraint was given a marker axis other than 0,
  1 or 2.

### MBD-K021

`TimeFunction: the function and both derivatives are needed`

- **Means:** a prescribed target s(t) was given without ds/dt or d²s/dt². The
  constraint's velocity and acceleration equations need both, and a wrong
  derivative makes the motion drift from the target.
- **What to do:** give all three, or use `TimeFunction::constant(value)`.

### MBD-K022

`kernel::Distance: the nominal length must be positive`

- **Means:** the distance constraint divides by its nominal length, so that its
  value is a length to first order.

### MBD-K023

`kernel::JointDriver: no such body` or `the joint must be revolute or prismatic`

- **Means:** a joint driver prescribes one coordinate of a revolute or
  prismatic joint, given by the joint's child body.
- **What to do:** to prescribe the motion of another joint or of a point, use
  the primitives with a time-dependent target (`Dot1`, `Dot2`, `Distance`).

### MBD-K024

`kernel::ConstraintSet::add: no constraint`

- **Means:** a null pointer was added to a set of constraints.

### MBD-K026

`kernel::JointCoordinateForce: no such body` or `the joint must have one coordinate (revolute or prismatic)`

- **Means:** a joint coordinate force acts on the coordinate of a joint with
  one coordinate, given by the joint's child body.
- **What to do:** for joints with more coordinates, use force elements between
  bodies (`SpringDamper`, or a bushing when it exists).

### MBD-K027

`kernel::JointCoordinateForce: friction, stop stiffness and damping must be >= 0, the friction velocity > 0, and the lower limit not above the upper`

- **Means:** a parameter of `JointCoordinateForceParams` is out of range. The
  friction velocity is the rate over which friction builds up; it must be
  positive (see the parameter's comment for how to choose it).

### MBD-K030

`kernel::ConstraintSolver: constraint <k> is empty`

- **Means:** entry `k` of the system's constraint list is a null pointer.

### MBD-K031

`kernel::ConstraintSolver: constraint <k> (<name>) refers to body <b>, but the model's bodies are 0 to <n>.`

- **Means:** one of the constraint's markers is on a body that does not exist.
- **Usual causes:** markers built with indices of another model, or before the
  bodies were added.
- **What to do:** `validate(system)` lists every such constraint at once
  (MBD-M031).

### MBD-K040

`kernel::Simulator: force element <k> is empty`

- **Means:** entry `k` of the system's force elements is a null pointer.

### MBD-K041

`kernel::Simulator: force element <k> (<name>) refers to body <b>, but the model's bodies are 0 to <n>.`

- **Means:** the force element acts on a body that does not exist; its forces
  would be written out of bounds. See also MBD-M034.

### MBD-K042

`kernel::Simulator: joint force <k> is empty`

- **Means:** entry `k` of the system's joint forces is a null pointer.

### MBD-K043

`kernel::Simulator: joint force <k> (<name>) refers to body <b>, but the model's jointed bodies are 1 to <n>.`

- **Means:** the joint force acts on the joint of a body that does not exist.
  The ground (body 0) has no joint. See also MBD-M036.

### MBD-K050

Warning: `Redundant constraints: <m> equations of rank <r>. The motion is unaffected; the multipliers of the redundant equations are set to zero.`

- **Means:** some constraint equations repeat others at the current state. The
  solver keeps an independent subset; the motion and the total constraint
  forces are unaffected, but the forces cannot be split uniquely among the
  redundant equations. Reported once per simulator.
- **Usual causes:** a planar loop closed with a three-dimensional joint (a
  four-bar closed by a revolute: 5 equations of rank 2), or a joint closed
  twice. Both are often harmless and intended.
- **When to look closer:** when the rank is lower than the mechanism should
  have, a closure may be degenerate (aligned points, parallel axes).
  `validate()` gives the counts (MBD-M051).

### MBD-K051

Warning: `Constraint projection did not converge at t = <t> s: residual <r> after <n> iterations. Further failures are counted, not reported.`

- **Means:** after a step, the state could not be brought back onto the
  constraints to the projection tolerance (`Simulator::projection_tolerance`,
  1e-10 by default).
- **Usual causes:** contradictory constraints (two drivers asking for
  different values of one quantity); a mechanism driven through a singular
  position; a driver target that jumps; a step too long for the motion.
- **What to do:** `Simulator::projection_failures()` counts the failures and
  `last_projection()` gives the last residual. Run `validate(system, sim.q)`
  at the state of the first failure; check the drivers' targets and their
  derivatives (MBD-K021); try a shorter step.

### MBD-K060

`<function>: the held <what> include <k>, but the velocity coordinates are 0 to <nv - 1>.` From `kernel::assemble` and `kernel::static_equilibrium`; also from `AssemblySpec::hold`, for a body without a joint or a coordinate its joint does not have.

- **Means:** an assembly or a static solution was asked to hold a coordinate
  that does not exist.
- **Usual causes:** a coordinate index counted in `q` rather than in `v`
  (they differ after a spherical or free joint, whose quaternion is four
  coordinates for three velocities); body 0, the ground, which has no joint.
- **What to do:** hold by body with `AssemblySpec::hold(model, body)` or
  `hold(model, body, k)`, which find the indices; by hand, the coordinates of
  body `i` are `model.idx_v[i]` to `model.idx_v[i] + model.nvs[i] - 1`.

### MBD-K061

`<function>: the held <what> hold part of the <spherical or free> joint of <body> (<n> of its 3 angular coordinates, <m> of its linear ones). Hold its rotation whole, or not at all; ...` From `kernel::assemble` and `kernel::static_equilibrium`.

- **Means:** a rotation's three angular velocity coordinates are not
  independent angles: a correction applied about one axis and not the others
  does not leave "the held angle" unchanged. Assembly and statics therefore
  hold a rotation whole or not at all. A free joint's translation is measured in its
  child-side axes, which turn with the body, so it can be held in part only
  while the rotation is held.
- **What to do:** hold all three angular coordinates, or none. To fix a body's
  height while it may rotate, use a constraint (a `Dot2` driver on the height)
  instead of a hold.

### MBD-K062

Report error: `The positions could not be assembled: |phi| is <r> after <n> Gauss-Newton steps (tolerance <tol>); the largest residual is in <constraint>, <value>. The velocities were left as given.`

- **Means:** no configuration near the given one satisfies the constraints
  with the held coordinates at their values. The state returned is the
  closest the iteration reached (the least `|phi|`); velocities and
  accelerations were not assembled.
- **Usual causes:** held coordinates that contradict the constraints (see
  MBD-K065, which then appears too); a loop that cannot close at all (links
  too short for the distance between their ends); a driver whose target is
  out of reach; a start so far from any closed configuration that the
  iteration stalls.
- **What to do:** hold fewer coordinates, or give the held ones consistent
  values; check the named constraint's markers; start from values nearer the
  intended configuration; `validate(system, q)` lists the constraints that are
  not satisfied.

### MBD-K063

Report error: `The velocities could not be assembled: |J v - nu| is <r> with the velocities held (tolerance <tol>).`

- **Means:** the held velocities ask for a motion the constraints do not
  allow, for instance two rates of a loop with one degree of freedom whose
  ratio is fixed by its geometry.
- **What to do:** hold as many velocities as the mechanism has degrees of
  freedom (`AssemblyReport::degrees_of_freedom`), and only ones that are
  independent; MBD-K065 names the conflict.

### MBD-K064

Report error: `The accelerations could not be assembled: |J a - gamma| is <r> with the accelerations held (tolerance <tol>).`

- **Means and what to do:** as MBD-K063, for the accelerations.

### MBD-K065

Report warning: `The <n> held <level> take <k> of the <r> independent constraint directions away from the coordinates left free: the held values must satisfy <k> constraint equation(s) by themselves.`

- **Means:** the coordinates left free cannot satisfy every constraint
  equation on their own. The held values must then satisfy the rest exactly,
  which they do only by chance or by construction. If they do not, MBD-K062,
  K063 or K064 follows.
- **Usual causes:** more coordinates held than the mechanism has degrees of
  freedom (all the angles of a closed loop, say); a held coordinate whose
  value the constraints alone determine (the angle of a driven joint).
- **What to do:** hold no more coordinates than `degrees_of_freedom`, chosen
  so that together they fix the mechanism (the crank of a four-bar, not its
  crank and rocker). When the held values come from a consistent
  configuration, the warning can be ignored.

### MBD-K066

Report note: `The held <level> leave <f> degree(s) of freedom; along them the assembly changed the given <level> as little as possible, in the kinetic-energy metric.`

- **Means:** the held coordinates do not fix the mechanism completely, so the
  assembly chose, among all the configurations (or velocities, or
  accelerations) that satisfy the constraints, the one nearest to the given
  values, weighted by the mass matrix. Light parts are moved more than heavy
  ones: a four-bar given all its angles slightly wrong may have its light
  crank turned furthest.
- **What to do:** nothing, if that is what was meant. Otherwise hold the
  coordinates that define the intended state (as many as
  `degrees_of_freedom`).

### MBD-K067

Warning: `Simulator assembly at t = <t> s did not succeed; the state is the closest it reached. <the report's errors>`

- **Means:** `assemble(sim, spec)` could not assemble the simulator's state;
  the codes that follow say which level failed and why. A simulation started
  from this state begins with a jump, as the first projection pulls it onto
  the constraints.
- **What to do:** as for the codes it contains. The full report, with the
  coordinate that changed most at each level, is the return value of
  `assemble`, and its `summary(model)` prints it.

### MBD-K070

Report error: `No static equilibrium found: the largest acceleration at rest is <a> (tolerance <tol>), in <coordinate>, after <n> Newton iterations and <m> relaxation steps.`

- **Means:** `static_equilibrium` stopped without reaching a state where
  nothing accelerates at rest. The simulator holds the best state reached.
- **Usual causes:** no equilibrium exists: a force with nothing to balance it
  (a vehicle on a slope with free-rolling tyres, a constant force on a
  sliding joint with no spring); a body that must fall further than the
  relaxation budget allows (`StaticsOptions::relaxation_max_steps`); a
  relaxation step too long for the stiffest motion, which makes it blow up
  (`relaxation_step`, below 2 / the highest natural frequency); a force law
  with a kink exactly at the equilibrium.
- **What to do:** check that every free direction has something to hold it,
  or hold it (`StaticsOptions::hold`); look at the coordinate named and at
  `history`, which shows whether Newton was converging; shorten the
  relaxation step or raise its budget.

### MBD-K071

Report error: `Statics could not start: the configuration could not be brought onto the constraints. Assemble it first (kernel::assemble), which reports why.`

- **Means:** the starting configuration is too far from any configuration
  that satisfies the constraints (with the held coordinates at their values).
- **What to do:** `kernel::assemble(sim)` with the same holds; its report
  names the constraint that cannot be met (MBD-K062).

### MBD-K072

Report warning: `The equilibrium is unstable: the stiffness is negative in <n> direction(s), so the smallest disturbance moves the system away from it (a pendulum balanced upright).`

- **Means:** the state found is an equilibrium, but not a resting state: the
  forces push away from it along `n` directions. Newton's method finds the
  nearest equilibrium, stable or not.
- **What to do:** if a resting state was meant, start nearer to it, or
  disturb the result slightly and run `static_equilibrium` again; a
  simulation from a slightly disturbed state shows where the system goes.
  For conservative forces the count is exact; with forces that are not (a
  follower force), it is taken from the symmetric part of the stiffness and
  is a guide only.

### MBD-K073

Report note: `<n> of the <d> degrees of freedom have no stiffness here: nothing pushes back along them (a vehicle's position and heading on a flat road), and statics left them as they were.`

- **Means:** along these directions neither force nor stiffness acts, so any
  position along them is an equilibrium; statics did not move them.
- **What to do:** nothing, when that is the physics (a car is free to roll
  and turn on a flat road). If a direction should be held (a body that
  should rest on a spring), check that the element acting along it exists
  and is engaged.

### MBD-K074

Report note: `Newton could not reduce the accelerations <n> time(s); dynamic relaxation took over for <m> steps.`

- **Means:** at least once, no step along Newton's direction reduced the
  accelerations: usually because a force acted where nothing was stiff yet (a
  body above the ground it will rest on), sometimes because of a kink in a
  force law. Dynamic relaxation (motion under the forces, stopped at each peak
  of kinetic energy) brought the system nearer, and Newton finished.
- **What to do:** nothing; it is information. Starting nearer to the
  equilibrium avoids it and is faster.

### MBD-K080

Warning in a linearisation: `The operating point is not at rest in equilibrium (largest acceleration <a>, largest velocity <v>): A describes the motion near it, but its eigenvalues are not modes of vibration about it. Run static_equilibrium first.`

- **Means:** `linearize` was called at a state that is moving or accelerating.
  The matrices are still the derivatives of the state equations there, which
  is what an implicit integrator or a controller design about a trajectory
  needs; but "natural frequency" and "damping ratio" describe oscillations
  about a state the system stays at, and this one it does not.
- **What to do:** for modes, call `static_equilibrium(sim)` first and
  linearise the state it leaves.

### MBD-K081

Note in a linearisation: `<n> of the <d> undamped modes have zero frequency: directions without stiffness (a vehicle's position and heading on a flat road). In A they appear as zero or real eigenvalues.`

- **Means:** nothing pulls the system back along these directions. In A,
  each appears as a zero eigenvalue, or as a real negative one when
  something damps motion along it (a tyre's slip at low speed acts as a
  viscous damper sideways).
- **What to do:** nothing, when that is the physics. Otherwise check the
  element that should provide the stiffness (MBD-K073 in statics says the
  same).

### MBD-K082

Warning in a linearisation: `<n> eigenvalue(s) of A have a positive real part: motion near the operating point grows. At an equilibrium, it is unstable.`

- **Means:** a small disturbance grows exponentially, at the rate of the
  eigenvalue's real part. At an equilibrium, this is an unstable one
  (statics reports it as MBD-K072 when the cause is negative stiffness; a
  negative damping, from a force that feeds energy in, shows here only).
- **What to do:** look at the modes with `eigenvalue.real() > 0` and their
  shapes, which name the coordinates involved.

### MBD-K090

`kernel::Simulator: event <i> (<name>) has no function.`

- **Means:** an `Event` in `Simulator::events` was added without its
  `function`, so there is nothing to watch.
- **What to do:** give every event a function of the simulator's state,
  `[](const Simulator& s) { return ...; }`.

### MBD-K091

Warning: `More than 100 events in one step at t = <t> s: the events are chattering. The rest of the step was taken without looking for events. Further cases are not reported.`

- **Means:** events followed each other a tolerance apart: an action that
  makes its own event's function cross again at once. The classic case is a
  switched (unregularised) friction at rest: each switch reverses the net
  force and the velocity crosses zero again. The rest of the step was taken
  without events, so whatever the actions maintain (the friction's
  direction) is no longer updated, and the motion after this point is not
  to be trusted.
- **Usual causes:** a discontinuous law modelled with events where a
  regularised one belongs (friction: use the regularised friction of
  `JointCoordinateForce` or `PlaneContact`); an event with direction 0 (both
  ways) whose action leaves the state on the switching surface.
- **What to do:** regularise the law, or give the event a direction, or a
  hysteresis (two events at different thresholds).

### MBD-K100

Trace diagnosis, warning: `The projection onto the constraints failed in <n> of <m> steps, first at t = <t> s, leaving |phi| up to <r>: the constraints cannot be met there (...)`

- **Means:** in the steps counted the state could not be put back onto the
  constraints (MBD-K051 is the same seen live).
- **What to do:** `validate(system, q)` at the first failure's state, from
  `dump_state(sim)` or by stepping to it; check the lengths and markers of the
  loop named there, the drivers' targets, and whether the mechanism passes a
  singular position.

### MBD-K101

Trace diagnosis, warning: `The energy balance drifts by <e> J (<f> of the energy exchanged), largest at t = <t> s: the integration or the projection makes or loses energy. Usually the step is too long for the fastest motion.`

- **Means:** kinetic + gravity's potential - the work of the applied and
  constraint forces should stay at its first value; it moved by more than
  1e-4 of the energy that changed hands during the run (event actions, such as
  an impact's rebound, are left out).
- **Usual causes:** a step too long for the stiffest element (a contact, a
  bushing, a tyre: the step should be well under 1 / its frequency);
  projections that move the state a long way each step (see MBD-K104).
- **What to do:** halve the step and compare; find the stiff element with
  `linearize` (its highest mode).

### MBD-K102

Trace diagnosis, note: `The energy's peaks grew by <e> J from the first quarter of the run to the last, and the applied forces did <w> J of work: a driver or a motor does that, and so does a force element with the wrong sign (a damper that pushes). ...`

- **Means:** something applied keeps putting energy in. Legitimate for a
  motor, a driver or a force that pushes on purpose; otherwise a sign error.
- **What to do:** if nothing should drive the system, look for a damper, a
  friction or a user force with the wrong sign (`law()` of each element
  gives its force against its rate).

### MBD-K103

Trace diagnosis, note: `<n> of the <m> constraint equations are redundant, from t = <t> s (see MBD-K050).`

- **Means and what to do:** as MBD-K050, found in the trace.

### MBD-K104

Trace diagnosis, warning: `Within a step the constraints drift by up to <d> before the projection (at t = <t> s): the step is long for the motion, and the dynamics within it were computed off the constraints.`

- **Means:** after the integrator's step and before the projection, |phi|
  exceeded 1e-6. The projection then moved the state back, which changes its
  energy and hides the error; the dynamics of the step were computed away
  from the constraint surface.
- **What to do:** a shorter step; if the drift comes with MBD-K100, the
  constraints cannot be met at all.

## M: findings of `validate()`

`validate(system)` checks a system at its neutral configuration, or
`validate(system, q, t)` at a given state, and reports without throwing.
Errors mean the system cannot be simulated as it is; warnings that it can,
but probably not as meant; notes are facts worth knowing.

### MBD-M001

Error: `The model's per-body arrays do not all have one entry per body: the model was changed other than through Model::add_body.`

- **Means:** the per-body vectors of `kernel::Model` were edited by hand.
  Build models through `add_body` only.

### MBD-M002

Error: `body <i> (<name>): its parent is body <p>, which does not come before it. Bodies must be in topological order, every parent before its children.`

### MBD-M003

Error: `body <i> (<name>): it has no joint model.`

### MBD-M004

Error: `body <i> (<name>): the coordinate and velocity indices of its joint do not match the joint model.`

- **Means:** `idx_q`, `idx_v`, `nqs` or `nvs` were edited by hand.

### MBD-M005

Error: `The model's totals (nq = <a>, nv = <b>) do not match its joints (<c> coordinates, <d> velocities).`

### MBD-M006

Error: `Gravity is not a finite vector.`

### MBD-M010

Error: `body <i> (<name>): its mass, centre of mass or rotational inertia is not a finite number.`

- **Usual causes:** a NaN or infinity from a division by zero where the
  inertia was computed.

### MBD-M011

Error: `body <i> (<name>): its mass is negative (<m> kg).`

### MBD-M012

Error: `body <i> (<name>): its rotational inertia is not symmetric.`

### MBD-M013

Error: `body <i> (<name>): its rotational inertia has a negative principal moment (<value> kg m^2).`

- **Usual causes:** products of inertia with the wrong sign convention (the
  inertia tensor has minus the products of inertia off the diagonal), or the
  parallel-axis theorem applied the wrong way.

### MBD-M014

Error: `body <i> (<name>): its principal moments of inertia (<a>, <b>, <c> kg m^2) break the triangle inequality: no real body has them.`

- **Means:** for any real body the two smaller principal moments add up to at
  least the largest. Moments typed in by hand are the usual source.

### MBD-M015

Error: `body <i> (<name>): its inertia was changed after the body was added, and the algorithms still use the old one. Give the inertia to Model::add_body.`

- **Means:** `model.inertia[i]` was edited, but the spatial inertia the
  algorithms use was computed when the body was added.

### MBD-M016

Error: `body <i> (<name>): its child-side joint frame X_CJ was changed after the body was added, and the algorithms still use the old one.`

### MBD-M020

Error: `The mass matrix is not finite at this configuration.`

### MBD-M021

Error: `body <i> (<name>): its joint can move in a direction in which neither this body nor the bodies it carries have mass or inertia, so the mass matrix is singular.`

- **Usual causes:** a massless body at the end of a chain on a sliding or
  rotating joint; a spherical joint carrying a point mass (no inertia about
  the axes through the point).
- **What to do:** give the body, or a body it carries, mass or inertia in that
  direction, or replace the joint by one with fewer freedoms.

### MBD-M022

Error: `The mass matrix is not positive definite at this configuration (smallest eigenvalue <a>, largest <b>).`

- **Means:** as MBD-M021, but no single joint could be named.

### MBD-M030

Error: `Constraint <k> is empty (a null pointer).`

### MBD-M031

Error: `constraint <k> (<name>): it refers to body <b>, but the model's bodies are 0 to <n>.`

- See MBD-K031.

### MBD-M032

Error: `constraint <k> (<name>): all its markers are on body <i> (<name>), so it cannot restrain anything.`

- **Usual causes:** the same body index given for both markers.

### MBD-M033

Error: `Force element <k> is empty (a null pointer).`

### MBD-M034

Error: `force element <k> (<name>): it refers to body <b>, but the model's bodies are 0 to <n>.`

### MBD-M035

Error: `Joint force <k> is empty (a null pointer).`

### MBD-M036

Error: `joint force <k> (<name>): it refers to body <b>, but the model's jointed bodies are 1 to <n>.`

### MBD-M040

Warning: `The constraints are not satisfied at this configuration: the largest residual, <r>, is in constraint <k> (<name>). Simulator::initialize moves the bodies onto the constraints.`

- **Means:** the configuration checked is not assembled. `Simulator::initialize`
  projects it onto the constraints; if the residual is large, check the
  geometry first: the projection finds the nearest assembled state, which may
  not be the one meant.

### MBD-M041

Error: `The configuration has <n> entries, or entries that are not finite; the model has nq = <m>.`

### MBD-M050

Note: `body <i> (<name>) floats freely: it is on a free joint to the ground, and no constraint or force element acts on it or on the bodies it carries.`

- **Means:** under gravity this body simply falls. Usually a connection or a
  force element was forgotten; for a probe or a projectile it is intended.

### MBD-M051

Note: `<n> of the <m> constraint equations are redundant at this configuration (a loop closed twice, or a planar loop closed in three dimensions). The solver drops them and gives them no force.`

- See MBD-K050.

## F: force elements

### MBD-F001

`SpringDamper parameters must be non-negative`

- **Means:** a `LinearSpringDamper` needs stiffness, damping and rest length of
  zero or more.

### MBD-F002

`Curve::table: x and y need the same length, at least 2`, `the points must be finite` or `x must be strictly increasing`

- **Means:** a tabulated characteristic needs at least two finite points with
  strictly increasing abscissae.
- **Usual causes:** a repeated x value, or a table sorted the wrong way.

### MBD-F003

`SpringDamper: free length and stop clearances must be >= 0`

- **Means:** a stop clearance is the travel before the stop engages; leave it
  at its default (infinite) for no stop.

### MBD-F004

`UserForce: no function given`

- **Means:** a user force needs the function that computes its forces.

### MBD-F010

`TireContactForce: stiffness must be >= 0`, `damping must be >= 0` or
`free radius must be > 0`

### MBD-F011

`FullTireForce: free radius must be > 0`, `vertical stiffness must be >= 0` or
`vertical damping must be >= 0`

### MBD-F020

`AerodynamicForce: CdA must be non-negative`, `ClA must be non-negative` or
`air_density must be positive`

- **Means:** drag and downforce areas are zero or positive (downforce is
  counted positive), and the air density positive.

### MBD-F030

`AntiRollBar: stiffness must be >= 0` or `damping must be >= 0`

### MBD-F040

`PlaneContact: stiffness, damping and friction must be >= 0, the exponent >= 1, and the damping depth and slip speed > 0`

- **Means:** a contact parameter is out of range. An exponent below 1 gives an
  infinite stiffness at first touch; a zero damping depth or slip speed
  divides by zero.
- **What to do:** 1 for a linear penalty, 1.5 for Hertz's sphere; a damping
  depth of the order of the static depth (0.1 mm for a stiff contact); a
  slip speed small against the speeds of interest (1 mm/s), with a time step
  short enough for it (see the contact section of the kernel page).

### MBD-F041

`PlaneContact: no contact points given`, or `a contact point is not finite, or its radius is negative`

- **Means:** the list of contact spheres is empty or malformed.
- **What to do:** give at least one `ContactSphere`, with radius 0 for a
  point.

## A: analysis

### MBD-A001

`Track::add_straight: length must be positive`, and the like for `add_arc`,
`add_arc_by_angle` and `add_clothoid`

- **Means:** a track segment needs a positive length, a non-zero radius and a
  non-zero sweep angle. A positive radius or sweep turns left.

### MBD-A002

`Track::from_polyline: need at least 2 points` or `zero-length segment`

- **Usual causes:** repeated points in the input.

### MBD-A003

`Track::query: empty track`

- **Means:** the track has no segments yet.

### MBD-A010

`sample_vmax_profile: n_samples must be >= 2` or `simulate_lap: n_samples must be >= 2`

### MBD-A020

`Recorder::add: channel "<name>" needs a name not used before, and a function`

### MBD-A021

`Recorder::add: channel "<name>" added after sampling began; add every channel first`

- **Means:** every channel must have a value at every sample, so all channels
  are added before the first `sample()`.

### MBD-A022

`Recorder::column: no channel named "<name>"`

- **What to do:** `names()` lists the channels.

### MBD-A023

`Recorder::write_csv: cannot open "<path>" for writing` or `writing "<path>" failed`

- **Usual causes:** a folder in the path that does not exist, a file open in
  another program (a spreadsheet keeps CSV files locked on Windows), or a full
  disk.

### MBD-A030

`write_trace_csv: cannot open <path> for writing`, `read_trace_csv: cannot open <path>`, `... does not start with the trace's header`, or `a row of <path> has <n> fields, not <m>`

- **Means:** a step trace could not be written or read back.
- **Usual causes:** a folder that does not exist; a file that is not a trace
  written by `write_trace_csv` (the header must match exactly); a file cut
  short.
- **What to do:** write traces with `write_trace_csv` and read them with
  `read_trace_csv` unchanged; a spreadsheet that saves the file again may
  change its format.
