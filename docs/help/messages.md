# Messages

Every error and warning the program reports starts with a code: `MBD-`, a
letter for the area, and three digits. Search this page for the code.

| Letter | Area | How it reaches you |
|---|---|---|
| K | The kernel: building models, the algorithms, constraints, the solver, the simulator | Errors: an exception `mbd::MbdError`, whose `what()` is the message. Warnings: the diagnostic sink, see below |
| M | `kernel::validate()`: findings in a model | Lines of the report: `errors`, `warnings` and `notes`, and `summary()` |
| F | Force elements | Errors, when the element is constructed |
| A | Analysis: tracks and lap simulation | Errors |

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

### MBD-K044

`kernel::Simulator::attach: the state element (<name>) is attached already`
or `... has <n> states`

- **Means:** a state element (task 4.1) was attached to the same simulator
  twice, which would integrate its states twice over, or reports a negative
  number of states.
- **What to do:** attach each element once, after building the simulator and
  before stepping.

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

### MBD-K052

Warning: `<n> events in one step at t = <t> s; the rest of the step ignores events. An action that sends its function back across zero makes a chain that never ends unless only one direction is watched. Further chains are not reported.`

- **Means:** within one step, events fired `Simulator::max_events_per_step`
  times (100 by default), and the step finished without watching them.
- **Usual causes:** an action that moves the state back across its own
  function's zero while `Event::direction` is 0, so that every crossing
  causes the next; or two events whose actions undo each other.
- **What to do:** watch one direction (`direction = -1` or `+1`), or make
  the action move the state clear of the crossing.
- **Related:** a function that goes out and back within one step (a bounce
  shorter than the step, as a ball bouncing ever lower makes near the end)
  is not seen at all, because only the signs at the ends of each stretch are
  compared: shorten the step, or give the action a rest condition.

### MBD-K053

`Simulator::write_trace: cannot open "<path>" for writing` or `writing "<path>" failed`

- **Usual causes:** a folder in the path that does not exist, a file open in
  another program (a spreadsheet keeps CSV files locked on Windows), or a full
  disk.

### MBD-K054

`kernel::Simulator: step_control needs absolute_tolerance > 0, relative_tolerance >= 0, min_step >= 0 and max_steps >= 1; it has <values>.`

- **Means:** `Simulator::step_control`, the settings of the Dormand-Prince
  integrator (task 4.2), holds a value it cannot work with. An absolute
  tolerance of zero would divide by zero wherever an entry of the state is
  zero.
- **What to do:** keep `absolute_tolerance` positive; to control the relative
  error mainly, make it small against the entries' sizes (1e-9 for a state
  of order one).

### MBD-K055

`kernel::Simulator: <Dormand-Prince, SDIRK4 or Radau IIA> needs a step of <h> s at t = <t> s, below step_control.min_step (<min> s); the error measure of the last try was <e>. The model is stiff or discontinuous here.`

- **Means:** to meet its tolerances the integrator would have to take a step
  shorter than `step_control.min_step` (1e-12 s by default), or so short that
  the time no longer changes. The state is the last one reached, before t.
  For the implicit methods an infinite measure means Newton failed even at
  the shortened step.
- **Usual causes:** a stiff force (a very stiff spring or contact, a tyre at
  low speed on a light wheel), whose stability limit is far shorter than the
  motion needs; a force that jumps (a sign that flips with a velocity, a
  table without interpolation) or a kink taken with very tight tolerances;
  a state that has become NaN or infinite (the measure is then infinite).
- **What to do:** look at the forces at that time (`dump_state`, the step
  trace); make a jump continuous (regularise it, or decide it at the end of a
  step as the drivetrain's brake does); for a stiff model use an implicit
  method (`RadauIIA` with the same step control, or `GeneralizedAlpha`);
  loosen the tolerances if they are tighter than the model deserves.

### MBD-K056

`kernel::Simulator: <Dormand-Prince, SDIRK4 or Radau IIA> took <n> steps within one step of <dt> s and reached only t = <t> s (step_control.max_steps). The model is stiff or discontinuous here, or the tolerances are too tight for it.`

- **Means:** within one call to `step(dt)` the integrator tried more steps,
  accepted or rejected, than `step_control.max_steps` (100000 by default).
  The state is the last one reached.
- **Usual causes:** as for MBD-K055, short of reaching `min_step`: a stiff
  model takes steps at its stability limit, which may be thousands of times
  shorter than `dt`. Also a `dt` much longer than the motion's time scale.
- **What to do:** as for MBD-K055; `integrator_stats()` gives the steps and
  rejections so far, and the step length reached.

### MBD-K057

`kernel::Simulator::interpolate: t = <t> s is outside the last Dormand-Prince step (<t0> to <t1> s)` or `(none taken yet)`

- **Means:** the continuous extension exists only within the last step the
  Dormand-Prince integrator took (task 4.2). That step may be much shorter
  than the last `step(dt)`: the error control chooses it.
- **What to do:** call `interpolate` right after `step`, for times within the
  step just taken; for other times, step there.

### MBD-K058

`kernel::Simulator: the <generalised-alpha, SDIRK4 or Radau IIA> step of <h> s from t = <t> s did not converge within <n> Newton iterations, or its corrections stopped falling, even with an iteration matrix just built there.` or `...: the iteration matrix is singular.`, then `The forces change too fast across the step, or are not finite.`

- **Means:** an implicit method (tasks 4.3, 4.4; SDIRK4 and Radau IIA only
  with `step_control.adaptive` off, since with error control they shorten
  the step instead) could not solve a step: Newton's corrections did not
  fall below the tolerance within `implicit_control.max_iterations`, or
  stopped falling, even with an iteration matrix just rebuilt there (at the
  step's start; for SDIRK4 at the stage that failed); or that matrix has a
  zero pivot. The state is the one at the start of the step.
- **Usual causes:** a force that is NaN or infinite (the matrix built from
  it has no pivot); a force that jumps within the step (contact that starts,
  a friction sign, a table without interpolation), which Newton on a smooth
  model of it cannot follow; a step so long that the forces change their
  character across it (a tyre leaving the ground, a mechanism passing a
  singular position). With SDIRK4 and Radau IIA, whose stages span the
  whole step, also a step long beside how fast the stiff forces' directions
  turn: on the bushed pendulum (k = 1e8 N/m) Radau IIA solves fixed steps of
  2 ms but not of 4 ms once the links swing fast, while generalised-alpha,
  one evaluation point per step, needs shorter steps for its accuracy anyway.
- **What to do:** check the forces at that state (`dump_state`); shorten
  the step there, or use error control, which shortens it where needed;
  make a jump continuous or decide it at the end of a step; raise
  `max_iterations` only if the corrections were still falling.

### MBD-K059

`kernel::Simulator: implicit_control needs rho_infinity from 0 to 1 and max_iterations >= 1; it has <rho> and <n>.` or, from SDIRK4 and Radau IIA, `implicit_control needs max_iterations >= 1; it has <n>.`

- **Means:** a setting of an implicit method is out of range. Generalised-
  alpha's parameters follow from `rho_infinity`, and outside [0, 1] the
  method loses its stability or its order.

### MBD-K060

`kernel::assemble: cannot hold velocity <k>; the model's velocities are 0 to <n>.`
or `AssemblyOptions::hold_joint: there is no jointed body <b>; the jointed bodies are 1 to <n>.`

- **Means:** a hold names no coordinate of the model. Holds are velocity
  indices (`model.idx_v[body]` is the first of a joint's), not coordinate
  indices: they differ after a spherical or free joint, whose orientation has
  four coordinates and three velocities. The ground (body 0) has no joint.
- **What to do:** use `AssemblyOptions::hold_joint(model, body)` to hold a
  whole joint by its body.

### MBD-K061

`<function>: velocity <k> (body <b>) moves no mass, so its correction cannot be weighed (mass matrix entry <w>). validate() names such joints.`

- **Means:** assembly and statics (`kernel::assemble`,
  `kernel::static_equilibrium`) weigh each coordinate's change by its
  diagonal entry of the mass matrix, and this one is zero (or not a number):
  the joint moves nothing with mass or inertia.
- **What to do:** run `validate()` (MBD-M021 names the joint); give the bodies
  below it a mass and an inertia.

### MBD-K062

Warning, in `AssemblyReport::warnings`: `Assembly: the positions were not assembled: |phi| = <r> after <n> iterations (tolerance <tol>).`, or the same for the velocities with `|J v - nu|`.

- **Means:** the constraints could not be met with the coordinates held. The
  assembled state is still written to the simulator, the nearest that was
  found; the given one is in the report (`q_given`, `v_given`).
- **Usual causes:** held coordinates that over-determine the mechanism (then
  MBD-K063 comes with it); given values too far from any assembled
  configuration, for instance a linkage on the wrong side of a dead centre;
  constraints that cannot all be met (two drivers asking for different
  values of one quantity, links too short to close the loop).
- **What to do:** read `report.summary(model)`; release held coordinates; start
  closer to an assembled configuration; run `validate(system, q)`.

### MBD-K063

Warning, in `AssemblyReport::warnings`: `Assembly: the coordinates held in position are not independent of the constraints: with them held the constraint equations have rank <rh> instead of <r>, so <n> of them are already determined by the others. Hold fewer, or give them consistent values.`, or the same for the velocities held.

- **Means:** the held coordinates fix something the constraints and the other
  held coordinates already fix. Holding both the crank and the rocker of a
  four-bar is an example: the crank alone determines the rocker. When the
  held values happen to agree, the state is assembled and this is the only
  warning; when they do not, MBD-K062 follows.
- **What to do:** hold only the coordinates you mean to prescribe, as many as
  the mechanism has degrees of freedom at most (`report.degrees_of_freedom`).

### MBD-K064

Warning, in `AssemblyReport::warnings`: `Assembly: the smallest correction was not found (<the steps grew | no convergence> after <n> iterations, last step <s>). The positions satisfy the constraints but may be farther from the given ones than necessary.`

- **Means:** the second stage of the position assembly, which makes the
  correction the smallest, did not converge; the result of the first stage is
  kept. The state is assembled, but some coordinates may have moved more than
  they needed to.
- **Usual causes:** a correction large against the curvature of the
  constraints (the second stage converges the faster the smaller the
  correction), or a configuration near a singular position of the mechanism.
- **What to do:** usually nothing; if the assembled values matter, start
  closer to them, or hold the coordinates you care about.

### MBD-K070

Warning, in `StaticReport::warnings`: `Static equilibrium not found: largest acceleration <a> after <n> Newton iterations and <m> relaxation steps (tolerance <tol>).`

- **Means:** `static_equilibrium` stopped before the accelerations at rest
  fell below the tolerance. The positions it reached are written to the
  simulator, at rest; the given ones are in the report (`q_given`).
- **Usual causes:** a force with nothing to resist it (a body on a free or
  sliding joint with no spring, a wheel that never reaches the ground), so
  that no equilibrium exists; a relaxation budget too small for the distance
  to fall (`max_relaxation_steps`); a relaxation step too long for the
  stiffest element (`relaxation_step`, which must stay below 2 / omega for
  the highest natural frequency omega); forces that are not smooth near the
  equilibrium (a stop or a contact switching on and off).
- **What to do:** read `report.summary(model)`: the history of the largest
  acceleration shows whether Newton stalled or relaxation ran out. Check
  that every degree of freedom with a force on it has something to hold it.

### MBD-K071

Warning, in `StaticReport::warnings`: `Static equilibrium: the given positions could not be brought onto the constraints (|phi| = <r>). assemble() reports why.`

- **Means:** before looking for an equilibrium, statics moves the given
  positions onto the constraints, and could not.
- **What to do:** run `assemble(sim)` on the same state: its report explains
  the failure (MBD-K062, MBD-K063).

### MBD-K072

Note, in `StaticReport::notes`: `<n> of the <d> degrees of freedom have no stiffness at the equilibrium: it is neutral along them, and the Newton steps leave them alone, so they keep their given values but for effects of second order in the other changes.`

- **Means:** along these directions nothing pushes back and nothing pushes:
  any position along them is an equilibrium, and statics does not move along
  them. A vehicle on tyres at rest has three: its position on the ground and
  its heading, because a tyre gives no horizontal force without slip.
  "Second order": a body that both rises and pitches moves its origin
  sideways by about half the product of the two (the rigid motion of the
  free joint is an exponential), so a sedan that rises 2 cm and pitches
  1.2 mrad moves 1.3e-5 m forward.
- **What to do:** nothing, unless you expected stiffness there; then look for
  the element that should provide it.

### MBD-K073

Note, in `StaticReport::notes`: `Dynamic relaxation took <m> steps in <r> runs, where Newton's method could not reduce the accelerations.`

- **Means:** at some point no Newton step reduced the accelerations, usually
  because a force had no stiffness against it (a wheel above the ground), and
  the motion under the forces, with kinetic damping, was used to find the
  support before Newton resumed.

### MBD-K080

Warning, in `LinearModel::warnings`: `Linearisation: the operating point did not satisfy the constraints (|phi| = <r>, |J v - nu| = <s>); it was linearised at the nearest point that does.`

- **Means:** the simulator's state was not on the constraints, so the
  linearisation moved it there first (positions by the least change in the
  weights of the mass matrix's diagonal, then velocities) and linearised
  about that point, which `LinearModel::q` and `v` hold.
- **What to do:** start from an assembled state (`assemble()`) or a static
  equilibrium (`static_equilibrium()`).

### MBD-K081

Warning, in `LinearModel::warnings`: `Linearisation: the operating point is at rest but not in equilibrium (largest acceleration <a>): the model describes motion about a point the system leaves. static_equilibrium() finds one.`

- **Means:** frequencies and damping describe small motions about an
  equilibrium; about a point where the system at rest accelerates, they
  describe a linear model of a motion that does not stay there.
- **What to do:** run `static_equilibrium(sim)` before `linearize(sim)`.

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

`PlaneContact: needs radius, stiffness, damping and friction coefficients >= 0, exponent >= 1, damping depth > 0 and 0 < v_static < v_dynamic`

- **Means:** a contact parameter is out of range. The exponent of the
  penetration must be at least 1 so that the stiffness is finite at first
  contact; the damping must ramp over a depth greater than zero; the friction
  coefficient rises to its static value at `v_static` and falls to its
  dynamic one at `v_dynamic`, so the second must exceed the first.

### MBD-F041

`PlaneContact: the spheres and the plane are on the same body`

- **Means:** a contact between a body and itself, which can never act.

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
