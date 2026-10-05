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
