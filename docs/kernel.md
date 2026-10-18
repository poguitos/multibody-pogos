# The multibody kernel

Plan Phase 2 (`documentation/Master_plan.md`). The kernel is the engine's
multibody core: one implementation of each joint's kinematics and one
recursive algorithm per quantity. It replaced the original `MultibodySystem`
(removed in task 2.7b), whose defects F1 to F4 all came from the same joint
kinematics being written out by hand in several places. In the kernel each
joint is described once, by its joint model, and every algorithm uses that
description.

| File | Contents |
|---|---|
| `include/mbd/spatial/spatial.hpp` | Spatial vectors, frame changes, cross products, spatial inertia, exponential maps |
| `include/mbd/kernel/joint_model.hpp`, `src/kernel/joint_models.cpp` | The joint interface and the eight joints |
| `include/mbd/kernel/model.hpp`, `src/kernel/model.cpp` | `Model` (what does not change) and `Data` (what the algorithms compute) |
| `include/mbd/kernel/algorithms.hpp`, `src/kernel/algorithms.cpp` | Kinematics, RNEA, CRBA, ABA, Jacobians, momentum, energy, configuration space |
| `include/mbd/kernel/constraints.hpp`, `src/kernel/constraints.cpp` | Markers, constraint primitives and joint closures |
| `include/mbd/kernel/constrained_dynamics.hpp`, `src/kernel/constrained_dynamics.cpp` | The constrained solve: accelerations, multipliers, redundancy, projection |
| `include/mbd/kernel/forces.hpp`, `src/kernel/forces.cpp` | Body states for the force elements, and their generalized forces |
| `include/mbd/kernel/simulator.hpp`, `src/kernel/simulator.cpp` | `System` (model, constraints, force elements) and `Simulator` |
| `include/mbd/kernel/validate.hpp`, `src/kernel/validate.cpp` | `validate()`: checks of a system before it is simulated, and its degrees of freedom |
| `src/kernel/checks.hpp`, `src/kernel/checks.cpp` | The argument checks of the entry points |
| `tests/kernel/` | Identities, invariants and closed-form cases |

The reference is R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer
2008 (RBDA below). The layout of `Model` and `Data` follows Pinocchio.

## Conventions

**Spatial vectors** are `Vec6`, ordered `[angular; linear]`.

* A motion vector *in frame A* holds the angular velocity and the velocity of
  the point at A's origin (the body point there, not A's origin itself), both
  in A's axes.
* A force vector *in frame A* holds the moment about A's origin and the force,
  both in A's axes.
* `X_AB` (a `Transform3`) places frame B in frame A: rotation `R = X_AB.q`,
  B's origin `p = X_AB.p` in A. It maps B coordinates to A coordinates.

| Function | Meaning |
|---|---|
| `motion_act(X_AB, m_B)` | motion vector given in B, expressed in A: `[R w; R v + p x R w]` |
| `motion_act_inv(X_AB, m_A)` | the inverse change |
| `force_act(X_AB, f_B)` | force vector given in B, expressed in A: `[R n + p x R f; R f]` |
| `motion_cross_motion(m1, m2)` | `m1 x m2` (RBDA's `crm`) |
| `motion_cross_force(m, f)` | `m x* f` (RBDA's `crf`) |
| `inertia_act(X_AB, I_B)` | `force_matrix(X_AB) I_B force_matrix(X_AB)^T` |

**Bodies.** Every body has a frame. Its velocity `data.v[i]` is a motion
vector in its own frame. The world is body 0 and does not move.

**Gravity** is `model.gravity`, default `(0, 0, -9.81)`: Z up, as in
ISO 8855 (decision D3).

## Joints

A joint connects a parent body P to a child body C. Two frames are fixed on
it: the parent-side joint frame Jp, placed by `X_PJ` in P, and the child-side
joint frame Jc, placed by `X_CJ` in C. At the neutral configuration the two
coincide. A joint model supplies, as functions of its coordinates q and
velocities v:

* `X_J(q)`, Jc placed in Jp;
* `S(q)`, the motion subspace: the velocity of Jc relative to Jp, in Jc, is
  `S(q) v`;
* `c(q, v) = (dS/dt) v`, with S differentiated as a matrix of Jc
  coordinates; zero when S is constant;
* `q_dot = G(q) v`;
* `q (+) dv` (`integrate`) and its inverse `q1 (-) q0` (`difference`).

Velocities are always **body-fixed**, in Jc. Coordinates and velocities need
not have the same size: a quaternion has four coordinates for three
velocities. Quaternions are stored `(x, y, z, w)`, Eigen's storage order.

The joint axis is Z of the joint frames.

| Joint | q | v | X_J | S (columns, in Jc) | c |
|---|---|---|---|---|---|
| revolute | angle | rate | `Rz(q)` | `[e_z; 0]` | 0 |
| prismatic | d | rate | `(I, d e_z)` | `[0; e_z]` | 0 |
| fixed | none | none | identity | none | 0 |
| universal | `(a, b)` | rates | `Rz(a) Rx(b)` | `[(0, sin b, cos b); 0]`, `[e_x; 0]` | `a_dot b_dot [(0, cos b, -sin b); 0]` |
| cylindrical | `(angle, d)` | rates | `(Rz(angle), d e_z)` | `[e_z; 0]`, `[0; e_z]` | 0 |
| planar | `(x, y, angle)` | rates; x, y in Jp | `(Rz(angle), (x, y, 0))` | `[0; Rz^T e_x]`, `[0; Rz^T e_y]`, `[e_z; 0]` | `[0; d(Rz^T)/dt (x_dot, y_dot, 0)]` |
| spherical | quaternion | w in Jc | `(Q, 0)` | `[I; 0]` | 0 |
| free | `(t, Q)`, t in Jp | `(w, v)` in Jc | `(Q, t)` | identity | 0 |

For the universal joint, the angular velocity in Jp is `e_z a_dot + Rz(a) e_x
b_dot`; rotated into Jc by `(Rz(a) Rx(b))^T` it becomes the S above. For the
planar joint the origin of Jc moves at `(x_dot, y_dot, 0)` in Jp, which is
`Rz(angle)^T (x_dot, y_dot, 0)` in Jc.

For the quaternion joints, `q_dot = Q (w, 0) / 2`: `u_dot = (q_w w + u x w)
/ 2`, `q_w_dot = -u . w / 2` with `u` the vector part. This is linear in the
quaternion, so it remains valid for the slightly non-unit quaternions inside
a Runge-Kutta step. The free joint adds `t_dot = R(Q) v`.

**`integrate` and `difference`.** `q (+) dv` is the configuration reached
from q by moving at the constant velocity dv for unit time. For the joints
with `q_dot = v` it is `q + dv`. The spherical joint gives `Q exp(dv)`, the
free joint `X_J exp(dv)` with the exponential of SE(3) (`exp6`), which is the
screw motion of constant body-fixed velocity. `difference` is the inverse,
through `log3` and `log6`. Both keep quaternions unit.

`kernel::integrate(model, q, v, dt, q_out)` applies `(+)` to every joint with
`dv = v dt`. A Runge-Kutta scheme does not use it: it combines the `q_dot` of
its stages linearly, as for any other ODE, and calls `normalize` once at the
end of the step. Normalizing (or retracting) the stages would cost the scheme
its order.

## Model and Data

`Model` holds the tree, the joint models, `X_PJ`, `X_CJ`, the inertias and
gravity, plus what follows from them (`X_CJ^-1`, the 6x6 forms). Bodies are
numbered in the order they are added, so a parent always has a lower index,
and every algorithm is a loop over indices. Joint models hold no state; one
instance may be shared by many joints, models and threads.

`Data` holds everything the algorithms compute, sized once for one `Model`.
After construction the algorithms do not allocate memory (to be confirmed by
the benchmarks, task 2.10). Several threads may use one `Model`, each with its
own `Data`.

## Algorithms

For body i with parent p, `iXp = motion_act_inv(liMi[i], .)` and `vJ = S_i
qd_i`:

    liMi[i] = X_PJ X_J(q_i) X_CJ^-1          oMi[i] = oMi[p] liMi[i]
    S_i     = motion_matrix(X_CJ) S_J         (the subspace in the body frame)
    v_i     = iXp v_p + vJ
    c_i     = motion_act(X_CJ, c_J) + v_i x vJ
    a_i     = iXp a_p + S_i qdd_i + c_i

**World quantities.** `body_velocity_world` returns `[R w; R v]`: the
angular velocity and the velocity of the body origin. The linear part of a
spatial acceleration is not the acceleration of any point; the acceleration
of the body origin is `R (a_lin + w x v)` (RBDA chapter 2), which is what
`body_acceleration_world` returns.

**Gravity** enters RNEA and ABA as an upward acceleration of the ground,
`a_0 = [0; -g]`. These algorithms store the result as `data.a_gf` and also the
physical acceleration, `data.a = a_gf + [0; R^T g]`.

| Algorithm | Function | Reference |
|---|---|---|
| Positions, velocities, accelerations | `forward_kinematics` | RBDA chapters 4, 5 |
| Inverse dynamics | `rnea` | RBDA chapter 5 |
| Mass matrix | `crba` | RBDA chapter 6 |
| Mass matrix, bias forces `rnea(q, v, 0)`, from the last kinematics pass | `mass_matrix`, `bias_forces` | |
| Derivative of the velocity products (Coriolis, centrifugal, gyroscopic) with respect to v | `bias_force_velocity_derivative` | RNEA linearized (D38) |
| Forward dynamics | `aba` | RBDA chapter 7 |
| Body Jacobian, world axes, body origin | `body_jacobian_world` | |
| Momentum about the world origin | `momentum_world` | |

## Constraints

The tree has no loops. A loop is closed by constraint equations `phi(q, t) =
0` between **markers**, frames fixed on bodies (`Marker{body, X_BM}`, body 0
for the ground). Each constraint returns its value, its Jacobian and the
right-hand sides of the velocity and acceleration equations,

    J v = nu,        J v_dot = gamma,
    nu = -d(phi)/dt at fixed q,
    gamma = -(dJ/dt) v - d2(phi)/dt2 at fixed q,

and needs `Data` after `forward_kinematics(q, v, 0)`: the body accelerations
at zero joint accelerations are exactly the velocity-product terms of
`gamma`.

Every closure is built from a few primitive equations (Haug 1989; Shabana,
*Computational Dynamics*, chapter 3). With p a marker origin and a one of its
axes, in world axes, and `d = p_j - p_i`:

| Primitive | phi | Equations |
|---|---|---|
| `PointCoincidence` | `p_j - p_i` | 3 |
| `Dot1` | `a_i . a_j - s(t)` | 1 |
| `Dot2` | `a_i . d - s(t)` | 1 |
| `Distance` | `(d . d - L(t)^2) / (2 L0)` | 1 |
| `NoTwist` | `x_i . y_j - y_i . x_j` | 1 |
| `JointDriver` | `q - s(t)` for a revolute or prismatic joint | 1 |

`Distance` divides by the nominal length L0 so that phi is a length, equal to
`|d| - L` to first order, without the singularity of `|d|` at zero.
`NoTwist` is zero when the relative rotation of the two markers has its axis
in their XY plane: a pure bend, which is what a constant-velocity joint
allows. The skew part of the relative rotation's XY block is `-2 sin(angle)
axis_z`, so it vanishes exactly for such rotations.

**Jacobian rows.** Every row is a sum of terms `c_w . w + c_p . v_P` over the
two bodies (w a body's angular velocity, v_P the velocity of a point P on it).
Such a term is the power of the world wrench `[c_w + P x c_p; c_p]` on the
body, so `add_wrench_row` turns it into generalized forces by walking up the
tree: `J_k += S_k^T f` for each joint k above the body, with f moved into
that body's frame. For example, for `Dot2`,
`d/dt (a_i . d) = (a_i x d) . w_i + a_i . (v_j - v_i)`.

**Closures.** The markers coincide when the joint is assembled; the joint axis
is Z, as for the tree joints.

| Closure | Built from | Equations | Relative freedom |
|---|---|---|---|
| spherical | point coincidence | 3 | 3 rotations |
| revolute | point coincidence, `z_i . x_j`, `z_i . y_j` | 5 | rotation about Z |
| universal | point coincidence, `z_i . z_j` (arms of the cross along `z_i`, `z_j`) | 4 | 2 rotations |
| cylindrical | point on the Z line, `z_i . x_j`, `z_i . y_j` | 4 | rotation about and slide along Z |
| prismatic | cylindrical, `x_i . y_j` | 5 | slide along Z |
| constant velocity | point coincidence, no twist | 4 | 2 bending rotations |

The tests check, for a free body tied to the ground by each closure, that the
velocities satisfying `J v = 0` are exactly the joint's motions.

## Constrained dynamics

`ConstraintSolver` (task 2.6) is the one place that forms and solves the
equations of a tree closed by constraints:

    M v_dot + b = tau + J^T lambda
    J v_dot     = gamma - 2 alpha (J v - nu) - beta^2 phi

with `b = rnea(q, v, 0)` (velocity products and gravity) and `J^T lambda`
the constraint forces. It uses the range-space method: with the Cholesky
factor `M = L L^T` and `Z = L^-1 J^T`, it solves
`(Z^T Z) lambda = rhs - J M^-1 (tau - b)`, since `Z^T Z = J M^-1 J^T`, and
adds `M^-1 J^T lambda = L^-T (Z lambda)` to the free accelerations; the
matrix `M^-1 J^T` itself is never formed. One kinematics pass,
`forward_kinematics(q, v, 0)`, serves the constraint equations, the mass
matrix (`mass_matrix`) and the bias forces (`bias_forces`);
`forward_dynamics_from_kinematics` takes it from a caller that has run it
already.

**Redundant constraints.** A planar loop closed in 3D, or a joint closed
twice, makes `J M^-1 J^T` singular. It is factorized as `L D L^T` with diagonal
pivoting (`RankRevealingLDLT`, shared with assembly and statics): each pivot
is the largest remaining diagonal, and the factorization stops when that falls
below 1e-10 of the first, the equations left being redundant; their
multipliers are set to zero. Stopping matters: eliminating past the rank, as
Eigen's own `LDLT` does (it is not rank-revealing), divides by pivots made of
rounding error, which was found when statics fed it a matrix with noise in
its null space (task 3.5).
`J^T lambda` and the accelerations are unique all the same, because the
dropped directions are those in which J has no rank. `info()` reports the
number of equations and the rank.

**Projection.** `project` moves q onto `phi = 0` by Gauss-Newton steps
`dq = -M^-1 J^T (J M^-1 J^T)^-1 phi`, the least change in the
kinetic-energy metric, applied through `integrate` so quaternions stay unit.
One mass matrix serves all the steps: that of the starting point or, with
`reuse_mass_matrix` (the simulator's setting), the one the last
`forward_dynamics` factorized, at the step's last stage. Any positive
definite weight gives a valid projection; the weight only decides which
nearby point of the constraint manifold is chosen. It then moves v onto
`J v = nu` the same way, with the J and nu of the last position evaluation
(nu does not depend on v). Starting a centimetre off, three steps reach
1e-16.

**Baumgarte stabilization** (`baumgarte_alpha`, `baumgarte_beta`) is off by
default; with it the constraint error obeys `phi'' + 2 alpha phi' + beta^2
phi = 0`.

**Memory.** The solver holds all its working storage, so `forward_dynamics`
and `project` allocate nothing after construction, and neither do the
algorithms. The hidden test `[alloc]` checks every kernel call in a Debug
build with `EIGEN_RUNTIME_NO_MALLOC`, where any Eigen heap allocation fails
an assertion; the CI job `kernel-no-malloc` runs it with GCC, and it has been
run with MSVC. A whole simulation step allocates nothing either (task 2.10):
`test_alloc` replaces the global `operator new` to count every allocation
while the double-wishbone sedan, with tyres and drivetrain, and a four-bar
take 100 steps, in every build; the same CI job runs it with Eigen's check
on as well. Like `Data`, a solver belongs to one thread.

One rule follows from what that check found. Eigen evaluates `x = y - A * b`
through a temporary shaped like `y`, so when `y` is a block of a dynamic
vector the temporary goes on the heap. Write `x = y; x.noalias() -= A * b;`
instead. Temporaries whose maximum size is fixed (`Mat6X`, vectors of at most
six) live on the stack.

## Forces and simulation

**Force elements** (springs, dampers, tyres, aerodynamics, anti-roll bars)
keep the interface they had before the kernel: they read world body states
(`RigidBodyState`) and add world forces at body origins, with the moment
about each origin (`RigidBodyForces`). `kernel/forces.hpp` connects them:
`body_states` fills the states from `Data`, and `generalized_forces` turns
the forces into generalized forces `tau = sum J_i^T [tau_W; f_W]` in one pass
from the leaves, without forming the Jacobians.

**The force library** (task 3.1, decision D24). Characteristics are
`Curve`s: a straight line, or a table interpolated by a monotone cubic
(PCHIP), which passes through its points without overshoot and continues as
a line beyond its ends. `SpringDamper` acts between two points: a spring curve
against the compression, a preload, a damper curve against the rate, and bump
and rebound stops beyond their clearances. `RotationalSpringDamper` acts
between two bodies about an axis; its coordinate is the twist from the
swing-twist split of the markers' relative rotation. `Bushing` acts on six
axes between two markers, each axis with its own curves. `UserForce` wraps a
function. `JointCoordinateForce` acts on the coordinate of a revolute or
prismatic joint directly (spring, preload, damper, regularised Coulomb
friction, limit stops that push but never pull); such elements go in
`System::joint_forces` and add to the generalized forces. Every element's
`law()` returns its force with the derivatives with respect to its own
coordinate and rate, for statics, linearisation and implicit integration.

**Contact** (task 3.8, decision D28, `forces/contact.hpp`). `PlaneContact`
puts spheres (radius zero: points) on one body against the XY plane of a
marker on another, the ground or any body. The normal force is ADAMS's
IMPACT function: `k delta^e + c(delta) delta_dot`, never pulling, the damping
rising along a cubic step from zero at the surface to its full value at a
chosen depth, so the force is continuous where contact begins and ends.
Friction opposes the sliding velocity of the contact point (the sphere's
lowest point) with a regularised coefficient: a cubic step from 0 at rest to
`mu_static` at `v_static`, another down to `mu_dynamic` at `v_dynamic`, then
constant. Nothing sticks exactly: below `v_static` a loaded contact creeps at
the speed where `mu(v)` balances the push. The force acts on both bodies at
the contact point, equal and opposite. `normal_law` and `friction` return
their derivatives, as every element's law does.

**`kernel::System`** is a `Model` with its loop constraints, force elements
and joint forces: what the vehicle builders fill. **`kernel::Simulator`** advances
one. It owns the state (`q`, `v`, `x`, `time`), `x` being the first-order states
of the attached state elements, and, at every evaluation, computes the
kinematics and body states, calls each state element's `begin_evaluation`
(the drivetrain hands its wheel spins to the tyres there), applies the force
elements and joint forces, adds `tau` and `force_callback`, solves the
constrained dynamics with `ConstraintSolver` from the same kinematics pass, and
asks each state element for its `rates`. RK4 carries `x` through its stages
like `q` and `v`, combines the stages' `q_dot` and normalizes once per step;
semi-implicit Euler moves q through `integrate` and `x` by explicit Euler.
After each step the state is projected onto the constraints (generalised-alpha
instead solves its steps on them: constraints per integrator, below) and each state
element's `end_step` makes its discrete changes.

**State elements** (task 4.1, decision D30, `kernel/state_element.hpp`)
replace the callbacks before the forces and after each step: an element that
owns first-order states (a drivetrain's wheel spins, a tyre's lag, an
actuator, a controller) is attached with `sim.attach(element)` and its states
are integrated by the same method and in the same stages as the bodies, so a
force that depends on them is as accurate as the integrator. The drivetrain is
one: its wheel spins are integrated states, its brakes lock a wheel at rest as
a discrete decision at the end of a step, and it shifts gear there.

**Integrators** (`Simulator::method`). `RK4` (fourth order, the default) and
`SemiImplicitEuler` (first order: velocities, then positions, the inertia's
velocity products linearly implicit and the new velocities on the
constraints, D38) take one step per `step(dt)`.
`DormandPrince` (task 4.2, decision D31) is the Dormand-Prince 5(4) pair: a
fifth-order step with an embedded fourth-order solution whose difference
estimates the error, and a fourth-order continuous extension. Within each
`step(dt)` it takes steps of its own: each step's error estimate, weighed
entry by entry over q, v and x by `absolute_tolerance + relative_tolerance
|entry|` (`Simulator::step_control`, 1e-6 both by default), must have a root
mean square of at most 1, or the step is taken again, shorter; the next
length aims at about 0.6 (safety factor 0.9 on the fifth root, growth at most
fivefold, none after a rejection). Steps end exactly at t + dt and the
length reached carries over to the next call, so the state after each call
is the integrator's own and inputs changed between calls apply from there.
Every step is projected onto the constraints, and events are watched at the
end of each and located on the continuous extension, which costs kinematics
only. `interpolate(t, q, v, x)` evaluates the extension within the last step;
`integrator_stats()` counts steps, rejections and evaluations for every
method. With `step_control.adaptive` off it takes one step per call, with no
error control, for measuring orders. The tolerances are local: on an
undamped oscillator the global error over a span T came out within 1 % of
0.287 T tol, the value derived from the method's coefficients
(`tests/kernel/test_kernel_integrators.cpp`). On the vehicles at output steps
of 1 ms it takes one step per call, seven evaluations against RK4's four.
Codes MBD-K054 to K057.

`GeneralizedAlpha` (task 4.3, decision D32) is implicit and second order,
for stiff models: the generalised-alpha method of Chung and Hulbert, stable
at any step on linear systems, its parameters set by one number,
`implicit_control.rho_infinity` (0.8 by default): the factor by which motions
far too fast for the step shrink at each step (1: the trapezoidal rule, no
damping; 0: removed within a step). Positions move through the joints'
exponential maps; the state elements' states follow the same formulas with
their rates. Each step is solved by Newton's method on an iteration matrix
from forward differences (2 nv + nx evaluations), kept from step to step and
rebuilt when Newton slows; the convergence test uses `step_control`'s
tolerances on the velocities and states, with a floor at the rounding of the
step's terms. A step costs one evaluation at its start and one per Newton
iteration: three on the sedan at 1 ms, against RK4's four. Its history
restarts where something other than its own step changed the state (q and v
together, each state element's state on its own). Constraints: projected
after each step, or, as an option, each step solved on them (constraints
per integrator, below). On a linear
system its steps equal the method's recurrence to rounding
(`tests/kernel/test_kernel_integrators.cpp`); the car
rolling at 0.5 m/s runs at 5 ms, where RK4's wheel spins chatter. Codes
MBD-K058 and K059.

`RadauIIA` and `SDIRK4` (task 4.4, decisions D33 and D34) are implicit
Runge-Kutta methods with the same step control as `DormandPrince` (their own
steps within each `step(dt)`, ending exactly at t + dt; with
`step_control.adaptive` off, one step per call). Both work on the stacked
state y = (q, v, x), quaternions carried as four numbers through the stages
and normalized once per step, with one derivative J = df/dy by forward
differences (n + 1 evaluations, n = nq + nv + nx), kept from step to step
while the Newton work a step costs, beyond that of the first step with it,
stays below rebuilding it. Their error estimates are of third order and
filtered through an iteration matrix, so that stiff components do not
inflate them; Newton stops at min(0.03, sqrt(tol)) of the tolerances
(RADAU5's rule), never below the rounding of the stage's terms and of the
state amplified by J. `RadauIIA` is the three-stage Radau IIA method of
RADAU5: fifth order, stage order 3, L-stable. Its 3n Newton equations per
iteration are transformed into one real and one complex system of n, each
iteration costing three evaluations; starting values come from the last
step's collocation polynomial. On the bushed pendulum (two links on stiff
bushings, `tests/kernel/bushed_pendulum.hpp`), with error control at 1e-6
it ends 1e-7 m from the reference after 1 s for less than half the
evaluations generalised-alpha needs for 1e-6 m. `SDIRK4` (Hairer and
Wanner's, five stages solved one at a time, fourth order) is cheaper per
step but its stage order is 1: as springs or bushings stiffen towards
joints its accuracy falls towards first order and its error estimate
understates the error (D33). It suits smooth models and stiffness in
first-order states; for stiff mechanical models use `RadauIIA` (tight
tolerances, error control) or `GeneralizedAlpha` (fixed steps, modest
accuracy). Both refuse a step Newton cannot solve with MBD-K058 at a fixed
step, and shorten it with error control. `integrator_stats()` counts their
Newton iterations and iteration matrices.

`LinearlyImplicitEuler` (task 4.5, decision D35) is the real-time
integrator: every step does the same work, one evaluation, one set of force
derivatives and one linear solve, whatever the state. It is backward Euler
on the velocities (positions moved with the new velocities), linearized
once per step in the applied forces: (M - h C - h^2 K) times the change from
an explicit Euler step equals h^2 (K (v + h a) + C a), the new velocities
on the constraints (J v+ = nu; D38), with the state elements' states solved
with it, implicit in themselves and coupled to the velocities through the forces
they share (a wheel's spin and its hub through the tyre's slip). K and C,
the derivatives of the applied generalized forces
(`Simulator::force_jacobian`), are built every step element by element:
each element's loads are differentiated with respect to its own bodies'
poses and velocities by differences of that element alone, and mapped
through the bodies' Jacobians; joint forces with respect to their
coordinates. The inertia's velocity products (Coriolis, centrifugal and
gyroscopic terms) are implicit too, through their exact derivative D
(`bias_force_velocity_derivative`, D38), which keeps a spinning body
stable: a free body's nutation is damped exactly as backward Euler damps it,
whatever its shape. Putting the new velocities on the constraints makes the
constraints' impulse do no work on them: with the change from explicit
Euler kept off them instead, a closed four-bar gained energy until it blew
up. The joints' geometry, `tau` and `force_callback` stay explicit. K,
C and the states' coupling enter by their symmetric parts (D35 says why
each is safe), D whole, so the matrix is solved by LU, its symmetric part
checked for positive definiteness. First order, L-stable on stiff springs: a
spring 1000 times too stiff for RK4 shrinks by 1 / sqrt(1 + (w h)^2) per
step, and every motion is damped by about (w h)^2 / 2 per step. On the
sedan at 1 ms it drives, brakes and stops as RK4 does (top speed within 0.1
%) at one evaluation a step; the car rolling at walking pace keeps its speed
at steps of 1 to 10 ms. A matrix that is not positive definite (negative
stiffness beyond the inertia) makes the step drop the stiffness, with MBD-K045.

**Multirate** (task 4.6, decision D36). With RK4, `Simulator::multirate`
= m > 1 integrates the state elements' states (the fast part: wheel spins,
lags) in m RK4 steps within each RK4 step of the bodies, as in Haug et
al.'s dual-rate method. A step evaluates the dynamics four times, as single
rate does, and the states' rates alone (kinematics, forces, rates) 4 m - 1
times (`integrator_stats().state_evaluations`): the dynamics, the dear
part, a sixth as often at a ratio of 6. The bodies' first stage comes first
and they are extrapolated from it over the fast steps; their other stages
then see the states the fast steps reached (fastest first). At walking pace,
where the spins hold RK4 under a millisecond, the sedan drives at 3 ms with
6 fast steps as RK4 does at 0.5 ms (top speed within 0.14 %), 2.8 times
faster. Each part must keep inside its own stability: the fast steps inside
the states', the step inside the bodies'.

**Constraints per integrator** (task 4.7, decision D37). Every evaluation
solves the loop constraints at the acceleration level, so a step's result
lies off them by the integrator's error across them, and left alone that
drift adds up (`enforce_constraints = false`: on the four-bar of the tests,
RK4 at 2 ms reaches 2.6e-5 in 20 s, semi-implicit Euler 0.46).
`enforce_constraints` (on by default) removes it: every method projects the
state after each of its steps, positions to `constraint_tolerance` (1e-10)
and velocities exactly. A projection keeps a one-step method's order
(Eich-Soellner and Fuhrer, section 5.3), and on the four-bar it kept or
improved the accuracy of RK4, Dormand-Prince, SDIRK4 and Radau IIA.
Generalised-alpha carries a history, its acceleration-like variable, which
lags the accelerations: each of its steps leaves the constraints by much
more than its error along them (in velocity, 300 times RK4's drift at the
same step), and a projection removes that from the state but not from the
history. Projected, it stays second order; with
`implicit_control.stabilized_index2` it solves each step on the constraints
instead, in the stabilized index-2 form of Gear, Leimkuhler and Gupta as
Arnold, Bruls and Cardona apply it to this method: Newton's unknowns gain,
per independent equation, a constraint force J^T kappa added to the
accelerations, which holds J v = nu at the step's end, and a position
correction M^-1 J^T eta, which holds phi = 0. Newton stops when, besides its
usual test, |phi| is within `constraint_tolerance` and |J v - nu| within
`step_control`'s absolute tolerance; the iteration matrix takes J at
Newton's first evaluation and the derivative of J v with respect to q from
the differences that build the rest of it, and eta is eliminated from it
through the factor of J M^-1 J^T, which leaves n + m unknowns (m the
constraint equations). On the four-bar the stabilized step converges at
second order (ratios 4.09 and 4.04), its positions 1.5 to 1.1 times and its
accelerations 1.7 to 1.2 times more accurate than projected at steps of 4
to 1 ms. On the sedan its positions came out 1.15 times more accurate, but a
step costs 146 us against 100 (its larger Newton matrix), and a steering
input that moves every step costs it 12 to 23 % more iterations: a shorter
projected step buys more for the same work, so it is not the default (ADAMS
makes the same choice between its I3 and SI2 formulations).

The drift channel: `constraint_drift()` gives what the last step left
before the projection (`position`, `velocity`: |phi|, |J v - nu|), the
residuals after it, and the largest of each since the simulator was built;
for the stabilized step the drift is the residual. `add_drift_channels(rec,
sim)` (`kernel/outputs.hpp`) adds them to a `Recorder`, and the step trace
has them as columns. Over 1000 simulated seconds of the four-bar and of a
heavy top held at its tip, every method kept the residuals within the
tolerance at every step and its drift from growing (the hidden case `[long]`
in `tests/kernel/test_kernel_drift.cpp`). Semi-implicit Euler and the
linearly implicit step, which took the gyroscopic terms explicitly until
D38, blew up on the top within 100 s.


Choosing a method:

| Model and need | Method |
|---|---|
| Smooth, fixed output steps | `RK4` |
| Stiff state elements only (wheel spins at walking pace), explicit steps | `RK4` with `multirate` 6 |
| Smooth, a tolerance to meet, or long output steps | `DormandPrince` |
| Stiff (stiff springs, bushings, contact, tyres at low speed), fixed steps, accuracy of the order of 1e-4 to 1e-6 | `GeneralizedAlpha` |
| As above, a closed mechanism at long steps, its accelerations and reactions wanted as accurately as one step allows | `GeneralizedAlpha` with `implicit_control.stabilized_index2` |
| Stiff, a tolerance to meet, tight accuracy | `RadauIIA` |
| Stiffness only in first-order states, or a smooth model at a tight tolerance | `SDIRK4` or `RadauIIA` |
| Real time: the same cost every step, stiff or not | `LinearlyImplicitEuler` |

**Outputs** (task 3.3, `kernel/outputs.hpp`). `compute_loads(sim)` evaluates
the accelerations at the simulator's state and returns them with the
multipliers, each constraint's wrench on each body, and each joint's reaction.
The constraint wrenches come from `ConstraintModel::add_wrenches`, which uses
the very rows the Jacobian is built from, weighted by the multipliers. A
joint's reaction is the spatial force it transmits to its child body, in the
child-side joint frame about its origin (`joint_reactions`: recursive
Newton-Euler with the external forces); it includes the joint's actuation,
whose component along the joint's motion is the generalized force there.
Energies: `kinetic_energy`, `potential_energy` (gravity), and the
`potential_energy` of each spring element. `Recorder`
(`analysis/recorder.hpp`) samples named channels, functions of anything, and
writes them as CSV.

**Drivers.** A constraint whose target depends on t drives a mechanism.
`JointDriver` holds a revolute or prismatic coordinate at s(t); its
multiplier is the force or torque the drive needs (plan task 3.2's driven
pendulum is a test). In kinematic analysis t is a parameter rather than a
time: a suspension's bump driver holds the wheel centre at its nominal
height plus t, so `Kinematics` (in `analysis/position_kinematics.hpp`)
sweeps bump travel by solving `phi(q, t) = 0` for successive t. Its
`velocities()` and `accelerations()` solve `J v = nu` and `J a = gamma` at the
solved configuration: for a mechanism with no freedom left, the first and
second derivatives of the motion with respect to t, from which a
suspension's motion ratios follow. A point is driven along a path by one
`Dot2` per coordinate, each with its own target function.

**Steering.** A steered axle with linkage suspension has a steering rack: a
light body sliding along the chassis's lateral axis, carrying the tie rods'
inner points and driven to `SteeringRack::travel` by a `JointDriver`.
`VehicleHandle::set_steering` sets that travel; the projection after the
next step moves the linkage, and each wheel's toe follows from the geometry.

## Initial conditions

Initial values rarely satisfy the constraints: a loop open by a few
millimetres, velocities that would stretch a link. `assemble(sim, options)`
(task 3.4, decision D25, `kernel/assembly.hpp`) makes the simulator's state
consistent, keeping the held coordinates exactly as given and changing the
others as little as possible:

    positions      min d^T W d   with  phi(q_given (+) d, t) = 0,  d_H = 0
    velocities     min e^T W e   with  J (v_given + e) = nu,       e_H = 0
    accelerations  the equations of motion at the assembled state

Holds are velocity indices; `AssemblyOptions::hold_joint(model, body)` holds a
whole joint. `W` is the diagonal of the mass matrix. Its entries carry each
coordinate's own units, so the result does not depend on the units chosen,
and because it is diagonal a coordinate that no constraint involves, a
pendulum hanging outside the loop or a wheel's spin, keeps its given value.
The full mass matrix, the metric of `ConstraintSolver::project`, would turn it
as well. The least change of each kind is `x_f = W_f^-1/2 B^T mu` with
`B = J_f W_f^-1/2` over the coordinates not held and `(B B^T) mu = r`: since
`B^T mu` lies in the range of `B^T` it is the least solution, whichever `mu`
solves the system. `mu` comes from the constraint solver's own rank-revealing
`L D L^T` (`RankRevealingLDLT`), so the ranks with and without the holds are
counted by the same rule as everywhere else.

The positions take two stages. Gauss-Newton with the same weights reaches the
constraints; but each of its steps is the least from where it starts, so
their sum is not the least correction. The second stage makes the whole
correction the least: at q, with `d = difference(q_given, q)`, it solves for
the least `d_new` with `J d_new = J d - phi` and steps by `d_new - d`. Its
fixed point satisfies `phi = 0` and `W d = J^T mu`, the conditions for the
least correction; it converges linearly, the faster the smaller the
correction against the curvature of the constraints. For spherical and free
joints, whose coordinates do not add, it is exact to second order in the
rotation. The accelerations and constraint forces at the assembled state come
from `compute_loads`.

The `AssemblyReport` gives the residuals before and after, the iterations,
the counts (equations, rank, degrees of freedom, and the ranks with the holds),
each coordinate's given and assembled value, the loads, and warnings with
their codes (MBD-K062 to K064): positions or velocities not assembled, held
coordinates that over-determine the mechanism (holding a four-bar's crank and
rocker), the smallest correction not found. `summary(model)` writes it all as
text. `Simulator::initialize()` still only projects, for a state already close
to the constraints.

## Static equilibrium

`static_equilibrium(sim, options)` (task 3.5, decision D26,
`kernel/statics.hpp`) finds the configuration on the constraints where the
system, at rest, does not accelerate: `a(q) = 0`, with `a` the accelerations
of the constrained dynamics at `v = 0`. Everything the simulator applies at
rest acts: gravity, force elements, joint forces, `tau`, the callbacks.
Time-dependent constraints are frozen where they are at the simulator's time:
`ConstraintSolver::quasi_static` takes `gamma` as zero, which at rest leaves
out exactly the drivers' prescribed accelerations, since statics has no
inertia.

It is Newton's method in the tangent space of the constraints. The Jacobian
`G = da/dq` comes from central differences, each perturbed configuration
projected back onto the constraints (the same Gauss-Newton projection as
assembly), so that `G` vanishes across them. The step is the least, in the
weights of the diagonal of the mass matrix, that zeroes `G step + a`: with
`S = W^1/2` and `Gs = S G S^-1`, `step = S^-1 Gs^T y` and `(Gs Gs^T) y = -S a`,
solved by `RankRevealingLDLT`. Directions with neither stiffness nor force
are therefore not moved (to first order: they shift by products of the other
changes), and the report counts them: a vehicle on tyres at rest has three,
its position on the ground and its heading, because a tyre gives no
horizontal force without slip. A line search halves the step until `|S a|`
decreases. When no step does, a force has no stiffness against it (a wheel
above the ground), and dynamic relaxation takes over: a motion driven by the
accelerations at rest, with kinetic damping (the velocities set to zero each
time the kinetic energy has passed a maximum), until those accelerations fall
below `relaxation_tolerance`; then Newton resumes. The forces are taken at
rest because damping and tyre slip forces play no part in the equilibrium and
are too stiff at low speed for an explicit step: a tyre's lateral force below
its `V_low` of 1 m/s acts as a damper of about 45 kN s/m.

The `StaticReport` gives the largest acceleration before and after and after
each iteration, the relaxation used, the degrees of freedom split into those
with stiffness and the neutral ones (from the Jacobian at the equilibrium),
each coordinate's change, the loads at the equilibrium (`compute_loads` with
the drivers frozen: constraint forces and joint reactions), and coded
messages (MBD-K070 to K073). The equilibrium is written to the simulator, at
rest.

## Linearisation

`linearize(sim, options)` (task 3.7, decision D27, `kernel/linearization.hpp`)
gives the linear model of small motions about the simulator's state, usually
a static equilibrium:

    d/dt [z; z_dot] = A [z; z_dot],   A = [[0, I], [A_q, A_v]],   dq = N z

in reduced coordinates z on the constraints. The n columns of N are columns
of the projector onto the allowed motions, orthogonal in the weights of the
mass matrix's diagonal, picked independent by the pivots of
`RankRevealingLDLT` (they need not be orthonormal). `A_q` and `A_v` are the
derivatives of the accelerations along them, by central differences with
each perturbed position projected back onto the constraints, reduced by the
weighted left inverse `(N^T W N)^-1 N^T W`. The eigenvalues of A are the
modes: `-zeta w +- i w sqrt(1 - zeta^2)` for an oscillation, real for an
overdamped or neutral motion; `LinearModel::modes` lists each complex pair
once, sorted by natural frequency, with its damping ratio and its shape in
velocity coordinates (the position part of the eigenvector, mapped by N,
largest entry 1). A state off the constraints is first moved onto them
(MBD-K080); a state at rest out of equilibrium is flagged (MBD-K081).

## Events

`Simulator::events` (task 3.9, decision D29) holds functions of the state
whose zero crossings matter: a ball's height above the ground, a gap, a
speed. At each step every function is evaluated at the start and the end;
where one has changed sign in its watched direction (`Event::direction`:
falling, rising or both), the crossing is found by the Illinois method on
the step length, each trial re-integrating from the start of the step, to
within `event_tolerance` in time (1e-10 s by default). The step is cut there,
just past the crossing, the event's action runs (it may change q and v), the
crossing is logged (`event_log()`), and the rest of the step follows, so
`step(dt)` still advances by dt. Simulations without events take the old path
and allocate nothing per step. Two limits: only the signs at the ends of each
stretch are compared, so a function that goes out and back within one step
is not seen; and a chain of events that never ends (an action that sends its
function back across zero while both directions are watched) is cut after
`max_events_per_step` with a warning (MBD-K052). With Dormand-Prince the
signs are compared at the ends of each of its own steps, and the crossing is
found on the step's continuous extension instead of by integrating again.

## Checks

**Always on.** Every entry point of the kernel checks the vectors it is
given against the model (q against nq; v, a and tau against nv) and that the
`Data` was made for the model; where a body is named, that it exists. These
checks stay on in release builds, where a wrong size would otherwise be read
out of bounds. Each costs a comparison; its message, which names the
function, the argument and both sizes, is built only when it fails. Building
a `ConstraintSolver` checks that every constraint acts on bodies of the
model, and building a `Simulator` does the same for the force elements:
both report their bodies through `bodies()`.

**`validate(system)`** (task 2.9) reports, without throwing, what would
otherwise surface later as a singular matrix, a NaN or a quietly wrong
answer, each with a message that names the body, constraint or force
element concerned:

* errors: per-body arrays out of step, bodies out of topological order, joint
  indices that do not match the joint models; inertias no real body has (not
  finite, negative mass, not symmetric, a negative principal moment,
  principal moments that break the triangle inequality); an inertia or joint
  frame changed after the body was added; a joint that moves nothing with
  mass in some direction, which makes the mass matrix singular; constraints
  or force elements on bodies that do not exist; a constraint whose markers
  are all on one body;
* warnings: constraints not satisfied at the configuration checked;
* notes: a body that floats freely (on a free joint to the ground, with no
  constraint or force element on it or on what it carries), and redundant
  constraint equations.

It also counts the degrees of freedom: the tree's velocities less the rank
of the constraint Jacobian as the solver itself sees it (the pivots of
`J M^-1 J^T`, with the same tolerance). `summary()` gives the counts and
every message as text.

**Debugging aids** (task H.4, `kernel/describe.hpp` and the simulator).
`describe(system)` writes the model as text: the counts with `validate()`'s
rank and degrees of freedom (at the neutral configuration, or at `q` with
`describe(system, q)`), and every body with its joint, coordinates,
velocities, mass and centre of mass, every constraint, force element and joint
force with the bodies it acts on. `dump_state(sim)` writes the state: q, v,
each body's world placement and velocity, the constraint residuals. With
`Simulator::trace_steps` set, every step adds a `StepRecord`: the drift the
projection corrected (positions and, since task 4.7, velocities), the
residuals after it, its iterations and whether it converged, the constraint equations and their rank, the kinetic and
gravitational energies, and the events that fired; `write_trace(path)` writes
them as CSV. A model with two drivers fighting over one coordinate shows it
in every row: rank 1 of 2, a projection that never converges, and a residual
equal to the distance between their targets.

Every message, from the checks, the solver, the simulator or `validate()`,
starts with a code (`MBD-K001`, `MBD-M021`, ...) explained in the message
catalogue, [docs/help/messages.md](help/messages.md).

## Tests

`tests/kernel/test_spatial.cpp` checks the identities of the spatial algebra
and the exponential maps on random frames and vectors.

`tests/kernel/test_kernel_invariants.cpp` runs every joint as root and as
child of every other, and two branching trees, through:

* velocities and accelerations against finite differences of the poses, and
  the Jacobian against the velocities;
* RNEA against d'Alembert's principle computed body by body from the
  kinematics alone;
* RNEA, CRBA and ABA against each other, and the kinetic energy against
  `v^T M v / 2`;
* gravity forces against the gradient of the potential energy;
* energy conservation and, for a floating tree, momentum conservation and
  straight-line motion of the centre of mass, over one second of RK4;
* a bead sliding on a spinning rod, `r = r0 cosh(w t)`;
* `integrate`, `difference` and `q_dot` against each other, and `integrate`
  against RK4 at constant velocity.

`tests/kernel/test_kernel_threads.cpp` runs one `Model` on two threads, each
with its own `Data`, and requires results identical bit for bit to a
single-threaded run.

`tests/kernel/test_kernel_constraints.cpp` checks every primitive and closure,
with constant and time-dependent targets and markers on every kind of body
pair, against finite differences of phi (first and second time
derivatives), and checks that each closure allows exactly its joint's motions.

`tests/kernel/test_kernel_constrained.cpp` checks that a body held by each
closure moves exactly like the same body on the matching tree joint, with
constraint forces equal to the tree's joint forces; that duplicated
constraints are reported as redundant and change nothing; that a planar
four-bar closed in 3D (rank 2 of 5) conserves energy over two seconds while
its crank turns twice; that projection and Baumgarte stabilization return
the state to the constraints; and, in the hidden `[alloc]` test, that nothing
is allocated.

`tests/kernel/test_kernel_analytic.cpp` holds closed-form cases: a free body
at constant velocity, in free fall, Euler's equations and the torque-free
symmetric top; a pendulum rod's force and the elliptic-integral period of a
large swing; a hinged bar's reaction. `tests/kernel/test_kernel_simulator.cpp`
checks the force bridge against `J^T f`, a spring-mass oscillator and the
convergence orders of both integrators, a constrained pendulum, the callbacks,
a driven pendulum's torque, and that a projection which cannot converge is
counted at every step and reported once.

`tests/kernel/test_kernel_joints_physics.cpp` gives each joint type a
closed-form case: the periods of pendulums on revolute, spherical and
universal joints and of a slider on a spring, a slider's acceleration on
inclines, the steady spin of a symmetric body, a fixed joint's placement, the
slow normal mode of a double pendulum, two bodies held together at a point
under a balanced load, and a height driver holding a point of a falling
body.

`tests/kernel/test_kernel_validate.cpp` gives each malformed system its own
expected message, and checks the counts on a planar four-bar closed in three
dimensions (one degree of freedom, three redundant equations); it also checks
that the algorithms, the solver and the simulator reject wrong sizes and
missing bodies. The template vehicles validate with ten degrees of freedom
whatever their suspension (`tests/vehicle/test_vehicle_template_dynamic.cpp`).

Until task 2.7b, `tests/kernel/test_kernel_vs_legacy.cpp` compared the kernel
with the original `MultibodySystem` on random chains and trees: poses,
velocities, mass matrices, energies and forward dynamics agreed to 5e-13.
Both were removed together; the analytic tests of the old joints and
algorithms live on in the files above.
