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
pivoting, which puts the largest pivots first; pivots below 1e-10 of the
largest are dropped and the multipliers of their equations set to zero.
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

**`kernel::System`** is a `Model` with its loop constraints and force
elements: what the vehicle builders fill. **`kernel::Simulator`** advances
one. It owns the state (`q`, `v`, `time`) and, at every evaluation of the
accelerations, computes the kinematics and body states, calls
`pre_force_callback` (the drivetrain hands wheel spins to the tyres there),
applies the force elements, adds `tau` and `force_callback`, and solves the
constrained dynamics with `ConstraintSolver`, from the same kinematics pass. RK4 combines the stages'
`q_dot` and normalizes once per step; semi-implicit Euler moves q through
`integrate`. After each step the state is projected onto the constraints.

**Drivers.** A constraint whose target depends on t drives a mechanism.
`JointDriver` holds a revolute or prismatic coordinate at s(t); its
multiplier is the force or torque the drive needs (plan task 3.2's driven
pendulum is a test). In kinematic analysis t is a parameter rather than a
time: a suspension's bump driver holds the wheel centre at its nominal
height plus t, so `Kinematics` (in `analysis/position_kinematics.hpp`)
sweeps bump travel by solving `phi(q, t) = 0` for successive t.

**Steering.** A steered axle with linkage suspension has a steering rack: a
light body sliding along the chassis's lateral axis, carrying the tie rods'
inner points and driven to `SteeringRack::travel` by a `JointDriver`.
`VehicleHandle::set_steering` sets that travel; the projection after the
next step moves the linkage, and each wheel's toe follows from the geometry.

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
