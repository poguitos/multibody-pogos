# The kinematics kernel

Plan Phase 2 (`documentation/Master_plan.md`). The kernel replaces the legacy
`MultibodySystem` algorithms with one implementation of each joint's
kinematics and one recursive algorithm per quantity. Defects F1 to F4 of the
review all came from the same joint kinematics being written out by hand in
several places; in the kernel each joint is described once, by its joint
model, and every algorithm uses that description.

| File | Contents |
|---|---|
| `include/mbd/spatial/spatial.hpp` | Spatial vectors, frame changes, cross products, spatial inertia, exponential maps |
| `include/mbd/kernel/joint_model.hpp`, `src/kernel/joint_models.cpp` | The joint interface and the eight joints |
| `include/mbd/kernel/model.hpp`, `src/kernel/model.cpp` | `Model` (what does not change) and `Data` (what the algorithms compute) |
| `include/mbd/kernel/algorithms.hpp`, `src/kernel/algorithms.cpp` | Kinematics, RNEA, CRBA, ABA, Jacobians, momentum, energy, configuration space |
| `tests/kernel/` | Identities, invariants, and the cross-check against the legacy path |

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
| Forward dynamics | `aba` | RBDA chapter 7 |
| Body Jacobian, world axes, body origin | `body_jacobian_world` | |
| Momentum about the world origin | `momentum_world` | |

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

`tests/kernel/test_kernel_vs_legacy.cpp` builds the same random models in the
kernel and in the legacy `MultibodySystem` and compares poses, velocities,
mass matrices, energies and the accelerations of forward dynamics. The legacy
path parameterizes spherical and free joints by a rotation vector r with
`w = E(r) r_dot`; with `qd_legacy = T v` the mass matrices satisfy
`M = T^T M_legacy T` and the forces `tau = T^T tau_legacy`. This test goes
with the legacy path (task 2.7).
