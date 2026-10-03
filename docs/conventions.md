# Conventions

One page for everything that must be the same everywhere: units, frames, signs and names. When code and this page disagree, one of them is a bug.

## Units

SI throughout: metres, kilograms, seconds, radians, newtons. Angles in degrees appear only at the boundary (input files, plots) and are converted with `deg2rad` and `rad2deg`.

## Frames and transforms

- `W` is the world frame. A body frame is named after its body.
- `X_AB` is the transform that takes coordinates in frame `B` to coordinates in frame `A`: `x_A = X_AB * x_B`. Its translation `X_AB.p` is the origin of `B` expressed in `A`, and its rotation `X_AB.q` rotates vectors from `B` to `A`.
- Transforms compose right to left: `X_AC = X_AB * X_BC`.
- Quaternions are Hamilton quaternions (Eigen's default). The constructor is `Quat(w, x, y, z)`.
- `Transform3` stores a quaternion and a translation. Rotation matrices are derived from it on request.

## Bodies

- Body 0 is ground: identity pose, zero velocity, never integrated.
- Bodies are stored in topological order. A body's parent always has a smaller index.
- `RigidBodyState` holds, for body frame `B`:
  - `p_WB`: origin of `B` in world coordinates;
  - `q_WB`: orientation of `B`;
  - `v_WB`: velocity of the origin of `B`, in world coordinates;
  - `w_WB`: angular velocity of `B`, in world coordinates.
- `RigidBodyInertia` holds the mass, the centre of mass `com_B` in the body frame, and the inertia tensor about the centre of mass, in body axes.
- `RigidBodyForces` holds a force `f_W` applied at the **body origin** and a torque `tau_W` **about the body origin**, both in world coordinates. A force acting at another point contributes its moment about the origin to `tau_W`.

## Joints

A joint connects a parent body to a child body and supplies the generalized coordinates of the child.

- Each joint has two fixed frames: `X_PJ` (joint frame in the parent body) and `X_CJ` (joint frame in the child body). The motion across the joint is `X_J(q)`, and the child pose is `X_PC = X_PJ * X_J(q) * X_CJ^-1`.
- The joint axis is the local Z axis of the joint frame. A joint about another direction is obtained by rotating `X_PJ` and `X_CJ`.
- The motion subspace `S(q)` maps `q_dot` to the relative velocity of the child, as a six-vector `[angular; linear]`, **expressed in the child-side joint frame** (the frame after `X_J(q)` has been applied). Every algorithm rotates `S` to the world with `R_WP * R_PJ * R_J(q)`.
- `bias_acceleration(q, q_dot)` returns the time derivative of `S` times `q_dot`, in the same frame.

| Joint | Coordinates |
|---|---|
| Revolute | angle about joint Z |
| Prismatic | displacement along joint Z |
| Universal | angle about joint Z, then angle about the rotated X |
| Spherical | rotation vector (exponential map), kept at norm ≤ π |
| Free | translation in the parent-side joint frame, then rotation vector |
| Fixed | none |

## Constraints

A constraint closes a loop that the joint tree cannot represent.

- `evaluate` returns the position-level residual `Phi`, zero when satisfied.
- `jacobian` returns two blocks, one per body, each with six columns ordered `[linear, angular]`, such that `dPhi/dt = J1 * [v1; w1] + J2 * [v2; w2]` with the body-origin velocities above.
- `velocity_bias` returns the part of `d2Phi/dt2` that does not multiply the body accelerations, so that `d2Phi/dt2 = J1 * [a1; alpha1] + J2 * [a2; alpha2] + gamma`.
- The equations of one constraint must be independent. A constraint that removes two degrees of freedom has two equations, not three of rank two.

Note the ordering: joints use `[angular; linear]`, constraints use `[linear, angular]`. This is historical and will be unified when the kernel is rewritten (plan, Phase 2).

## World axes and vehicle axes

**Decided on 2 October 2026: ISO 8855.**

| | ISO 8855 (target) | Code today |
|---|---|---|
| X | forward | forward |
| Y | left | up |
| Z | up | labelled "left" |
| Gravity | along −Z | along −Y |

With X forward and Y up, a right-handed frame has +Z pointing to the right of the vehicle. The present code calls +Z "left", so it is a mirror image of the physical car: internally consistent, but wrong the moment real hardpoints, track data or CFD results are imported.

Until task 7.1 of the master plan is done, the vehicle layer (tyre, aerodynamics, anti-roll bar, vehicle builders, equilibrium, and their tests) still uses the legacy frame, and "up is world Y" is written into those files. The core (joints, constraints, dynamics, integration) assumes nothing: it takes gravity as a vector.

Signs in the ISO 8855 frame, which the migration will establish:

- Steering angle, yaw angle and yaw rate are positive counter-clockwise seen from above, that is, for a left turn.
- Lateral acceleration is positive to the left.
- Roll is positive when the right side goes down. Pitch is positive nose down.
- The wheel frame has X along the intersection of the wheel plane with the road, Z along the road normal, Y to the left.
- Longitudinal slip is positive under traction. The slip angle is defined as `alpha = -atan(v_y / |v_x|)`, with `v_y` the lateral velocity of the contact point, so that a positive slip angle produces a positive (leftward) lateral force. This is Pacejka's sign convention, not the ISO 8855 one, and matches the tyre model.
- Camber is positive when the top of the wheel leans outward from the car.

## Naming in code

- Types are `PascalCase`, functions and variables are `snake_case`, constants start with `k`.
- A quantity carries its frames in its name: `p_WB`, `R_WJ`, `X_PJ`.
- Classes ending in `CoordJoint` are joints of the tree. Classes derived from `Constraint` are loop closures. (Two constraint classes are still called `RevoluteJoint` and similar; they are renamed in plan task 2.7.)

## Decisions on record

| # | Decision | Status |
|---|---|---|
| D1 | One kinematics kernel based on spatial vectors | Adopted, Phase 2 |
| D2 | Quaternion coordinates for free and spherical joints | Adopted, Phase 2 |
| D3 | ISO 8855 axes | **Decided 2 Oct 2026.** Migration is task 7.1 |
| D4 | Wheels as rotating bodies with a spin joint | Adopted, Phase 7 |
| D5 | Compiled library instead of header-only | Adopted, Phase 2 |
| D6 | CFD by coupling to OpenFOAM | **Decided 2 Oct 2026.** A CFD solver written from scratch is a stated goal for later, tracked as task 11.12 |
| D7 | Real-time target | **Decided 2 Oct 2026.** Both: Windows soft real time first, then Linux hard real time |
| D8 | JSON model files, Python scripting, CasADi and IPOPT for optimal control | Adopted |
