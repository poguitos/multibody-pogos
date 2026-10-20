# How to run the analyses

Initial conditions made consistent, static equilibrium, kinematics along a
driving parameter, natural frequencies and damping (Phase 3, tasks 3.4 to
3.7). Every example is compiled and run, its result checked, in
`tests/howto/test_howto_analyses.cpp`, and this page quotes that code
exactly (task H.6). How each analysis works:
[the kernel](../../kernel.md#initial-conditions).

The examples use `using namespace mbd; using namespace mbd::kernel;`. Two of
them use the four-bar of the tests, `mbd_test::FourBar fb` from
`tests/kernel/four_bar.hpp`: a crank, a coupler and a rocker on hinges
along Z, a tree closed at the fourth hinge by a revolute closure, with
`FourBar::configuration(angle)` its closed configuration at a crank angle.
The others use the 10 kg block hanging from a spring of 1e4 N/m of
[forces and drives](forces-and-drives.md#a-spring-damper-between-two-points).

## Make the initial conditions consistent

Given positions and velocities rarely satisfy the constraints: a loop is
open by a few millimetres, a velocity would stretch a link. `assemble`
makes them consistent, keeping what you hold exactly as given and changing
the rest as little as possible:

<!-- example: assembly -->
```cpp
Simulator sim(fb.sys);                       // the four-bar: three hinges, a closure at the fourth
sim.q.head(3) = closed + Eigen::Vector3d(0.0, 0.05, -0.04);   // the loop a few degrees open
sim.v(0) = 2.0;                              // the crank's rate [rad/s]; the others unknown
AssemblyOptions options;
options.hold_position = {0};                 // keep the crank's angle as given
options.hold_velocity = {0};                 // and its rate
const AssemblyReport report = assemble(sim, options);
// report.ok(): positions and velocities on the constraints; sim.q, sim.v assembled.
// report.summary(fb.sys.model) lists every coordinate's given and assembled value.
```

- The indices are velocity indices (`model.idx_v`); `options.hold_joint(model,
  body)` holds every coordinate of a body's joint.
- "As little as possible" is weighted by each coordinate's own mass or
  inertia, so the units chosen do not matter and a coordinate no constraint
  involves never moves.
- Holding too much (the held coordinates fixing the loop on their own, in a
  way it cannot close) is reported in `report.warnings` with MBD-K063; the
  report is the result, nothing is printed.
- `Simulator::initialize()` only projects onto the constraints, holding
  nothing: use `assemble` when some coordinates must stay as given.

## Find the static equilibrium

<!-- example: statics -->
```cpp
Simulator sim(sys);                          // the block hanging from its spring
sim.q(0) = 0.2;                              // where the spring is at its free length
const StaticReport report = static_equilibrium(sim);
// report.converged; sim.q is the equilibrium, at rest; report.loads the forces there;
// report.summary(sys.model) says how it went.
```

The block ends 9.81 mm lower, where k (L - L0) = m g, at rest.

- Newton's method on the accelerations at rest, from the given positions,
  each step limited (`StaticOptions::max_translation`, `max_rotation`, as
  ADAMS's TLIMIT and ALIMIT) so that a preloaded spring cannot throw a body
  past its travel; dynamic relaxation where Newton stalls.
- Directions with no stiffness at all (a vehicle's position on the ground)
  are left where they are and reported as neutral (`report.neutral`,
  MBD-K072).
- MBD-K070 if it does not converge: usually a force with nothing to
  balance it ([troubleshooting](../troubleshooting.md)).

## Solve a mechanism along a driving parameter

A mechanism whose drivers leave it no freedom is solved by its geometry
alone. In `Kinematics`, the drivers' time `t` is the driving parameter: here
the four-bar with a driver that sets its crank's angle to `t`:

<!-- example: kinematics -->
```cpp
Kinematics kin(fb.sys);                      // the four-bar with a driver turning its crank to angle t
kin.q.head(3) = FourBar::configuration(0.9); // a start near the solution
kin.t = 1.0;                                 // the driving parameter: here the crank's angle [rad]
const bool closed = kin.solve();             // phi(q, t) = 0
const VecX& rates = kin.velocities();        // dq/dt: each joint's rate per unit rate of t
```

- `velocities()` gives the derivatives of the motion with respect to the
  parameter: a suspension's motion ratios when `t` is the wheel's travel.
  `accelerations()` the second derivatives.
- Sweeping `t` in small steps, each solve starting from the last, follows
  the mechanism through its travel; `sweep_bump_travel` does that for a
  vehicle corner (`tests/test_suspension_kinematics.cpp`).
- `Kinematics` is in `mbd/analysis/position_kinematics.hpp`, namespace
  `mbd`.

## Natural frequencies, damping and mode shapes

<!-- example: linearize -->
```cpp
const LinearModel lm = linearize(sim);       // about the simulator's state: here the equilibrium
const Mode& slowest = lm.modes.front();      // the modes come by natural frequency
// slowest.natural_frequency [rad/s], slowest.damping_ratio, slowest.shape (velocity coordinates),
// slowest.energy_fraction (its kinetic energy's share in each coordinate, adding up to 1);
// lm.summary(sys.model) lists every mode with its largest shape components and energy shares.
```

The hanging block with a damper of 63.2 N s/m has one mode: 31.6 rad/s
(sqrt(k / m)) at a damping ratio of 0.1 (c / (2 sqrt(k m))).

- Linearise about an equilibrium: find it first with `static_equilibrium`.
  About a state in motion the same formulas are used, exact for joints
  whose coordinates add.
- `energy_fraction` is the modal energy distribution, from the whole mass
  matrix: how pure a mode is (an engine's bounce on its mounts, a car's
  roll). Where coordinates are coupled through the mass matrix a share can
  be negative; the shares still add up to 1.
- `lm.A` is the state matrix in reduced coordinates on the constraints
  (`lm.basis` maps them back); a real eigenvalue appears as an overdamped
  mode, a zero one as a neutral direction.
- In `lm.warnings`: MBD-K080 if the state was not on the constraints (it
  is linearised at the nearest point that is), MBD-K081 if it is at rest
  but not in equilibrium.
