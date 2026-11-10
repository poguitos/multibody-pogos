# Data: Phase 6, the interface tuned with the author

Raw measurements behind the journal entry `2026-11-10_phase-6-tuning.md`.

## The survey before (the installed package of task 6.2)

`api_survey.py` over `multibody._mbd`:

- 59 classes, 3 with a `__repr__` of their own (`Transform3`, `JointModel`,
  `ConstraintModel`; the joints and constraints inherit theirs);
- callables with an unnamed argument (pybind11's `arg0`): 0;
- callables without a docstring: 71, of which 10 the constructors of the
  settings classes: AssemblyOptions.__init__, AssemblyReport.ok,
  AssemblyReport.summary, BushingParams.__init__, ConstraintModel.name,
  ConstraintModel.bodies, ContactParams.__init__, Curve.value, Curve.slope,
  Curve.is_zero, CylindricalJointModel.__init__, Dot1.__init__,
  Dot2.__init__, FixedJointModel.__init__, ForceElement.name,
  ForceElement.bodies, FreeJointModel.__init__, ImplicitControl.__init__,
  JointCoordinateForce.__init__, JointCoordinateForce.law,
  JointCoordinateForce.potential_energy, JointCoordinateForceParams.__init__,
  JointDriver.__init__, JointForce.name, JointForce.bodies, JointModel.name,
  Kinematics.phi, LinearModel.summary, LinearizationOptions.__init__,
  Mode.oscillatory, NoTwist.__init__, PlanarJointModel.__init__,
  PlaneContact.normal_law, PlaneContact.friction, PlaneContact.penetrations,
  PointCoincidence.__init__, PrismaticJointModel.__init__,
  Recorder.__init__, Recorder.samples, Recorder.names, Recorder.times,
  Recorder.write_csv, RevoluteJointModel.__init__, RigidBodyState.pose_WB,
  RotationalSpringDamper.__init__, RotationalSpringDamper.law,
  RotationalSpringDamper.potential_energy,
  RotationalSpringDamperParams.__init__, Simulator.integrator_stats,
  Simulator.constraint_drift, Simulator.projection_failures,
  Simulator.event_names, SphericalJointModel.__init__, SpringDamper.__init__,
  SpringDamper.law, SpringDamper.potential_energy,
  SpringDamperParams.__init__, StateElement.__init__,
  StaticOptions.__init__, StaticReport.summary, StepControl.__init__,
  System.__init__, System.add_constraint, System.add_force_element,
  System.add_joint_force, TimeFunction.value, TimeFunction.rate,
  TimeFunction.acceleration, Transform3.inverse,
  UniversalJointModel.__init__, ValidationReport.summary.
- `mbd.StaticOptions(max_iterations=50)`, `add_body` without frames, and
  `mbd.Marker(0, [1.0, 0.0, 0.0])`: each a `TypeError`.

## What prints after

`repr_tour.py` with the package installed in `mbd-try` (the parallelogram of
the tests, assembled and run 0.5 s with RK4 at 1 ms; a 2 kg slider on
400 N/m and 10 N s/m under gravity, brought to rest and linearised):

```
coverage: 61 classes; 19 module functions and 147 methods checked
<Model: 3 bodies and the ground (crank, coupler, rocker); nq = 3, nv = 3; gravity [0, -9.81, 0]>
<System: 3 bodies and the ground, 1 constraint, 0 force elements, 0 joint forces>
<AssemblyReport: ok; position residual 0 (given 0); velocity residual 0 (given 0); 1 degree of freedom; 0 warnings>
<Simulator: t = 0 s, RK4; 3 coordinates, 3 velocities, 0 states>
<Simulator: t = 0.5 s, RK4; 3 coordinates, 3 velocities, 0 states>
IntegratorStats(steps=500, rejected=0, evaluations=2000, step_size=0.001, error=0.0, newton_iterations=0, jacobians=0, state_evaluations=0)
ConstraintDrift(position=0.0, velocity=0.0, position_residual=0.0, velocity_residual=0.0, max_position=3.885780586188048e-16, max_velocity=2.7755575615628914e-15, max_position_residual=3.885780586188048e-16, max_velocity_residual=2.9976021664879227e-15)
RigidBodyState(p_WB=[0, 0, 0], q_WB=[0.446222, -0, 0, -0.894922], v_WB=[0, 0, 0], w_WB=[0, 0, -10.979])
<ValidationReport: 0 errors, 0 warnings, 1 note; 3 bodies, 3 coordinates, 5 constraint equations (1 independent, 4 redundant), 2 degrees of freedom>
<Loads: v_dot = [22.1131, -22.1131, 22.1131], lambda_ = [40.5892, 56.9538, 0, 0, 0]; constraint wrenches and joint reactions of 4 bodies>
<StaticReport: converged in 18 iterations, residual 5.61e-11 (given 69.8); 1 degree of freedom; 0 warnings>
<LinearModel: 1 degree of freedom at t = 0 s; 1 mode: 14.14 rad/s (2.251 Hz), damping ratio 0.177; 0 warnings>
<Mode: 14.14 rad/s (2.251 Hz), damping ratio 0.177>
<LinearSpringDamper: spring-damper, bodies [0, 1]>
StaticOptions(tolerance=1e-06, max_iterations=50, perturbation=1e-06, max_translation=0.02, max_rotation=0.17453292519943295, relaxation_step=0.001, max_relaxation_steps=20000, relaxation_tolerance=0.01)
SpringDamperParams(free_length=0.3, spring=<Curve: f(0) = 0, f'(0) = 30000>, preload=0.0, damper=<Curve: zero>, bump_clearance=inf, bump_stop=<Curve: zero>, rebound_clearance=inf, rebound_stop=<Curve: zero>)
RigidBodyInertia(mass=1.0, com_B=[0, 0, 0], I_com_B=[[0.0433333, 0, 0], [0, 0.0333333, 0], [0, 0, 0.0166667]])
revolute closure constraint (5 equations) revolute joint
<Recorder: 1 channel (crank), 1 sample>
error: MBD-P010: StaticOptions has no setting 'max_iteration'; its settings are tolerance, max_iterations, perturbation, max_translation, max_rotation, relaxation_step, max_relaxation_steps, relaxation_tolerance
error: MBD-P008: no body is named 'wheel'; the bodies are 'ground', 'crank', 'coupler', 'rocker'
```

The coverage line counts what the two general tests check. The mode
matches its closed form: omega = sqrt(400 / 2) = 14.142 rad/s, 2.2508 Hz,
zeta = 10 / (2 sqrt(800)) = 0.17678.

## The install into mbd-try

`scripts\build.ps1 -Install -Python <mbd-try>\python.exe`: 161 s, the eight
binding files compiled (the library's objects kept from the earlier
install in the same build folder, `build-python\cp314-cp314-win_amd64`);
`python -m pytest python/tests`: 43 passed in 0.23 s. The package's
docstring example: q from 0.5 to -0.49914528 rad after 1 s.
