# How to integrate in time

How to choose and set up an integrator: a fixed step, a tolerance, a stiff
model, real time, states of your own, a closed mechanism (Phase 4). Every
example below is compiled and run, with its result checked, in
`tests/howto/test_howto_time_integration.cpp`; a test checks that this page
quotes that code exactly, so it cannot go stale (task H.6). What each method
costs for a given accuracy is measured in
[the integrators, verified](../../integrators.md); how each works, in
[the kernel](../../kernel.md#forces-and-simulation).

The examples use `using namespace mbd; using namespace mbd::kernel;` and
`mbd/kernel/simulator.hpp`; a `System sys` holds the model, built before
the simulator, and must outlive it.

## Which method

| Model and need | Method |
|---|---|
| Smooth, fixed steps | `RK4` (the default) |
| Smooth, a tolerance to meet | `DormandPrince` |
| Stiff (stiff springs, bushings, contact, tyres at low speed), fixed steps, modest accuracy | `GeneralizedAlpha` |
| Stiff, a tolerance to meet | `RadauIIA` |
| Stiffness only in first-order states (a vehicle's wheel spins), a tolerance to meet | `SDIRK4` or `RadauIIA` |
| Real time: the same cost every step | `LinearlyImplicitEuler` |
| Fast states and slow bodies, explicit | `RK4` with `multirate` |

The kernel's [table for choosing one](../../kernel.md#forces-and-simulation)
says more, and the report shows the costs behind it.

## Run with a fixed step

A rod 1 m long hanging from a hinge (a revolute joint, whose coordinate is
the angle), released 1 mrad from the vertical:

<!-- example: fixed-step -->
```cpp
Simulator sim(sys);
sim.method = Integrator::RK4;   // the default
sim.q(0) = 0.001;               // the hinge's angle [rad]
sim.initialize();
sim.run(2.0, 0.001);            // 2 s in steps of 1 ms
// sim.q, sim.v and sim.time now hold the state at 2 s.
```

- `initialize()` puts the state on the constraints (if there are any) and
  computes the kinematics; call it after setting `q` and `v`.
- `run(duration, dt)` takes `round(duration / dt)` steps; `step(dt)` takes
  one, so you can read or change anything between steps.
- An explicit method is stable only while the step is short against the
  fastest motion: RK4 needs w h below about 2.8. A model that blows up at a
  step but not at half of it is at that limit; see the stiff model below.

## Run to a tolerance

`DormandPrince` chooses its own steps within each `step(dt)` to meet a
tolerance, and ends exactly at the step's end:

<!-- example: error-control -->
```cpp
Simulator sim(sys);
sim.method = Integrator::DormandPrince;
sim.step_control.relative_tolerance = 1e-8;   // per step; the global error follows them roughly
sim.step_control.absolute_tolerance = 1e-8;
sim.q(0) = 0.001;
sim.initialize();
for (int k = 0; k < 40; ++k) sim.step(0.05);  // output every 50 ms; the method picks its own steps
const IntegratorStats& stats = sim.integrator_stats();   // steps, rejected, evaluations, step_size
VecX q_mid, v_mid, x_mid;
const Real t_mid = sim.time - 0.5 * stats.step_size;     // halfway through the last step taken
sim.interpolate(t_mid, q_mid, v_mid, x_mid);
```

- The tolerances are local, per step: the error at the end is roughly
  proportional to them, by a factor that depends on the model and the span.
- `interpolate` gives the state anywhere within the last step taken, from
  the method's fourth-order continuous extension, without stepping again.
- A step that would have to be shorter than `step_control.min_step` stops
  the run with MBD-K055, too many steps within one `step(dt)` with
  MBD-K056: usually a force that jumps, or a model much stiffer than
  expected.
- `step_control.adaptive = false` makes it take one step of `dt`, for
  measuring its order.

## A stiff model

A 1 kg block on a spring of 1e6 N/m (1000 rad/s) at steps of 10 ms, ten
times beyond RK4's limit:

<!-- example: stiff -->
```cpp
Simulator sim(sys);                          // a 1 kg block on a spring of 1e6 N/m: 1000 rad/s
sim.method = Integrator::GeneralizedAlpha;
sim.implicit_control.rho_infinity = 0.5;     // what is left per step of motions far too fast for it
sim.q(0) = 0.001;
sim.initialize();
sim.run(0.1, 0.01);                          // w h = 10: RK4 would need steps under 2.8 ms
```

- `GeneralizedAlpha` is stable at any step and second order; the motions
  too fast for the step are damped, towards `rho_infinity` per step (0.8 by
  default; 1 keeps them, 0 removes them at once). Here the vibration falls
  under a tenth of its start in ten steps.
- Each step is solved by Newton's method to the tolerances of
  `step_control` (1e-6 by default), with an iteration matrix from
  differences, kept while Newton converges fast. A step Newton cannot
  solve stops the run with MBD-K058: shorten the step there, or check for a
  force that jumps. A tolerance far tighter than the step's accuracy makes
  it likelier.
- For a tolerance to meet, `sim.method = Integrator::RadauIIA;` with error
  control (the default) chooses its steps as Dormand-Prince does; on stiff
  mechanical models it was the cheapest at every accuracy measured.

## Real time

The same block at 1 ms with the linearly implicit step, which costs the
same every step:

<!-- example: real-time -->
```cpp
Simulator sim(sys);
sim.method = Integrator::LinearlyImplicitEuler;
sim.q(0) = 0.001;
sim.initialize();
for (int k = 0; k < 10; ++k) sim.step(0.001);   // in real time: one step per tick of the clock
// integrator_stats().evaluations is 10: one evaluation per step, every step.
```

- One evaluation, one set of force derivatives and one linear solve per
  step, whatever happens: what a real-time budget needs.
- First order, and it damps every motion by about (w h)^2 / 2 per step, as
  backward Euler does: here the vibration halves its amplitude every two
  steps. Slow motions (a car's roll, its speed) are hardly touched.
- On very stiff springs or bushings between bodies that turn, keep k h^2 /
  m below about 25: beyond it the single linearization per step feeds
  energy in (D40, [troubleshooting](../troubleshooting.md)).
- MBD-K045 warns once if the step's matrix is not positive definite (a
  negative stiffness beyond the inertia); such steps leave the stiffness
  explicit.

## States of your own

An actuator whose force lags its command, as an element with a state of
its own, integrated with the bodies by the same method and in the same
stages:

<!-- example: actuator -->
```cpp
/// An actuator whose force follows its command with a first-order lag:
/// f_dot = (command - f) / tau. Its one state is the force.
class Actuator : public StateElement {
public:
    explicit Actuator(Real tau) : tau_(tau) {}
    const char* name() const override { return "actuator"; }
    int size() const override { return 1; }
    void initial_states(VecRef x) const override { x(0) = 0.0; }
    void begin_evaluation(const Simulator&, Real, ConstVecRef x) override { force = x(0); }
    void rates(const Simulator&, Real, ConstVecRef x, VecRef x_dot) override
    {
        x_dot(0) = (command - x(0)) / tau_;
    }

    Real command{0.0};   ///< Set between steps [N]
    Real force{0.0};     ///< The force of the evaluation in progress [N]

private:
    Real tau_;
};
```

Attached to a simulator of a 2 kg block on a slider, it pushes the block:

<!-- example: states -->
```cpp
Actuator actuator(0.2);                  // a time constant of 0.2 s
Simulator sim(sys);                      // a 2 kg block on a slider, no gravity
sim.attach(actuator);                    // its state joins sim.x, integrated with q and v
sim.force_callback = [&](Simulator&, Real, VecX& tau) { tau(0) += actuator.force; };
actuator.command = 3.0;                  // [N]
sim.run(1.0, 0.01);
// sim.x(0) is the actuator's force at 1 s, sim.q(0) the block's position.
```

- `begin_evaluation` receives the states of each evaluation (each stage of
  the step) before the forces; `rates` gives their derivatives after the
  dynamics; `end_step` (optional) makes discrete changes after a step (a
  gear shift, a brake locking). The drivetrain's wheel spins are such
  states.
- The element must outlive the simulator; attach it before stepping.

When the states are fast and the bodies slow, multirate RK4 steps the
states several times per step of the bodies:

<!-- example: multirate -->
```cpp
sim.method = Integrator::RK4;            // multirate works with RK4 only
sim.multirate = 4;                       // the states take 4 steps for each step of the bodies
sim.run(1.0, 0.01);
```

Each part must stay within its own stability: the fast steps within the
states', the step within the bodies' (MBD-K046 for another method).

## A closed mechanism

A rod held at its top end by a spherical closure instead of a joint (a
pendulum closed by a constraint), released at rest tilted by `tilt` (a
quaternion, 0.3 rad about X). Every method keeps the state on the
constraints by projecting after each step, and records what each step
left before it, the drift channel:

<!-- example: drift -->
```cpp
Simulator sim(sys);                          // a rod held at its top end by a spherical closure
sim.q.head<3>() = tilt * Vec3(0.0, 0.0, -0.5);   // the free joint: position, then the quaternion
sim.q.tail<4>() = tilt.coeffs();
sim.constraint_tolerance = 1e-10;            // the default: |phi| left after each projection [m]
sim.initialize();                            // projects onto the constraints
Recorder rec;
add_drift_channels(rec, sim);                // constraint_drift, constraint_residual, and the velocities'
for (int k = 0; k < 1000; ++k) {
    sim.step(0.001);
    rec.sample(sim.time);
}
const ConstraintDrift& drift = sim.constraint_drift();   // max_position: the most a step left
```

- `rec.write_csv(path)` writes the channels with the time; the step trace
  (`sim.trace_steps`) has the drift as columns as well.
- The drift is the integrator's error across the constraints; if it grows,
  shorten the step. A projection that cannot converge is counted and
  reported (MBD-K051).
- `GeneralizedAlpha` can instead solve each step on the constraints:
  `sim.implicit_control.stabilized_index2 = true;` (more accurate
  accelerations and reactions at long steps, about 1.5 times the cost).
- `sim.enforce_constraints = false` turns the projection off: the state
  then drifts as the errors add up.

## The mass matrix's factorization

The constraint solver factorizes the mass matrix along the tree where that
is faster (every branched model, and chains of up to 40 velocities) and
densely otherwise. `sim.solver().mass_factorization` chooses `Automatic`
(the default), `Dense` or `Tree`; the results agree to rounding, so the
setting matters only for speed, or to reproduce results from before task
4.8 to the last digit.
