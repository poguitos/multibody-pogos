# External sources reviewed, 9 November 2026

Three sources the author recommended, read with the same question as the
first review ([review_2026-10-20](../review_2026-10-20/external_sources.md)):
what would make this program better, and where in the plan would it go?

1. Nick McCleery's three-part series on suspension kinematic simulation.
2. Suspension Explorer: the web application, its documentation, and its
   open-core solver `suspension-explorer-core`.
3. AVL's pages on vehicle engineering and vehicle simulation, which the
   author holds up as the industry's standard: "usually not very technical,
   but they show what can be achieved".

Following AVL's claim of a "correlated" vehicle model led to a fourth
source: the ISO standards that define how a vehicle-dynamics simulation is
validated against tests (section 4).

The proposals, P1 to P14, are at the end. The author adopted them all the
same day; they are decision D51 and are in the plan in their tasks.

## How the review was done

- The pages were read on the web, while CI ran on the Phase 5 gate. Local
  copies are in `C:\dev\references\web\`, and a shallow clone of the solver
  (commit `3af4097`, 4 October 2026) is in
  `C:\dev\references\repositories\suspension-explorer-core\`. The clone was
  read, not installed or run. The index is `C:\dev\references\README.md`.
- Our plan and code were checked first, so that nothing already planned is
  proposed again. Phase 9's tasks (9.1 the suspension rig, 9.4 manoeuvres,
  9.5 reports, 9.7 correlation), 8.4 (setup), 8.6 (reference cars) and 14.5
  (FMI) already cover much of what AVL describes.
- **Licences.** `suspension-explorer-core` is AGPL-3.0, and this repository
  has no licence yet. Nothing from it may enter the project, neither its
  code nor its data (its example hardpoint files included). Its ideas are
  described below in our own words, with citations, and its mathematics is
  textbook: Mozzi and Chasles, Lynch and Park's *Modern Robotics*. Running it
  as a separate program for a comparison, with files exchanged, keeps it
  apart from our code, as Chrono is in `crosscheck/`. McCleery's posts are
  cited for their ideas. AVL's pages are marketing material: their numbers
  are the vendor's claims, unverified, and are quoted as such.

## The sources

### 1. McCleery, "Suspension kinematic simulation", parts I to III (nickmccleery.com)

**What it is.** A development diary of the solver behind Suspension
Explorer, in Python. Part I sets up the problem: the free points' coordinates
are the unknowns, rigid links are distance residuals, the upright is rigid
through distances and an angle, a sweep adds target residuals, and each step
is solved by SciPy's Levenberg-Marquardt from the previous step's result. The
previous step is what keeps the solution on one branch ("you don't magically
flip the upright around halfway through a bump event"). Part II adds camber
shims, analytic Jacobians (SymPy-generated, with dual numbers through the
derived points) and the first metrics, defined with ISO 8855 signs. Part III
handles a steered multi-link corner, which has no physical kingpin.

**What is useful to us.**
- **The steering axis as an instantaneous screw axis** (Part III). The
  upright's motion per unit of rack travel, with a chosen suspension
  coordinate held, is a twist. Its axis is the steering axis, its pitch says
  how far the motion is from a pure rotation, and caster, KPI, ground offset
  and trail follow from it with the same formulas as for a physical kingpin.
  He fits the twist to the rates of seven upright points by least squares
  (21 equations, 6 unknowns, centroid as reference); the axis point is
  c = p + (rho x r) / |rho|^2 and the pitch h = (rho . r) / |rho|^2. On a
  double wishbone the motion-derived axis is the ball-joint line to 1e-10,
  with pitch below 1e-8 mm/rad. That is the check.
- **The hold decides the axis.** The rates depend on what is held while the
  rack moves. The sweep's own tangent, with the wheel-centre height held,
  gave a pitch up to 25 mm/rad. It moved ground offset by up to 7.6 mm and
  trail by up to 10.6 mm, while caster and KPI changed by only 0.017 and
  0.007 degrees. Holding a wishbone's angle recovers the kingpin. On a
  multi-link only the damper length can be held, and the axis then differs
  from the textbook link-pair construction by up to 0.4 degrees of KPI and
  3.5 mm of ground offset at full lock.
- **Derivatives restricted to the mechanism's motions.** The constraint
  Jacobian's null space N (by SVD) gives the allowed motions, and the
  mobility is the number of its columns (2 for a steered corner: travel and
  rack). The response to target j solves (g_q N) alpha = e_j, a small
  system. A response that is not unique is reported, not resolved silently
  by a least-norm fallback. A target's coupling to the mechanism is measured
  by |g_q,i N| / |g_q,i|.
- **Each frame gets a status** (valid, too few points, degenerate geometry,
  fit error too large, near pure translation, rank deficient, not finite),
  so one bad frame does not stop a sweep.
- **Two corrections of his own Part II.** Trail must be measured along the
  steered tyre's heading, not chassis X. "Scrub radius" in Part II was the
  signed ground offset; scrub radius is its magnitude. Both are easy to get
  wrong and are worth a test each.
- **Shims as a local constrained solve** (Part II, version 2). A camber
  shim is a small mechanism of 7 unknowns and 10 residuals, solved once
  before the sweep. His first version, a rigid rotation about the lower
  ball joint, did not keep the joints on their spheres, and he replaced it.

**What we already do, or do otherwise.**
- Our Jacobians come from the kernel's spatial algebra (task 2.5), exact and
  without SymPy or dual numbers.
- His distance residual |d| - L needed a smoothed norm near d = 0. Ours,
  (d^2 - L^2) / (2 L0), is smooth everywhere.
- Our kinematic analysis (3.6) already gives the exact tangent of a sweep,
  `Kinematics::velocities()` (J v = nu), and the curvature. Two of its
  choices are the ones he warns against. Where freedom is left it returns
  the least-norm solution. `KinematicSweepResult::camber_gain()` averages
  differences across the sweep instead of using the tangent. A failed step
  is already kept and flagged (`converged`), not fatal.
- Our twist needs no fit: the kernel gives the upright's spatial velocity
  directly from the joint velocities.

### 2. Suspension Explorer: the web application, its documentation, and `suspension-explorer-core` (AGPL-3.0)

**What it is.** The solver of section 1, grown into a product. It has a web
application and a Python package with a command line. It models double
wishbones, MacPherson struts, five-link corners and semi-trailing arms, as one
corner or as an axle with the right side mirrored. Actuation is direct or by
pushrod and rocker, with coilovers or torsion bars, U-bar and T-bar anti-roll
mechanisms and heave links. Shims change camber, toe and pushrod length.
Bump, roll and steer sweeps write points, metrics, solver statistics and
diagnostics to CSV or Parquet. It is kinematics only: no compliance, loads or
dynamics.

**What is useful to us.**
- **A sweep's rules, checked before solving.** Every actuator is controlled
  exactly once, either held or swept. Targets are paired step by step, not
  expanded into a product. A held target captures its value at the sweep's
  reference state. The contact centre is an output and cannot be a target.
- **Diagnostics that keep the frame** (read in `diagnostics.py` and
  `suspensions/axle/mechanisms.py`):
  - a *branch snap*: a point moves farther between two steps than a
    multiple of its median step;
  - *chirality*: a body's points come out as the mirror image of the design;
  - a *transmission margin*: the cosine between a link and the direction its
    pickup can move, warned below 0.15 because at zero the linkage toggles
    (locks);
  - weak target coupling and poor conditioning.
- **A metric catalogue.** Each metric has an identity, unit, scope (corner
  or axle) and the architectures it is available for. Derivatives are named
  `deriv_<metric>_wrt_<driver>` with both units kept (deg/mm). The catalogue
  is generated from the code, written out as JSON with a schema, and checked
  by a test, as our message catalogue is.
- **One axle without a vehicle.** The road plane is constructed tangent to
  both wheels, which gives heave, roll and ride-height change of one axle.
  Pitch is left to a full vehicle.
- **Metric reference frames stated per metric.** Camber relative to the
  vehicle; caster and KPI in chassis axes; ground offset, scrub radius and
  trail in the tyre axes on the local road plane. Anti-dive, anti-lift and
  anti-squat use the line from the reaction point to the side-view instant
  centre, with braking and traction anti ratios. Metrics include wheel-centre
  recession, contact-patch lateral migration, swing-arm lengths and angles,
  and Ackermann percentage.
- **The setup state before any sweep.** A "static pose" shows the shims'
  effect on the assembled corner.
- **Examples labelled honestly.** The race-car examples (FSAE-, IndyCar- and
  F1-style) are "based on published architecture descriptions, not measured
  team data", and a test checks that every example's links clear each other
  through all sweeps.
- **A second code for task 9.1.** It is independent of ours (different
  author, formulation and language), and it is open. Run separately, on our
  hardpoints, it can be compared with our rig curve by curve, as Pinocchio
  and Chrono were in task 5.3. Its README warns that it is experimental, so
  as in 5.3 it would first be checked on cases with known answers.

### 3. AVL: vehicle engineering and vehicle simulation (avl.com, avlracetech.com)

**What it is.** The pages of AVL's vehicle-engineering services and its
simulation products: Vehicle Composer (AI-based concept development), VSM
(vehicle simulation, with the Vehicle Model Factory and VSM RACE), EXCITE M
(multibody dynamics of powertrains), virtual vehicle development and
functional development, plus two blog posts (the 2025 R1 release and
off-road testing). As the author said, they show scope, not method: none
gives a model equation, a degree-of-freedom count or an error measure.

**What they show can be achieved** (vendor claims):
- **Concept development from targets** (Vehicle Composer). Architecture
  variants are evaluated "in real time" by physics-informed neural networks
  trained on AVL's simulations and benchmark data, more than 10 attributes
  at once (energy, dynamics, crash, cost). Vehicle-level targets are
  cascaded into subsystem requirements, "from weeks to hours".
- **Models identified from road data** (Vehicle Model Factory). A
  manoeuvre matrix is driven on a proving ground: acceleration, braking,
  coast-down, constant load, constant and transient cornering, slalom, step
  steer, weave. Parameters are identified per subsystem: weight
  distribution, driving losses, suspension, drive unit, battery, brakes,
  steering, and tyres (automatic since 2023 R1). Where no vehicle exists, a
  model is fitted to target KPIs instead (acceleration time, braking
  distance). AVL claims up to 60 % less modelling time, more than 300
  validated models, and "up to 97 % model correlation", with no stated
  definition of correlation.
- **One model from office to simulator.** The same model is used in the
  office, SiL, HiL, test benches and the driving simulator.
- **Lap time.** More than 40,000 laps of the Nordschleife per hour in the
  cloud. Driver models are tuned automatically to the vehicle and the track
  (grip, banking). Tracks come from GPS or IMU logs or maps. Setup changes
  and ambient conditions are assessed against lap time, handling and energy.
- **Post-processing templates.** Standard plots per manoeuvre (constant
  radius cornering, for instance), with reports "within minutes".
- **EXCITE M:** a fast frequency-domain analysis for quick assessments
  beside non-linear time-domain runs, and flexible bodies from FE models.
- **Off-road** (blog): the on-road twin is converted by a soft-soil model
  tuned to measured torque and slip. Lowering grip and raising rolling
  resistance is "a weak substitute".

**What is useful to us.** The ideas are in the proposals below: targets
cascaded to subsystems (P12), a model fitted to KPIs when no measurement
exists, and coast-down and weave in the identification sequence (P11), the
frequency domain from the linearisation (P13), and the driver tuned to the
track (P14). The rest (one model from office to HiL, report templates, a
library of reference vehicles, tracks from logs) is in the plan already, in
tasks 12.x and 14.5, 9.5, 8.6 and 10.1.

### 4. ISO 19364 and ISO 19365: validating a vehicle-dynamics simulation

Found by asking what "model correlation" should mean.
**ISO 19364:2016** (*Passenger cars — Vehicle dynamic simulation and
validation — Steady-state circular driving behaviour*) specifies how results
of a vehicle model are compared with measurements of an existing vehicle
from steady-state circular tests (ISO 4138). The comparison uses cross plots
with tolerances at boundary points (section 9.3 of the standard).
**ISO 19365:2016** does the same for the sine-with-dwell manoeuvre (stability
control) and requires ISO 19364 first. The texts are sold by ISO and have not
been read: the tolerances themselves are not known here.

**What is useful to us.** Task 9.7 asks for "per-channel errors normalised by
each channel's spread", which is our own measure. These standards are the
recognised acceptance procedure for exactly the comparison of 9.6 and 9.7.
A third standard on lateral transient response may exist; it was not
confirmed and is not cited.

## What the sources point to

- **Kinematics is only half of a suspension tool, but the half where
  definitions decide everything.** Both McCleery and Suspension Explorer
  spend most of their effort on what is held, in which frame each metric is
  measured, and when a derivative is not defined. Our rig (9.1) should write
  those choices down per output, as our message catalogue does for
  messages.
- **Derivatives at a state, not differences between states.** Both use
  tangents of the solved mechanism, never differences of neighbouring sweep
  frames. Our kernel has them already; the rig should use them throughout.
- **Report, don't guess.** Suspension Explorer reports a derivative as
  unavailable rather than return an unreliable one. That is our D44 and D50
  (name the contradiction, never return a silent minimum-norm answer)
  carried over to sensitivities.
- **AVL sells the workflow, not the solver:** identification from a
  standard manoeuvre matrix, targets cascaded down, one model everywhere.
  Phases 9, 12 and 13 are that workflow.

## Proposals

Each is an amendment to a task already in the plan, except P12, which
extends 13.3. None changes Phase 6's order.

### Phase 9: the suspension rig (9.1) and its neighbours

- **P1. The steering axis from the motion (9.1).** The instantaneous screw
  axis of the upright's twist per unit of rack, with a declared hold, for
  every architecture: axis point, direction, pitch and angular rate. Caster,
  KPI, ground offset, scrub radius and trail are computed from it and from
  the physical kingpin through one representation of an axis, reported as
  two families. *Added to "done when": on a double wishbone the motion
  axis is the ball-joint line to rounding with zero pitch, and on the
  five-link (7.6) its difference from the link-pair construction is
  reported.*
- **P2. Holds declared per output (9.1).** Each architecture lists the
  coordinates that can be held while the rack moves (wishbone angle, strut
  length, damper length, contact centre on the road), with a default. Every
  steering output names its hold. *A test shows that holding the
  wheel-centre height instead changes ground offset and trail by
  millimetres, but caster and KPI hardly at all.*
- **P3. Responses in the mechanism's null space (9.1, 13.3, 13.9).** The
  plan's dq/dp = -phi_q^+ phi_p becomes a solve restricted to the allowed
  motions. It reports the mobility, each target's coupling, and a response
  that is not unique as unavailable, with an MBD- code. Derivative channels
  are named with both units (camber gain in deg/mm, bump steer, motion
  ratios) and come from tangents (`Kinematics::velocities()`), not from
  differences between frames. `camber_gain()` is replaced. The same applies
  to `Kinematics::velocities()` itself where freedom is left: it reports the
  freedom instead of returning the least-norm motion.
- **P4. Each metric defined once, with its frame and sign (9.1,
  `docs/conventions.md`).** ISO 8855 axes. Signed ground offset and unsigned
  scrub radius are separate. Trail is measured along the steered wheel's
  heading. Wheel-centre recession, contact-patch lateral migration,
  anti-lift, braking and traction anti ratios and swing-arm angles are
  added. A test each for the two traps McCleery met. *Done when* still
  "matches Dixon chapters 4 to 6".
- **P5. Sweep rules and per-step diagnostics (9.1, 3.6).** Before solving,
  check that every driver is controlled exactly once and that targets pair
  by step. Per step, keep the frame and record a status with an MBD- code:
  branch snap (a point moving more than a stated multiple of its median
  step), chirality (a body solved as its mirror image), transmission margin
  (a link within a stated angle of toggling), poor conditioning.
- **P6. Shims as lengths inside the assembly (8.1, 8.4).** Camber, toe and
  pushrod shims are given as design and setup stack thicknesses and change
  link lengths. They are applied through the assembly of 3.4 with the
  installed coordinates held, never as rigid rotations, and the assembled
  setup state can be inspected before any sweep.
- **P7. A metric and channel catalogue (9.5).** Every metric and channel
  has an identity, unit, scope and availability, written out from the code
  and checked by a test that every reported metric is in it, like the
  message catalogue (H.2).
- **P8. An axle's road plane (9.1).** The plane tangent to both wheels
  gives one axle's heave, roll and ride-height change without a full
  vehicle; the full vehicle's pitch from both axles stays in 9.2.
- **P9. A second code for the rig (9.1).** `suspension-explorer-core`, run
  in its own environment on our hardpoints (files in, files out), compared
  curve by curve, as in 5.3. Its own examples and code stay out of the
  repository. It is checked first on a case with a known answer (the
  double-wishbone kingpin).
- **P10. Validation by ISO 19364 and 19365 (9.6, 9.7).** The standards'
  comparison procedure (cross plots with tolerance boundary points, steady
  state first, then sine with dwell) becomes the acceptance test where a
  measurement exists. The author would need to obtain the two standards.
- **P11. The identification sequence (9.7).** Coast-down (rolling
  resistance and drag) and weave are added to "straight line, then steady
  cornering, then a lap". Where no measurement exists, the model is fitted
  to target KPIs (acceleration time, braking distance, understeer gradient)
  by the same machinery, checked the way 9.7 already is: KPIs measured on an
  altered model must give its parameters back.
- **P13. The frequency domain from the linearisation (9.4, from 3.7).**
  Transfer functions of the linear model, such as body acceleration per
  road input or yaw rate per steer, computed directly as H(jw) =
  C (jw I - A)^-1 B. *Checked against the time-domain frequency sweep of
  9.4 at small amplitude.*
- **P14. The driver tuned to vehicle and track (9.3, 10.5).** The path
  follower's preview and gains are set automatically from the vehicle and
  the track's grip and banking. *Checked by a lap whose speed profile
  approaches the quasi-steady envelope of 10.2.*

### Phase 13

- **P12. Targets cascaded to subsystems (13.3).** Vehicle-level targets
  (understeer gradient, roll gradient, ride frequencies, braking distance)
  are turned into subsystem requirements (roll-stiffness distribution,
  spring and bar rates, anti percentages, hardpoint ranges) by the
  sensitivities of 9.1 and 13.9 and the optimisers of 13.3. This is
  Vehicle Composer's idea with our physics in place of its neural network.
  *Done when a reference car's targets, moved, are met by the cascade
  within stated tolerances.*

### Noted, not proposed

- **Neural networks trained on millions of runs** for concept attributes
  (AVL): they need data we do not have; 13.5's surrogates cover the method.
- **Snow and soft-soil tyre models** (AVL): outside the plan, which is for
  road and race cars. Worth recalling if the HMMWV ever leaves the road.
- **Lap throughput** (40,000 laps per hour, AVL): a number to measure our
  own against once 10.2 exists, not a task.
- **Tracks from map services** (AVL's Google Maps and HERE): licensing; the
  CSV, TUM and Fast-F1 sources of 10.1 suffice.
- **Dual numbers or SymPy for Jacobians** (McCleery): the kernel's
  Jacobians are already exact.
- **Fitting a twist to points** (McCleery): the kernel gives the twist
  directly.

## For the book

- Part on vehicle analyses: the virtual kingpin. A five-link has no
  kingpin, yet the steering geometry exists as the screw axis of a motion,
  and which coordinate is held decides it. The figure: the axodes, the
  ruled surface the instantaneous axis sweeps along a steer or bump sweep.
- Part X: "report, don't guess", from D44 and D50 to sensitivities that
  refuse to be computed where they are not unique.
- A comparison of two formulations of the same problem: coordinates of
  points with distance residuals (Suspension Explorer) against joint
  coordinates on a tree with loop closures (ours). What each makes easy:
  his targets on any derived point, our exact Jacobians and dynamics.
