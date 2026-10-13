# Decision log

Every design decision, with its context, the alternatives and its
consequences, numbered in the order taken. D1 to D8 were set out in the review
and are written in full in [Master_plan.md](Master_plan.md), section 4; they
are indexed here. From D9 on, decisions are written here when they are taken,
and the journal entry of the work refers to them by number
(see [journal/README.md](journal/README.md)).

## Index

| # | Decision | Taken | Status |
|---|---|---|---|
| D1 | One kinematics kernel on spatial vectors | 2 Oct 2026 | Done, Phase 2 |
| D2 | Quaternions for free and spherical joints, body-fixed angular velocity | 2 Oct 2026 | Done, task 2.2 |
| D3 | ISO 8855 axes | 2 Oct 2026, by the author | Kernel done; vehicle layer is task 7.1 |
| D4 | Wheels as rotating bodies on a spin joint | 2 Oct 2026 | Planned, Phase 7 |
| D5 | A compiled library instead of header-only | 2 Oct 2026 | Done, task 2.8 |
| D6 | CFD by coupling to OpenFOAM first; an in-house solver later (task 11.12) | 2 Oct 2026, by the author | Planned, Phase 11 |
| D7 | Real time on Windows first, then hard real time on Linux | 2 Oct 2026, by the author | Planned, Phase 12 |
| D8 | JSON model files, Python, Rerun or Meshcat, CasADi with IPOPT | 2 Oct 2026 | Planned |
| D9 | One compiler at a time, enforced by the build system | 3 Oct 2026 | In force |
| D10 | Precompiled headers rather than include surgery | 3 Oct 2026 | In force |
| D11 | Invariant tests are the gate for every joint, constraint and force | 3 Oct 2026 | In force (working rule 2) |
| D12 | Body-fixed velocities, quaternions `(x, y, z, w)`, one normalisation per step | 3 Oct 2026 | In force |
| D13 | Every constraint built from five primitives on markers | 4 Oct 2026 | In force |
| D14 | Range-space solve with a pivoted factorisation; projection in the kinetic-energy metric | 4 Oct 2026 | In force |
| D15 | Prescribed motion as constraints, with t as a parameter in kinematic analysis | 4 Oct 2026 | In force |
| D16 | Steering of linkage axles by a rack body | 4 Oct 2026 | In force |
| D17 | Delete the legacy core instead of keeping two engines | 4 Oct 2026 | Done, task 2.7b |
| D18 | A precompiled header for the library, without the vehicle headers | 5 Oct 2026 | In force |
| D19 | Argument checks in every build; `validate()` reports without throwing | 5 Oct 2026 | In force |
| D20 | One kinematics pass per evaluation; `Z = L^-1 J^T`; projection weighted by the last stage's mass matrix | 5 Oct 2026 | In force |
| D21 | Timings pinned to a performance core, versions compared alternately | 5 Oct 2026 | In force |
| D22 | A development record kept in the repository, for the book | 5 Oct 2026 | In force |
| D23 | Help documentation grown with each phase, with message codes | 5 Oct 2026 | In force (H.1, H.2 done) |
| D24 | Force laws in each element's own coordinates, with their derivatives; joint-coordinate forces act on generalized forces directly | 5 Oct 2026 | In force, task 3.1 |

## D9. One compiler at a time, enforced by the build system

- **Context.** In mid-2026 two all-core builds restarted the PC; whether memory
  or heat was the cause was never established. The rule "one compiler at a
  time" existed only in the assistant's memory (finding F20).
- **Decision.** A Ninja job pool in the top-level `CMakeLists.txt`
  (`MBD_COMPILE_JOBS`, default 1) limits every build, whatever Ninja would choose.
- **Alternatives.** Rely on `-j1` in instructions (fails silently when forgotten);
  allow six compilers, as memory alone would (16 GB, 1.6 GB per compiler, 4 GB
  spare), which ignores the unexplained restarts.
- **Consequences.** Slow builds, which made build time worth engineering (D10,
  D18). Raising the limit is the author's call; the README says how, with
  temperatures watched.
- **Record.** Journal: Phase 0. Commit `d811521`.

## D10. Precompiled headers rather than include surgery

- **Context.** The engine was header-only; each of the 44 test files parsed all
  of it (92 s per file).
- **Decision.** One precompiled header shared by every test executable. Every
  file still includes what it uses, and the Linux CI job builds without
  precompiled headers to keep that honest.
- **Alternatives.** Trim includes file by file (long, fragile, and the engine
  headers were the bulk); move to a compiled library first (D5, the larger
  change, done later in task 2.8).
- **Consequences.** A test file 92 s to 23 s, a full rebuild an hour to 21 min.
- **Record.** Journal: Phase 0. Commit `d811521`.

## D11. Invariant tests are the gate

- **Context.** Months of scenario tests had not caught F1 to F4; a
  finite-difference check found them in an afternoon (finding F21).
- **Decision.** Every joint, constraint and force passes tests that hold for
  any correct implementation, at random states: velocities and Jacobians
  against finite differences of positions, the dynamics algorithms against
  each other and against Lagrange's equations, conservation of energy and
  momentum. Nothing is built on an element before it passes (working rule 2).
  Every other tolerance comes from a hand calculation (working rule 3).
- **Alternatives.** Scenario tests with tuned thresholds, which record what the
  code does rather than what it should do.
- **Consequences.** The suite failed on the old code in exactly the places F1 to
  F4 predicted (315 of 845 assertions), and on nothing else.
- **Record.** Journal: Phase 1. Commit `d811521`.

## D12. Body-fixed velocities, quaternions, one normalisation per step

- **Context.** D2 chose quaternions; how to store them, which velocity to
  pair them with, and how an explicit integrator keeps them unit remained.
- **Decision.** Joint velocities are body-fixed, expressed in the child-side
  joint frame; quaternions are stored `(x, y, z, w)`, Eigen's order; RK4
  combines its stages' `q_dot` linearly and normalises once at the end of the
  step. `integrate` (the exponential map) is used where a configuration must
  stay on the manifold: semi-implicit Euler and the projection.
- **Alternatives.** Normalise or retract each stage (costs RK4 its order);
  world-frame velocities (the motion subspace would depend on the
  configuration).
- **Consequences.** Coordinates and velocities differ in size (`nq` against
  `nv`); every algorithm handles both.
- **Record.** Journal: the kinematics kernel. Commit `cbab561`; `docs/kernel.md`.

## D13. Every constraint built from five primitives on markers

- **Context.** The legacy constraints were separate classes, each with its own
  derivatives, which is where F3 and F4 lived.
- **Decision.** Five primitives on markers (frames fixed on bodies): point
  coincidence, dot-1, dot-2, distance, no-twist, each with an optional
  time-dependent target. Point on a line and on a plane, and every joint
  closure, are compositions. Jacobian rows are filled as wrenches carried up
  the tree, with no body Jacobian formed.
- **Alternatives.** One class per joint closure (more code to get right, each
  with its own second derivatives).
- **Consequences.** Five sets of derivatives to verify, after which every
  closure is right by construction; the tests check each closure allows
  exactly its joint's motions.
- **Record.** Journal: constraints. Commit `d33a128`.

## D14. Range-space solve with a pivoted factorisation

- **Context.** Vehicle models contain redundant constraints (planar loops
  closed in 3D, joints closed twice); the legacy solve could fail on them
  (task 1.6).
- **Decision.** Solve for the multipliers with `J M^-1 J^T` factorised as
  `L D L^T` with diagonal pivoting; pivots below 1e-10 of the largest are
  dropped and their multipliers set to zero; the rank is reported. Project q
  and v after each step by Gauss-Newton in the kinetic-energy metric, each step
  halved until the residual falls.
- **Alternatives.** An augmented (KKT) system with a general solver; coordinate
  partitioning (choosing independent coordinates, which change with
  configuration).
- **Consequences.** `J^T lambda` and the accelerations are unique even when
  lambda is not. `validate()` reports the rank the solver sees (D19).
- **Record.** Journal: constraints. Commit `d33a128`.

## D15. Prescribed motion as constraints; t as a parameter in kinematics

- **Context.** Bump sweeps used to edit a constraint's target between solves,
  and steering moved constraint anchors between steps (F10): neither enters
  the velocity and acceleration equations.
- **Decision.** A driven motion is a constraint whose target depends on t
  (`JointDriver`, time-dependent primitives), with its rates in `nu` and
  `gamma`. In kinematic analysis t is a parameter: a bump sweep solves
  `phi(q, t) = 0` for successive t.
- **Consequences.** The driver's multiplier is the force the drive needs. One
  formulation serves kinematic sweeps and driven dynamics (plan tasks 3.2, 3.6).
- **Record.** Journal: the simulator and the vehicles. Commit `24faade`.

## D16. Steering of linkage axles by a rack body

- **Context.** F10: the tie-rod inner points were moved by different amounts by
  editing constraints.
- **Decision.** A light rack body on a prismatic joint along the chassis's
  lateral axis carries the tie rods' inner points, driven to the commanded
  travel by a `JointDriver`; each wheel's toe follows from the geometry. The
  travel per radian of toe is calibrated per corner. Simple corners still
  steer their tyre directly.
- **Consequences.** One more body and one more equation per steered linkage
  axle; the steering rate enters the equations.
- **Record.** Journal: the simulator and the vehicles. Commit `24faade`.

## D17. Delete the legacy core

- **Context.** After task 2.4 the kernel agreed with the legacy algorithms to
  5e-13 on random trees; everything had been ported by task 2.7.
- **Decision.** Delete the legacy core and its tests, keeping their closed-form
  cases on the kernel, instead of keeping both engines.
- **Alternatives.** Keep the legacy path as a reference (two engines to
  maintain, and a reference already known to be wrong in places).
- **Consequences.** 176 legacy test cases went; their physics lives on in
  `test_kernel_joints_physics.cpp`. One behaviour changed (contradictory
  constraints) and its test was rewritten to the new, documented behaviour.
- **Record.** Journal: library and checks. Commit `860b264`.

## D18. A precompiled header for the library

- **Context.** Task 2.8 moved 17 headers' code into sources. The build log
  showed each library source file spent 10 to 46 s, nearly all parsing Eigen.
- **Decision.** The library has its own precompiled header with the standard
  library, Eigen, the core and the kernel, and without the force, vehicle and
  analysis headers, so editing one of those recompiles only its users. One
  option, `MBD_PCH`, covers the library and the tests.
- **Consequences.** Library files 0.4 to 3 s; a full rebuild 7.4 min at one job.
- **Record.** Journal: library and checks. Commit `32f2075`.

## D19. Checks in every build; `validate()` reports without throwing

- **Context.** Size checks were debug-only asserts (F16); in release a wrong
  size was read out of bounds. Malformed models failed later, far from the
  cause.
- **Decision.** Every kernel entry point checks its arguments in every build:
  one comparison each, the message built only on failure. Solver and simulator
  reject references to missing bodies when built. `validate()` collects every
  problem it can find into a report with errors, warnings and notes, and
  counts the degrees of freedom.
- **Alternatives.** Checks only in Debug (the release user gets corruption);
  `validate()` that throws on the first problem (the user fixes one at a time).
- **Consequences.** Each message names the function, body or constraint at
  fault; these messages are the start of the help catalogue (D23).
- **Record.** Journal: library and checks. Commit `253e788`.

## D20. One kinematics pass per evaluation

- **Context.** The sedan needed to step in under 250 us (task 2.10). Each
  evaluation ran the kinematics four times.
- **Decision.** `forward_kinematics(q, v, 0)` serves the forces, the constraint
  equations, the mass matrix (`mass_matrix`) and the bias forces
  (`bias_forces`, from `a_gf = a - [0; R^T g]`). The constraint matrix is
  `Z^T Z` with `Z = L^-1 J^T`. The projection after a step is weighted by the
  mass matrix of the step's last stage and reuses its last evaluation's `J`
  and `nu` for the velocities.
- **Consequences.** 181 to 124 us per step. Callbacks must not run the
  simulator's kinematics while an evaluation is in progress (documented).
- **Record.** Journal: performance. Commit `9472d8d`.

## D21. How timings are measured

- **Context.** On the hybrid i7-13700H the same unpinned benchmark varied by
  more than a factor of two, and pinned runs an hour apart by 2.2.
- **Decision.** Timings are taken pinned to a performance core at high priority;
  two versions are compared alternately in one session, the older one built
  separately from `git archive`.
- **Consequences.** Pinned runs agree within 1 %. Absolute numbers carry the
  date and machine state; targets are judged in the machine's normal state,
  and the slow state is reported.
- **Record.** Journal: performance; `docs/performance.md`.

## D22. A development record for the book

- **Context.** The author will write a long account of the project, a thesis
  or a book of hundreds of pages. Git keeps what changed and the plan which
  tasks are done; neither keeps the reasoning, the alternatives, the dead ends
  or how numbers were measured. Conversations hold that, but they are
  temporary and outside the repository.
- **Decision.** Keep a record in the repository, written as the work happens
  and committed with it: a journal (one entry per task or session, from a
  template, with a retrospective per phase), this decision log, the raw data
  behind every number, and the book's outline and bibliography, kept current.
  Earlier periods are reconstructed from git and the plan, and say so.
- **Alternatives.** Rely on commit messages and the plan (too terse for a book);
  keep conversation transcripts (unstructured, temporary, outside the
  repository); write the book now (premature: its structure depends on work
  not yet done).
- **Consequences.** Every task ends with a journal entry in the same commit
  (working rule 7). The author completes the entries on the period before the
  review, which only the author knows.
- **Record.** Journal: Phase 2 retrospective; `documentation/journal/README.md`.

## D23. Help documentation grown with each phase, with message codes

- **Context.** The author wants the help documentation every program has, to
  use and debug the program. The plan left all documentation to Phase 15, by
  which time hundreds of messages and options would exist undocumented.
- **Decision.** A documentation track (H in the plan) that runs alongside the
  phases: a help folder `docs/help/` (getting started, concepts, how-to guides,
  troubleshooting, reference); a stable code on every error and warning
  message (for instance `MBD-K012`), each explained in a catalogue, with a test
  that every code in the sources is documented; debugging aids in the program
  (a model description, a step trace); an API reference generated from the
  header comments; and a rule that each task updates the help it affects
  (working rule 8). Phase 15 then completes and polishes rather than starts.
- **Alternatives.** Write all documentation at the end (the plan before);
  only generate an API reference (explains functions, not problems).
- **Consequences.** Message codes are introduced before Phase 3 adds more
  messages, so all new code uses them from the start.
- **Record.** Journal: Phase 2 retrospective; `documentation/Master_plan.md`,
  documentation track H.

## D24. Force laws in their own coordinates, with their derivatives

- **Context.** Task 3.1 asks for a force library whose elements return their
  derivatives with respect to position and velocity, which statics (3.5),
  linearisation (3.7) and implicit integration (Phase 4) need. Two forms are
  possible: each element's derivatives in its own coordinates (a spring's
  length and its rate, a joint's coordinate and rate, a bushing's relative
  displacement and rotation), or each element's tangent stiffness and damping
  matrices in generalized coordinates, geometric terms included.
- **Decision.** Every element states its force law in its own coordinates and
  returns the force with its two derivatives (`law(x, x_dot)` returning the
  force, `d/dx` and `d/dx_dot`), tested against finite differences of the law.
  Generalized stiffness and damping are assembled by the kernel from these and
  the elements' kinematics when statics and linearisation need them, starting
  from finite differences of the generalized forces, which the element
  derivatives then replace term by term. Tabulated characteristics are
  monotone cubic (PCHIP) curves, so that a monotone table gives a monotone
  force with a continuous stiffness. Forces on a joint coordinate (spring,
  damper, friction, limit stop on a revolute or prismatic joint) are a kernel
  element of their own that adds to the generalized forces directly: they
  need the joint's coordinate, which the world-force interface does not see,
  and are exact for it.
- **Alternatives.** Tangent matrices per element now: correct and fastest in
  the end, but a large amount of derivation (geometric stiffness of every
  element) before any statics exists to use it. Natural cubic splines for
  tables: smooth, but they overshoot between points, so a monotone damper
  table could give a force of the wrong sign.
- **Consequences.** Element derivatives are verified independently of any
  assembly. Phase 4 decides whether the analytic tangent matrices are worth
  their cost.
- **Record.** Journal: the force library.

## D25. Initial conditions: held coordinates, the diagonal of the mass matrix as the metric

- **Context.** Task 3.4 asks for initial positions and velocities solved
  "with chosen coordinates held, with the smallest correction to the rest".
  Three things had to be chosen: how a hold is expressed, in what measure a
  correction is "smallest", and whether "smallest" is meant exactly or to
  first order. The solver already had a projection (`ConstraintSolver::project`,
  task 2.6) that moves a state onto the constraints by Gauss-Newton steps,
  each the least in the kinetic-energy metric (the full mass matrix).
- **Decision.** Holds are velocity indices whose component of the correction
  is kept at zero; a joint is held whole by `hold_joint`. The metric is the
  diagonal of the mass matrix, at the given positions for positions and at the
  assembled positions for velocities. "Smallest" is exact: after a
  Gauss-Newton stage reaches the constraints, a second stage iterates on the
  whole correction from the given values until it meets the optimality
  conditions (`phi = 0`, `W d = J^T mu`). Accelerations and constraint forces
  come from the equations of motion at the assembled state, not from holds.
  Problems go to the report's warnings with codes; nothing is printed, and
  the function is `[[nodiscard]]`.
- **Alternatives.** Holds as extra constraint equations: general, but a
  constraint sees the joint transforms, not the coordinates, so a hold on an
  arbitrary coordinate could not be written as one. The identity metric:
  never changes a coordinate no constraint involves, but weighs metres
  against radians, so the result depends on the units. The full mass matrix:
  unit-consistent and the metric of the projection (and, for velocities, the
  response to an impulse), but it couples every coordinate to the bodies it
  carries, so it changes coordinates that no constraint involves; a user who
  gives a wheel's spin rate or a pendulum's angle would see it altered. The
  diagonal keeps the units right and leaves such coordinates alone. The
  Gauss-Newton stage alone: cheaper, but its result is the least correction
  only to first order. On a four-bar off by several hundredths of a radian
  its optimality condition is off by 6e-3 where the second stage reaches
  8e-14; the distance it gives away is small (3.7e-5 of the squared
  distance), as a second-order effect should be. The second stage costs a
  few iterations at setup and makes "smallest" a property that can be stated
  and tested exactly.
  Acceleration holds: the accelerations of a dynamic model follow from its
  forces, so prescribing them would contradict the equations of motion.
- **Consequences.** `Simulator::initialize()` keeps projecting, for states
  already near the constraints; `assemble()` is the initial-condition
  analysis. The rank-revealing `L D L^T` of the constraint solver became a
  class of its own (`RankRevealingLDLT`) that both use, so the assembly
  counts rank by the solver's rule (a complete orthogonal decomposition was
  tried first and cost 31 s of compile time per file). For spherical and free joints the second stage treats the
  logarithm of the relative rotation as additive, so the smallest correction
  is exact to second order in the rotation there.
- **Record.** Journal: initial-condition assembly; `docs/kernel.md`, initial
  conditions.

## D26. Statics: Newton on the accelerations at rest, least-norm steps, relaxation as fallback

- **Context.** Task 3.5 asks for Newton iteration on the static residual with
  a line search and dynamic relaxation as a fallback, done when the detailed
  sedan settles in under 20 iterations to accelerations below 1e-6. Three
  things had to be chosen: the residual and its unknowns, the Jacobian, and
  what to do where the Jacobian is singular. It is singular for every
  vehicle on tyres at rest: a tyre gives no horizontal force without slip,
  so the position on the ground and the heading have neither force nor
  stiffness. Constraints may also be redundant (the planar four-bar closed in
  3D, 5 equations of rank 2).
- **Decision.** The residual is the plan's own criterion, the accelerations
  of the constrained dynamics at rest, with time-dependent constraints frozen
  at the current time: a solver flag, `quasi_static`, takes gamma as zero,
  which at rest removes exactly the drivers' prescribed accelerations. The
  unknowns are the coordinates on the constraints. The Jacobian comes from
  central differences, each perturbed configuration projected back onto the
  constraints, so it vanishes across them and needs no null-space basis. The
  step is the least, in the weights of the diagonal of the mass matrix, that
  zeroes the linearized accelerations, solved by the shared
  `RankRevealingLDLT`: directions with neither stiffness nor force are not
  moved, and are counted (from a last Jacobian at the equilibrium) and noted.
  Each step is shortened so that no translation changes by more than 0.02 m
  and no rotation by more than 10 degrees (ADAMS's TLIMIT and ALIMIT), then
  halved until the weighted norm of the accelerations decreases; when none
  does, dynamic relaxation with kinetic damping runs until the accelerations
  at rest are small, and Newton resumes. Relaxation is driven by the
  accelerations at rest, not by the full dynamics: damping and tyre slip
  forces play no part in the equilibrium and are too stiff at low speed for
  an explicit step. `RankRevealingLDLT` became a pivoted factorization that
  stops at the numerical rank, because Eigen's LDLT, which eliminates past
  it, misjudged the rank of the statics Jacobian (journal).
- **Alternatives.** Newton on the equilibrium equations and the constraints
  together, in q and the multipliers (Haug, ADAMS): needs the derivative of
  `J^T lambda`, and its matrix is singular with redundant constraints and
  with neutral directions, both present here. Minimising the potential
  energy: needs a potential for every force (tyres, user forces) and cannot
  hold forces that have none. Coordinate partitioning (independent
  coordinates chosen from J): needs a rank-revealing decomposition of J and a
  choice that changes along the motion; the projected differences avoid both.
  Analytic stiffness from the elements' derivatives (D24): not yet assembled;
  D24 already planned differences first. Dynamic relaxation alone: robust,
  but it takes hundreds to thousands of steps where Newton takes a few.
- **Consequences.** Each iteration costs 2 nv evaluations of the
  accelerations and as many projections (measured in the journal), fine for
  setup and too slow for anything per step. Neutral directions keep their
  given values but for second-order effects, which for a vehicle means where
  it was put and where it points. Analytic tangent stiffness can replace the
  differences later without changing the method. The solver has one more
  setting, which statics sets and restores. Statics follows the model: a
  template vehicle started with its wheels in the air (no rebound stops, a
  contact point fixed on the upright) settles with its rear suspension folded,
  an equilibrium of the model but not of a car.
- **Record.** Journal: static equilibrium; `docs/kernel.md`, static
  equilibrium.

## D27. Linearisation in reduced coordinates from projected differences

- **Context.** Task 3.7 asks for state matrices about an operating point,
  natural frequencies, damping and mode shapes, done when the quarter-car
  frequencies match the analytic ones. With constraints, the coordinates are
  not independent: the full first-order system in (q, v) has spurious
  eigenvalues for the directions across the constraints, so a minimal set of
  coordinates on them is needed. The derivatives of the accelerations had to
  come from somewhere (D24: differences first).
- **Decision.** Reduced coordinates along n independent columns of the
  W-orthogonal projector onto the allowed motions (W the diagonal of the mass
  matrix), picked by the pivots of the shared `RankRevealingLDLT`; the
  derivatives of the accelerations along them by central differences, each
  perturbed position projected back onto the constraints (as in statics);
  reduction by the W-weighted left inverse of the basis; `A = [[0, I], [A_q,
  A_v]]`; modes from Eigen's general eigensolver, a complex pair listed once.
  Everything the simulator applies acts, as at the operating point.
- **Alternatives.** A null-space basis from an SVD or QR of J: orthonormal,
  but another heavy decomposition to compile, and the projector's pivoted
  columns serve as well (they need not be orthonormal). Coordinate
  partitioning (independent velocities chosen from J): the basis would be
  made of unit vectors, but the choice changes from one configuration to
  another. The symmetric generalized eigenproblem `K phi = w^2 M phi`:
  cheaper and exact for undamped conservative systems, but it has no damping
  and the plan asks for it.
- **Consequences.** The general eigensolver costs compile time in one source
  file only. Linearising about a state in motion uses the same formulas, which
  are exact for joints whose coordinates add (for quaternion joints the
  moving frame adds terms not modelled, as noted in the header). Input
  matrices (B, for control) are left for when a controller needs them.
- **Record.** Journal: linearisation; `docs/kernel.md`, linearisation.

## D28. Contact: ADAMS's IMPACT law and a regularised friction coefficient

- **Context.** Task 3.8 asks for point and sphere contact with a plane, with
  penalty stiffness, nonlinear damping and regularised friction, done when a
  block on an incline slides at the correct friction angle. Two choices
  shape everything else: how the normal force starts and ends, and how
  friction behaves near zero sliding speed, where Coulomb's law is
  discontinuous.
- **Decision.** The normal force is `k delta^e + c(delta) delta_dot`, clamped
  at zero, with the damping coefficient rising along a cubic step from zero
  at the surface to its full value at a chosen depth: the IMPACT function of
  ADAMS, so the force is continuous at first contact and at separation. The
  friction coefficient is a function of the sliding speed alone: a cubic step
  from 0 at rest to `mu_static` at `v_static`, another to `mu_dynamic` at
  `v_dynamic`, constant beyond (ADAMS's friction with its two transition
  velocities). The plane is a marker's XY plane on any body; the force acts on
  both bodies at the sphere's lowest point. Both laws return their
  derivatives (D24).
- **Alternatives.** Damping at full strength from first contact (a linear
  Kelvin-Voigt contact): the force jumps on landing and pulls as the bodies
  part. A bristle model (LuGre, Dahl) or a stick-slip state machine: true
  sticking, but they need an extra state per contact or events, which arrive
  with Phase 4 (extra first-order states) and task 3.9. A single tanh
  regularisation, as for joint friction (task 3.1): no difference between
  static and dynamic friction, and the incline test is about exactly that.
- **Consequences.** A loaded contact below the friction limit creeps at a
  speed below `v_static`, predictable from the step (the test checks it to
  rounding); a smaller `v_static` creeps less and stiffens the equations.
  Sticking proper, and contact between general shapes, are later work.
- **Record.** Journal: contact; `docs/kernel.md`, forces and simulation.
