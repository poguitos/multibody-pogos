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
| D23 | Help documentation grown with each phase, with message codes | 5 Oct 2026 | In force (H.1 to H.5 done) |
| D24 | Force laws in each element's own coordinates, with their derivatives; joint-coordinate forces act on generalized forces directly | 5 Oct 2026 | In force, task 3.1 |
| D25 | Initial conditions: held coordinates, the diagonal of the mass matrix as the metric | 13 Oct 2026 | In force, task 3.4 |
| D26 | Statics: Newton on the accelerations at rest, least-norm steps, relaxation as fallback | 13 Oct 2026 | In force, task 3.5 |
| D27 | Linearisation in reduced coordinates from projected differences | 13 Oct 2026 | In force, task 3.7 |
| D28 | Contact: ADAMS's IMPACT law and a regularised friction coefficient | 13 Oct 2026 | In force, task 3.8 |
| D29 | Events: sign changes at step ends, located by re-integration | 13 Oct 2026 | In force, task 3.9 |
| D30 | State elements: first-order states integrated with the bodies | 14 Oct 2026 | In force, task 4.1 |
| D31 | Dormand-Prince within the fixed output step, with events on its interpolant | 14 Oct 2026 | In force, task 4.2 |
| D32 | Generalised-alpha on the state-space accelerations, with Newton on difference Jacobians | 14 Oct 2026 | In force, task 4.3 |
| D33 | SDIRK4 on the stacked first-order system, kept for smooth models | 14-16 Oct 2026 | In force, task 4.4 |
| D34 | Radau IIA for stiff mechanical models, with Newton held to a tolerance-dependent fraction | 16 Oct 2026, the method by the author | In force, task 4.4 |
| D35 | Linearly implicit Euler, with force derivatives element by element | 16 Oct 2026 | In force, task 4.5 |

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

## D29. Events: sign changes at step ends, located by re-integration

- **Context.** Task 3.9 asks for the detection of sign changes in user
  functions, with the step shortened to the event in variable-step mode,
  done when a bouncing ball's impact times are correct to the tolerance. The
  simulator has fixed-step integrators only until Phase 4, and no dense
  output (an interpolant of the state within a step).
- **Decision.** Each event is a function of the simulator's state, an
  optional action and a watched direction. Its value is compared at the
  start and the end of each stretch of a step; on a sign change the crossing
  is located by the Illinois method on the step length, each trial
  re-integrating one step from the start (the same method, so the located
  time is that of the numerical solution), to a tolerance in time. The state
  is left just past the crossing, the action runs, and the step continues to
  its end. A step of dt still advances by dt. A chain of more than
  `max_events_per_step` events within one step is cut with a warning.
- **Alternatives.** Dense output: one interpolant per step instead of a
  re-integration per trial, but it needs the embedded methods of Phase 4,
  which will reuse the same locating code with the interpolant. Ending the
  step at the event (the caller sees a shorter step): simpler, but every
  caller of `step()` and `run()` would have to cope with steps of varying
  length. Leaving the state just before the crossing: actions that reverse
  the motion (bounces) then see no false crossing, but every action that does
  not would fire twice; "just past" fires once, and a bounce watches one
  direction.
- **Consequences.** Only signs at the ends of stretches are compared, so a
  function that goes out and back within one step is missed; near the end of
  a bouncing chain whose bounces shrink below the step, the ball passes
  through. Smaller steps, a rest condition in the action, or Phase 4's
  variable steps address it. Each located event costs one extra step per
  trial of the root finder.
- **Record.** Journal: events; `docs/kernel.md`, events.

## D30. State elements: first-order states integrated with the bodies

- **Context.** Task 4.1 asks for one integrator interface, with extra
  first-order states (tyre lag, actuators, controllers) registered by the
  elements that own them, replacing the callbacks before the forces and after
  each step; done when the drivetrain no longer uses callbacks. The
  drivetrain advanced its wheel spins once per step in its own linearly
  implicit Euler step (Phase 1), outside the integrator, which left an O(dt)
  error of about 0.2 % in braking and could not benefit from a better
  integrator.
- **Decision.** A `StateElement` owns a number of states, gives their initial
  values, receives the states of every evaluation before the forces
  (`begin_evaluation`), returns their rates after the dynamics (`rates`), and
  makes discrete changes after every completed step (`end_step`). The
  simulator's state is (q, v, x); every integrator carries x through its
  stages; events save and restore it. Elements are attached to a simulator
  (`attach`), not owned by it, as the callbacks captured their owners. The
  callbacks before the forces and after the step are removed;
  `force_callback`, a plain input of generalized forces, stays. The
  drivetrain becomes a state element: its spins are integrated states, its
  engine torque follows the spins within a step, its brake is Coulomb
  friction on a turning wheel and a lock decided at the end of a step for a
  wheel at rest, and it shifts gear there.
- **Alternatives.** Keeping the drivetrain's own implicit step behind the new
  interface (a "discrete state"): robust at standstill, but the spins would
  still not be integrated, and the plan's later tasks (implicit methods,
  multirate) need them to be. A regularised brake (friction times
  tanh(omega / omega_0)): continuous, but stiff enough near zero to defeat an
  explicit step unless omega_0 is so large that a braked wheel creeps; the
  lock keeps a stopped wheel exactly stopped. Owned elements (shared
  pointers in the System): uniform with force elements, but the drivetrain is
  built after the simulator, from its state, and its users hold it by value.
- **Consequences.** At low speed the spin equation is stiff (the tyre's slip
  stiffness over the wheel's inertia is about 2700 per second on an undriven
  40 kg wheel below V_low), just inside RK4's limit at 1 ms (2.8 / dt) and
  outside explicit Euler's (2 / dt): with the drivetrain, semi-implicit Euler
  at 1 ms is not stable at standstill, and lighter wheels or longer steps
  need the implicit methods of tasks 4.3 to 4.5. Locking is decided once per
  step, so a wheel stops to within a step.
  *Correction, 14 Oct 2026 (task 4.3):* the figure of 2700 per second was an
  estimate. Measured on the simple vehicle at 0.5 m/s, RK4 keeps the front
  spins at 0.9 ms but not at 1 ms, so the rate is 2800 to 3100 per second,
  just outside RK4's limit at 1 ms, not inside it; beyond the limit the spins
  chatter by a few per cent, bounded by the tyre's saturation, rather than
  blow up. Generalised-alpha (D32) carries them at any step.
- **Record.** Journal: the integrator interface and state elements;
  `docs/kernel.md`, forces and simulation.

## D31. Dormand-Prince within the fixed output step, with events on its interpolant

- **Context.** Task 4.2 asks for an embedded pair (Dormand-Prince) with error
  control and dense output; done when the measured convergence order matches
  theory. The simulator is driven by `step(dt)` calls, between which users
  change inputs (`tau`, the drivetrain's pedals), state elements make
  discrete changes, and events act: the state at the end of each call must be
  the integrator's own, not an interpolation.
- **Decision.** `Integrator::DormandPrince` advances each `step(dt)` in steps
  of its own, chosen by the standard controller (root-mean-square error
  measure over q, v and x, weights `atol + rtol |y|`, safety 0.9 on the fifth
  root, growth at most 5, none after a rejection), the last one cut to end at
  t + dt exactly; the length reached carries over to the next call. Each step
  is projected onto the constraints. The seventh stage is reused as the next
  step's first only when nothing changed the state after it (no quaternions to
  normalise, no projection, no event) and never across calls. Events are
  checked at the end of each of its steps and located on the step's
  continuous extension (fourth order, Hairer's DOPRI5 form), costing
  kinematics but no evaluation of the forces; the state at the crossing is
  the interpolated one, projected. `StepControl` holds the settings,
  `IntegratorStats` counts steps, rejections and evaluations for every
  method, `interpolate()` gives the extension within the last step. With
  `adaptive` off it takes one step per call, for measuring orders. The
  drivetrain's brake keeps the direction it had at the start of each step
  through the step and decides it again at the end.
- **Alternatives.** Running ahead of the output times and interpolating back
  (as ODE codes with dense output usually do): fewer steps cut short, but the
  inputs set between calls, the state elements' discrete changes and the
  events' actions would all apply to a state the integrator has passed.
  Locating events by integrating again, as RK4 does: correct, but each trial
  costs a full step of seven evaluations, where the extension needs none.
  Keeping the brake's sign at each evaluation: under error control, a spin
  crossing zero flips it between the stages, the error estimate does not
  shrink with the step, and the step falls to about 1e-9 s and below (the
  sign is discontinuous in the state); a regularised brake was already
  rejected in D30.
- **Consequences.** Output steps much shorter than the steps the error
  control would choose cut every step short: at 1 ms on the vehicles,
  Dormand-Prince costs seven evaluations per output step against RK4's four,
  and is chosen for its error control, not its speed. The tolerances mix
  units (m, rad, m/s, quaternion entries), one value for all. A crossing that
  only the projection makes is placed at the step's end. With the brake's
  direction fixed per step, a wheel whose spin passes zero within a step
  ends it turning slightly backwards unless the lock takes it, as before;
  the lock decides at the step's end in both cases.
- **Record.** Journal: the Dormand-Prince pair; data
  `2026-10-14_dormand-prince.md`; `docs/kernel.md`, integrators.

## D32. Generalised-alpha on the state-space accelerations, with Newton on difference Jacobians

- **Context.** Task 4.3 asks for an implicit method for stiff models,
  starting with generalised-alpha (HHT), with finite-difference Jacobians
  first and the force elements' analytic derivatives later; done when a
  spring 1000 times too stiff for RK4 at 1 ms runs stably. The simulator's
  accelerations already include the constraint forces (the constrained
  dynamics at acceleration level, then a projection after each step), and
  since task 4.1 the state elements' states are integrated with the bodies.
- **Decision.** `Integrator::GeneralizedAlpha`: the Chung-Hulbert method in
  the Arnold-Bruls form, its four parameters from one number, the spectral
  radius at an infinite step (`rho_infinity`, 0.8 by default), applied to the
  accelerations the simulator already computes. Positions move through the
  joints' exponential maps, so quaternions stay unit (the Lie-group form).
  The state elements' states follow the same formulas a level down, with
  their rates for the accelerations. Each step solves for the new
  acceleration-like variables by Newton's method; the iteration matrix
  comes from forward differences of the accelerations and rates with respect
  to q (along the velocities), v and x (2 nv + nx evaluations), is kept from
  step to step, re-formed for each step length (cheap: only the matrix
  arithmetic and its LU), and rebuilt when Newton slows (four iterations, or
  a contraction worse than one half) or fails (then the step is retried
  once). The accelerations at each step's start are evaluated afresh, so
  inputs changed between steps take effect; the method's history is kept
  only while nothing but its own step and the projection has changed the
  state. The LU is a short in-place factorization with partial pivoting,
  allocation-free. Constraints stay as for the explicit methods:
  accelerations from the constrained dynamics, projection after each step.
- **Alternatives.** HHT-alpha proper (alpha_m = 0): the member of the family
  at rho_infinity = 0.5; Chung-Hulbert's choice damps the slow motions less
  for the same high-frequency damping. An index-3 formulation with the
  multipliers as unknowns (as in Arnold-Bruls): the natural form for
  generalised-alpha with constraints and the subject of task 4.7, but a
  second solver path beside the one every other method uses. Analytic
  Jacobians from the force elements' `law()` derivatives now: they need the
  derivative of the constrained forward dynamics as well, and the
  difference Jacobian, built rarely, is not where the time goes in these
  tests; deferred to task 4.5, where a fixed cost per step depends on it.
  Eigen's `PartialPivLU` as a member: it reallocates when resized and
  brings its own compile cost to the simulator's header.
- **Consequences.** A stiff model runs at any step, with its fast motions
  damped by rho_infinity per step; second order, with a phase lag of 0.088
  (w h)^2 per radian at the default (1/12 at rho_infinity = 1). The first
  step of a run, and a step after a lock, an event or a user's change,
  restarts the history (q and v together, each state on its own). The
  iteration matrix costs 2 nv + nx evaluations (24 on the simple vehicle),
  paid rarely: one matrix served a whole 2.3 s run of the rolling car. A
  step costs one evaluation plus one per Newton iteration: three on the sedan
  at 1 ms against RK4's four, plus forming and factorizing an (nv + nx)
  square matrix (not timed). Newton's test has a floor at the rounding of
  the step's terms, without which a stiff spring at a long step cannot
  converge. A discrete decision a state element must hold exactly (a lock)
  is enforced in its end_step: Newton solves only to the tolerance. No error
  control yet (fixed steps; task 4.4 brings an error estimate).
- **Record.** Journal: generalised-alpha; data
  `2026-10-14_generalized-alpha.md`; `docs/kernel.md`, integrators.

## D33. SDIRK4 on the stacked first-order system, kept for smooth models

- **Context.** Task 4.4 asks for the five-stage, fourth-order, L-stable
  SDIRK scheme used by Haug and co-workers; done when it beats
  generalised-alpha on the bushed benchmark at tight tolerance. No benchmark
  of that name existed.
- **Decision.** `Integrator::SDIRK4`: Hairer and Wanner's SDIRK4 (gamma =
  1/4, stiffly accurate, embedded third-order solution), applied to the
  stacked system y = (q, v, x), y' = (q_dot, accelerations, rates), with q
  in its embedding and normalized once per step, as RK4 and Dormand-Prince
  do. Each stage is solved by Newton with the matrix I - gamma h J, one LU
  per step length and J; J by forward differences (n + 1 evaluations), kept
  across steps while the Newton evaluations of a step, beyond those of the
  first step with it, stay below n + 1, and rebuilt at a stage whose Newton
  fails. Newton stops at a fraction of the tolerance (D34) and fails when a
  correction is not smaller than the one two iterations before. The error
  estimate is filtered through (I - gamma h J)^-1 and drives the same step
  control as Dormand-Prince (own steps ending at t + dt, the fourth root for
  a third-order estimate). If Newton fails even with a fresh matrix, the
  step is quartered (with error control) or the run stops with K058
  (without). The benchmark: a double pendulum whose hinges are stiff damped
  bushings between two free bodies (`tests/kernel/bushed_pendulum.hpp`), with
  Dormand-Prince at 1e-12 as the reference.
- **Alternatives.** Positions through the exponential map, as
  generalised-alpha: beyond second order a Lie-group Runge-Kutta method
  needs the inverse of the exponential's differential in its stages
  (Munthe-Kaas). Unknowns of the size of the accelerations only: smaller
  matrices, but with quaternions the stage positions depend on themselves
  through q_dot = G(q) v. Unfiltered error estimates: the embedded solution
  is not A-stable (|R_hat(iy)| reaches 3.3). Hairer and Wanner's Newton
  tests on the rate of contraction (stop when theta / (1 - theta) times the
  correction is small; fail early when the iterations left will not do):
  tried and dropped, because on the bushed pendulum the first correction
  from the predictor contracts by 0.25 to 0.8 and the next ones by 1e-2 to
  1e-4, so the tests gave up on solves about to converge, even with J built
  at the stage, and J was rebuilt almost every stage.
- **Consequences.** Measured, not foreseen: SDIRK4 does not beat
  generalised-alpha on the bushed benchmark, and cannot. It is fourth order
  on smooth models (the oscillator; the pendulum with soft bushings: error
  ratios 14.0 and 16.1 when the step halves), but its stage order is 1 (the
  stage conditions A c = c^2 / 2 fail by 0.031 to 0.046 in its first four
  stages), and as the bushings stiffen towards joints the model tends to a
  constrained (index-3) system, for which the order of such a method falls
  towards its stage order (Hairer and Wanner, section VII.4). Order measured
  on the pendulum at h = 2, 1 and 0.5 ms: about 3.6 with k = 1e6 N/m, 2.5
  with 1e8 (the benchmark), 1 with 1e10. On the benchmark at 0.5 ms it is 4.2e-5
  m out after 1 s, against generalised-alpha's 1.7e-6 at the same step, and
  its error estimate understates the error (tolerance 1e-4: 7e-4 m).
  Asked how to go on, the user chose (16 October 2026) to keep SDIRK4 for
  smooth models and stiff first-order states, and to add Radau IIA within
  task 4.4 (D34). `Integrator::SDIRK4`'s documentation says where not to use
  it.
- **Record.** Journal: SDIRK4 and Radau IIA; data
  `2026-10-16_sdirk4-radau.md`; `docs/kernel.md`, integrators.

## D34. Radau IIA for stiff mechanical models, with Newton held to a tolerance-dependent fraction

- **Context.** SDIRK4 loses its order on stiff bushings (D33), the case the
  benchmark stands for and the vehicles are full of. A Runge-Kutta method
  of higher stage order keeps more of its order as the model tends to a
  constrained one: three-stage Radau IIA, stage order 3, fifth order, is the
  standard (Hairer and Wanner's RADAU5).
- **Decision.** `Integrator::RadauIIA` on the same stacked system and the
  same J as SDIRK4 (one shared `stacked_f`, one Jacobian, one cost rule).
  The 3n Newton equations of a step split, through A^-1 = T diag(gamma,
  [alpha -beta; beta alpha]) T^-1, into one real and one complex system of
  n (`gamma I - h J`, `(alpha + i beta) I - h J`), factorized once per step
  length and J by the same LU, templated for complex matrices. T and the
  eigenvalues are computed once from closed forms (the roots of z^3 - 9 z^2
  + 36 z - 60) and the eigenvectors as cross products of rows, not copied
  from RADAU5's constants. Starting values from the last step's collocation
  polynomial when nothing has changed the state since; RADAU5's error
  estimate (gamma I - h J)^-1 (h f(y_n) + sum dd_i Z_i), O(h^4), evaluated
  once more at y_n + estimate after a first step or a rejection; one extra
  evaluation per step for f(y_n), none without error control (which then
  makes no estimate). Newton failure: rebuild J at the step's start and
  retry; then halve the step (with error control) or K058. For both SDIRK4
  and Radau IIA, Newton stops when its corrections fall below min(0.03,
  sqrt(tol)) of the tolerances (RADAU5's rule, tol the larger of the two),
  and never below a floor at the rounding of the stage's terms plus the
  rounding of the state amplified by J (64 eps h sum_j |J_ij| |y_j|).
- **Alternatives.** A fixed tenth of the tolerance for Newton: measured on
  the benchmark, Newton's errors then dominated the global error (0.4 to 7
  times the tolerance, because the fifth-order truncation error is far
  below its third-order estimate and Newton's errors add up undamped);
  a hundredth or a thousandth brought it to 0.01 to 0.3 times the tolerance
  for 15 to 35 per cent more evaluations. RADAU5's rule without the J term
  in the floor: at tolerance 1e-10 it asked for corrections of 1e-15, below
  the noise a 1e8 N/m bushing makes of the positions' rounding, and every
  fixed step failed (K058) at once. A J rebuilt at the step's middle when
  Newton fails: rescued steps of 4 ms but not 8 ms, at more cost than 2 ms
  steps. Solving the 3n system directly: n^3 work nine times over.
  BDF (ADAMS's GSTIFF): a multistep history to restart at every event and
  lock; a later option if Radau IIA proves too costly on large models.
- **Consequences.** On the benchmark, with error control at 1e-6, Radau IIA
  ends 1.0e-7 m from the reference after 1 s for 11548 evaluations, where
  generalised-alpha needs fixed steps of 0.25 ms (4.6e-7 m) and 26057;
  at 1e-8, 4.1e-9 m for 13986, where generalised-alpha needs about 48600
  for 7.8e-8 m: the "done when". At fixed steps of 2 ms it is within 5e-11 m
  but fails at 4 ms once the links swing fast (Newton, K058): its stages
  span the step, and the bushings' coupling of torsion and bending turns
  with the swing. Per step: three evaluations per Newton iteration, a real
  and a complex LU of n when h or J changes (n = 26 here; not timed for
  large models). SDIRK4's Newton changed with this decision (the fraction
  and the floor), not its order. Generalised-alpha's Newton now judges
  progress over two iterations, as the two Runge-Kutta methods do, and
  rebuilds its matrix when the average contraction is worse than a half.
- **Record.** Journal: SDIRK4 and Radau IIA; data
  `2026-10-16_sdirk4-radau.md`; `docs/kernel.md`, integrators;
  troubleshooting.

## D35. Linearly implicit Euler, with force derivatives element by element

- **Context.** Task 4.5: a fixed-cost implicit step, the real-time
  integrator; done when its cost per step does not vary. Task 4.3 deferred
  analytic Jacobians to this task. A difference Jacobian of the whole
  dynamics costs 2 nv + nx + 1 evaluations (59 on the sedan, about 1.5 ms),
  too much for every step at 1 ms; kept across steps, as generalised-alpha
  and Radau IIA keep theirs, it makes the cost vary (generalised-alpha on the
  sedan: 2 to 76 evaluations a step).
- **Decision.** `Integrator::LinearlyImplicitEuler`: backward Euler on the
  velocities (positions moved with the new velocities), linearized once per
  step about its start, as a correction to the explicit Euler step:
  (M - h C - h^2 K) delta = h^2 (K (v + h a) + C a), with J delta = 0 and
  the projection after the step; the state elements' states implicit in
  themselves (X) and coupled to the velocities (Xv, Tx), eliminated by a
  small Schur complement. One evaluation, one set of derivatives and one
  solve per step, always. The derivatives every step: K and C of the force
  elements by forward differences of each element's apply() alone on its
  bodies' states (poses and velocities, world axes), mapped through the
  bodies' Jacobians at the columns they can reach; joint forces by
  differences in their coordinates; X and Tx by differences in each state
  with the forces applied again; Xv by taking the states' rates again after
  each velocity perturbation of an element. Left explicit: the inertial and
  gyroscopic terms, the joints' geometry, `tau`, `force_callback`, and the
  states' dependence on positions (Xq). K, C and Tx G enter by their
  symmetric parts, so that `ConstraintSolver::constrained_solve` uses the
  Cholesky and rank-revealing factorizations of forward dynamics. If the
  matrix is not positive definite the step drops K, then everything
  (MBD-K045, once). `Simulator::force_jacobian` makes K and C available.
- **Alternatives.** The whole dynamics differentiated every step: fixed
  cost, but 59 evaluations. Analytic derivatives of the whole dynamics
  (Carpentier and Mansard's derivatives of RNEA and ABA, with those of the
  constraints): exact and fast, a large piece of work; not needed for
  stability, which rests on the stiff forces. Analytic derivatives of each
  element's loads (its law() derivatives and its geometry): faster than
  differences, but one derivation per element type, the tyres' among them;
  differences of one element give the same derivatives from apply() alone,
  cover every element, and leave room for analytic versions where timing
  asks. A non-symmetric constrained solve, keeping the antisymmetric parts
  implicit: needs a rank-revealing factorization of a non-symmetric matrix
  for redundant constraints. The states implicit in themselves only:
  measured, the sedan accelerated a fifth too slowly at 1 ms (the tyre's
  slip pulls the hub towards the spin of the step's start, a strong damper
  that brakes the car). The states coupled through Xq as well: measured, a
  car coasting at 0.5 m/s gained 0.5 % of its speed in 2 s at 5 ms and 5 %
  at 10 ms (the tyre's rolling radius depends on its deflection, so the
  wheel's height changes the slip: a non-symmetric loop the symmetric matrix
  cannot carry); without Xq it keeps its speed to 2e-5 at every step to 10
  ms. A second-order linearly implicit method (a trapezoid, or
  generalised-alpha with one Newton iteration on these derivatives): more
  accurate, less damping, not L-stable or not tested; an open end.
- **Consequences.** On linear systems each step is backward Euler's to 2e-13;
  a spring 1000 times too stiff for RK4 shrinks by 1 / sqrt(1 + (w h)^2) per
  step; first order (ratio 2.07 on a spatial pendulum, 2 mm after 1 s at 1
  ms on a swing of a metre: a real-time method, not an accurate one). On the
  sedan at 1 ms every step of a drive takes one evaluation and one set of
  derivatives, the drive matches RK4's (top speed 1.4684 against 1.4685
  m/s), and a step takes 106 us (median, pinned) against RK4's 121 and
  generalised-alpha's 80, with 99 % of steps within 1.3 times the median;
  generalised-alpha's longest steps take 2.1 to 2.3 ms (a new iteration
  matrix). Every motion is damped by about (w h)^2 / 2 per step. The
  projection after the step still varies (a Gauss-Newton iteration or
  none): task 4.7. The element-by-element derivatives can serve the other
  implicit methods' iteration matrices too.
- **Record.** Journal: the fixed-cost step; data
  `2026-10-16_linearly-implicit-euler.md`; `docs/kernel.md`, integrators;
  `docs/performance.md`; troubleshooting.
