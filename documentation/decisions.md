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
| D32 | Generalised-alpha on the state-space accelerations, with Newton on difference Jacobians | 14 Oct 2026 | In force, task 4.3; its constraints: D37 |
| D33 | SDIRK4 on the stacked first-order system, kept for smooth models | 14-16 Oct 2026 | In force, task 4.4 |
| D34 | Radau IIA for stiff mechanical models, with Newton held to a tolerance-dependent fraction | 16 Oct 2026, the method by the author | In force, task 4.4 |
| D35 | Linearly implicit Euler, with force derivatives element by element | 16 Oct 2026 | In force, task 4.5; inertial terms: D38; derivatives whole: D40 |
| D36 | Multirate RK4, fastest first, for the state elements' states | 17 Oct 2026 | In force, task 4.6 |
| D37 | Constraints per integrator: projection after every method's steps, generalised-alpha's stabilized index-2 form as an option | 17 Oct 2026 | In force, task 4.7 |
| D38 | The inertia's velocity products implicit, and the new velocities on the constraints, in the two Euler steps | 18 Oct 2026 | In force, task 4.7's open end |
| D39 | The mass matrix factorized along the tree, the dense Cholesky kept for long chains | 18-19 Oct 2026 | In force, task 4.8 |
| D40 | The linearly implicit step with its force derivatives whole, not their symmetric parts | 19 Oct 2026 | In force, task 4.9 |
| D41 | The integrators verified by a program that writes the report's figures and tables | 19 Oct 2026 | In force, task 4.9 |
| D42 | Ideas from external sources adopted into the plan | 20 Oct 2026 | In force, plan amended (5.1, 5.2, 5.5, 7.2, 7.6, 7.8, 8.2, 8.6, 9.1, 9.3, 9.5, 9.7, 10.1, 10.7, 13.2, 13.3, 13.5, 13.8, 13.9, 14.7, H.7) |
| D43 | Analytic cases: manufactured trajectories checked by one step and over a span; the modal energy from the whole mass matrix | 20 Oct 2026 | In force, task 5.1 |
| D44 | The IFToMM benchmarks against closed forms; singular positions passed by holding the constraints near rounding; conflicting equations named by a deletion filter | 20 Oct 2026 | In force, task 5.2 |
| D45 | A sanitizer report fails its test; an Eigen factorization sized by computing it, never by moving an empty one | 20 Oct 2026 | In force, task 5.2's close |

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

## D36. Multirate RK4, fastest first, for the state elements' states

- **Context.** Task 4.6: advance the fast states (wheel spin, tyre lag)
  several times per main step, slowest first, as in Haug, Negrut, Serban and
  Solis (1999), section 6; done when a ratio of 6 reproduces the single-rate
  result. The paper's partition is the one the simulator already has (the
  bodies, and the states x' = Gamma(q, q', x, t)), and its slow integrator a
  constant-step third-order Adams-Bashforth method: one evaluation per step,
  needing the fast states only at the step's start, which is what makes
  slowest first natural there. The paper itself warns that extrapolating
  the fast states into the slow step "is subject to numerical errors".
- **Decision.** `Simulator::multirate` (RK4 only; MBD-K046 otherwise): per
  step of H, the bodies' first RK4 stage (a full evaluation), then m RK4
  steps of H / m for the states, whose rates are evaluated without the
  dynamics (kinematics, force elements, rates; counted in
  `IntegratorStats::state_evaluations`), the bodies extrapolated from that
  first stage (v + s a, q (+) s (v + s a / 2)); then the bodies' other three
  stages, which see the states the fast steps reached at H / 2 and H.
  Fastest first: the slow part, being smooth, is the one extrapolated.
- **Alternatives.** Slowest first with the states held through the bodies'
  step: measured, braking from 20 m/s came out 2.2 % short at 6 ms (each
  tyre's slip drifts over the step as the hub slows and the spin is held).
  Slowest first with the states extrapolated along the last step's change:
  equal on braking and slightly better on a car rolling at 0.5 m/s (wheels
  within 5.7e-5 against 6.4e-5), but three times worse on the sedan's drive
  (top speed 0.41 % off against 0.14 %), and in principle unsafe for a stiff
  state in a transient, whose last change overshoots. The paper's own
  scheme, an explicit multistep method for the bodies: one evaluation per
  step, but a new integrator whose history must restart at every lock,
  event and gear change, and Adams-Bashforth's small stability region; it
  may come with task 12.4. A chosen subset of the states as the fast part:
  every state element is fast so far.
- **Consequences.** Where the states drive the bodies without depending on
  them (a lag pushing a mass) the step keeps RK4's fourth order. At walking
  pace with a ratio of 6 (3 ms, fast steps of 0.5 ms) it reproduces
  single-rate RK4 at 0.5 ms: the rolling car's wheels within 6.4e-5 of free
  rolling (single rate at 0.5 ms 3.7e-5; at 3 ms its spins chatter by 48
  %); the sedan's drive within 0.14 % on the top speed, at rest at the end,
  for a sixth of the dynamics' evaluations and 2.8 times less time. Each
  part must keep within its own stability: fast steps of 1 ms, where the
  sedan's front spins chatter, made the car drift (0.5 m/s two seconds after
  it had stopped), because the bodies' stages sample the chatter that single
  rate averages; a step of 6 ms on the simple vehicle at 0.5 m/s was
  unstable through the hubs' side of the tyres' slip. The coupling error is
  of second order in H (0.04 % at 1.5 ms, 0.14 % at 3 ms on the sedan).
- **Record.** Journal: multirate; data `2026-10-17_multirate.md`;
  `docs/kernel.md`, integrators; troubleshooting.

## D37. Constraints per integrator: projection after every method's steps, generalised-alpha's stabilized index-2 form as an option

- **Context.** Task 4.7: projection for explicit methods, a stabilised
  index-2 form for implicit methods, and a drift channel in the output; done
  when the drift stays under tolerance for 1000 simulated seconds. Every
  evaluation solves the loop constraints at the acceleration level (index
  1), and since task 2.7 every method projected q and v after each step.
  Measured on a four-bar with a pendulum before any change: the projection
  keeps or improves the accuracy of RK4, Dormand-Prince and SDIRK4 (and of
  Radau IIA down to the projection's own floor of 1e-10), as the theory
  says (coordinate projection keeps a one-step method's convergence;
  Eich-Soellner and Fuhrer, section 5.3). Generalised-alpha is different:
  its acceleration-like variable lags the accelerations, so each step
  leaves the constraints by far more than its error along them (velocity
  drift per step 3.3e-3 m/s at 5 ms, 300 times RK4's), and a projection
  removes that from the state but not from the method's history. Left free
  it drifts by 1.3e-3 in 20 s.
- **Decision.** `Simulator::enforce_constraints` and `constraint_tolerance`
  (renamed from `project_constraints` and `projection_tolerance`). Every
  method, explicit and implicit, projects after each of its steps, as
  before. Generalised-alpha can instead solve each step on the constraints,
  `implicit_control.stabilized_index2`, off by default: the stabilized
  index-2 form (Gear, Leimkuhler and Gupta 1985; for generalised-alpha
  Arnold, Bruls and Cardona 2015, 2016), in which Newton's unknowns gain,
  per independent equation, kappa, a constraint force J^T kappa added to
  the index-1 accelerations, which holds J v+ = nu at the step's end, and
  eta, a position correction M^-1 J^T eta in the position update, which
  holds phi(q+) = 0. kappa is the difference between the multipliers and
  those of the index-1 solve, so the discrete solution is the one with the
  multipliers themselves as unknowns. The iteration matrix takes J, M^-1
  J^T and the independent equations from Newton's first evaluation, at the
  predictor, and the derivative G of J v - nu with respect to q from the
  differences that build the rest of it; eta is eliminated through the
  Cholesky factor of A = J_c M^-1 J_c^T, leaving n + m unknowns; redundant
  equations are dummies, so the matrix keeps one size. Newton stops when,
  besides its usual test, |phi| <= constraint_tolerance and |J v - nu| <=
  step_control.absolute_tolerance. The drift channel:
  `Simulator::constraint_drift()` (what each step left before the
  projection, the residuals after, the largest of each),
  `add_drift_channels(recorder, sim)`, and a velocity drift column in the
  trace. MBD-K047 for a tolerance that is not positive.
- **Alternatives.** The stabilized form for every implicit method, as the
  plan put it: SDIRK4 and Radau IIA are one-step methods that the
  projection does not hurt, and Hairer and Wanner's comparison on a seven-
  body mechanism (section VI.9) found RADAU5 alike on the index-1, index-2,
  index-3 and GGL forms, the GGL form needing the most linear algebra; the
  linearly implicit step is one-step and first order. The stabilized form as
  generalised-alpha's default: chosen at first, then measured on the sedan
  (n = 31, m = 17), pinned, against the previous commit in one session: a
  step cost 169 us against 98 with the Newton matrix of n + 2 m, 145 us with
  eta eliminated, of which about 17 us the larger LU and most of the rest
  forming its blocks (Eigen's blocked LU was only a quarter faster than
  ours at these sizes); on a manoeuvre with a steering ramp it took 12 to 23
  % more evaluations (each step absorbs the driver's change) for positions
  1.15 times more accurate and the same velocities. A projected step 1.45
  times shorter costs the same and gains more. ADAMS makes the same choice
  between its I3 (default) and SI2 formulations. Within the stabilized form:
  the matrix's constraint rows from the step's start, without G (Newton
  contracted by 100 per iteration at best and failed at 4 and 5 ms on the
  four-bar: J turns by about h |v| / length across a step, and G is of the
  same size); |J v - nu| held to constraint_tolerance (5.3 Newton iterations
  a step instead of 3.3, and the last correction, through the simplified
  matrix, left 4e-9 all the same; a velocity residual does not add up, since
  the next step's positions are held to phi = 0 again). Baumgarte
  stabilization: kept for runs with `enforce_constraints` off; its rates
  depend on the model and the step.
- **Consequences.** The one-step methods and generalised-alpha by default
  behave as before task 4.7; the drift channel shows, for each, the drift
  each step leaves (on the four-bar at 5 ms: RK4 1.8e-7, Dormand-Prince
  3e-9, Radau IIA 4e-10 in positions; generalised-alpha 2.6e-5) and the
  residuals after (within 1e-10, and 5e-15 in velocities). Over 1000
  simulated seconds of the four-bar and of a heavy top held at its tip, no
  residual left the tolerance and no drift grew. Stabilized,
  generalised-alpha converges at second order on the four-bar (ratios 4.09
  and 4.04 at 2, 1 and 0.5 ms), its steps ending within 1.2e-15 of the
  constraints with nothing projected, more accurately than projected at
  every step (1.34, 1.18, 1.09 times there), for the same Newton iterations
  on the four-bar (3.3 a step at the default tolerances); its velocity
  residual is of the order of its Newton tolerance (1.8e-7 at most over
  1000 s at 1e-6), not of rounding as after a projection. A user's settings
  `project_constraints` and `projection_tolerance` must be renamed. Found
  with the long runs: semi-implicit Euler and the linearly implicit step
  take the gyroscopic terms explicitly and blow up on a top spinning at 30
  rad/s within 100 s (a limit of D35's step, not of the constraints).
- **Record.** Journal: constraints per integrator; data
  `2026-10-17_constraints-per-integrator.md`; `docs/kernel.md`, constraints
  per integrator; `docs/performance.md`, task 4.7; troubleshooting; messages
  K047, K051, K058.

## D38. The inertia's velocity products implicit, and the new velocities on the constraints, in the two Euler steps

- **Context.** An open end of task 4.7: semi-implicit Euler and the linearly
  implicit step (D35) took the inertia's velocity products (the Coriolis,
  centrifugal and gyroscopic terms, b(q, v): rnea(q, v, 0) less gravity)
  explicitly. On a top spinning at 30 rad/s, stepped at 5 ms, both blew up
  within 100 s: explicit Euler multiplies a nutation of rate W by sqrt(1 +
  (W h)^2) per step. The vehicles had not shown it because their wheels'
  spins are the drivetrain's states, not bodies; a model with spinning
  bodies (a motorcycle's wheels, a flywheel, a rotor) would, and the
  linearly implicit step is the real-time integrator.
- **Decision.** Both steps take these terms linearly implicitly through
  their exact derivative D = d b / dv, a new kernel algorithm,
  `bias_force_velocity_derivative`: RNEA linearized along each velocity over
  that joint's subtree and back to the root, exact because the terms are
  quadratic in v (a joint's own velocity product, in the universal and
  planar joints, by a central difference with a unit step, exact for the
  same reason; `JointModel::constant_motion_subspace` says which joints have
  none). In the linearly implicit step C becomes C - D: W = M - h C_s - h^2
  K_s - h (Tx G)_s + h D, r gains -h^2 D a; semi-implicit Euler solves (M +
  h D) delta = -h^2 D a. And both put the new velocities on the
  constraints, J v+ = nu, where they had kept the explicit step's (J delta =
  0): the constraints' impulse then does no work on v+. D enters whole, so
  W is not symmetric: `ConstraintSolver::constrained_solve_lu` (LU with
  partial pivoting, the multipliers on the independent equations of the
  last forward dynamics, a target for J dv), after a Cholesky of W's
  symmetric part as the positive-definiteness check behind MBD-K045 (now
  worded for both steps; its second fallback drops every derivative).
- **Alternatives.** All measured on the closed four-bar with a pendulum
  (semi-implicit Euler, 100 s at 5, 2 and 1 ms), the heavy top (300 s at 5
  ms) and two free bodies spinning at 30 rad/s with a small wobble, one
  flat, one long (5 s at 5 ms). Explicit, as before: the four-bar lost
  energy (11.9 J of 17.8 at 5 ms), the top blew up. D with the old
  restriction J delta = 0: the top and both free bodies right (the wobble
  damped exactly as backward Euler damps it, the spin kept), but the
  four-bar gained energy until it blew up at every step size (within 10, 40
  and 80 s): the new velocities kept the explicit part of the constraints'
  own velocity terms (gamma), which the implicit inertial terms no longer
  balanced. D's skew part only: the four-bar stable, but the top's spin fell
  from 30 to 18.7 rad/s. A Coriolis matrix B frozen at the step's start (B v
  = b, B + B^T = dM/dt, built from the bodies as sum J^T (I dJ/dt + (I v)x^
  J); implemented and verified, then removed), with which an unconstrained
  step provably cannot gain kinetic energy: the flat body's wobble fell to
  0.003 in a second where backward Euler gives 0.15, the long body tumbled
  end over end (its spin from 30 to 0.002 rad/s in 5 s), and the top lost
  spin, with or without the velocities on the constraints. D by differences
  of RNEA over v: 27 RNEAs a step on the sedan, about 130 us. Per-body
  implicit gyroscopic torques, as game engines do: in joint coordinates the
  bodies are not independent.
- **Consequences.** A free body's nutation is damped exactly as backward
  Euler damps it, by 1 / sqrt(1 + (W h)^2) per step, flat or long, its spin
  kept; the heavy top runs 1000 s with both steps (data); the closed
  four-bar only loses energy, at every step checked. Both methods stay
  first order and dissipative: the four-bar at 5 ms lost 14.7 J of 17.8 in
  100 s (explicit: 11.9). On the sedan the linearly implicit step costs
  about 160 us instead of 131 (pinned, against the previous commit): D (no
  joint-model call where c is zero, the backward pass over each joint's
  subtree and path only) and the LU solve (blocked triangular solves and one
  product) took it from a first 186 us. The vehicle tests did not change
  beyond their tolerances. `bias_force_velocity_derivative` is a kernel
  algorithm, checked against differences of RNEA on every joint type;
  `ConstraintSolver::constrained_solve_lu` takes a target for J dv.
- **Record.** Journal: the inertia's velocity products implicit; data
  `2026-10-18_implicit-inertia.md`; `docs/kernel.md`, integrators;
  `docs/performance.md`, D38; troubleshooting; message K045.

## D39. The mass matrix factorized along the tree, the dense Cholesky kept for long chains

- **Context.** Task 4.8: "sparse assembly and factorisation above a size
  threshold; done when dense and sparse results agree and the crossover
  size is measured". Every evaluation of the constrained dynamics
  factorized the mass matrix by Eigen's dense Cholesky, n^3 / 6
  multiply-adds for n velocities. A tree's mass matrix is zero wherever
  neither of two velocities is the other's ancestor; a car's corners, each
  hung from the chassis, leave most of it zero. The composite-rigid-body
  algorithm already computes only the nonzero entries (its work follows the
  same paths), so the assembly was sparse; the factorization was not.
- **Decision.** Featherstone's LTL factorization (RBDA section 6.5),
  `TreeCholesky`: M = L^T L along the velocity-level tree, eliminating from
  the last velocity to the first, which leaves no fill-in when parents are
  numbered first (the model's order); work sum_k (d_k (d_k + 1) / 2 + d_k +
  1), d_k the number of velocity k's ancestors. L is stored transposed, so
  that walking a velocity's ancestors runs down a column, and a path is
  walked as runs of consecutively numbered velocities, each by a plain loop.
  The range-space solve is unchanged with R = L^T in place of the dense
  factor (M = R R^T): Z = L^-T J^T, M^-1 J^T x = L^-1 (Z x).
  `ConstraintSolver::mass_factorization`: Automatic (default), Dense, Tree.
  Automatic takes the tree's unless the model has more than 40 velocities
  and the tree's work is at least 0.8 of the dense one's: a long chain. M
  stays a dense nv x nv matrix of which the pattern is read.
- **Alternatives.** All measured pinned in one session, alternately (data).
  The dense Cholesky everywhere (as before): on the sedan (27 velocities,
  tree work 0.26 of dense) the constrained forward dynamics 23.3 to 23.7 us
  against 19.7, an RK4 step 123 to 125 us against 105 to 107. The same
  factorization with L stored as in the book, scalar loops along the parent
  pointers: each step along a path a stride of n doubles, 78 us on a chain
  of 64 against 21 dense (the same work); as fast as the kept version on
  short runs (the binary tree of 64: 2.0 against 2.6 us). Each run as an
  Eigen segment operation: the chain of 64 at 34 us, but a fixed cost per
  run that made the binary tree of 64 13.1 us and the sedan's forward
  dynamics slower than dense (24.3 to 24.7 us against 23.6 to 24.1).
  Eigen's general sparse Cholesky (SimplicialLLT, natural ordering, the
  pattern analysed once, M's values copied in at each call): slower than
  the dense one on every model measured (1.1 to 2.9 times) and than the
  tree's by 1.6 to 13.7 times; its indexing costs more than the zeros save
  at these sizes. A rule on the work ratio alone (the first, the tree's
  below half the dense work): it chose dense for small models and chains up
  to 32 velocities, where the tree's was measured faster, because the dense
  factorization's speed per multiply-add grows with size. The tree's
  always: 8 to 11 % slower than dense on a chain of 64, and more on longer
  ones.
- **Consequences.** The results agree to rounding: L^T L = M to 2e-16 of
  |M|, the solver's accelerations to 2e-15 relative, the projected
  positions identical, the sedan's state after 500 steps of RK4 within
  1e-14. On the sedan the constrained forward dynamics takes 16 % less time;
  against the previous commit a step takes 13 % less with RK4, 12 % with
  generalised-alpha and 3 % with the linearly implicit step (which
  factorizes its own W densely). The tree's factorization is faster on every
  branched tree measured (up to 64 velocities: 3 to 8 times on a binary
  tree, about 2 times with two chains) and on chains up to 32 velocities;
  the dense one on longer chains, by 2 to 4 % at 48 and 8 to 11 % at 64.
  Vehicles are trees of short branches. The rest of the constrained solve
  (J M^-1 J^T, dense even where J is sparse) and the implicit steps'
  iteration matrices stay dense.
- **Record.** Journal: the mass matrix along the tree; data
  `2026-10-19_sparse-solve.md`; `docs/kernel.md`, constrained dynamics;
  `docs/performance.md`, task 4.8.

## D40. The linearly implicit step with its force derivatives whole, not their symmetric parts

- **Context.** Task 4.9's verification ran every integrator on the bushed
  pendulum (two free links held by bushings of 1e8 N/m and 1e6 N m/rad, the
  hinge axis free). The linearly implicit step, the real-time integrator
  (D35), whose description promised stability at any step for stiff springs
  and bushings (checked on linear springs only), diverged within the first
  second at 2, 1 and 0.5 ms. Probed: the bushings' rotational stiffness
  decides it (at 1e2 N m/rad even 1e8 N/m ran at 2 ms; at 1e6 N m/rad even
  1e6 N/m diverged there). The force derivatives were right (without
  damping, K equal to differences of the whole generalized forces to 5e-8
  of its norm) but far from symmetric: |K - K^T| 1.46e6 of |K| 5.6e8 after
  0.3 s, a bushing's rotational stiffness, carrying the links' large
  relative rotation about the free axis, coupling its other two axes by
  about k_r times half the rotation. The step's matrix took only the
  symmetric parts of K, C and the states' coupling (D35, so that a Cholesky
  could solve it); h^2 times the part it dropped, 1.46 at 1 ms, was larger
  than the links' inertia.
- **Decision.** K, C and Tx G enter whole: W = M - h C - h^2 K - h Tx G + h
  D, r = h^2 (K (v + h a) + C a - D a) + h Tx dx0. What D35 named as the
  obstacle, a non-symmetric constrained solve that copes with redundant
  constraints, D38 provided (`ConstraintSolver::constrained_solve_lu`), and
  the step already used it because of D. W's symmetric part is still
  checked for positive definiteness (MBD-K045). The assembly has fewer
  terms; the solve is the same.
- **Alternatives.** The symmetric parts (D35): divergence. The whole
  dynamics differentiated (2 nv evaluations a step): not needed, since K
  element by element was right; with damping it leaves out about c |v| (5e3
  of 5.6e8 here: how a body's velocity turns with its pose at fixed joint
  rates, part of the joints' geometry D35 left explicit), which is not the
  cause (without damping the step diverged as well). Shorter steps: 0.25 ms
  ran, a burden on every user of such a model.
- **Consequences.** The pendulum runs at every step from 4 ms to 62.5 us,
  of first order from 0.25 ms. The step still gains energy on it at 1 and
  2 ms (kinetic plus gravity's energy up to +0.21 and +8.6 J, where it can
  only fall): one linearization per step moves each attachment point along
  its tangent, the true motion along an arc, and the stiff translational
  spring turns their second-order gap, k L (w h)^2 / 2, into a force. The
  gain depends on k h^2 alone (1e7 N/m: only at 2 ms; 1e6: none; the
  rotational stiffness plays no part): the limit of a method that
  linearizes once, which generalised-alpha's Newton iterations do not have.
  Every other test passed unchanged, the vehicle's drives with the step
  among them. A test keeps the pendulum bounded at 2 and 1 ms and without
  energy gain at 0.5 ms.
- **Record.** Journal: the integrators verified; data
  `2026-10-19_integrators-verified.md`; `docs/integrators.md`;
  `docs/kernel.md`, integrators; message K045.

## D41. The integrators verified by a program that writes the report's figures and tables

- **Context.** Task 4.9: "convergence plots, energy drift tables and
  work-against-precision curves for each method, stored in
  `docs/integrators.md`". The project builds with CMake and a C++ compiler
  only; no Python with a plotting library is installed, and figures pasted
  in by hand go stale when an integrator changes.
- **Decision.** `bench/verify_integrators.cpp` (`mbd_verify`, built on
  request like the benchmark) runs every method on six models, from an
  oscillator with an exact solution to the sedan, and writes the figures as
  SVG (`bench/svg_plot.hpp`, a small log-log plotter), the data as CSV into
  `docs/integrators/`, and the report's tables to standard output. The
  report quotes them; running the program again reproduces them. The
  oscillator's figures are checked against each method's own recurrence,
  and a test keeps that check (every method's energy over many steps equal
  to its recurrence's, to 1e-13).
- **Alternatives.** Python and matplotlib: a second toolchain for one
  report. Tests that print: they run in every build and would make the
  suite minutes longer. Tables only: the plan asks for plots, and a
  work-precision comparison of eleven series reads better as a figure.
- **Consequences.** Text-based figures that show up in a diff; the report
  regenerated in about two minutes, pinned. The plotter is small and
  general enough for later reports.
- **Record.** Journal: the integrators verified; `docs/integrators.md`.

## D42. Ideas from external sources adopted into the plan

- **Context.** Before Phase 5 the author gave nine links (five GitHub
  repositories, a course, a product site, a plot catalogue, an article) and
  five documents, then two more repositories, asking which ideas would make
  the program better. Each was read and judged against what the program
  already has (`documentation/review_2026-10-20/external_sources.md`); the
  review proposed fourteen changes, ranked, and the author adopted them all
  (20 October 2026).
- **Decision.** The plan is amended in the phases where each belongs, every
  amended task marked D42:
  - **5.1** manufactured trajectories (a prescribed motion, its forces by
    inverse dynamics, every integrator required to return it at its order),
    from the verification methods of Barba's numerical-methods course; a rigid
    body on three bushings against eig(K, M) in closed form, with the modal
    energy distribution added to the linearisation, from Seddighi's
    motor-mount optimisation. **5.2** contradictory constraints reported by
    the conflicting equations (JAOT's minimal conflicting set). **5.5**
    optional mutation testing (JAOT).
  - **7.2** ISO 8608 random roads (the SciML brief). **7.6** a damper with a
    series stiffness and a Bouc-Wen bushing as state elements, checked on a
    virtual damper dyno (the SciML brief, Dobre et al. 2026); parametric
    bushing curves (Chindamo et al.). **7.8** a hysteretic driveline isolator.
  - **8.2** unknown keys are errors; **8.6** a test that every parameter of a
    reference car changes the built model (JAOT's input-sensitivity gate).
  - **9.1** exact hardpoint sensitivities from the constraint Jacobians, and
    Magri et al. 2024 as a compliance reference (Chindamo et al.). **9.3**
    pure pursuit or Stanley with a PI speed loop, then LQR (Yano's guide),
    tested by derived results. **9.5** channels named with their units and the
    derived vehicle-dynamics channels, laps, corners and phases (TUM).
    **9.7, new:** correlation with measured data (MOVEdot, TUM, Fast-F1).
  - **10.1** an elevation profile from public telemetry; **10.7** real laps as
    a realism check (Fast-F1).
  - **13.2, 13.3** kinematic sensitivities and the motor-mount benchmark;
    **13.5** discipline for fitted models (held-out test used once, constant
    baseline, errors with the range, a coded out-of-range warning; the
    Díaz Rivera thesis, the Siemens white paper); **13.8** compact results
    with handles, structured errors from the message catalogue, an MCP
    server, skills for users' agents (headroom, REA, JAOT, ampl-agents);
    **13.9, new:** parameter sensitivities of a simulation.
  - **14.7, new:** Adams/Car property files.
  - **H.7, new:** the working rules in the repository (`CLAUDE.md`,
    `AGENTS.md`; ampl-agents, the course's `AGENTS.md`).
  - Effort estimates raised by about 100 to 150 hours; the risk "no measured
    vehicle data" and the reference appendix updated.
- **Alternatives.** Not adopted: installing headroom (it routes the sessions
  through a proxy and compresses logs that are read exactly); using REA on
  any program (the commercial tools forbid reverse engineering, and the
  formats needed are documented text); anyplot (the plots already use a
  colour-blind-safe palette); field surrogates on meshes and PINNs (relevant
  only with our own CFD; PINNs not proven at scale); ingestion of ROS, MCAP
  and CAN logs; path planning. Copying code: nothing is copied; ideas are
  reimplemented from their published descriptions and cited (licences: Apache
  2.0, MIT, CC-BY 4.0 and BSD-3).
- **Consequences.** Phase 5 gains two strong checks (manufactured
  trajectories; the mount case) at small cost. The plan now validates the
  car against measured data (9.7), not only against other models. Several
  later tasks have sharper "done when" conditions. Local copies of the
  sources are in `C:\dev\references\` (outside the repository), as the review
  records; the five PDFs stay in `documentation/`, which git does not track
  for PDFs.
- **Record.** Journal: ideas from external sources;
  `documentation/review_2026-10-20/external_sources.md`;
  `documentation/Master_plan.md`.

## D43. Analytic cases: manufactured trajectories checked by one step and over a span, and the modal energy from the whole mass matrix

- **Context.** Task 5.1 asks for analytic cases, among them, from D42, a
  rigid body on three bushings with the modal energy distribution added to
  the linearisation, and manufactured trajectories: a smooth motion
  prescribed for random trees and closed loops, its forces by inverse
  dynamics, and every integrator required to return it at its order. Three
  things had to be decided: how the energy shares are defined, how a
  prescribed motion is built for joints with quaternions and for loops, and
  what "at its order" is measured on.
- **Decision.**
  - **Energy shares.** `Mode::energy_fraction`: for velocity coordinate k,
    `Re(conj(phi_k) (M phi)_k) / Re(phi^H M phi)`, M the whole mass matrix
    at the operating point, phi the mode's shape; the shares add up to 1, a
    cross term of M is split equally between its two coordinates (so a share
    can be negative), and `summary()` lists the three largest.
  - **Prescribed motion.** Scalar coordinates as `c + a sin(w t + p)`.
    Rotations (spherical joints, the free joint's) as a product of three
    exponentials about fixed axes, `R0 E1 E2 E3`, whose body-fixed angular
    velocity and its rate follow in closed form from the factors (the test
    file writes them out). The free joint's translation as sines in the
    parent's frame, its velocity coordinate `R^T p'`. The motion's own rates
    are checked against its differences before it is used.
  - **Loops.** Only some joints are driven, and the drives and the
    constraint multipliers solve `B tau_a + J^T lambda = RNEA` together, so
    that the constraints carry part of the load (from 0.27 to 2.6 times the
    drives' size on the two loops) and the forward dynamics must return the
    constraint forces as well as the accelerations. The loops are built so
    that the prescribed motion meets the constraints exactly in closed form:
    two identical chains side by side tied at their tips by a distance
    constraint (their tips stay a fixed vector apart), and a parallelogram
    four-bar (the coupler's angle minus the crank's).
  - **Orders.** Two measures. One step from the exact state at t = 0.2 s:
    its error falls by 2^(p+1) when the step halves, for every one-step
    method; it does not depend on how the motion amplifies errors, so it
    settles at long steps. Over a span of 0.5 s: the error falls by 2^p,
    for every method, with the steps of each chosen from tables of five
    steps (kept as a hidden test that prints them) where the ratio has
    settled and the error is far above rounding; within 0.25 of the order in
    both.
  - **Two cases left out, with their reasons.** Generalised-alpha has no
    one-step check: started at the true acceleration, its first step carries
    an O(h^2) velocity error that later steps damp (Arnold and Brüls 2007);
    its order over the span is 2.00 on every model. Dormand-Prince on the
    spherical-root tree has no span check: its ratio there has not settled
    at 32 above that model's rounding floor (2^4.39, 2^4.31, 2^4.70 from 12.5
    to 1.6 ms), though its one-step order there is 6.17 and falling towards
    6, and every other method, Radau IIA at fifth order included, settles on
    that model.
  - **The mounts without gravity.** The closed-form K of three bushings is
    the stiffness at zero load. With gravity the bushings carry the weight
    at rest, and a preload turning with the body adds a stiffness the closed
    form does not have; the test runs without gravity, so that K is exact.
- **Alternatives.**
  - Shares from the diagonal of M only, `phi_k^2 M_kk / sum`, as
    Seddighi's slides compute them: simpler and never negative, but they
    ignore the products of inertia and an offset centre of mass, which an
    engine or a car body always has; the shares would no longer add up to
    the mode's kinetic energy. Rejected.
  - A prescribed orientation from a time-varying rotation vector,
    `R0 exp(phi(t))`: its angular velocity needs the derivative of the
    exponential map and its rate the derivative of that, longer formulas
    to get right in a test. Rejected for the product of exponentials, each
    factor about a fixed axis, whose rates are one line each.
  - Feedback towards the prescribed motion, adding `K_p (q_ref - q) +
    K_d (v_ref - v)` to the forces: the prescribed motion stays an exact
    solution and errors stop growing, so long spans would be usable. But the
    forces then depend on the state, through a callback that the implicit
    methods' iteration matrices and the linearly implicit step's force
    derivatives see differently, and the gains damp the integrators' own
    errors. Not adopted; a short span and the one-step check do without it.
  - Every closed loop from a random tree with a closure "cut" out of it (a
    free joint plus the closure of the joint it replaces): topologically
    still a tree, not a loop. Rejected for the two constructions above.
- **Consequences.** The integrators are now checked against an exact
  reference on spatial trees with every joint type and on loops, not only on
  the oscillator and on models whose reference was itself a tight-tolerance
  run. The span check needs short spans: forced motions are unstable (on
  the spherical-root tree the fifth-order methods stop converging over 0.5 s
  at about 1e5 times the state's rounding), which is worth knowing
  for any later use of manufactured motions (the regression harness of 5.5).
  `linearize()` gains one mass matrix per call and the shares in its
  summary. Seddighi's decoupling target (90 % of a mode's energy in one
  coordinate) can be written directly on `energy_fraction` in 13.3.
- **Record.** Journal: Phase 5, analytic cases;
  `tests/kernel/test_kernel_manufactured.cpp`,
  `tests/kernel/test_kernel_linearization.cpp`,
  `tests/kernel/test_kernel_analytic.cpp`;
  `documentation/journal/data/2026-10-20_analytic-cases.md`.

## D44. The IFToMM benchmarks against closed forms; singular positions passed by holding the constraints near rounding; conflicting equations named by a deletion filter

- **Context.** Task 5.2 runs the IFToMM benchmark problems of group A (the
  simple pendulum, the N-four-bar mechanism, Andrews' mechanism, Bricard's
  mechanism, the flyball governor), "done when each meets its published
  reference", and, from D42, names the conflicting equations when a
  mechanism cannot be closed. The library's site and its reference files
  could not be reached from the machine the work was done on. The problems'
  parameters were taken from a third-party reimplementation of the library
  for OpenSim (Tagliapietra et al., MBS-BOS, GPL; read for the problem
  statements only) and, for Andrews' mechanism, from the Test Set for IVP
  Solvers (through the R package deTestSet), which also publishes a
  reference solution. On the N-four-bar the kernel failed: after a few
  passes through the singular horizontal position Dormand-Prince stopped
  with MBD-K055.
- **Decision.**
  - **References.** The pendulum and the N-four-bar against their closed
    forms in Jacobi's functions (the four-bar of identical bars moves as one
    parallelogram, a pendulum turning over); Andrews' mechanism against the
    test set's values at 0.03 s; Bricard's mechanism against the closed form
    of its path, tan(psi / 2) the small root of u w^2 + (u + 1)^2 w + u = 0
    with u = tan(phi / 2) (derived for this task from the closure equations),
    and against its dynamics reduced to one coordinate from the geometry
    alone; the flyball governor against its exactly kept angular momentum
    about the spindle, its energy balance with the dampers' work integrated
    as a state, and three integrators. The IFToMM files themselves remain
    to be compared with for A04 and A05, whose motions have no closed form.
  - **Singular positions.** The solver is not changed. The failure was
    traced to a residual of the positions of about 1e-13 m, below the
    default `constraint_tolerance` (1e-10) and so never projected away, which
    makes the model a slightly imperfect parallelogram: its coupler must turn
    at a rate growing as delta theta' / (L theta^2) towards the singular
    position. Held to 1e-14 (1e-12 for the 40 loops, whose coordinates reach
    40 m), the N-four-bar passes its 19 singular positions in 10 s on its
    closed form. The default stays 1e-10; the help says when to hold the
    constraints tighter (MBD-K055, the troubleshooting guide, the kernel's
    documentation).
  - **Conflicting equations.** When positions cannot be assembled (MBD-K062,
    `AssemblyReport::conflicting_equations`) or a projection fails
    (MBD-K051), a diagnosis goes from the nearest positions found to the
    least-squares nearest by Levenberg-Marquardt, the held coordinates kept,
    and there runs a deletion filter (Chinneck and Dravnieks 1991) on the
    linearised equations J dq = -phi: each equation in turn is left out for
    good if the rest still have no solution. The equations left, a smallest
    set that cannot all hold, are named by constraint index, kind, part and
    bodies. An empty set is reported as such: nothing contradicts.
  - **Messages.** MBD-K047, K051, K054 and K055 wrote small numbers with
    `std::to_string` (six decimals: a step of 1e-13 s read "0.000000"); they
    now use the kernel's scientific notation.
- **Alternatives.**
  - **A lower rank cut** in `RankRevealingLDLT` (1e-13 instead of 1e-10,
    nearer LAPACK's n eps): it narrows the window around a singular position
    where the accelerations are wrong (rank 4 of 8 kept: 0.75 rad/s^2 off on
    four four-bars) from |theta| < 5e-6 to < 1e-7, and the whole kernel suite
    passed with it; but measured on the runs it made them worse (with ctol
    1e-14, N = 4 at tolerance 1e-10: 5e-4 rad against 2e-8; with ctol 1e-13,
    failures at 2.6 s), the nearly dependent equations kept amplifying the
    residual. Rejected.
  - **A tighter default** `constraint_tolerance`: every model would pay for
    projections it does not need, and large models cannot reach 1e-14 at
    all (the 40 four-bars stopped near 1e-12). Rejected; a mechanism meant
    to pass singular positions sets it.
  - **Regularised multipliers or an augmented-Lagrangian formulation**, the
    usual robust answer to singular positions: not tried; the measured fix
    above suffices for the benchmark. Open.
  - **Naming the equations with the largest residuals**: in a loop the
    residual spreads over every equation of the loop, contradicted or not.
    **The support of a left null vector of J**: a dependent set, not a
    smallest one. **An exact irreducible infeasible subsystem by mixed
    integer programming**: far more machinery than a diagnosis needs.
    Rejected for the deletion filter.
  - **The deletion filter at the positions the projection stopped at**,
    without going to the least-squares point first: tried first, it found
    nothing on the test case (Gauss-Newton had reduced |phi| only from 1.20
    to 0.96 of a least possible 0.88, where the two contradicting equations,
    linearised, still have a solution). Rejected.
- **Consequences.** The benchmarks run in the suite (about 3 s; the 40
  four-bars, 17 s, hidden). A user whose mechanism will not close is told
  which constraints contradict. A user whose mechanism stalls at a singular
  position is told how to hold its constraints. Task 5.2 stays open until
  the IFToMM reference files for A04 and A05 are compared, and A05's
  definition confirmed against the library's own (the OpenSim version's
  parameters were used).
- **Follow-up, 20 October 2026.** The author downloaded the library's
  results for A04 (three) and A05 (two) and gave them to the project
  (`tests/kernel/data/iftomm/`, with their contributors). They confirm the two
  definitions taken from the OpenSim version, and the tests now compare with
  each: Bricard's point P2 within 2.5e-5 m of the most accurate result
  (Tagliapietra's; the other two drift, by 3.5e-4 and 7.7e-3 m, as their own
  energy columns or phase show), the governor's sleeve within 2.0e-6 m and
  6.8e-6 m of González's two. Each bound comes from how that result's error
  behaves, the model being checked independently to about 1e-9. Task 5.2 is
  done. The comparison also corrected a claim of this work (the governor's
  "slow swing near 40 s" was a swing of about 1 s sampled every 2 s).
- **Record.** Journal: Phase 5, the IFToMM benchmarks;
  `tests/kernel/iftomm.hpp`, `tests/kernel/test_kernel_iftomm.cpp`,
  `tests/kernel/data/iftomm/`, `src/kernel/conflicts.cpp`;
  `documentation/journal/data/2026-10-20_iftomm-benchmarks.md`.

## D45. A sanitizer report fails its test; an Eigen factorization sized by computing it, never by moving an empty one

- **Context.** CI's Linux job (task 0.5) builds `test_core`, `test_model`
  and `test_kernel` with GCC's address and undefined-behaviour sanitizers
  and runs them with `ctest --output-on-failure`. The undefined-behaviour
  sanitizer, by default, prints a report and lets the program go on, so the
  test passes; and ctest prints a test's output only when it fails. A report
  in a passing test was therefore shown nowhere: not in CI's log, not in the
  summary a developer reads. A check of task 5.2's records, after its
  closing commit (`85de229`) was pushed, read the full log of the run
  recorded there as "175 of 175 passed with no sanitizer report"
  (`build-asan/Testing/Temporary/LastTest.log`) and found two reports in it,
  in "Constraints per integrator: semi-implicit and linearly implicit Euler
  only lose energy on a closed loop" and "Manufactured trajectories: every
  integrator returns the motion at its order":

  ```
  Eigen/src/Cholesky/LLT.h:66:49: runtime error: load of value 32697, which is not a valid value for type 'ComputationInfo'
  ```

  With `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` the first test
  fails, and the stack names `Simulator::step_semi_implicit_euler`,
  `src/kernel/simulator_linear.cpp:490`, `w.llt_sym =
  Eigen::LLT<MatX>(nv);`. Eigen's sized constructor `LLT(Index)` sets the
  storage and `m_isInitialized` but not `m_info`; the move assignment then
  copies `m_info`, reading an enum that was never set. The same line is at
  `simulator_linear.cpp:153` and `simulator_implicit.cpp:299`. It came with
  tasks 4.7 (`8ed4c6c`, 17 October) and its open end (`576df64`, 18
  October). It did no harm in practice (each step's `compute` sets `m_info`
  before `info()` is read), but it is undefined behaviour, and every
  sanitizer run since had reported it unseen.
- **Decision.**
  - **The sanitizer job stops at the first report.** It builds with
    `-fno-sanitize-recover=undefined`, so an undefined-behaviour report ends
    the test with an error: the test fails, and ctest prints its output.
    `UBSAN_OPTIONS=print_stacktrace=1` adds the stack. The address
    sanitizer already stops at its first report.
  - **A factorization is sized by computing it.** The three places compute
    the factorization of the identity of the right size
    (`compute(MatX::Identity(n, n))`), which sets every member, instead of
    moving in a sized, empty one. They stay in the set-up branches, which
    allocate anyway; the steps after set-up do not allocate (CI's
    no-allocation job).
  - **A run is called clean from its full log**, not from the ctest
    summary.
- **Alternatives.**
  - **`UBSAN_OPTIONS=halt_on_error=1` in CI's environment only**: the same
    in CI, but a build made on another machine with the same flags, without
    the variable, keeps recovering. The compile flag puts it in the build.
    Rejected.
  - **A CI step that searches the full log for "runtime error"**: it finds
    the reports but leaves the tests passing, and the failure would name a
    log, not a test. Rejected.
  - **Constructing the factorization of the identity and moving it in**
    (`Eigen::LLT<MatX>(MatX::Identity(n, n))`): correct as well, but it
    builds a second factorization to move; computing in place does not.
    Rejected.
- **Consequences.** Any later report fails CI on the commit that brings it.
  The records of earlier sanitizer runs that said "no sanitizer report"
  (tasks 5.1 and 5.2, read from the ctest summary) were wrong in the same
  way and are corrected in their journal entries; their full logs were
  overwritten, so whether those runs had the same reports can no longer be
  checked, but the code that caused them was there.
- **Record.** Journal: Phase 5, the IFToMM benchmarks ("What went wrong");
  `.github/workflows/ci.yml`, `src/kernel/simulator_linear.cpp`,
  `src/kernel/simulator_implicit.cpp`; `CLAUDE.md` (pitfalls).
