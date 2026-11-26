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
| D46 | Comparison with independent codes: Pinocchio's joints added depth first; Chrono's equations at rest solved from its own matrices, its motion checked by convergence | 7 Nov 2026 | In force, task 5.3 |
| D47 | Statics: the Newton step by a rank-revealing QR, not the normal equations; a relaxation that ends worse is undone | 7 Nov 2026 | In force, task 5.4 |
| D48 | The HMMWV benchmark: Chrono's HMMWV data where the paper gives none, corrected; one tyre spring to the nearest surface; runs without events, as the paper, and a second table with them | 7 Nov 2026 | In force, task 5.4 |
| D49 | The regression harness: deterministic scenarios against stored references at 1e-8 of each channel, a nightly long suite, and mutation tests | 9 Nov 2026 | In force, task 5.5 |
| D50 | The diagnosis of contradicting constraints damps its steps alike in every direction | 9 Nov 2026 | In force, Phase 5 gate |
| D51 | Ideas from a second review of external sources adopted into the plan: suspension kinematics, validation and target cascading | 9 Nov 2026 | In force, plan amended (8.1, 8.4, 9.1, 9.3, 9.4, 9.5, 9.6, 9.7, 13.3) |
| D52 | The Python interface: pybind11 3, one extension module inside a package, the C++ names, the state as views, errors and warnings with their codes | 9 Nov 2026 | In force, task 6.1 |
| D53 | The Python package: scikit-build-core, only the module built, a build folder kept between installs, one compiler by default | 10 Nov 2026 | In force, task 6.2; amended 10 Nov 2026 (the build script's own folder) |
| D54 | The Python interface tuned with the author: the C++ names kept, shorthands for frames and bodies, settings by keyword, objects that show what they hold | 10 Nov 2026 | In force, Phase 6 (plan amended) |
| D55 | Parity tests: C++ test cases reproduced through Python, their numbers compared bit for bit in the same build | 10 Nov 2026 | In force, task 6.3 |
| D56 | Model files in the C++ library (JSON, nlohmann/json), exact and strict; results as pandas tables with two column levels | 10 Nov 2026 | In force, task 6.4 |
| D57 | Viewing with Rerun: a motion played back from the model's own parts, its channels plotted on the same time line | 12 Nov 2026 | In force, task 6.5; amended 12 Nov 2026 by D58 (a recording per replay) |
| D58 | Notebooks that check themselves: three worked examples, run as tests | 12 Nov 2026 | In force, task 6.6 |
| D59 | Rerun's viewer inside notebooks, where rerun-notebook is installed | 13 Nov 2026 | In force, Phase 6 (added at the author's request) |
| D60 | The vehicle layer in ISO 8855 axes, its labels kept; up decided in one place | 15 Nov 2026 | In force, task 7.1 |
| D61 | The road: one interface for height, normal and friction; the tyres take the ground only from it | 24 Nov 2026 | In force, task 7.2 |
| D62 | The wheel as a body on a spin joint; the contact from the wheel plane and the road; the drivetrain without states, the engine's inertia left out until 7.8 | 25 Nov 2026 | In force, task 7.3 |
| D63 | The Magic Formula tyre: TNO's MF-Tyre 6.1 as published, where the book differs; MF 5.2 as 6.1 with 5.2's forms; checked against MFeval | 25 Nov 2026 | In force, task 7.4 |
| D64 | The tyre on a wheel: the Magic Formula of a parameter set, its slips lagging by the semi-non-linear model, mirrored by side, faded at rest; every vehicle's tyre | 25 Nov 2026 | In force, task 7.4 |
| D65 | Simple tyres behind one interface: linear with saturation, Adams's Fiala, Pacejka's brush model | 26 Nov 2026 | In force, task 7.5 |
| D66 | Hysteresis as state elements: a damper with a series stiffness and a Bouc-Wen bushing, checked on a virtual dyno; bushing curves by their shape | 26 Nov 2026 | In force, task 7.6 |
| D67 | Suspensions from the kernel's joints, their parts' masses from their joints, springs set by virtual work to hold the design position | 26 Nov 2026 | In force, task 7.6 |
| D68 | Steering: a rack geared to a pinion by a joint coupler, a compliant column to a steering wheel, the input the rack's travel, the wheel's angle or a torque, as held values or functions of time | 27 Nov 2026 | In force, task 7.7 |
| D69 | The powertrain as a state element: an engine with its own speed, a converter or clutch, a gearbox with shift time, differentials and compliant half-shafts, reacting on the housing and the chassis | 27 Nov 2026 | In force, task 7.8 |
| D70 | The brakes as a state element: line pressures lagging the pedal, a proportioning valve, dry friction that locks a wheel, a slip-switched anti-lock controller | 27 Nov 2026 | In force, task 7.9 |

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

## D46. Comparison with independent codes: Pinocchio's joints added depth first; Chrono's equations at rest solved from its own matrices, its motion checked by convergence

- **Context.** Task 5.3 compares the library with Pinocchio (tree dynamics,
  to 1e-8) and Project Chrono (constrained mechanisms), the scripts to run
  in CI. Both codes have Python packages; this library's Python bindings
  come in Phase 6. Measured on the way (journal and data of 7 November
  2026):
  - Pinocchio 4.1.0's CRBA assumes the joints are numbered depth first, each
    joint's subtree taking the ids just after it, as the C++ documentation
    of `addJoint` says; the Python docstrings do not, and a release build
    checks it nowhere (`model.check(data)` would return false; the CRBA's
    assert is compiled out). On 37 random trees of 40 not so numbered, its
    mass matrix differed from the one its own RNEA gives by up to 7.7, while
    its ABA and RNEA agreed with ours to rounding.
  - Chrono 10.0's assembly at the acceleration level (the only way it
    offers accelerations at a given state) differentiates the velocities of
    a step of 1e-6 s. It leaves out the velocity products (a pendulum at 3
    rad/s lacked omega^2 r), and it carries the constraints' residual over
    the step squared into the accelerations: one rounding error, 1e-16 m,
    becomes about 1e-4. It also leaves the velocities advanced by its step.
    Its integrators start from the accelerations stored in the bodies (zero
    unless assembled, which makes them first order), and their
    accelerations and reaction forces oscillate (HHT with alpha = 0) or
    converge at first order to about 1e-4 (alpha = -0.2). Its positions
    converge cleanly: HHT with alpha = 0 at second order.
  - On Windows, conda gave NumPy MKL 2026.1.0 as its BLAS, and NumPy's
    first matrix product stopped Python.
- **Decision.**
  - **Exported models, compared in each code's language.** A program built
    on request, `mbd_crosscheck`, writes random models and this library's
    answers as JSON with 17 significant digits; a Python script per code
    rebuilds the models there and compares. Each script's header states the
    conventions it maps.
  - **Pinocchio:** 40 random trees of every joint type, numbered as our
    generator numbers them (any parent before its child), three states
    each; ABA, RNEA, CRBA and placements, relative difference (over 1 + the
    largest entry of Pinocchio's value) within 1e-8. The joints are added to
    Pinocchio depth first and the coordinates mapped between the two orders.
  - **Chrono at rest:** 12 random trees closed by one to three
    independent constraints of six kinds. The script lets Chrono assemble at
    rest, reads the system descriptor's matrix blocks (mass matrix,
    constraint Jacobian, forces; 13 significant digits), sets the
    constraints' right-hand side to its exact value at rest, zero, and
    solves those equations; each link's rows give its wrench on its bodies.
    Accelerations, the closures' loads on each body and the joints'
    reactions within 1e-10 (the files' floor is near 1e-12).
  - **Chrono's motion:** from rest, assembled and the velocities put back,
    HHT with alpha = 0 at two steps, h and h/2. It passes when Chrono's
    difference from our motion falls by between 3 and 5 as the step halves
    and the Richardson extrapolation of its two runs is within a tenth of
    its difference at h/2 (or below a floor set by our own error estimate
    and Chrono's Newton tolerance): Chrono converges to our motion at its
    order, so no tolerance is guessed.
  - **The environment from conda-forge, pinned** (`crosscheck/environment.yml`:
    Python 3.12.16, Pinocchio 4.1.0, PyChrono 10.0.0, NumPy 2.5.4, BLAS
    OpenBLAS); a Linux CI job, `crosscheck`, installs it with micromamba and
    runs both scripts.
- **Alternatives.**
  - **Compare against the mass matrix of Pinocchio's RNEA**: it agrees, but
    it compares RNEA twice and CRBA not at all. **Generate only depth-first
    trees**: Pinocchio would then pass, but the comparison would stop
    covering our numbering, which the library allows and uses. Rejected.
  - **Through this library's Python bindings**: they do not exist yet
    (Phase 6), and JSON keeps the answers readable and the C++ side free of
    any Python. Rejected for now.
  - **Chrono's assembled accelerations and its links' reactions**: limited
    to about 1e-4 by the residual over the step squared, and only at rest.
    **Accelerations and reactions from its integrators**: worse. **Its
    sparse matrices through the API**: PyChrono does not wrap
    `ChSparseMatrix`. Rejected; the matrix blocks give the same equations.
  - **One Chrono run with a fixed tolerance on the motion**: the tolerance
    would be guessed, and too loose or too tight depending on the mechanism.
    Rejected for the convergence test.
  - **Starting the motion with velocities**: the assembly would give
    accelerations without their velocity products, and the integrator would
    start from them. Rejected; the velocity products are exercised by the
    motion that gravity starts.
  - **PyPI instead of conda**: Pinocchio's `pin` has no Windows wheels, and
    PyPI's `pychrono` is a different program. Rejected.
- **Consequences.** The library agrees with two independent codes: with
  Pinocchio to 2.1e-13 on 563 bodies of every joint type, with Chrono to
  1.4e-12 at rest and in a motion to which Chrono converges at second
  order. The comparison covers gravity and constraints, not force elements,
  drivers or contact. A perturbation of one part in a million in one
  inertia makes it fail. CI runs it on every push (about two minutes for
  Chrono).
- **Record.** Journal: Phase 5, comparison with independent codes;
  `documentation/journal/data/2026-11-07_independent-code.md`; `crosscheck/`
  (README, `environment.yml`, the scripts, the exporter);
  `.github/workflows/ci.yml`; `CLAUDE.md`.

## D47. Statics: the Newton step by a rank-revealing QR, not the normal equations; a relaxation that ends worse is undone

- **Context.** `static_equilibrium` (task 3.5, D26) takes Newton steps of
  least change in the weights of the mass matrix's diagonal: with `Gs = S G
  S^-1`, the step solved `(Gs Gs^T) y = -S a` by `RankRevealingLDLT`, whose
  rank cut is 1e-10 of the largest pivot. Directions below the cut count as
  neutral and are left alone. On the HMMWV with the paper's bushings (task
  5.4: 2e7 N/m on upper arms of 5.8 kg), started from its design position,
  Newton went from 1.7e3 to 2.37 m/s^2 in two iterations and stopped there,
  the chassis still 0.02 m above where its tyres hold it. The singular values
  of `Gs`, printed by a temporary probe, run from 1 to 4.8e-6 for the
  directions with stiffness (the three softest: the chassis's heave, pitch
  and roll on its tyres), then drop to the differences' noise, at most
  9e-12 (at most 2e-10 on the HMMWV with joints, and once 7e-8 at a
  design position where the tyres had no load). Squared, 4.8e-6 is
  2.3e-11: below the cut, so 12 of the 15 directions were seen, and the
  chassis's three never moved. Dynamic relaxation then took over with its
  explicit step of 1 ms, beside modes of 1,860 rad/s, and blew up: the
  chassis ended 17 m in the air, and on a two-block test model the positions
  came back as NaN.
- **Decision.**
  - **The step from a QR factorization of `Gs^T` with column pivoting**
    (Eigen's `ColPivHouseholderQR`): `Gs^T P = Q R`, so `Gs = P R^T Q^T`;
    the least `x` with `Gs x = b` has `Q^T x = (w, 0)` with `R_r^T w =
    (P^T b)_r` over the rank `r`. No product squares the stiffness ratio.
  - **A direction has stiffness when its pivot is above 1e-7 of the
    largest** (`kStiffnessCut`): 50 times below the softest real direction
    measured (4.8e-6), and 500 times above the noise (2e-10); the transient
    7e-8 falls below it.
  - **A relaxation that ends with larger accelerations than it began with,
    or with none, is undone**: the positions go back to where it started,
    `StaticReport::relaxation_undone` counts it, and MBD-K073 says so and
    suggests a shorter `relaxation_step`.
- **Alternatives.**
  - **A lower cut on the normal equations** (1e-14 on `Gs Gs^T`): the
    squared values sit within two orders of rounding (1e-16), so the rank
    would depend on rounding. Rejected.
  - **A singular value decomposition**: the most reliable rank, but a
    heavier factorization to compile (the pitfalls note that Eigen's
    complete orthogonal decomposition costs 30 s per file), and the pivoted
    QR separates these spectra by orders of magnitude. Rejected for now.
  - **Starting the bushed model from the joint model's equilibrium**: it
    would hide the defect for this model and leave it for the next.
    Rejected.
  - **Detecting a blow-up during relaxation and stopping it**: the end
    comparison is simpler and catches the same, NaN included. Rejected.
- **Consequences.** The bushed HMMWV settles in 4 iterations (15 directions
  with stiffness, 3 neutral) to the joint model's chassis height within
  5 digits. Two tests: a soft spring below a stiff one (stiffness ratio
  1e-6) settles in one Newton step, where the old step failed and its
  relaxation returned NaN; a relaxation too long for a stiff element is
  undone, and a shorter one succeeds. Every earlier statics test passes
  unchanged. A direction whose stiffness lies between 1e-7 and 1e-5 of the
  largest, neutral before, now has stiffness.
- **Record.** Journal: Phase 5, the HMMWV benchmark; `src/kernel/statics.cpp`,
  `include/mbd/kernel/statics.hpp`, `tests/kernel/test_kernel_statics.cpp`,
  `docs/kernel.md` (static equilibrium), `docs/help/messages.md` (MBD-K073).

## D48. The HMMWV benchmark: Chrono's HMMWV data where the paper gives none, corrected; one tyre spring to the nearest surface; runs without events, as the paper, and a second table with them

- **Context.** Task 5.4 rebuilds the fourteen-body HMMWV of Haug, Negrut,
  Serban and Solis (1999): its bodies and joints, tyres of 296,325 N/m and
  3,502 N s/m, 10 mph over a half cylinder 0.1 m across, and the variant
  with bushings of 2e7 N/m and 2e6 N s/m at the upper arms. The paper gives
  no geometry, masses or suspension laws. Project Chrono's HMMWV comes from
  the same research group; its chassis rests 0.709 m up at the design
  position, the paper's 0.71 m. Read as data, it had two faults: control arm
  inertias that break the triangle inequality, and springs whose free
  lengths lift the chassis 0.085 m onto their rebound stops. The road's
  bump makes the tyre forces jump, which fixed-step methods straddle.
- **Decision.**
  - **Chrono's HMMWV data** (hardpoints, masses, inertias, the progressive
    spring table, the JSON files' linear shock dampers, the rack) where the
    paper is silent; the paper's own figures where it gives them.
  - **Two corrections, stated in the model:** the arms as flat bodies (Izz
    = Ixx + Iyy: 0.8 and 0.06 kg m^2 for Chrono's 0.8938 and 0.06276), and
    the springs' free lengths those that rest the arms at their design
    angles (0.294734288 and 0.332951200 m, found by `mbd_hmmwv calibrate`;
    Chrono's 0.339 and 0.382), the table's shape kept.
  - **Fourteen bodies, as the paper:** each spindle body is Chrono's
    upright, spindle, wheel and tyre together (no wheel spin, as the
    paper's ten degrees of freedom imply); the tie rods massless distance
    constraints; the steering rack held by a driver.
  - **The bushings:** each upper arm on a spherical joint at its front
    point, and at its back point two spring-dampers across its axis,
    perpendicular to each other, 0.1 m long and free at the design
    position. Their fastest mode is -2.06e6 s^-1, eight times the paper's
    -2.6e5, whose geometry is not given.
  - **The tyre:** one spring-damper from the wheel centre to the nearest
    point of the road, along its normal there, pushing only, without
    friction. The vehicle starts at 10 mph from rest on its tyres and
    coasts; only the bump's slope takes speed from it. The bump is placed
    where the front tyres' force turns to it at 0.5 s.
  - **The benchmark:** 5 s, every body's position every 10 ms against Radau
    IIA at a tolerance of 1e-10; every method at several fixed steps or at
    tolerances from 1e-2 to 1e-8 (the paper's 1e-2 to 1e-5, and two
    tighter); on the bushed model the explicit methods only near their
    stability limit. Costs: wall-clock time of the stepping, pinned to a
    performance core at high priority, from one run; steps, evaluations,
    Newton iterations, Jacobians. Without events, as the paper ran it; a
    second table with the tyres' events (`Hmmwv::add_events`) on the joint
    model.
- **Alternatives.**
  - **Chrono's data as it stands**: the vehicle would rest on its spring
    stops, 0.085 m above the paper's height. Rejected.
  - **A wheel body on a spin joint, tyres with friction, a driving torque**:
    a different model from the paper's, with more states and its own
    stiffness. Rejected for this benchmark (Phase 7 builds such tyres).
  - **A tyre that sums its contacts with both surfaces, or follows the
    bump's envelope**: smoother at the switch but not the paper's single
    spring-damper; the damper's jump on meeting a step-like bump remains
    either way. Rejected.
  - **Events by default**: they make fixed-step methods accurate at the
    jumps, but generalised-alpha fails with them on this model and they
    cost the implicit methods' tight runs many times more. Kept as the
    second table.
  - **Bushings placed to match the paper's eigenvalue**: its geometry is
    unknown; matching one number by tuning would not make the model the
    paper's. Rejected; the difference is stated.
- **Consequences.** A reproducible HMMWV-14 in `tests/kernel/hmmwv.hpp`,
  whose numbers come with their sources; the chassis rests at 0.6875 m
  (paper: about 0.71 m) and settles after the bumps within about 3.5 s
  (paper: about 4 s). The costs of all methods on both variants in
  `docs/hmmwv.md`. The model becomes a reference car file in task 8.6.
- **Record.** Journal: Phase 5, the HMMWV benchmark; `tests/kernel/hmmwv.hpp`,
  `tests/kernel/test_kernel_hmmwv.cpp`, `bench/hmmwv.cpp`, `docs/hmmwv.md`,
  `documentation/journal/data/2026-11-07_hmmwv-benchmark.md`.

## D49. The regression harness: deterministic scenarios against stored references at 1e-8 of each channel, a nightly long suite, and mutation tests

- **Context.** Task 5.5: "Stored reference results with tolerances. A fast
  suite for every commit and a long suite nightly. Optionally, mutation
  testing of the kernel in a Linux CI job: a deliberately altered source line
  must make some test fail (D42). Done when a deliberate physics change is
  caught." The suite's tests check results against what is known to be
  right (closed forms, invariants, other codes); nothing yet noticed a
  result that changed where nothing says what it should be: a vehicle's
  motion, a lap, a mode. CI runs on Windows (MSVC) and Linux (GCC), whose
  rounding differs; its logs cannot be read without logging in, its
  annotations can.
- **Decision.**
  - **Scenarios, not stored states.** `tests/regression/scenarios.cpp`
    runs nine models of the test suite deterministically (fixed steps, or a
    solve to a tight tolerance: a step controller's accept-or-reject could
    turn on rounding) and records named channels; together they cover the
    kernel, the force elements, the vehicle template with its tyres and
    drivetrain, statics and linearisation, suspension kinematics and the lap
    model. Three more run longer, for the nightly suite.
  - **References in CSV** (`tests/regression/data/`, 17 significant digits,
    a header saying what wrote them), written by the test itself when
    `MBD_REGRESSION_UPDATE=1`. Writing one again is allowed only for a
    change meant and understood, recorded in the task's journal entry and
    commit with the channels' differences (`tests/regression/README.md`).
  - **The tolerance: 1e-8 of each channel's largest magnitude**, set
    between two measurements (the hidden `[sensitivity]` case): a relative
    change of 1e-12 in one physical parameter, standing for rounding
    differences, moves no scenario by more than 5e-3 of it; a change of
    1e-6 moves every scenario by 26 times it or more. A fast test requires
    the second for five scenarios, so the harness's power is itself tested.
  - **Platforms reported, not assumed:** each CI test job sets
    `MBD_REGRESSION_REPORT`, and a step turns each scenario's worst channel
    into an annotation.
  - **The nightly workflow** (`.github/workflows/nightly.yml`, Linux): the
    hidden `[long]` tests, the long scenarios, the sensitivity probe, and the
    mutation tests.
  - **Mutation tests** (`scripts/mutation_test.py`): nine one-line changes
    to the physics of the kernel and the force elements (signs, missing
    terms, factors of 1 or 2 %), each applied, built, tested with the kernel
    and regression executables (stopping at the first failure), and undone.
    A mutation no test catches, or one whose line has gone from the source,
    fails the job.
- **Alternatives.**
  - **References of whole states at every step**: large files, and a
    failure would say only that something changed. Channels with names and
    units say what changed. Rejected.
  - **Adaptive integrators in the scenarios**: a rejected step that turns on
    rounding would make the platforms disagree by the tolerance of the
    integration. Rejected; fixed steps throughout.
  - **A tolerance per channel set by hand**: each would need its own
    reasoning; one relative rule, measured against both bounds for every
    scenario, is checkable. Rejected for now.
  - **A mutation tool (Mull, Dextool)**: thousands of generated mutants,
    most equivalent or irrelevant to physics, and a toolchain to maintain.
    A short list of meaningful mutations, each saying what it breaks, is
    enough to show the tests notice physics. Rejected for now.
  - **Mutation tests on every commit**: each costs a rebuild; nightly is
    enough. Rejected.
- **Consequences.** Ten more test cases on every commit (1.4 s in all on the
  laptop). A physics change of one part in a million in any scenario fails
  CI. Every planned change that moves results (the vehicle layer's axes in
  Phase 7, a new tyre) will fail these scenarios on purpose, and the
  references will be written again with the reasons recorded. On the
  laptop the nine mutations were all caught, in 3.5 minutes: seven by the
  regression scenarios, which run first, and two by the kernel tests alone
  (gravity in the recursive Newton-Euler pass and the articulated-body
  pass, which the simulator does not use). Run against the kernel tests
  alone, six of those seven are caught too; the tyres' peak force is caught
  only by the regression scenarios.
- **Record.** Journal: Phase 5, the regression harness;
  `tests/regression/` (README, scenarios, references), `tests/CMakeLists.txt`,
  `scripts/mutation_test.py`, `.github/workflows/ci.yml`,
  `.github/workflows/nightly.yml`, `CLAUDE.md`.

## D50. The diagnosis of contradicting constraints damps its steps alike in every direction

- **Context.** The nightly workflow's first runs (D49), on 9 November 2026,
  on branch `phase-5`. In GCC's Release build (-O3), run only by the
  nightly's mutation job, "Assembly: constraints that contradict each other
  are named, and only they" named nothing: MBD-K062 said the linearised
  equations could all hold. Every other build (MSVC, GCC at -O2, the
  sanitizers) passed. The diagnosis (D44) goes from where the projection
  stopped to the least-squares point of |phi|^2 / 2 by Levenberg-Marquardt,
  then runs a deletion filter on the linearised equations. A probe in that
  loop on the laptop showed that, from the test's one starting position,
  the search used all 200 tries and stopped with the gradient still 0.90,
  |phi| 0.960 (the least possible is 0.884), lambda 3.7e10, and the bob
  2.1e-6 m from the anchors' axis. The filter's rank test (pivots below
  1e-10 of the first dropped) drops one of the two distances' rows within
  about 7.5e-6 m of the axis (for those rows, A = B B^T has pivots 2.25 and
  about 4 rho^2), and then the contradiction is certain to be seen: that
  stop was inside it by chance. Farther out both rows are kept and have a
  solution, mu = A^-1 r of size 0.88 / (2 rho^2), and the conflict is named
  only if the solve's rounding leaves more than the filter's tolerance
  (1e-10) of r unexplained; that rounding, about 1e-16 |A| |mu| = 2e-16 /
  rho^2, falls below 1e-10 near rho = 1e-3. Started from 48 positions (a
  new test), the same code on the laptop named nothing from 40 of them:
  20 places of the bob, each with both crank angles (the crank changes
  nothing of the bob's path). The 16 of those places whose stop was printed
  ended 1.0e-3 to 0.11 m from the axis; the 4 places that named the
  distances ended 2.8e-9, 2.1e-6, 6.4e-4 and 7.8e-4 m from it.
- **Cause.** The damping was Marquardt's: each direction damped by its own
  diagonal entry of J^T J (with a floor of 1e-12 of the largest). At the
  least-squares point of a contradiction, the columns of J that move across
  it vanish: here the bob's offset rho from the axis, whose diagonal entry
  is 2 rho^2. |phi|^2 / 2 keeps a curvature there that Gauss-Newton's J^T J
  leaves out, sum phi_i grad^2 phi_i = 1.25 in every direction of the bob
  (each distance's phi = (d^2 - 1) / 2 is 0.625 there). A step across the
  axis overshoots unless its damping is about that curvature, which makes
  lambda so large that the steps along the axis, damped by lambda times
  4.5, stop. Why GCC's -O3 build stopped elsewhere than its -O2 build was
  not found (no FMA or fast-math flag is set); the outcome did not need a
  compiler difference to go wrong.
- **Decision.** Levenberg's damping: (J^T J + lambda m I) dq = -J^T phi, m
  the largest diagonal entry of J^T J, lambda's schedule and the stops as
  before (divided by 3 after a step that lowers |phi|, multiplied by 4
  after one that does not, given up above 1e12). From all 48 starts the
  search reaches the least-squares point, |phi| = 0.625 sqrt(2) =
  0.883883476483184 to every digit printed, in 39 to 51 tries, and stops
  where no step lowers |phi| any more, 0.4e-9 to 1.7e-8 m from the axis:
  more than 400 times nearer than the rank test needs to drop a row.
- **Alternatives.**
  - **Moré's scaling** (MINPACK's: each direction damped by the largest
    diagonal entry it has had so far): it keeps the damping a direction had
    at the start, but the start is where the projection stopped, which may
    already be near the point. Not measured. Rejected.
  - **A looser rank cut in the filter** (1e-6 instead of 1e-10): it would
    see the contradiction from farther away, but the search would still
    stop short of the point, and nearly dependent equations that do have a
    solution would be called contradictory. Rejected.
  - **More tries**: the stalled search's steps along the axis were divided
    by lambda, 3.7e10 when it stopped. Rejected.
  - **Newton's method with phi's second derivatives**: the constraints do
    not provide them. Rejected.
- **Consequences.** The diagnosis runs only after a failure (MBD-K062,
  MBD-K051), so no simulation's results or cost change. Coordinates of
  different kinds (lengths, and angles of long levers) are now damped
  alike: the steps of those with small diagonal entries are shortened
  more, which slows the search but does not stop it. The 5.2 entry's
  account, that going "first to the least-squares point by
  Levenberg-Marquardt made the filter return exactly the two distances",
  went further than its evidence: the search had stopped short of that
  point, near enough to the axis for the filter.
- **Record.** Journal: the nightly's first runs; `src/kernel/conflicts.cpp`,
  `tests/kernel/test_kernel_assembly.cpp`, `docs/kernel.md`.

## D51. Ideas from a second review of external sources adopted into the plan: suspension kinematics, validation and target cascading

- **Context.** At the Phase 5 gate the author recommended three sources:
  Nick McCleery's series on suspension kinematic simulation, Suspension
  Explorer (its web application, documentation and open-core solver), and
  AVL's pages on vehicle engineering and simulation. Read against the plan
  and the code (`documentation/review_2026-11-09/external_sources.md`), they
  gave fourteen proposals, P1 to P14, and the author adopted them all
  (9 November 2026).
- **Decision.** The plan is amended where each belongs, every amended task
  marked D51:
  - **9.1** (P1 to P5, P8, P9): the steering axis also from the motion,
    the instantaneous screw axis of the upright per unit of rack with a
    declared hold, for every architecture; every steering output names its
    hold; responses solved in the null space of the constraint Jacobian,
    with the mobility reported and a response that is not unique reported
    as unavailable rather than returned at least norm, and derivative
    channels from tangents, not differences of sweep frames; each metric
    defined once with its frame and sign; sweeps checked before solving and
    a status per step (branch snap, mirror image, link near toggle, poor
    conditioning); an axle's road plane; more outputs (ground offset,
    anti-lift and the anti ratios, swing arms, wheel-centre recession,
    contact-patch migration); `suspension-explorer-core` as a second code.
    Its "done when" adds the double wishbone's motion axis equal to its
    ball-joint line with zero pitch, a test of what the hold changes, and
    the comparison recorded.
  - **8.1, 8.4** (P6): shims as design and setup stack thicknesses that
    change link lengths, applied through the assembly of 3.4.
  - **9.5** (P7): a catalogue of every metric and channel, checked by a
    test like the message catalogue.
  - **9.6, 9.7** (P10, P11): ISO 19364 and ISO 19365 as the comparison
    procedure where tests or another vehicle code are compared; coast-down
    and weave in the identification sequence; a model fitted to target
    figures where no measurement exists.
  - **9.4** (P13): frequency responses from the linearisation, checked
    against the time-domain sweep.
  - **9.3** (P14): the driver's preview and gains set from the vehicle and
    the track.
  - **13.3** (P12): vehicle-level targets cascaded into subsystem
    requirements through the sensitivities of 9.1 and 13.9.
  - Effort raised by 50 to 75 hours (Phases 8, 9 and 13); the appendix of
    references updated.
- **Alternatives.** Not adopted (the review's "noted, not proposed"):
  neural networks trained on millions of runs for concept attributes (no
  such data; 13.5 covers the method); snow and soft-soil tyre models
  (outside the plan's road and race cars); a lap-throughput target (to be
  measured once 10.2 exists); tracks from map services (licensing);
  generated Jacobians and dual numbers, and fitting a twist to points (the
  kernel already gives both exactly).
- **Consequences.** Task 9.1 grows by about a third, and its outputs gain
  definitions, holds and diagnostics that make them checkable. Two choices
  of the existing kinematic analysis are to change when 9.1 is done: the
  least-norm velocities where freedom is left, and the camber gain
  averaged across a sweep. `suspension-explorer-core` is AGPL-3.0 and this
  repository has no licence: nothing of it, code or data, may enter the
  repository; a comparison runs it separately and exchanges files. The ISO
  standards must be bought before P10 can be applied. AVL's figures are the
  vendor's claims and are not used as evidence.
- **Record.** Journal: ideas from external sources, second review;
  `documentation/review_2026-11-09/external_sources.md`;
  `documentation/Master_plan.md`; local copies in `C:\dev\references\`.

## D52. The Python interface: pybind11 3, one extension module inside a package, the C++ names, the state as views, errors and warnings with their codes

- **Context.** Phase 6, task 6.1: "Model building, data, all analyses,
  recorder output as NumPy arrays, errors as Python exceptions." What
  existed was the project template's stub: a module `pycore` with two
  functions of no use (`compute_mean`, `make_greeting`), on pybind11 2.12.0,
  built for whatever Python the Windows Store alias gave, with no Python
  test.
- **Decision.**
  - **pybind11, raised to 3.1.0.** It supports Python 3.14 (Miniforge's
    base here). It finds Python through CMake's FindPython. Its
    `py::smart_holder` passes the `shared_ptr<const T>` the kernel keeps for
    joints, constraints and forces, and will let Python subclass C++
    classes later. Its `py::native_enum` makes C++ enums Python `enum.Enum`
    types. It is fetched only when `MBD_BUILD_PYTHON` is on.
  - **A package around one extension module.** `import multibody as mbd`.
    `python/multibody/` holds the Python files (the error classes now; plots
    and file helpers later, 6.4 and 6.5). The compiled module
    `multibody._mbd` is built from `python/src/`, one file per part of the
    engine. The two are laid out together in `<build>/python/multibody/`.
  - **The C++ names.** Classes, functions and members keep their C++ names
    (`RevoluteJointModel`, `add_body`, `step_control.relative_tolerance`), so
    the C++ documentation and the help apply as written. Only what Python
    needs and C++ lacks is added: `Transform3.from_axis_angle`, and
    `Simulator.run_recorded`, which runs as `run()` does and returns the
    states as arrays without a Python call per step.
  - **Arrays.**
    - Vectors and matrices are NumPy arrays, and quaternions are
      [w, x, y, z].
    - The simulator's state (`q`, `v`, `x`, `tau`) reads as a view of its own
      memory, so `sim.q[0] = 0.3` writes the state. It can be assigned only
      whole and at its size (MBD-P001), because assigning an Eigen vector of
      another size would resize the state.
    - Rotation matrices and quaternions typed in Python are checked (MBD-P002
      to P004), since the C++ constructors trust their callers.
  - **Errors and warnings.**
    - `mbd::MbdError` becomes `multibody.MbdError`, a `RuntimeError` with the
      message unchanged and its code in `.code`.
    - The diagnostic sink sends warnings to Python's `warnings` module as
      `multibody.MbdWarning`, so Python's filters apply. The previous sink is
      restored when the interpreter exits.
    - A new area P in the catalogue holds the interface's own codes, and the
      catalogue test now scans `python/`.
  - **Lifetimes.** A simulator keeps its system alive. A system's model is a
    reference into it. The model's per-body tables come back as copies, so
    the tree changes only through `add_body`.
  - **The interpreter and the tests.**
    - The interpreter is chosen by CMake's `Python_EXECUTABLE`
      (`scripts\build.ps1 -Python <path>`).
    - On the laptop it is a pinned conda environment,
      `python/environment.yml` (mbd-python: Python 3.12.16, NumPy 2.5.4 on
      OpenBLAS, pytest 9.2.0). CI's Windows job uses Python 3.11.
    - Each file in `python/tests/` is one CTest test, run by pytest under that
      interpreter. Without NumPy or pytest the tests are left out, with a
      note.
  - **The version.** The CMake project's version follows the tags (0.6.0
    now, raised at each gate) and is the module's `__version__`.
  - **Scope of 6.1:** the kernel, its analyses, the recorder, and the
    general force elements (springs, dampers, bushings, contact, forces
    written in Python, joint forces). The vehicle layer, with the tyre,
    aerodynamic and anti-roll-bar elements still on the old Y-up axes, is
    bound when Phases 7 and 8 rebuild it.
  - **Python callbacks** (user forces, prescribed functions, events, the
    force callback, recorder channels) run with the interpreter lock taken.
    A Python list or array handed to C++ code would be a copy, so where a
    callback must change something it is given a handle that writes through.
    `BodyForces` adds loads and is valid during its call only (MBD-P005).
    The force callback gets `tau` as a writable view. Lists and curve
    triples read as tuples, which refuse an item assignment that would
    otherwise be lost.
  - **The interpreter lock** is held through calls into C++ for now, which
    keeps Python callbacks simple. Releasing it for long runs is left for
    later.
- **Alternatives.**
  - nanobind: smaller modules and faster builds, but a second binding
    library to learn, with stricter ownership rules. Rejected for now.
  - Staying on pybind11 2.12: no Python 3.14, and the legacy finder.
    Rejected.
  - PEP 8 renames: two vocabularies to document and keep in step. Rejected.
  - The state as copies: `sim.q[0] = 0.3` would change a copy and be lost
    without a word, the best-known trap of Eigen bindings. Rejected.
  - Views with resizing assignment: arrays held by the user would dangle.
    Rejected.
  - The Store's Python: unpinned, with no NumPy. Rejected.
- **Consequences.** The template's stub is gone (`include/core.hpp`,
  `src/core.cpp`, their three tests, `pycore`). Building the interface needs
  a Python with its development files. Each later phase binds what it adds
  (the plan's Phase 6 goal: extended in every later phase).
- **Record.** Journal: Phase 6, the bindings; `python/` (the package, the
  bindings in `python/src/`, the tests in `python/tests/`, the environment
  file), `docs/python.md`, `CMakeLists.txt`, `scripts/build.ps1`,
  `.github/workflows/ci.yml`, `docs/help/messages.md` (area P).

## D53. The Python package: scikit-build-core, only the module built, a build folder kept between installs, one compiler by default

- **Context.** Phase 6, task 6.2: "`pyproject.toml`, an editable install,
  and a wheel built in CI. *Done when `pip install -e .` works on a clean
  machine.*" Until then the module was built only by the project's CMake
  build, for the interpreter given with `-Python`, and imported through
  `PYTHONPATH`.
- **Decision.**
  - **scikit-build-core 1.1.2, pinned, as the build backend.** pip runs the
    project's own CMake, so the package's module is built by the same
    `CMakeLists.txt` as the suite's. pybind11 stays fetched by CMake at its
    pinned commit.
  - **Only the module.** `pyproject.toml` configures with the interface on
    and the tests off and builds the target `_mbd` (with the library it
    needs). CMake installs only its component `python`, the module, into the
    package; the Python files come from `python/multibody/`. Catch2 is now
    fetched only when the tests are built.
  - **Release**, where the suite builds RelWithDebInfo. Task 6.3's parity
    tests belong to the suite's build, with the C++ tests' flags.
  - **A build folder kept between installs,** `build-python/<wheel tag>/`.
    An install after a change compiles only what changed.
  - **One version.** pip's metadata reads it from `CMakeLists.txt`'s
    `project()` line, which the module reports as `__version__`. A test
    compares the two.
  - **NumPy 2.0 or later**, the only dependency; CI tests that floor.
    pytest is the extra `test`.
  - **One compiler by default.** The build system's job pools hold for
    pip's build where it uses Ninja (Linux). On Windows pip uses Visual
    Studio's generator, which ignores them; on the laptop the build script's
    `-Install` sets Ninja, and pip is not run by hand there.
  - **The suite checks what it imports.** An editable install's finder is
    inserted at the head of `sys.meta_path`, before `PYTHONPATH`, so a
    package installed into the suite's interpreter would be tested instead
    of the build. `python/tests/conftest.py` stops the tests then, from
    `MBD_TEST_PACKAGE_DIR`, which CMake sets.
  - **CI.** A package job installs from the checkout with plain pip on
    Linux (Python 3.11 and NumPy 2.0, the oldest allowed) and on Windows
    (Python 3.14), tests the installed package, then builds a wheel, tests
    it in an empty environment and keeps it 14 days.
  - **Not published.** A wheel suits machines like the one that built it (a
    Linux wheel for PyPI needs a manylinux build), and the repository has no
    licence. Publishing is a later decision.
- **Alternatives.**
  - setuptools with a CMake extension written in `setup.py`: the build
    logic in two places. Rejected.
  - meson-python: the project builds with CMake. Rejected.
  - pybind11 from PyPI as a build requirement: two sources for one
    dependency. Rejected.
  - A new temporary build folder per install (scikit-build-core's default):
    every install compiles everything. Rejected.
  - Rebuilding at import (`editable.rebuild`): on the laptop a compiler
    would start from an import, outside the script, perhaps beside another
    build. Rejected.
  - cibuildwheel and manylinux wheels: needed only to publish. Deferred.
- **Consequences.** A machine needs Python, git and a C++20 compiler; CMake
  and, on Linux, Ninja come from PyPI when it has none. The build folder
  takes 726 MB on the laptop. The laptop's rule, to build only through the
  script, gains `-Install`.
- **Record.** Journal: Phase 6, packaging; `pyproject.toml`,
  `python/CMakeLists.txt`, `python/tests/conftest.py`, `CMakeLists.txt`,
  `scripts/build.ps1`, `.github/workflows/ci.yml`, `docs/python.md`.
- **Amended 10 Nov 2026, after the author's first install.** The author
  typed `pip install -e .`, the first command `docs/python.md` then showed,
  in a miniforge3 Prompt, in the base environment and in `mbd-try`. Both run
  Python 3.14, so both builds went to `build-python/cp314-cp314-win_amd64/`,
  where the script's install had built with Ninja that morning. pip's build
  used Visual Studio's generator: the top-level cache was written again for
  it, and the dependencies' sub-builds, which keep the generator they were
  made with, refused it ("CMake configuration failed" at the first
  `FetchContent_MakeAvailable`). Keeping a folder was not the fault; one
  folder for two routes with different generators was.
  - **The script's install builds in a folder of its own,**
    `build-python/ninja-<wheel tag>/` (pip's `-C build-dir=...`). Plain pip
    keeps `build-python/<wheel tag>/`.
  - **The laptop's instructions are steps from a miniforge3 Prompt**, the
    script started with `powershell -NoProfile -ExecutionPolicy Bypass
    -File`, since the laptop's execution policy is Windows' default and runs
    no script; tried as written before they were committed.
  - Alternatives: FetchContent without sub-builds (policy CMP0168, from
    CMake 3.30, above the project's minimum of 3.25; and the top-level cache
    would still change generator at every switch of route, each switch a
    full build). Rejected. A temporary folder for plain pip: rejected above.
  - Journal: Phase 6, the author's first install.

## D54. The Python interface tuned with the author: the C++ names kept, shorthands for frames and bodies, settings by keyword, objects that show what they hold

- **Context.** With task 6.2 done the author asked to tune the interface
  together, before task 6.3's parity tests, 6.4's files and 6.6's notebooks
  are written against it. A survey of the installed package found every
  argument already named; 71 functions and methods without a docstring;
  joint frames that had to be typed even when they are the identity;
  settings set one attribute at a time; and most results printing as a bare
  address (joints and constraints had readable reprs through their base
  classes). Four sets of changes were proposed; the author chose three, and
  chose to keep the C++ names.
- **Decision.**
  - **The C++ names stay** (D52, confirmed): one vocabulary for C++, the
    help and Python, the frames carried by the monograms (`p_WB`, `x_pj`).
  - **Frames.** `add_body`'s joint frames default to the identity. Three
    numbers convert to a translation wherever a `Transform3` is expected
    (a constructor `Transform3(translation)` and pybind11's implicit
    conversion from a sequence); anything else still fails to convert, and
    the call reports the types it takes.
  - **Bodies by name.** `add_body` takes the parent by its name as well as
    its index, and `Model.body(name)` gives an index: MBD-P008 when no body
    has the name, MBD-P009 when several do. In the bindings, not in the C++
    `Model`, whose interface stays one of indices.
  - **Settings by keyword.** The ten settings classes (the options of
    assembly, statics and linearisation, `StepControl`, `ImplicitControl`,
    the parameters of five force elements) take their settings as keywords
    over their defaults, each through the setting's own setter; any other
    name raises MBD-P010, since a misspelt one would otherwise be lost
    without a word. Each prints as the call that builds it, which reads
    back to the same object for numeric settings.
  - **Objects that show what they hold.** Records print as a call listing
    their fields; reports and objects holding a model or a state print as
    `<Name: what matters>`, with numbers to the digits that matter and
    arrays on one line. A report's `summary(model)` keeps the full text.
    Curves show only whether they are zero, or their value and slope at 0:
    their points are private in C++, and task 6.4 makes them readable for
    files.
  - **Docstrings** for the 71, from the headers' comments;
    `ValidationReport.ok()` bound.
  - **Where each lives.** The reprs and the keyword constructors are Python
    (`python/multibody/_display.py`, `_settings.py`), installed on the
    classes when the package is imported: formatting needs NumPy anyway,
    and a change needs no compiler. The frames and the names are in the
    bindings, since they change how arguments convert.
  - **Kept general by tests:** every class prints more than its address, and
    every function and method has a docstring (61 classes, 166 functions and
    methods today); the message catalogue's test now reads the package's
    Python files.
- **Alternatives.**
  - Python-style names, as aliases or as the only names: rejected by the
    author.
  - The reprs and keyword constructors as C++ templates in the bindings:
    a rebuild for every wording, and NumPy called from C++ for arrays.
    Rejected.
  - Names in the C++ `Model`: deferred, the C++ interface being one of
    indices throughout.
  - Results of `run_recorded` as an object with named columns: not chosen;
    task 6.4 brings pandas.
  - Curve accessors now: a header most of the library includes, a near-full
    rebuild on the laptop; 6.4 needs them for JSON and adds them.
- **Consequences.** A class bound later needs a repr and docstrings, or the
  tests fail; a settings class bound later is added to
  `_settings.install()`. The plan gains the tuning as a done item after 6.2.
- **Record.** Journal: Phase 6, the interface tuned with the author;
  `python/src/bind_*.cpp`, `python/multibody/`, `python/tests/test_shorthands.py`,
  `docs/python.md`, `docs/help/messages.md` (P008 to P010),
  `tests/core/test_message_catalogue.cpp`.

## D55. Parity tests: C++ test cases reproduced through Python, their numbers compared bit for bit in the same build

- **Context.** Task 6.3: "Ten C++ tests reproduced through Python. *Done
  when the numbers are identical.*" The package calls the same compiled
  library as the C++ tests; what could differ is what the bindings add: a
  conversion, a default, a step done another way, a callback that sees
  another state. The Python tests of task 6.1 check the package against
  closed forms to tolerances, which such a difference can pass.
- **Decision.**
  - **Identical means the same double.** The numbers are compared as
    hexadecimal, so that no tolerance enters and -0.0 differs from 0.0. Both
    sides run the same machine code of the library on the same inputs, and
    the code they compile themselves keeps MSVC's default floating point
    (`/fp:precise`, SSE2: nothing fused or reordered; the project sets no
    flag), so anything else is a difference to explain.
  - **The C++ numbers come from the C++ test case itself, run then.**
    `tests/support/parity.hpp` writes what a case records when
    `MBD_PARITY_FILE` names a file, each number in hexadecimal floating
    point; otherwise it does nothing. `python/tests/parity.py` runs the case
    (it must pass, and run alone), reads the file, keeps the same names from
    Python and compares. A name's numbers are kept in order, so a case that
    runs several times (sections, step sizes) compares run by run.
  - **In the suite's build only.** The C++ test executables and the module
    are built together, with the same flags; CTest gives the Python tests
    `MBD_TEST_BIN_DIR`. Against an installed package (CI's package job, an
    author's environment) the tests are skipped: there are no C++ tests to
    run, and a Release wheel is not the suite's build (D53).
  - **Only the engine's numbers.** A case records what the engine computed
    (states, reports, loads, modes, statistics), not its own arithmetic on
    them (a speed along a slope, a spring's length), which Python would
    redo in another order.
  - **Setups copied expression for expression.** An input one rounding
    apart moves almost every number after it: Eigen's norm of a short
    vector is written as the root of the sum of squares in order, a vector
    over a scalar entry by entry, and a frame from an axis and angle through
    `Transform3.from_axis_angle`, which builds the quaternion as the C++
    `Quat(AngleAxisd(...))` does.
  - **The ten**, one or more per part of the interface: assembly of a
    four-bar with a held crank; a limit stop; statics and linearisation of a
    four-bar on a spring; a bouncing ball's events (their function and action
    in Python); a lag written in Python pushing a mass through the force
    callback (RK4 and semi-implicit Euler); a pendulum on a distance
    constraint (projection, energies); a hinge's reaction (loads); a block on
    an incline (contact with friction, two sections); springs with preloads
    (nonlinear spring, energies); the bushed pendulum (bushings,
    Dormand-Prince, generalised-alpha, Radau IIA). 516 numbers.
  - **What the reproductions needed is bound:** `Simulator.last_projection()`
    and `ProjectionInfo`. A test whose C++ case needs what is not bound was
    not chosen: the heavy top (the kernel's algorithms), the driven pendulum
    and the driven point (the solver's multipliers), the Radau oscillator
    (a joint force written in C++; `JointForce` cannot be subclassed from
    Python).
- **Alternatives.**
  - Reference numbers stored in the repository: they differ between
    Windows and Linux in the last digits, as the regression harness's
    annotations show; they would be refreshed by hand at every change of
    the engine, and the comparison would be with an old run, not this
    build's. Rejected.
  - The Python test checking the C++ test's conclusions to its tolerances:
    parity of conclusions, not numbers, and the plan asks for identical
    numbers. Rejected.
  - A tolerance of a few ulps: it would pass the differences the tests are
    for, which begin that small and grow along a run. Two frames for the
    same hinge, one from an axis and angle and one from the rotation matrix,
    have quaternions one ulp apart in one coefficient; built from the second,
    the hinge's case moved 48 of its 54 numbers, half of them by 1024 ulps
    or more. Rejected.
  - Parsing Catch2's output (`-s`): fragile, and decimal. Rejected.
- **Consequences.** Ten C++ test cases call `parity()`; their checks are
  unchanged and the calls do nothing without the variable. `test_parity.py`
  runs in the suite on the laptop and in CI's Windows job (Linux CI builds
  without the interface). A change to one of the ten cases, or to its
  model, is made on both sides. The parity tests took 8 to 15 s in three
  runs on the laptop (once 218 s; all in energy-saving mode; the journal),
  most of it the bushed pendulum's reference at tolerance 1e-12, run once
  in C++ and once from Python (as fast from Python as from C++).
- **Record.** Journal: Phase 6, parity tests; `tests/support/parity.hpp`,
  `python/tests/parity.py`, `python/tests/test_parity.py`,
  `python/CMakeLists.txt`, `python/src/bind_simulator.cpp`,
  `docs/python.md`.

## D56. Model files in the C++ library (JSON, nlohmann/json), exact and strict; results as pandas tables with two column levels

- **Context.** Task 6.4: "Models to and from JSON. Results to pandas, Parquet
  and CSV. *Done when a model survives a round trip unchanged.*" D8 chose
  JSON for model files; task 8.2 will extend the format to the vehicle, with
  a schema and units, where "an unknown key is an error, never ignored"
  (D42); task H.6 foresees a command-line program reading model files. Most
  elements and every constraint kept their settings private, so a model
  could be built but not read back out.
- **Decision.**
  - **In the C++ library, chosen by the author:** `mbd/io/model_file.hpp`
    (`model_to_json`, `model_from_json`, `save_model`, `load_model`), bound
    to Python with the same names. One reader for Python, C++ programs, the
    command line and the real-time runner to come.
  - **nlohmann/json 3.12.0** (MIT), fetched and pinned by its archive's
    hash like every dependency, used in one source file of the library and
    private to it; its implicit conversions are switched off there.
  - **Read accessors** for what the elements, constraints and curves kept
    private: a curve's points or line, a time function's constant, the
    constraints' markers, axes and functions, the elements' anchors,
    markers, geometry and settings, a joint force's body. A constraint set
    owns its name (it held a pointer to a literal), so a set read from a
    file can have any name.
  - **Exact.** Numbers in nlohmann's shortest round-trip form; infinite
    values as the texts "inf" and "-inf"; transforms set member by member,
    since the `Transform3` constructor normalises its quaternion and could
    move its last bit. A model read back writes the same text and runs bit
    for bit as the original.
  - **Strict.** Every object's keys are checked: an unknown key or one given
    twice is an error (MBD-I004), and every error names its place in the
    file (`bodies[2].x_pj.quaternion`). Values then go through the same
    constructors as code, whose own errors keep their codes with the place
    added. What a file cannot hold is refused by name when writing
    (MBD-I008): a force written as code, a time function given as code,
    elements the format does not know. Messages in a new area, I.
  - **Readable.** Keys and type names are those of the C++ and Python
    classes and arguments (`Bushing`, `X_1M`, `revolute_closure`); joints by
    their kind (`revolute`). A closure is written as the call that makes
    it, but only when that call makes exactly the set (compared as
    written); other sets part by part. Keys with defaults may be left out
    when writing by hand. Short objects and lists are printed on one line,
    within 100 columns. `docs/model-files.md` describes version 1.
  - **Results as tables, in Python** (pandas; pyarrow for Parquet, the extra
    `results`): `dataframe(run, sim)` indexed by time with two column levels,
    chosen by the author: the vector (q, v, x) and the coordinate, named
    after its body (`df["q"]["crank"]`); several coordinates numbered, an
    unnamed body `body<i>`, a shared name followed by `#<i>`.
    `dataframe(recorder)` for a recorder's channels. `write_results` and
    `read_results` keep a table in Parquet or CSV exactly. `import
    multibody` needs neither pandas nor pyarrow.
  - **Curves show the call that makes them**, numbers in full, now that
    their points can be read (the open end of D54).
- **Alternatives.**
  - The files in Python only (the json module): quicker, but C++ programs
    could not read them, and Phase 8 would need a second reader. Offered to
    the author, not chosen.
  - A JSON reader written here: the crosscheck programs write JSON by hand,
    but reading needs a parser with exact numbers and positioned errors.
    Rejected.
  - Writing only what differs from the defaults: shorter files, but a
    default changed later would silently change old files' models.
    Rejected; written files are complete, hand-written ones may be brief.
  - Closures as their primitives always: exact but unreadable, and a reader
    of the file would not see a hinge. Rejected.
  - A results table with one column level (`q[0]`) or by body only: offered
    to the author with the two levels; the two levels chosen.
- **Consequences.** A new kind of element, constraint or joint is added to
  the format, or is refused by name when a model holding it is written;
  the format's test covers the kinds it builds, so a kind added to the
  format is added to that test's system. The suite's environment and the author's gain pandas 3.1.1 and
  pyarrow 26.0.1; CI installs them. The catalogue's test accepts two forms
  of a coded throw it did not know: a caught error's message rethrown with
  its place, and a code a function is given by its callers.
- **Record.** Journal: Phase 6, files; `include/mbd/io/model_file.hpp`,
  `src/io/model_file.cpp`, `tests/io/test_model_files.cpp`,
  `python/src/bind_files.cpp`, `python/multibody/_results.py`,
  `python/tests/test_files.py`, `docs/model-files.md`, `docs/python.md`,
  `docs/help/messages.md` (area I, P012 to P014).

## D57. Viewing with Rerun: a motion played back from the model's own parts, its channels plotted on the same time line

- **Context.** Task 6.5: "Three-dimensional playback of recorded poses, and
  channel plots. *Done when the four-bar and the car can be replayed.*" D8
  named Rerun or Meshcat. The model has no geometry: bodies carry an
  inertia, joints frames, elements points. The car of the project in the
  kernel is the HMMWV of task 5.4, whose tyres are forces written in C++
  against its road (`UserForce`), which no model file can hold (D56).
- **Decision.**
  - **Rerun, chosen by the author** (rerun-sdk 0.43.0, the extra `view`):
    one viewer for the three-dimensional view and the plots, on one time
    line; recordings saved as `.rrd` files or shown in the viewer. Imported
    only by `multibody.replay` and `plot`.
  - **Drawn from the model's own parts:** a body as the uniform box with its
    mass and moments of inertia, on its principal axes at its centre of
    mass (a point for a point mass); lines from the centre of mass to its
    joints; springs, dampers, bushings and distances as lines between their
    points; contact spheres and closures' markers as points. Up is against
    gravity. What has no points (a `UserForce`) is not drawn.
  - **A motion is its coordinates over time**, from a run, a table (two
    levels, or one with columns `q[0]` ... as a C++ program writes them) or
    arrays; poses come from the engine's kinematics at each frame drawn
    (60 per second by default); the plots take every sample, q and v and
    any channels given.
  - **The scene is computed apart from the drawing,** so that tests check
    what is drawn against closed forms, without a viewer; Rerun's own
    rendering is not tested. The tests also require Rerun to print no
    warning while recording.
  - **Read accessors bound** for what is drawn: the elements' anchors,
    markers and contact geometry, the constraints' markers, axes and time
    functions, a set's parts, a time function's constant.
  - **The car is recorded in C++ and replayed from Python:**
    `mbd_hmmwv record <folder>` writes its model file, the tyres left out,
    and its coordinates every 10 ms; `python/examples/replay_hmmwv.py` plays
    them back. The four-bar is built, run and replayed in Python
    (`python/examples/replay_four_bar.py`).
- **Alternatives.**
  - Meshcat: three-dimensional only, plots elsewhere; offered to the author
    with Rerun; Rerun chosen.
  - Shapes given by the user (meshes, cylinders): the format has none yet;
    later, with the vehicle's geometry (Phase 7 and 8).
  - Rebuilding the HMMWV's tyres in Python to run the car there: a second
    copy of a model the benchmark owns, to replay what C++ already records.
    Rejected.
  - Reading the recording back in the tests: Rerun 0.43's package has no
    `rerun.dataframe` any more, and no other reader of `.rrd` files was
    found in it. The scene is tested instead.
- **Consequences.** The suite's environment and the author's gain rerun-sdk
  0.43.0; CI installs it. The viewer's tests skip without it. A new kind of
  element with points is drawn once the viewer knows it; otherwise it is
  left out of the view, not refused.
- **Record.** Journal: Phase 6, viewing; `python/multibody/_viewer.py`,
  `python/tests/test_viewer.py`, `python/examples/`, `bench/hmmwv.cpp`
  (`record`), `python/src/bind_forces.cpp`, `python/src/bind_constraints.cpp`,
  `docs/python.md`, `docs/hmmwv.md`, `docs/help/messages.md` (P013, P015).

## D58. Notebooks that check themselves: three worked examples, run as tests

- **Context.** Task 6.6: "Pendulum, four-bar and quarter car as notebooks",
  with no "done when". The author had seen 6.5's playback and asked to go on
  in that direction. A notebook is read by people (the author, the book's
  readers) and goes out of date unseen when the interface changes, unless
  something runs it. Working rule 1: tests assert values derived
  independently.
- **Decision.**
  - **Done when** (added here): each notebook runs from top to bottom in the
    suite and in CI, its numbers checked against closed forms.
  - **Three notebooks in `python/notebooks/`** (nbformat 4.5): the pendulum,
    the four-bar linkage and the quarter car. Each says what it checks,
    builds its model with the documented interface, prints each result
    beside its closed form through a five-line `check`, which stops the
    notebook past an allowance derived in the text above it, plots with
    matplotlib, and plays its motion back in Rerun.
  - **Run as tests without Jupyter** (`python/tests/test_notebooks.py`): the
    code cells in order, in one namespace, the plots drawn off screen (Agg),
    `replay` and `plot` redirected to recordings in a temporary folder, and
    no warning from Rerun on the process's stderr. The notebooks therefore
    use no magic commands and no IPython display calls; a cell's last
    expression is shown by Jupyter and ignored by the test.
  - **Stored without outputs.** The code is the source; outputs go stale with
    any change and each figure adds 25 to 50 kB to a diff.
  - **The quarter car is a shaker rig:** the road a plate on a driven
    prismatic joint (a `TimeFunction` written in Python), the tyre a contact
    sphere of exponent 1 against it. Linear while it touches, unilateral when
    it does not, so one model serves the linear checks and the bump that
    throws the wheel off.
  - **Viewing amended (D57) for notebooks:** each `replay` and `plot` is a
    recording of its own (Rerun draws its id) under an application named by
    `name`. In 6.5 the name was the recording's id under one application,
    `multibody`: Rerun merges what arrives under one id, so a cell run twice
    showed both runs in one recording, and a later model would have kept an
    earlier one's layout (blueprints are kept per application). The viewer
    is started from the binary in Rerun's package (`rerun_cli`), not looked
    for on the PATH, which a kernel started by an editor may not share.
  - **matplotlib** (matplotlib-base 3.11.2) joins the suite's environment;
    the extra `notebooks` lists pandas, pyarrow, rerun-sdk, matplotlib and
    ipykernel; CI's Windows job installs matplotlib and the package jobs
    `.[notebooks]`.
- **Alternatives.**
  - Running the notebooks through Jupyter in the tests (nbclient): it would
    test the kernel too, at the cost of a kernel process and more packages in
    the suite and CI, for the same code. Used once in the task's session,
    through `jupyter_client`, to read the outputs and see the figures; not
    kept.
  - Storing the notebooks with their outputs, readable on GitHub without
    running them: stale outputs, large diffs, and a check needed that the
    outputs match the code. Rejected.
  - Notebooks kept as scripts with cell markers (jupytext): one more tool for
    the author, while VS Code opens `.ipynb` directly. Rejected.
  - Rerun's viewer inside the notebook (`rerun-notebook`): on PyPI only, not
    on conda-forge, and its rendering in VS Code cannot be checked without a
    display. Left for the author to ask for.
  - The tyre as a spring fixed to the road: linear throughout, but it cannot
    leave the road.
- **Consequences.** A change to the interface that breaks a notebook fails
  the suite. Text in a notebook that quotes a number must stay within what
  its checks hold. Running a notebook in place writes its outputs into the
  file, which git then shows as changed (`docs/python.md` says how to put it
  back). The notebooks' test skips without matplotlib, pandas or Rerun, as in
  CI's wheel job.
- **Record.** Journal: Phase 6, examples; `python/notebooks/`,
  `python/tests/test_notebooks.py`, `python/multibody/_viewer.py`,
  `python/tests/test_viewer.py`, `python/environment.yml`, `pyproject.toml`,
  `.github/workflows/ci.yml`, `docs/python.md`,
  `docs/help/troubleshooting.md`.

## D59. Rerun's viewer inside notebooks, where rerun-notebook is installed

- **Context.** After task 6.6 the author asked for the viewer inside the
  notebook (13 Nov 2026), instead of in a window of its own. Rerun shows its
  viewer in a notebook through `rerun-notebook` (0.43.0, matching rerun-sdk
  0.43.0), a package on PyPI only, not on conda-forge. Its widget loads the
  viewer, a WebAssembly program of 70 MB, from `app.rerun.io` by default,
  or from a server on the machine with `RERUN_NOTEBOOK_ASSET=serve-local`.
  `notebook_show` waits up to 10 s for the viewer to report ready; data
  sent before that is queued.
- **Decision.**
  - **Where the viewer appears, with no new parameter:** with `path`, a
    recording file, as before; else, in a Jupyter kernel (IPython's
    `ZMQInteractiveShell`: JupyterLab, VS Code's notebooks) with
    `rerun-notebook` installed, below the cell, as wide as the cell at 16:9
    (`notebook_show(width="auto", height="auto")`); else, or with
    `spawn=True`, in a window of its own. In a notebook without
    `rerun-notebook` the window opens with the warning MBD-P016, which says
    what to install. Each call in a notebook shows a viewer of its own.
  - **Rerun's default source for the viewer** (`app.rerun.io`), the one
    Rerun maintains; `serve-local` documented for offline use.
  - **`rerun-notebook` in the author's environment** (`mbd-try`, by pip, the
    documented command tried as written) and in the extra `notebooks` as
    `rerun-sdk[notebook]`, which pins it to Rerun's version. Not in the
    suite's environment: the tests replace Rerun's `notebook_show` and
    `spawn` with recorders bound to their real signatures, and check which
    was called.
- **Alternatives.**
  - A parameter (`notebook=True`): one more choice to make in every call,
    where the place the code runs already says it. Rejected; `spawn=True`
    keeps the window.
  - `serve-local` by default (nothing fetched from Rerun's site): it starts
    a web server at the package's import, and whether VS Code's notebook
    view loads from it could not be checked here. Rejected as the default.
  - `inline` (the viewer sent through the notebook's connection, compressed
    and encoded: 23.7 MB, measured from the package): heavy, and it may be
    stored in the notebook. Rejected.
- **Consequences.** The viewer's drawing in VS Code is not tested; in a
  kernel without a front end (the session's check), the cell shows the
  widgets and goes on after the 10 s wait. A notebook needs the network the
  first time a viewer is shown, unless `serve-local` is set. A notebook
  stored with its outputs would hold the viewer's widgets; the project's are
  stored without outputs (D58).
- **Record.** Journal: Phase 6, the viewer in notebooks;
  `python/multibody/_viewer.py`, `python/tests/test_viewer.py`,
  `python/notebooks/` (text), `pyproject.toml`, `docs/python.md`,
  `docs/help/messages.md` (P016), `docs/help/troubleshooting.md`.

## D60. The vehicle layer in ISO 8855 axes, its labels kept; up decided in one place

- **Context.** Task 7.1 (finding F11): "Move the vehicle layer to ISO 8855.
  Remove every hard-coded 'up'. *Done when the same manoeuvre gives the same
  physical result before and after.*" The layer used X forward, Y up, and
  called +Z "left", which with Y up is geometrically the right: the mirror
  image of the car its labels described, consistent with itself. World Y as
  up was written into the tyres, the aerodynamics, the anti-roll bar, the
  builders and the equilibrium. D3 had decided ISO 8855 and the signs to use
  (docs/conventions.md).
- **Decision.**
  - **Keep every label:** a left corner stays left, a positive steer stays a
    left turn. Data given in the legacy axes (x, up, left) is converted to
    ISO (x, left, up) by swapping its last two components: hardpoints,
    inertia boxes in frames aligned with the chassis (the chassis, the
    uprights, the arms, the rack), offsets. That swap is a reflection, so
    the new model is the mirror image of the old one, which is the car the
    old labels described. Every element of the layer is mirror-symmetric
    (rigid bodies, springs, the tyre's law odd in its slip angle and even in
    its combined-slip weights), so every result read through the labels is
    unchanged: positions and velocities with their last two components
    swapped, angular velocities swapped and of opposite sign (axial
    vectors), a tyre's lateral force of opposite sign (its lateral axis is
    now the left), toe, camber, loads, frequencies and states unchanged.
  - **Up in one place:** `mbd/vehicle/axes.hpp` gives `iso_gravity()` and
    `ground_up(gravity)`. The builders set the model's gravity from the
    first and give the tyres and the aerodynamics the ground's normal from
    the second; those elements hold the normal as a setting (default +Z)
    and measure penetration, vertical speed, ride height and horizontal
    speed along it. The anti-roll bar measures along the chassis's own Z,
    the vehicle's up. Task 7.2 replaces the flat ground with a road.
  - **Proof by the regression scenarios:** two scenarios were added before
    the move, with references from the legacy code (`d9f9466`): the
    double-wishbone sports car with anti-roll bars and off-centre downforce
    turning left, and the McPherson hatchback braking in a right turn. After
    the move, every vehicle scenario's new results were compared with the
    legacy references through the mapping above (a script, in the task's
    journal data).
- **Alternatives.**
  - Keep the physical car (a rotation of the old frame about X by 90
    degrees, under which the old "left" wheels become ISO's right ones):
    exact for any data, but every label would change meaning (the old left
    front wheel becomes the right front). Rejected: the labels are what
    users and tests read.
  - Take up from gravity inside each element: the elements see only body
    states, not the model; and a flat road's normal is not gravity's
    direction on a slope. Rejected for a setting given by the builder.
  - A world constant for up used directly in the elements: the hard-coded
    up the task removes. Rejected.
- **Consequences.** A model or test written for the legacy axes must be
  converted (x, y, z) -> (x, z, y), with rotations about the old axes
  replaced by their ISO counterparts (a positive steer is now a positive
  rotation about Z). The kinematic helpers keep their names and signs:
  `extract_toe` and `extract_camber` return the angle toward +Y (the left),
  as they returned it toward the old "left". `VehicleModel::chassis_ty_idx`
  is now `chassis_tz_idx`; `KinematicSweepPoint::wheel_y` is `wheel_height`.
- **Record.** Journal: Phase 7, axes; `include/mbd/vehicle/axes.hpp`, the
  tyre, aerodynamic and anti-roll-bar elements, the builders and suspension
  corners, `position_kinematics`, the lap vehicle; the vehicle tests;
  `tests/regression/` (two scenarios, four references written again);
  `docs/conventions.md`.

## D61. The road: one interface for height, normal and friction; the tyres take the ground only from it

- **Context.** Task 7.2: "One interface returning height, normal and friction
  at a point. Flat, analytic (ramp, bump, kerb, sine), gridded height map,
  and a three-dimensional track surface. ISO 8608 random roads, classes A to
  H, generated from the standard's displacement spectrum with a seed (D42).
  *Done when the tyre uses only this interface, and a generated ISO 8608 road
  reproduces its class's spectrum.*" After task 7.1 the tyres met a flat
  plane given by a height and a normal.
- **Decision.**
  - **The interface:** `Road::at(x, y)` returns a `RoadPoint`: the height z
    of the surface above the ground point (x, y), its unit normal, and its
    friction, in ISO 8855 world axes (`mbd/forces/road.hpp`). A road is a
    height field: one surface above each ground point.
  - **Friction is a scale** on the tyre's own friction, 1 being the surface
    its data was measured on. The tyre multiplies its peak force D by it and
    keeps its slip stiffness K = BCD (B rises as D falls), as the Magic
    Formula's friction scaling does.
  - **The tyres take the ground only from their road:** at the contact
    point's (x, y), the depth below the road's tangent plane, (h - z) n_z,
    the force along the road's normal, the heading and slip in the tangent
    plane, the friction into the tyre law. Each tyre starts on a level road
    at height 0; `VehicleHandle::set_road` and `VehicleModel::set_road` put
    a vehicle's tyres on another; a null road is refused (MBD-F051).
  - **The roads:** `FlatRoad`; `RampRoad`, `BumpRoad` (raised cosine),
    `KerbRoad` (a chamfered strip along the road), `SineRoad`; `GridRoad`
    (bilinear within each cell, its edges continuing level); `Iso8608Road`;
    and in `mbd/analysis`, beside the track it is built on,
    `TrackSurfaceRoad`: a query point projected onto the centreline (the
    nearest of samples every metre, found through a grid of buckets, then
    Newton on the arc length), its height the elevation less the lateral
    distance times the banking's tangent.
  - **ISO 8608 roads:** one profile along X, the same across the road, a
    sum of cosines at n_k = k / L between n_min and n_max (0.011 to 2.83
    cycles/m by default) with amplitudes A_k = sqrt(2 G_d(n_k) / L), so
    that the variance in each frequency interval is the class's, and phases
    uniform from the seed. The 64-bit Mersenne twister's outputs are fixed by
    the C++ standard, its distributions are not, so the phases are its top
    53 bits mapped to [0, 1) by hand: a seed gives the same road on every
    compiler. The profile is sampled once (ten samples per shortest
    wavelength) and interpolated by Catmull-Rom cubics, which pass through
    the samples with a continuous slope; it repeats every L.
- **Alternatives.**
  - Random amplitudes as well as phases (each component Rayleigh): the
    spectrum then holds on average over many roads, not on each. With the
    amplitudes fixed, every road's periodogram over one period is the
    class's spectrum exactly, which is what the "done when" asks and what the
    test checks.
  - The sum evaluated at every query: about 2,800 cosines per tyre per
    evaluation for a 1 km road. Rejected for sampling once.
  - Two tracks, left and right, with a coherence between them: ISO 8608
    defines one track's spectrum; the coherence needs a model of its own.
    For the four-post rig (task 9.4).
  - A stateful projection for the track (the last arc length as a hint):
    cheaper, but a road queried by four tyres and by several threads would
    need one hint each. The buckets need none.
  - A three-dimensional mesh road (OpenCRG and the like, overhangs): later
    if needed; a height field serves ground vehicles.
- **Consequences.** The tyre's contact geometry is unchanged: the contact
  point is still the lowest point of the wheel's frame (finding F8, task
  7.3). The ramp's and the kerb's edges, the grid's cell edges and the track's
  segment ends (where the banking changes) are where a road's slope or
  height jumps; the tyres meet those as they are. The aerodynamics still
  measure ride height above flat ground (`ground_up`). On flat ground every
  regression scenario reproduces its reference bit for bit.
- **Record.** Journal: Phase 7, the road; `include/mbd/forces/road.hpp`,
  `src/forces/road.cpp`, `include/mbd/analysis/track_surface.hpp`,
  `src/analysis/track_surface.cpp`, the tyres, the tyre law (`friction`),
  the builders (`set_road`); `tests/forces/test_road.cpp`,
  `tests/analysis/test_track_surface.cpp`,
  `tests/vehicle/test_vehicle_road.cpp`, the regression scenario
  `sedan-rough-road`; `docs/help/messages.md` (F050, F051).

## D62. The wheel as a body on a spin joint; the contact from the wheel plane and the road

- **Context.** Task 7.3: "Wheel and contact (F6, F8). Wheel as a rotating
  body on a spin joint. Contact point and frame from the wheel plane and the
  road. *Done when a pitching upright no longer moves the contact point, and
  a spinning wheel shows gyroscopic precession.*" D4 had decided the spin
  joint in October: "Spin becomes part of the multibody state, gyroscopic
  effects appear, and torque reactions go to the right bodies." Until now
  the four spins were the drivetrain's first-order states (tasks 1.7, 4.1),
  handed to the tyres, which sat on the uprights with their contact point
  fixed in them at (0, 0, -R).
- **Decision.**
  - **The wheel** (`mbd/vehicle/wheel.hpp`): a body on a revolute joint
    about the Y axis of the body that carries it, its origin at the axle
    centre, its axes the carrier's at a joint angle of zero, so that a
    forward roll is a positive rotation. Its inertia is a uniform thin disc
    of the wheel's mass and the tyre's free radius: m R^2 / 2 about the axle
    (the drivetrain's value since task 1.7), m R^2 / 4 about a diameter.
  - **In the builders** every corner has an upright (the linkage corners'
    upright; on a simple corner the slider, now massless, the wheel holding
    all the unsprung mass) and a wheel on it. The upright steers, takes the
    spring and the anti-roll bar, and gives toe and camber; the tyre is on
    the wheel. `CornerHandle::upright_body` is new and `wheel_body` is now
    the wheel; the simple vehicle has `carrier_bodies`, `wheel_bodies` and
    `spin_v`. A linkage corner's wheel adds its mass, which the model did not
    have before (F9's first item). `wheel_half_extents` is gone.
  - **The contact** (`wheel_contact`, `mbd/forces/tire.hpp`): the wheel's
    lowest point is the point of its circle (the free radius, in the plane
    normal to the spin axis) furthest against the road's normal; the road is
    asked under the centre for that normal, then at the lowest point for the
    height, normal and friction there; the depth is the lowest point's below
    the road's tangent plane, (h - z) n_z; the contact point is the lowest
    point brought onto that plane, and the forces act there. The heading is
    the wheel plane's line in the road plane, a x n. A simple corner's
    steering turns the wheel plane about the road's normal (its upright does
    not steer).
  - **The slips:** the spin Omega is the wheel's angular velocity along its
    axis (absolute: a pitching upright with a locked wheel turns it); Vx and
    Vy are the contact point's velocity carried by the wheel without that
    spin. The tyre law and its slip definitions are unchanged.
  - **The drivetrain** has no states. Its torques act through a force element
    that `connect` adds to the system: a driven wheel's share of the engine
    torque between the wheel and the chassis (an inboard differential), the
    brake's between the wheel and its upright (an outboard brake). Spins are
    read as the wheel's rate relative to its upright. A locked wheel's brake
    gives the opposite of the other torques on it (drive and the tyre's
    moment about the axle), plus a damping of the wheel's inertia over
    `lock_time` (5 ms), within the brake's torque; at the end of each step a
    locked wheel's spin joint is set to rest. `end_step` still locks,
    releases and shifts.
  - **The engine's inertia is left out** until task 7.8 gives the engine a
    speed of its own: its share, seen through the gears, was added to each
    driven wheel's spin inertia while the spins were the drivetrain's
    states; a body's inertia cannot change with the gear, and the kernel has
    no rotor inertia on a joint coordinate. `EngineParams::inertia` is not
    read. The plan's 7.8 is amended to say it returns there.
  - `roll_wheels` (both builders) and `set_free_rolling` set a moving
    vehicle's wheels at free rolling; `Drivetrain::initialize` calls it.
- **Alternatives.**
  - The engine's reflected inertia in the wheel body: it would change with
    the gear (a model is fixed during a run), break the triangle inequality
    of a disc's inertia (validate(), MBD-M014) and spin gyroscopically about
    the axle, which the engine does not.
  - A rotor inertia ("armature") on joint coordinates in the kernel
    (Featherstone's geared rotors, Pinocchio's and MuJoCo's armature): sound
    and general, but a kernel feature with its own invariant tests, model
    files and bindings, for an approximation (each driven wheel its own share
    of one engine) that 7.8 replaces with an engine speed and half-shafts.
  - Locking a wheel by setting its joint's velocity to zero at the end of a
    step, as the drivetrain set its own states: it would change the
    multibody state behind the integrator's back (generalised-alpha's
    history, Dormand-Prince's last stage) and needs write access the state
    elements do not have. The holding torque keeps everything a force.
  - The contact point by iterating to a fixed point on a curved road: on a
    plane two queries are exact; on curved roads a single point is an
    approximation anyway (enveloping is for later).
  - Steering a simple corner by a steer joint: that is task 7.7's.
- **Consequences.**
  - The vehicles have four more bodies and four more degrees of freedom
    (14); statics and linearisation find four more neutral directions (the
    spin angles).
  - Multirate RK4 (D36) has no fast states on a vehicle until 7.4 (the
    tyre's lag) or 7.8; its vehicle test was removed. The integrator report's
    sedan measurements describe the model before.
  - The spin's stiffness at low speed is unchanged (the same inertia, the
    same tyre), now in the bodies' step: RK4 still needs under a millisecond
    at walking pace; the implicit methods carry it.
  - A locked wheel's spin joint is set to rest at the end of each step, as
    the drivetrain's spin state was (a force-only lock let wheels turn back
    by up to 0.11 rad/s): the drivetrain keeps the simulator it was
    connected to, since `end_step` receives it as const.
  - Every driven vehicle accelerates faster without the engine's inertia (in
    first gear about 12 %), and its tyre's force turns the wheel with the
    arm to the road; the linkage cars are heavier by their wheels. The
    regression scenarios sedan-drive, sedan-slalom, sports-car-turn,
    hatchback-brake, sedan-rough-road and sedan-modes were written again;
    the journal's data show by how much and why.
- **Record.** Journal: Phase 7, the wheel; `include/mbd/vehicle/wheel.hpp`,
  `src/vehicle/wheel.cpp`, the tyres, the builders, the drivetrain;
  `tests/vehicle/test_wheel.cpp`, `tests/forces/test_full_tire.cpp`;
  `docs/help/messages.md` (F052).

## D63. The Magic Formula tyre: TNO's MF-Tyre 6.1 as published, where the book differs; MF 5.2 as 6.1 with 5.2's forms; checked against MFeval

- **Context.** Task 7.4: "Tyre (F7). Magic Formula 5.2 with its full
  coefficient set, a `.tir` file reader, camber, aligning and overturning
  moments, rolling resistance, combined slip, and scaling factors. Lag
  through relaxation length, valid to standstill. *Done when curves match
  the reference curves in Pacejka chapter 4 and an independent Magic Formula
  implementation, and a car held on its brakes on a slope stays put.*" The
  author chose, when asked, both MF 5.2 and MF 6.1, the file choosing, and
  MFeval 4.3.1 (Marco Furlan, BSD-3, MATLAB File Exchange 63618) as the
  independent implementation, downloaded outside the repository and run
  there by a script that writes its curves to CSV. This decision covers the
  steady-state tyre; the lag and the slope come in the task's second part.
  The library has had a simpler Magic Formula since before the plan
  (`mbd/forces/pacejka.hpp`: a few coefficients per axis, no file).
  Three sources were at hand: Pacejka's Tire and Vehicle Dynamics, 3rd ed.
  (2012), section 4.3.2, (4.E1) to (4.E78), MF 6.1 only; the appendix of
  Besselink, Schmeitz and Pacejka, "An improved Magic Formula/Swift tyre
  model that can handle inflation pressure changes" (Vehicle System
  Dynamics 48, 2010; the accepted manuscript is open at TU Eindhoven),
  equations (33) to (90), TNO's MF-Tyre 6.1, with a list of what 6.1 changed
  from 5.2; and MFeval's code and documentation, whose author compared it
  with TNO's solver (MF-Tyre/MF-Swift 6.1.2.1) and documents seven places
  where the book's forms did not match it and the paper's did. No MF 5.2
  equation manual was within reach (TNO's is not public; a DTIC report
  refused the download). The book's chapter 4 figures are fits of measured
  tyres whose parameters it does not give.
- **Decision.**
  - **The equations are the paper's where it and the book differ**, and the
    book's where the paper is silent (the epsilons of 1e-6 with their
    neighbour's sign, LMUV, the sign of the forward speed in the trail and
    the residual torque). The slip angle and the inclination enter as they
    are, not as the book's alpha* = tan(alpha) sgn(Vcx) and gamma* =
    sin(gamma); the trail's camber terms are (84) and (86), and its peak
    takes Fz0, not Fz0'; cos(alpha) appears once in the residual torque, in
    Mzr (88), not also in Dr (90); the combined equivalent slips are (80)
    and (81); the rolling resistance is (50), -R0 Fz0 (...), against the
    rolling; the overturning couple is (49), with qsx12 to qsx14; the
    degressive friction factor is not applied (the book's A_mu taken as 1;
    the paper's (42) and (65) carry lambda_mu). The book's qBz6 is not a
    parameter. One point follows TNO's solver as MFeval found it, against
    both written sources: the combined side force in Mz takes Gyk at the
    inclination ((4.E74) and the paper's note say at zero), which matters
    only when rBy4 is not 0.
  - **MF 5.2 is the same evaluation with 5.2's forms where the paper says
    6.1 changed them**: the camber shift pHy3 gamma in place of the camber
    stiffness; the aligning torque's arm on Fy - SVyk; the rolling
    resistance -R0 Fz (qsy1 ... qsy4); the relaxation lengths from pTx and
    pTy; no inflation pressure; the inclination scaled by LGAX, LGAY and
    LGAZ. For the rolling resistance and the relaxation lengths MFeval's
    code (which cites TNO's 5.2 manual) is the only source: there it is
    source and check at once. FITTYP chooses (6, 21: MF 5.2; 61, 62: MF
    6.1), or Adams's PROPERTY_FILE_FORMAT in a file without FITTYP.
  - **"Match the reference curves in Pacejka chapter 4"** is read as: the
    chapter's equations, with the one parameter set the book publishes
    (Appendix 3's 205/60R15 tyre, Table A3.1, written out as a .tir file),
    against MFeval, plus identities the equations imply (slip stiffnesses
    are the slopes, peaks are mu Fz plus the shifts, combined slip leaves a
    force alone at zero of the other slip). The chapter's figures cannot be
    redrawn without their tyres' parameters.
  - **The comparison:** three tyres (Appendix 3; Appendix 3 with every
    scaling factor other than 1; a synthetic MF 5.2 tyre with scaling
    factors), 1812 points over load, both slips, combined slip,
    inclination, speed and pressure, thirteen channels. MFeval runs in mode
    211 (no limit checks or low-speed reduction, no alpha*, no turn slip).
    Coefficients a file leaves out take this tyre's defaults (0; PKY4 2) in
    the script, not MFeval's sample tyre's. Tolerances: 1e-10 of each
    channel's largest magnitude, 1e-6 for the trail and Mz, where MFeval's
    cos(alpha') = Vcx / (Vc + 1e-6) differs from cos(alpha) by 1e-6 / Vc.
  - **The inputs are taken as they are:** a file's ranges are read and not
    applied, and there is no low-speed reduction in the steady state; the
    second part's relaxation keeps the slips finite at rest.
- **Alternatives.**
  - The book's equations as printed: they differ from TNO's solver, with
    which .tir files are fitted. Put back one at a time on the test tyres
    they change a channel by 0.09 % to 9 % of its largest magnitude (the
    rolling resistance by 75 %, and its sign); `docs/tyres.md` has the
    table.
  - Both forms, chosen by an option (as MFeval's alpha* mode): two formulas
    for one quantity, and the book's alpha* matters only beyond the slip
    angles the tyres are fitted at.
  - MFeval's code as the source throughout: the comparison would check a
    copy against its original. It is the source only where nothing else
    was (MF 5.2's rolling resistance and relaxation lengths), and for the
    one observation about TNO's solver above.
  - Refusing qsx12 to qsx14 (as the first version of this code did, after
    the book): the paper has them, and MF 6.1 and 6.2 files carry them.
  - Comparing with the book's figures, digitised: their tyres' parameters
    are not published, so no curve here could be expected to lie on them.
- **Consequences.**
  - `MagicFormulaTyre` (`mbd/forces/magic_formula.hpp`) and `read_tir`
    (`mbd/io/tir_file.hpp`); messages MBD-F053 and MBD-I020 to MBD-I024;
    `docs/tyres.md`.
  - Each line of the equations names its source in both numberings.
  - The comparison needs no MATLAB to run: the CSVs are in
    `tests/forces/data/mfeval/`. Writing them again needs MATLAB, MFeval and
    the script (kept with MFeval, quoted in the journal).
  - Rolling backwards is not compared: MFeval's conventions for it differ.
    Here the trail and the residual torque change sign with Vcx (the book's
    sgn(Vcx)) and the rolling resistance opposes the motion; the second part
    settles reverse running with the vehicle.
  - The vehicles keep the simpler tyre until the second part.

## D64. The tyre on a wheel: the Magic Formula of a parameter set, its slips lagging by the semi-non-linear model, mirrored by side, faded at rest; every vehicle's tyre

- **Context.** Task 7.4's second part: "Lag through relaxation length,
  valid to standstill. *Done when ... a car held on its brakes on a slope
  stays put.*" Finding F7 (October 2026): the default tyre had no load
  sensitivity, no aligning moment, camber thrust, rolling resistance or lag,
  "and below 1 m/s the slip calculation is clamped, so a car held on its
  brakes on a slope will creep instead of standing still". The tyre on a
  wheel (`FullTireForce`, task 7.3's contact) took the simple formula of
  `mbd/forces/pacejka.hpp`, its slips -Vsx and Vsy over max(|Vx|, 1 m/s): at
  rest a damper, not a spring. D63 gave the Magic Formula; the kernel has had
  state elements since task 4.1 (D30), designed with a tyre's lag in mind.
  Pacejka's chapter 8.6 compares four transient models starting from rest on
  a slope; the paper of D63 gives TNO's linear transients, (21) and (22).
- **Decision.**
  - **The slips lag** by Pacejka's semi-non-linear model (8.110 to 8.114):
    two states per tyre, the tread's deflections, du/dt = -Vsx - |Vx| u /
    sigma_kappa and dv/dt = Vsy - |Vx| v / sigma_alpha, the Magic Formula
    taking kappa' = u / sigma_kappa and alpha' = atan(v / sigma_alpha), so
    that a steady state gives the steady-state tyre at the true slip angle.
    The relaxation lengths are the Magic Formula's at the present load (MF
    6.1: the slip stiffnesses over the carcass's, the paper's (19), (20); MF
    5.2: pTx, pTy), at least 0.02 m (Table 8.4's sigma_min). At rest the
    deflections follow the contact's motion: the tyre is a spring of the
    carcass's stiffness.
  - **Below VXLOW** (the file's, 1 m/s for the Appendix 3 tyre): the
    low-speed damping of (8.127, 8.128), k_V_low0 = 770 N s/m at rest
    (Table 8.4), fading to 0 at VXLOW, added to the slips as -k Vs / K; and
    the deflection limit of (8.112): past the slip of full sliding, 3 D / K,
    a deflection is not let grow. Off the road the deflections relax over
    the nominal lengths at |Vx| + VXLOW.
  - **TNO's low-speed model** in the Magic Formula, as MFeval has it: below
    VXLOW the shifts (SHx, SVx, SHy, SVy, SVyk), the trail and the residual
    torque fade out by (1 - cos(pi |Vcx| / VXLOW)) / 2, so that a tyre at rest
    and undeflected pushes nothing. The rolling resistance moment acts in
    full once the wheel rolls at V_roll = 0.01 m/s (Omega R_eff), in
    proportion below it: dry friction, smoothed; it opposes the spin and
    vanishes at rest.
  - **Mirroring by side.** A parameter set describes a tyre of one side
    (TYRESIDE, left by default). A tyre mounted on the other side is
    evaluated mirrored, as TNO's MF-Tyre does: the slip angle and the
    inclination negated going in, the side force, the overturning and the
    aligning moments negated coming out. The builders mount every corner
    (`FullTireForce::mount`).
  - **The parameter set is the tyre's one source:** the free radius
    (UNLOADED_RADIUS), the vertical stiffness and damping
    (VERTICAL_STIFFNESS, VERTICAL_DAMPING) of task 7.3's contact, VXLOW, and
    the forces and moments. `VehicleParams` and `AxleConfig` hold a `tyre`,
    Pacejka's Appendix 3 tyre by default (`pacejka_appendix3_tyre()`), in
    place of `tire_free_radius`, `tire_k_z`, `tire_c_z` and `tire_params`.
  - **The inclination** is asin(a . n), a the spin axis and n the road's
    normal.
  - **The simulator attaches** a force element that is also a state element
    when it is made: every simulator of a vehicle integrates its tyres'
    deflections without being asked.
  - Without lag (`lag = false`) the tyre keeps the old slips over max(|Vx|,
    VXLOW), for comparison and for tests of the slips.
  - The bicycle model reduces a vehicle's tyre to the simple formula about
    its nominal load (`PacejkaTireParams::FromMagicFormula`): slip
    stiffnesses and their load slope, friction and its load sensitivity,
    shape and curvature factors.
- **Alternatives.**
  - The restricted fully non-linear model (8.119 to 8.121), the slip itself
    the state: no division by the relaxation length, but it does not answer
    load changes; the deflection is the physical state, and the book takes
    the semi-non-linear model under "less demanding conditions", which a
    car's handling and parking are.
  - The enhanced model with a contact-patch mass (8.122 to 8.126): a mass of
    a kilogram on the carcass's spring, stiff; it belongs with a rigid-ring
    model, later.
  - TNO's linear transients with the slip angle as the state (the paper's
    (22)): the same equation, but its steady state feeds tan(alpha) where the
    steady-state formula takes alpha; the arctangent keeps the two models
    one.
  - The vehicle's radius and stiffnesses kept beside the parameter set: two
    sources for one tyre, and the rolling resistance's and moments' radius
    would differ from the wheel's (the Appendix 3 tyre's 0.3135 m against the
    vehicles' 0.35 m).
  - Attaching the tyres by hand (`sim.attach` in every caller): a simulator
    made without it would integrate nothing, and the tyre would sit at zero
    deflection with no error.
  - No mirroring: every tyre of a car would push the same way at zero slip
    (the Appendix 3 tyre's ply steer and conicity, 73 N a tyre), the car
    drifting sideways and turning; the symmetric bounce test showed it.
  - No fade at rest: the shifts push at zero slip and zero deflection (19 N
    forwards a tyre), so a car at rest has no static equilibrium, and a free
    wheel on its edge tips over.
  - A damping in proportion to critical: the book's value is light (3 % of
    critical on a quarter of the car), and a car rings on its tyres after a
    stop. Kept, as the book's; a parameter of the tyre (k_V_low0).
  - The rolling resistance ramped in over VXLOW, as first written: at
    walking pace a tyre then had almost none, and the double-wishbone sedan,
    left with a side force of about 300 N on each tyre by its settling (the
    treads deflected as the contacts moved across) on wheels toed out by
    0.8 to 0.95 degrees, rolled away backwards at 4 cm/s within 5 s. Over
    0.01 m/s it creeps at under 2 mm/s; a real tyre's rolling resistance
    acts from the first centimetre, and TNO applies it in full with the
    sign of the speed.
- **Consequences.**
  - The vehicles are the Appendix 3 tyre's: 3.65 cm lower (radius 0.3135
    against 0.35 m), their tyres stiffer vertically (209651 against 200000
    N/m) and damped less (50 against 500 N s/m), their wheels' spin inertia
    lower (1.97 against 2.45 kg m^2), with load sensitivity, aligning and
    overturning moments, camber thrust and rolling resistance. Every vehicle
    scenario of the regression harness was written again (the journal entry
    says by how much and why).
  - A vehicle has fast states again (the deflections): multirate RK4 has
    work on a vehicle. Linearisation holds them at the operating point: at
    rest the sedan's wheels' spins decay at about 170 per second (the
    low-speed damping and the rolling resistance's ramp; about 2800 before,
    the old tyre a stiff damper), and its motions along the road appear as
    real modes of 1.6, 2.0 and 4.1 per second.
  - The stiff spin mode at walking pace is gone: RK4 steps a car at 0.5
    m/s in 5 ms (every wheel within 1.3e-3 of free rolling). The tests of
    the implicit methods on that stiffness run on the tyre without lag.
  - After a stop, sliding tyres give back their treads' deflection, up to
    the limit of (8.112), three times the sliding slip over the relaxation
    length (about 3.5 cm): the car rocks back, a rear wheel can slip back
    against its brake for a few milliseconds (11.7 rad/s measured), and the
    car rings to rest on its tyres (2.4 mm/s a second after the
    double-wishbone sedan's stop).
  - A free car on level ground can creep at millimetres per second when
    its settling leaves it a push (above): the ramp of the rolling
    resistance is not a static friction.
  - The tyre evaluates the Magic Formula twice per evaluation (the slips need
    the relaxation lengths and stiffnesses at the load first).
  - Tests that compare with hand calculations at rest or rolling set aside
    what they do not test, and say so (rolling resistance in the forward
    force test; the shifts and the overturning couple for a wheel balanced
    on its edge).
  - Rolling backwards: the deflections' equations use |Vx| and give slips
    that oppose the sliding; the trail and residual torque change sign with
    Vcx. Not compared with any reference.

## D65. Simple tyres behind one interface: linear with saturation, Adams's Fiala, Pacejka's brush model

- **Context.** Task 7.5: "Simple tyres. Linear with saturation, Fiala and
  brush models, kept for real-time use and for teaching." The plan gives it
  no "done when". After task 7.4 the tyre on a wheel (`FullTireForce`) held a
  Magic Formula; the old simple formula (`mbd/forces/pacejka.hpp`) was left to
  the bicycle model. "Fiala" names two things: in Pacejka's book (section
  3.1) Fiala's 1954 model has a flexible carcass and pure slip only, its
  formulas not printed; in multibody codes (MSC Adams, Project Chrono's
  `ChFialaTire`) the Fiala tyre is the brush model's pure side slip with a
  longitudinal force of its own and a friction falling with slip. The book
  gives the brush model in full (section 3.2), with combined slip for the
  isotropic case. PyChrono's vehicle module, which would have let Chrono's
  Fiala be run, is not in the cross-check environment.
- **Decision.**
  - **One interface**, `TyreModel` (`mbd/forces/tyre_model.hpp`): evaluate()
    the forces and moments from load, slips, inclination, speed, pressure
    and road friction; characteristics() the slip stiffnesses, friction and
    relaxation lengths at a load (by default evaluate() at zero slip); body()
    the radius, vertical stiffness and damping, nominal load, low speed and
    side. The Magic Formula implements it; its input and output became the
    interface's (`MagicFormulaInput` and `MagicFormulaOutput` are names for
    them). `FullTireForce` holds a `std::shared_ptr<const TyreModel>`; its
    lag, low-speed terms and mirroring apply to any model.
  - **Three simple tyres** (`mbd/forces/simple_tyres.hpp`): `LinearTyre`
    (stiffnesses in proportion to the load, the friction circle, a constant
    trail); `FialaTyre`, the industry's Fiala as Chrono has it (read for
    this, BSD-3, kept outside the repository); `BrushTyre`, the book's
    isotropic brush model with a parabolic pressure distribution, its
    contact length growing as the square root of the load. Common to the
    three: rolling resistance f_r R0 Fz, the relaxation lengths the slip
    stiffnesses over the carcass's stiffnesses, defaults near the Appendix 3
    tyre at its nominal load; MBD-F055 for a parameter they cannot use.
  - **Vehicles** take any model: `VehicleParams::tyre_model` and
    `AxleConfig::tyre_model`, null for the Magic Formula of `tyre`;
    `tyre_body()` and `make_tyre()` give the one in use. The bicycle model
    reduces any model to its simple formula
    (`PacejkaTireParams::FromTyreModel`).
  - **"Done when"**, set here as the plan gave none: each model reproduces
    its closed forms, its stiffnesses and its friction limit; carries a
    vehicle (the car held on its brakes on a slope stays put on each); and
    its cost per evaluation and per vehicle step is measured against the
    Magic Formula's.
- **Alternatives.**
  - The book's Fiala (flexible carcass): its formulas are not in the book,
    and no multibody code uses it under that name.
  - The brush model with different stiffnesses in x and y: its combined slip
    has no closed form (the book restricts itself to the isotropic case);
    the Magic Formula serves for a realistic tyre.
  - A friction ellipse for Fiala's combined slip: Adams's and Chrono's Fiala
    have none; a tyre called Fiala should behave as theirs.
  - Simple tyres as separate force elements: the lag, the low-speed terms,
    the mirroring and the contact would be written again for each.
  - A virtual call per evaluation against a template: the call costs
    nanoseconds against microseconds of the Magic Formula and the step.
- **Consequences.**
  - `FullTireForce::tyre` is a pointer (`tire->tyre->evaluate`); the Magic
    Formula's `characteristics()` is the interface's.
  - The cost, in `docs/performance.md`: one evaluation 0.034 to 0.100 us
    against the Magic Formula's 1.35 (13 to 40 times less); one RK4 step of
    the double-wishbone sedan 289 to 291 us against 325, 11 % less, because
    the tyre on a wheel evaluates its model twice per evaluation and the
    rest of the step is the vehicle's (measured on battery power, one run:
    the rows compare with each other).
  - The simple tyres have no shifts, so at rest they need no fade; their
    aligning torque has no residual part and does not change sign.

## D66. Hysteresis as state elements: a damper with a series stiffness and a Bouc-Wen bushing, checked on a virtual dyno; bushing curves by their shape

- **Context.** Task 7.6 carries three items that D42 adopted from external
  sources: a damper with a stiffness in series (a Maxwell element: the top
  mount's compliance and the oil's and gas's) and a Bouc-Wen bushing, both as
  state elements (task 4.1, D30), checked on a virtual damper dyno (a
  sinusoidal stroke; force against stroke and against velocity); and bushing
  curves given by a few shape parameters (linear range, progressive end,
  stop), so that optimisation can move their shape. Half of the task's "done
  when" is the series damper's loop against its closed-form steady-state
  response. The force library had curves as lines or tables (D24), a
  two-point spring-damper with stops and a six-axis bushing, none with
  states of its own. The task's other half, the suspensions, is D67.
- **Decision.**
  - **A bushing's curve by its shape** (`Curve::bushing`, `BushingShape`):
    odd in the deflection, straight with slope k over |x| <= a, then its
    slope rising from k to k_s along a cubic that is flat at both ends,
    until |x| = b, then straight with slope k_s. Four numbers, each a
    physical quantity; the force has a continuous second derivative, so
    Newton's method and the implicit integrators meet no kink. A new kind of
    `Curve`, not a resampled table: the parameters stay what they are, for
    an optimisation to move. Model files hold it (`{"shape": {...}}`), and
    Python makes it (`Curve.bushing`) and shows it as that call.
  - **A rising curve has an inverse** (`Curve::inverse`, to rounding;
    `Curve::is_increasing`; MBD-F057 for a curve that does not rise
    everywhere): a Newton step kept inside the bracket of a table's interval
    or a shape's range.
  - **`MaxwellDamper`** (`mbd/forces/hysteresis.hpp`): two points and one
    state, the series stiffness's extension s. Its force is series(s); the
    state's rate is the length's rate less the damper's own rate,
    damper^-1(series(s)) (MBD-F058 for curves that do not rise). The state
    starts at zero: a Maxwell element carries no force at rest, and statics,
    which holds the states where they are, sees none from it.
  - **`BoucWenBushing`**: the six-axis bushing, its own curves kept, with
    Bouc-Wen hysteresis k z on any axis, z one state per axis following
    Wen's equation; A > 0, beta + gamma > 0, beta >= gamma and n >= 1
    (bounded for any motion; MBD-F059 otherwise). Bouc-Wen's usual alpha k x
    is the bushing's elastic curve, so no parameter is given twice.
  - Both take what their rates need from the bodies (the length's rate, the
    deflections' rates) in apply() and keep it for rates(), as the tyre's
    lag does, so that the linearly implicit method's derivatives reach it.
  - **The virtual damper dyno** (`run_damper_dyno`,
    `mbd/analysis/damper_dyno.hpp`): any force element between the ground
    and a body on a prismatic joint driven through a sine, run by the
    kernel's simulator with any method and step control; after each step
    the forces are evaluated at the step's end state and the element's
    force on the driven body read, against the stroke; the energy a loop
    takes in is the integral of force times velocity over its cycle
    (MBD-A030 for settings it cannot use).
- **Alternatives.**
  - The shape as a resampled table (as Chindamo, Magri and Gadola resample
    into Adams/Car tables): its derivatives with respect to the parameters
    would be piecewise, and a file would hold numbers no one chose.
  - The Maxwell element's state as the damper's stroke, with a reference
    length: it needs the length at which the element is relaxed, and
    statics would see the series stiffness as a spring around the starting
    position.
  - The series stiffness inside `SpringDamper`: every spring-damper would
    become a state element, and the plain element's model files and tests
    would change for a feature most do not use.
  - The dyno's energy as the sum of force times the stroke's changes: it is
    short by sin(w h) / (w h), 6.6e-6 at 1000 steps a cycle; force times
    velocity by the trapezoidal rule is exact for a periodic loop.
  - The dyno's force from its driver's multiplier, as a rig's load cell
    reads it: the element's own force on the driven body is the same number
    without a constraint solve, for any element.
  - The Python interface for the new elements: as for the tyres (task 7.5),
    left for later; `Curve` is bound because curves come from model files.
- **Consequences.**
  - The series damper's loop is its closed-form steady state: on the dyno
    (25 mm at 10 Hz, tau = 0.01 s, RK4 at h = tau / 100) the force follows
    the exact response from rest within 7.8e-8 N of 2666 N; in the third
    cycle the loop is the steady state within 4.6e-6 N (the transient's
    exp(-20), as predicted), its in-phase and quadrature parts U K' =
    1415.22 N and U K'' = 2252.39 N and its energy pi U^2 K'' = 176.902 J
    to 1e-8; the energy a cycle peaks at w tau = 1 as K'' does. Radau IIA
    at steps ten times tau, where RK4 cannot step, misses the steady state
    by 1.748e-7 N, the error its coefficients give for this system.
  - Bouc-Wen's force follows its exact branches (n = 1, composed along the
    stroke; n = 2, the tanh of a first loading) within 2.4e-6 N of the 250
    N hysteretic force, through two loops, against a bound of 2.3e-4 N from
    RK4's error at the rate's kinks; its loop is the same at 0.1 and 10 Hz
    to 8e-10 N; about a rotation axis through a driven hinge, the same.
  - A Maxwell element in parallel with a spring leaves the spring's static
    deflection, in statics and after a released motion; with a tabulated
    damper the rig's work equals the damper's loss and the mount's energy
    to 1.7e-4 J of 195 J, the trapezoidal rule's estimate.
  - The new elements are not in model files (MBD-I008 by their names) nor
    in Python.
- **Record.** Journal: Phase 7, suspensions; `include/mbd/forces/curve.hpp`,
  `include/mbd/forces/hysteresis.hpp`, `include/mbd/analysis/damper_dyno.hpp`,
  `tests/forces/test_hysteresis.cpp`, `docs/kernel.md`, `docs/model-files.md`.

## D67. Suspensions from the kernel's joints, their parts' masses from their joints, springs set by virtual work to hold the design position

- **Context.** Task 7.6 asks for double wishbone, McPherson, five-link and
  trailing-arm suspensions, a solid axle as an option, built from the
  constraint primitives with real masses and centres of mass; coil-over and
  pushrod layouts; damper curves, bump stops and an anti-roll bar acting
  through drop links; done when the motion ratio from the geometry equals
  the one from virtual work (with the series damper's loop, D66). Finding F9
  had measured the double-wishbone and McPherson cars: each arm's centre of
  mass at its pivot, fixed boxes for inertias, springs preloaded for 4000 N
  whatever the car, and set_vehicle_equilibrium leaving the chassis
  accelerating at 1.08 m/s^2 with a coordinate at 73. The five-link existed
  only as a kinematic corner on the ground; there was no trailing arm or
  solid axle; the spring was a linear spring-damper and the bar a force
  element on the wheels' heights.
- **Decision.**
  - **Mechanisms** (`src/vehicle/corners.cpp`,
    `mbd/vehicle/suspension/*.hpp`): the double wishbone and McPherson as
    they were; the five-link a free upright held by five distances, its toe
    link on the rack when steered; the trailing arm one hinge carrying the
    hub; the solid axle a free axle carrying both wheels on two lower
    trailing links, a central upper link and a Panhard rod (heave and roll).
    A trailing arm or solid axle is not steered (MBD-V002). One builder
    makes each corner's mechanism for the vehicle, for the rack's
    calibration and for analysis (`build_corner_rig`: the corner on a fixed
    chassis with bump drivers).
  - **Masses from joints** (`mbd/vehicle/suspension/mass.hpp`): an arm a
    rod from its pivot to its outer joint; an upright its mass among its
    joints and its wheel centre; a solid axle a rod between its wheel
    centres; a five-link's links massless; the template's total mass every
    body's.
  - **The spring unit**: coil-over or pushrod (a distance to a rocker on a
    hinge, double wishbone only), a `SpringDamper` with the damper's curve,
    a second one for the stops measured from the design length (a stop's
    clearance from the spring's free length would be negative for a
    preloaded spring), and a `MaxwellDamper` in place of the damper when a
    top mount is given (D66).
  - **Springs set by virtual work**: each axle's tyre load at rest from the
    masses the builder made and where they lie; each corner's motion at its
    design position, per unit of wheel travel, from Kinematics' velocities;
    the spring force at which the virtual powers of the spring, the tyre's
    load, its overturning moment at rest and the parts' weights add to zero;
    the spring's free length from it. set_vehicle_equilibrium puts every
    corner at its design position, the chassis raised and pitched to the
    tyres' loaded radii; a mixed car's simple corners are mounted so that
    their wheel centres too are at the chassis frame's Z = 0 there. A car of
    simple corners only keeps its earlier placement.
  - **The bar through drop links**: two halves on hinges about the lateral
    axis, a torsion spring between them of k_arb / r^2 (r the half's turn
    per unit of wheel travel), drop links as distances; opt-in
    (`arb_links.enabled`), the force element kept otherwise.
  - **The motion ratio's check**: from the geometry, the unit's rate of
    extension along the motion (a velocity solve); from virtual work, the
    bump drivers' multipliers against a known spring force at rest (a force
    solve).
- **Alternatives.**
  - Each link of a five-link as a body: with ball joints at both ends it
    spins freely about its own axis (a freedom with no stiffness), and its
    mass is a few per cent of the upright's.
  - The springs' preload from statics on the whole car: it gives an
    equilibrium but not the design position, which the user chose with the
    hardpoints.
  - Keeping the 4000 N preload and placing the car by statics alone: every
    car would settle away from its hardpoints' design, differently for each.
  - The tyre's moments left out of the virtual work: the Magic Formula's
    overturning moment at rest (R0 Fz QSX1, about 10 N m) does work along a
    corner whose camber changes, and left the double wishbones and
    McPherson 0.13 to 0.8 m/s^2 from rest.
  - The bar through drop links by default: it would move the regression
    scenario with bars for no gain in what it tests; it is one flag.
  - A mixed car's simple corners mounted at Z = 0, as before: the chassis
    must then pitch by about 4.5 degrees to put every wheel on the road.
- **Consequences.**
  - The plan's test: the motion ratio from the geometry and from virtual
    work agree within 8.9e-16 for the double wishbone (coil-over -0.990,
    pushrod -0.704), McPherson (-0.889), five-link (-1.002), trailing arm
    (-10/9 by its levers) and solid axle (-2 for its two units in heave).
  - Every linkage car is placed within 0.028 m/s^2 of rest (gravity turned
    by the chassis's pitch of about 3.5e-4 rad); statics moves the chassis by
    under 1e-5 m in one or two iterations.
  - The regression scenarios `sports-car-turn` and `hatchback-brake` were
    written again: both had started far from rest (the sports car with 4596
    N on a front tyre 50 ms in, where it settles near 3600, rising 2.7 cm
    before its manoeuvre; the hatchback placed 11 cm above its McPherson's
    design, its front tyres off the ground 50 ms in, with 0.061 rad of
    camber). `wishbone-sweep` moved by 1.3e-15
    (the masses weigh Kinematics' Newton steps) and was left as it was.
  - The vehicle layer has its own message area, V.
  - Joints are ideal: no bushings in the linkages (task 9.1); steering is
    still the rack of task 2.7 (F10, task 7.7); the chassis's centre of mass
    still at its origin (F9's cg_offset, task 7.11).
- **Record.** Journal: Phase 7, suspensions; `docs/suspensions.md`,
  `src/vehicle/corners.cpp`, `include/mbd/vehicle/suspension/`,
  `tests/vehicle/test_suspensions.cpp`.

## D68. Steering: a rack geared to a pinion by a joint coupler, a compliant column to a steering wheel, the input the rack's travel, the wheel's angle or a torque, as held values or functions of time

- **Context.** Task 7.7: "A rack body with tie rods as links, pinion ratio,
  and column compliance. Input as angle or as torque. Done when Ackermann
  behaviour emerges from geometry and the rack moves as one piece." Finding
  F10 had measured the legacy steering: the two tie rods' inner points moved
  by different amounts, set by editing constraints between steps, so the
  steering's rate never entered the equations of motion. Task 2.7 had
  already made the rack one body on a prismatic joint, driven by a
  `JointDriver` to a travel set between steps (its rate zero), with the tie
  rods as distances; F9 had found `steering_ratio` and `max_steer_angle`
  declared and never read. The kernel had no constraint between two joints'
  coordinates.
- **Decision.**
  - **A joint coupler in the kernel** (`kernel::JointCoupler`): q_a - r q_b =
    0 for two revolute or prismatic joints, each read from its joint
    transform, an angle on the branch nearest what the other side gives it
    (a pinion may turn many times); its multiplier the force on joint a;
    MBD-K023 for what it cannot couple.
  - **A steering command** (`SteeringCommand`): a value held, or a function
    of time followed, whose rate and acceleration then enter the equations
    (F10). The rack's driver and the steering wheel's follow one.
  - **Three inputs** (`SteeringConfig::input`): `RackTravel`, the default,
    the rack driven as before (set_steering unchanged, and `steer` for a
    function of time); `WheelAngle` and `WheelTorque`, a steering column on
    the front rack: the pinion and the wheel on hinges about the column's
    axis, the pinion geared to the rack, the column's torsion spring
    between them, the wheel's angle driven or a torque on it
    (`set_steering_torque`, `steer`); MBD-V005 for a command that does not fit.
  - **The ratios from the linkage**: the pinion's ratio is the linkage's
    mean rack travel per radian of toe, as calibrated, over `steering_ratio`;
    the rack's stops at the travel for `max_steer_angle` (F9's two fields
    now read).
  - **Ackermann from the geometry**: nothing in the code computes Ackermann
    angles for a linkage axle; the test compares the mechanism with the
    planar linkage's closed form and reads the behaviour from it.
- **Alternatives.**
  - A column in every car by default: it adds the column's twist, a stiff
    mode (by estimate some 470 rad/s: the column's stiffness over the rack's
    and the uprights' inertia brought to the pinion), to every vehicle, and
    moves every steered scenario; the rack driven by its travel is what the
    existing users steer with.
  - A stiff spring between rack and pinion instead of a constraint: a mode
    far above the others for a gear that does not slip.
  - The coupler's angles from the state vector: constraints see only the
    joints' transforms, and reading them as the driver does keeps one way.
  - A set_steering that ramps by itself: a held value is what it was; a
    ramp is a function of time, which `steer` takes.
  - A default geometry changed to give Ackermann: the presets' tie rods stay
    as they are (their scenarios unchanged); the test builds its own.
- **Consequences.**
  - The rack moves as one piece: its tie rods' inner points within 3e-17 m
    of its travel; geared to the pinion within 4e-18 m and 9e-19 m/s; its
    rate its travel's rate of change within 1.3e-8 m/s (estimate 8e-9).
  - Ackermann emerges: for three arm geometries the mechanism's toes are
    the closed form's within 9e-12 rad; arms turned outward along the line
    from the rear axle's centre give 56 % of the ideal difference at 0.5 rad
    with the rack ahead of the axle (Ackermann's rule is for a track rod
    behind it), parallel arms -7 %, arms turned inward -127 %.
  - The column's statics and its two modes match their closed forms; a car
    with a column has a freedom more (the twist), two with a torque.
  - No regression scenario moved: the default input builds what it built.
  - The coupler is not in model files (refused by name) or in Python.
- **Record.** Journal: Phase 7, steering; `docs/suspensions.md` (Steering),
  `docs/kernel.md`, `tests/vehicle/test_steering_system.cpp`,
  `tests/kernel/test_kernel_constraints.cpp`.

## D69. The powertrain as a state element: an engine with its own speed, a converter or clutch, a gearbox with shift time, differentials and compliant half-shafts, reacting on the housing and the chassis

- **Context.** Task 7.8: "The engine's inertia, left out by task 7.3 (D62),
  returns with the engine's own speed. Engine map with engine braking,
  clutch or torque converter, gearbox with shift time and hysteresis, open,
  locked and limited-slip differentials, half-shaft compliance, a centre
  differential, and an electric motor option. A hysteretic driveline
  isolator (Bouc-Wen) as a state element (D42). Reactions applied to the
  correct bodies. Done when anti-squat matches the geometric value." The
  template's `Drivetrain` (tasks 1.7, 4.1, 7.3) has no states: an engine
  speed tied to the driven wheels, equal torques to them, the brakes, and
  every vehicle scenario steered and braked by it.
- **Decision.**
  - **A new element, `Powertrain`** (`mbd/vehicle/powertrain.hpp`), beside
    the `Drivetrain`, which stays the vehicles' default: a state element
    whose states are the engine's speed, each driven axle's carrier and
    spider speeds and its half-shafts' twists (and Bouc-Wen variables). The
    wheels stay bodies; the powertrain acts on them through the shafts'
    torques.
  - **The parts' laws**: an engine map of a full-load and a drag curve
    blended by the throttle (fuel cut above the redline, idle control
    below idle) or a motor's (peak torque, then constant power, with
    regeneration); a torque converter of pump torque w_e (w_e - w_t) / K^2
    and a torque ratio falling linearly to 1 at the coupling point, or a
    smoothed clutch, or a direct coupling for a motor; a gearbox shifting at
    two speeds (the gap is the hysteresis) whose torque falls to zero and
    back over the shift time, the ratio changing midway; open, locked or
    limited-slip differentials (viscous and a clutch pack); a centre
    differential's torque split with a viscous coupling; half-shafts as
    springs and dampers, with optional Bouc-Wen hysteresis.
  - **Inertias that keep it explicit**: the gearbox's input side and a
    direct motor carried by the carriers as J g^2.
  - **Reactions**: each shaft's torque on its wheel and against the
    differential's housing (the chassis, or a solid axle); the rate of the
    hidden parts' angular momentum against the chassis.
  - **The test of anti-squat**: on trailing arms, the shafts' torques
    reacting on the chassis, from the bump drivers' multipliers, against the
    wheel centre's path.
- **Alternatives.**
  - Replacing the `Drivetrain`: every vehicle scenario and dozens of tests
    would move for a model that the existing users do not ask for; the
    powertrain is one connect away.
  - The driveline's shafts as bodies on joints, geared by couplers (task
    7.7): a position-level gear between unbounded spins is ambiguous (each
    angle is read modulo a turn), and three-way relations (a differential)
    need a constraint the kernel lacks.
  - Massless side gears with an algebraic split: it needs a damper in every
    shaft to be solvable; small inertias keep everything an ODE.
  - A torque converter by tables of K and TR: the closed forms let the
    test be exact; tables can replace them later.
- **Consequences.**
  - Anti-squat: 40 % from the multipliers and from the geometry, to
    rounding; 640 % both ways when the torque reacts on the arm.
  - The converter's stall (207.846 rad/s, 3381 N m a shaft) to 1e-9; the
    differentials' rigid limits to 1e-6; engine braking to 1e-9; the shift's
    torque through zero.
  - The template's `engine.inertia`, unread since task 7.3, is read by
    `EngineMap::from`.
  - Not in model files or Python; the `Drivetrain`'s brakes remain the
    vehicles' (task 7.9 gives brakes of their own).
- **Record.** Journal: Phase 7, the powertrain; `docs/powertrain.md`,
  `tests/vehicle/test_powertrain.cpp`.

## D70. The brakes as a state element: line pressures lagging the pedal, a proportioning valve, dry friction that locks a wheel, a slip-switched anti-lock controller

- **Context.** Task 7.9: "Brakes (F5, F6). Pedal to pressure to torque per
  wheel, bias, locked-wheel handling, and an optional anti-lock controller.
  Done when stopping distance matches a hand calculation." The template's
  `Drivetrain` brakes inside it, without states: a torque equal to the pedal
  times a maximum, split by a bias, as dry friction that locks a wheel (D62).
  The powertrain of task 7.8 (D69) has no brakes.
- **Decision.**
  - **A new element, `BrakeSystem`** (`mbd/vehicle/brakes.hpp`), connected
    to a vehicle's four corners like the powertrain: a state element whose
    four states are the wheels' line pressures, each following its target
    as p' = (target - p) / tau. The target is the master cylinder's pressure
    (the pedal times a maximum); the rear's passes a proportioning valve
    (above a knee it rises by a fraction of the master's).
  - **Torque per wheel** is the axle's gain times the pressure; the bias is
    the ratio of the gains.
  - **Locking** is the `Drivetrain`'s dry friction, copied: the torque
    opposes the spin's direction at the step's start; a wheel that reaches
    rest while the brake can hold the other torques is locked, its spin set
    to rest at each step's end, and released when they exceed the brake.
    The other torques are the tyre's moment about the axle and an optional
    drive (`drive_torque`, for the powertrain's shafts). The torque reacts
    on the upright.
  - **The anti-lock controller** switches each wheel's target to zero when
    its braking slip (from its tyre, Vsx / |Vx|) passes a release threshold,
    and back to the pedal's below an apply threshold, above a minimum
    speed.
  - **The test of the stopping distance**: the simple car on stiff linear
    tyres without rolling resistance, against the closed form with the
    pressure's lag and the wheels' inertia.
- **Alternatives.**
  - Moving the brakes out of the `Drivetrain`: every scenario that brakes
    would move; the two coexist, the `Drivetrain`'s for the existing cars.
  - Brakes inside the powertrain: a car without the powertrain (the
    template's drivetrain, or none) could not have them, and the anti-lock
    controller belongs with the brakes.
  - A smooth friction law (tanh of the spin) instead of locking: it needs a
    slip speed small against the wheels' speeds, which makes the wheel's
    equation stiff where the car stops; the lock logic was already proven
    in the `Drivetrain`.
  - Pressure as an algebraic function of the pedal: the response time is
    what makes the stopping distance differ from v^2 / (2 d), and what the
    anti-lock controller cycles against.
  - An anti-lock controller on the wheels' decelerations, as production
    units do: more thresholds and no closed form to test against; the slip
    switch shows the effect on the stopping distance.
- **Consequences.**
  - The stopping distance from 20 m/s at half pedal: 31.6219 m and 3.109 s
    against 31.5954 m and 3.11036 s by hand (8.4e-4 and 4.4e-4 relative,
    allowed 3e-3). Locked wheels on friction 0.3 decelerate at 2.94202
    m/s^2 against 2.943. On the Magic Formula tyre the controller shortens
    the stop from 100.5 m to 85.3 m.
  - MBD-V007 for settings and corners the brakes cannot use.
  - Not in model files or Python; the `Drivetrain`'s brakes remain the
    template's.
- **Record.** Journal: Phase 7, the brakes; `docs/brakes.md`,
  `tests/vehicle/test_brakes.cpp`.
