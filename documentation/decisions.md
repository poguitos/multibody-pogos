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
| D25 | Work in cloud sessions through git: rules in `CLAUDE.md`, a branch per session, the record unchanged | 5 Oct 2026 | In force |
| D26 | Assembly holds coordinates by masking J and the metric; the least correction is in the kinetic-energy metric, refined to optimality | 5 Oct 2026 | In force, task 3.4 |
| D27 | Statics by Newton on the constraint surface with a finite-difference stiffness, residual measured as accelerations at rest, kinetic-damping relaxation as fallback | 5 Oct 2026 | In force, task 3.5 |
| D28 | Linearisation in the coordinates of the constraint surface, by central differences fitted against the points' actual coordinates | 5 Oct 2026 | In force, task 3.7 |
| D29 | Contact by penalty with a ramped damper and tanh-regularised friction, as a force element | 5 Oct 2026 | In force, task 3.8 |

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

## D25. Working in cloud sessions, through git

- **Context.** Until 5 October 2026 the work was done in sessions on the
  author's laptop: the assistant had the working tree, a memory that outlived
  each session, and the MSVC build. From then on it is also done in cloud
  sessions, each starting from a fresh clone of the GitHub repository on a
  Linux container, allowed to push only to a branch of its own
  (`claude/...`), with no access to the laptop's memory. The rule "every task
  ends with its journal entry" had been saved to that memory; a cloud session
  would never have seen it. The first cloud session also found that the
  latest work (Phase 3 so far) was on the branch `phase-3`, not on `main`.
- **Decision.** Everything a session must know is committed: the working rules
  are in `CLAUDE.md` at the repository root, which every Claude Code session
  reads, and in the plan (section 7). Each cloud session starts its branch
  from the latest work branch and pushes at the end of every task. The record
  itself (journal, decisions, data, help) is kept exactly as before, in the
  commit that finishes each task. Journal entries say whether they were
  written on the laptop or in a cloud session; timings say which machine.
  Merging to `main` and tagging at a phase gate stay with the author.
- **Alternatives.** Keep the rules in session memory (invisible to cloud
  sessions); rely on the user to repeat them at the start of each session
  (easy to forget, and the reason the rules are written down at all).
- **Consequences.** A cloud session builds with GCC only, so CI on the pushed
  branch is the first MSVC build of its work and must be checked. Timings
  taken in a container are not compared with the laptop's (D21). A journal
  entry cannot cite its own commit's hash; the index gets it in the next
  commit.
- **Record.** `CLAUDE.md`; journal README, "Where the work happens".

## D26. Assembly: held coordinates masked, least correction in the kinetic-energy metric

- **Context.** Task 3.4 asks for positions, velocities and accelerations
  solved onto the constraints with chosen coordinates held and the smallest
  correction to the rest. The projection of task 2.6 already moved q onto the
  constraints by Gauss-Newton steps in the kinetic-energy metric, but held
  nothing, gave the least correction only to first order, and reported only
  a residual.
- **Decision.** A held coordinate k is removed from the problem by zeroing
  column k of J and replacing row and column k of the metric by those of the
  identity. Every correction `W^-1 J^T mu` then has a zero in k exactly, and
  the one range-space solve with its rank-revealing factorization serves
  projection and assembly alike. The metric is the mass matrix: at the given
  configuration for positions, at the assembled one for the rates. Positions
  are first brought onto the constraints by Gauss-Newton, then refined to the
  least correction by re-linearized least-norm steps whose fixed point
  satisfies `W d = J^T mu`. A rotation is held whole. Everything is reported:
  per level, residuals, iterations, the coordinate changed most, the rank
  left to the free coordinates, and the freedom left, with codes.
- **Alternatives.** Select the free columns into smaller matrices: the same
  result with index bookkeeping and a second copy of the solve. Unit weights
  on the coordinates: mixes radians and metres, so the answer depends on
  units. User weights per coordinate, as some commercial codes offer: more
  to specify, and hard limits are what users mean by "keep this value"; can
  be added later as a diagonal scaling of W. Large weights instead of exact
  holds: approximate, and ill-conditioned when large. Stopping after the
  Gauss-Newton stage: the correction is the least only to first order (13 %
  more costly in the four-bar test).
- **Consequences.** Light parts move more than heavy ones when they are not
  held, which can surprise (the four-bar's crank turns furthest); the help
  says so and recommends holding as many coordinates as the mechanism has
  degrees of freedom. The least correction is exact for scalar coordinates and
  second-order for rotations. The projection now stops at the best point when
  no step reduces `|phi|`. When the equations are dependent and the plain
  step fails, assembly tries a least-squares step; the simulator's projection
  does not, so that it drops the same equations as the dynamics and never
  allocates.
- **Record.** Journal: Phase 3, assembly. `docs/kernel.md`, "Assembly of
  initial conditions".

## D27. Statics: Newton on the constraint surface, residual as accelerations

- **Context.** Task 3.5 asks for Newton's method on the static residual, with
  line search and dynamic relaxation as a fallback, settling the detailed
  sedan in under 20 iterations to accelerations below 1e-6. Choices: what the
  residual is and how it is measured, how the constraints enter Newton's
  system, where the stiffness comes from, what to do with directions in which
  nothing is stiff, and which relaxation.
- **Decision.** The residual is `F = f + J^T lambda` with time frozen, and it
  is measured as the accelerations at rest, `a = W^-1 F` with `J a = 0`,
  computed by the constraint solver's range-space solve (one place for that
  formula); its tolerance is the plan's 1e-6 on `|a|_inf`, and the line
  search's merit is `a^T M a`. Newton works on the constraint surface: the
  step is `N y` with `(N^T K N) y = -N^T f`, N an orthonormal basis of the
  allowed motions from an SVD of the constraint and hold rows; after each
  trial step the configuration is projected back. K is `dF/dq` at fixed
  lambda by central differences of the simulator's own force evaluation
  (`Simulator::applied_forces`), as D24 planned. Directions without
  stiffness are dropped by a complete orthogonal decomposition with a
  relative threshold, and reported; the stability of the result is read from
  the eigenvalues of the symmetric part of the reduced stiffness. Dynamic
  relaxation is pseudo-dynamics under the forces at rest with kinetic damping
  (velocities zeroed at each peak of kinetic energy), which needs no damping
  parameter, run until the accelerations fall a hundredfold.
- **Alternatives.** The full KKT system `[K J^T; J 0]` in q and lambda: one
  linear solve, but mixed units (N/m against dimensionless) make its rank
  decisions unreliable, and directions without stiffness make it singular in
  ways that are hard to tell from redundancy. Minimizing the potential
  energy: needs every force to have a potential, which tyres and user forces
  do not. Analytic tangent stiffness: decided against for now in D24.
  Viscous dynamic relaxation: needs a damping coefficient tuned to the model.
  Running the simulator with its dampers until the car settles: measured, it
  approaches the static state to 3e-5 m after 8 s but the car keeps rolling
  at a few mm/s, since free-rolling tyres resist nothing (finding F7).
- **Consequences.** About `2 nv` force evaluations per iteration (54 for the
  sedan), cheap at this size; the analytic derivatives can replace the
  differences later. Newton finds the nearest equilibrium, stable or not, so
  the report says which. Eigen's complete orthogonal decomposition must have
  its threshold set before it decomposes: set afterwards, the rank it reports
  changes but its solve does not (found in this task; the least-squares step
  of D26 had the same mistake).
- **Record.** Journal: Phase 3, statics. `docs/kernel.md`, "Static
  equilibrium".

## D28. Linearisation in the coordinates of the constraint surface

- **Context.** Task 3.7 asks for state matrices about an operating point and
  the modes they give. A constrained system's state (q, v) has more entries
  than degrees of freedom; its Jacobian in those coordinates mixes the
  physical modes with the constraints' own directions (zero eigenvalues of
  drift), and quaternions add a coordinate per rotation.
- **Decision.** Reduce first: with N the orthonormal basis of allowed motions
  that statics already computes, the state is `y = N^T (q (-) q0)`,
  `z = N^T v`, and A is `d(y', z') / d(y, z)` by central differences. Each
  point is placed by moving along N and projecting back onto the
  constraints (positions, then velocities); A is then fitted against the
  reduced coordinates the projected points actually have, `A = dF dX^-1`,
  so that the projection's small displacements do not bias it. `dy/dt` is
  `N^T v` where that is exact (scalar coordinates, rotations at q0 or at
  rest) and a central difference in time of `q (-) q0` otherwise. B is the
  exact affine response to generalized forces. The second-order matrices
  and the undamped modes are formed at equilibrium; the modes of A are
  reported with natural frequency, damping ratio and shape in the velocity
  coordinates. An operating point that is not at rest in equilibrium is
  linearised all the same and reported.
- **Alternatives.** Linearise the full state and project out the
  constraint directions afterwards: twice the evaluations and an eigenvalue
  problem polluted with zero eigenvalues that look like rigid-body modes.
  Analytic Jacobians: D24's decision stands; the differences cost 4d + nv
  evaluations, 8 ms for the sedan. A minimal set of independent coordinates
  chosen among q (as some codes do, by pivoting J): coordinates that change
  meaning when the pivoting changes, and fail near singular positions; an
  orthonormal basis does neither.
- **Consequences.** Mode shapes come out in the velocity coordinates, where a
  suspension arm's angle and the chassis height share one vector; the
  coordinate labels say which is which. Linearisation about steady motions
  in a moving frame (cornering, Phase 9) is a vehicle-level task built on
  this.
- **Record.** Journal: Phase 3, linearisation. `docs/kernel.md`,
  "Linearisation".

## D29. Contact by penalty, ramped damping and regularised friction

- **Context.** Task 3.8 asks for points and spheres against a plane with
  penalty stiffness, nonlinear damping and regularised friction. The
  choices: the damping law, the friction law, and whether contact is a force
  element or a constraint.
- **Decision.** A force element. Normal force `k d^e + c s(d / d_c) d_dot`,
  clamped at zero: the damper's coefficient ramps by a smooth step from zero
  at first touch to c at the depth `d_c`, as in the IMPACT function of
  commercial codes, so the force is continuous at contact. Friction
  `mu F_n tanh(|u| / v_s)`, the same regularisation as the joint friction of
  D24, smooth through zero slip. Each law returns its derivatives; the
  spring has a potential.
- **Alternatives.** Hunt and Crossley's damping `c d^e d_dot`: also
  continuous, with a restitution coefficient that depends on impact speed in
  a known way; worth adding as an option when impacts matter (Phase 9
  kerbs). A linear damper: a jump in force at first touch, and a pull just
  before separation. Unilateral constraints with complementarity (exact
  sticking, exact impacts): a different solver class (LCP, time-stepping
  schemes), out of scope until a case needs it. A stick-slip model with
  a bristle state (LuGre): real sticking without creep, at the cost of a
  state per contact, which the integrator interface of task 4.1 will make
  possible.
- **Consequences.** A body that should stick creeps at a speed of order the
  slip speed, exactly predictable on an incline; a small slip speed makes the
  friction stiff (`mu F_n / v_s`) and the time step must resolve it (the
  implicit integrators of Phase 4 will). Contact is with planes only; a
  terrain or road surface (Phase 7) generalizes the plane to a height field.
- **Record.** Journal: Phase 3, contact. `docs/kernel.md`, "Contact".
