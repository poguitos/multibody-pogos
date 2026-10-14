# The book: outline

A working outline for the account of the project, a thesis or a book. It
changes as the project does: when a journal entry adds a topic worth a
section, the topic is added here with its sources. Each chapter lists where
its material is: journal entries (`J:`), decisions (`D`), project documents,
code and tests.

Working title: *Building a vehicle multibody solver: from a failing prototype
to a validated engine.*

## Part I. The problem

1. **Why build a multibody solver for vehicles.** Goals: an ADAMS-class
   solver, a vehicle builder, lap-time optimisation, aerodynamics from CFD,
   real time, a Python and AI layer. What exists, and why this project.
   *J: before the review. Master plan, section 2.*
2. **Background: rigid multibody dynamics.** Joint coordinates on a tree and
   constraints to close loops; spatial vector algebra; recursive algorithms;
   differential-algebraic equations and their stabilisation.
   *Featherstone 2008; Haug 1989; Shabana 2010; Baumgarte 1972; `docs/kernel.md`.*
3. **Background: vehicle dynamics.** Suspensions as mechanisms, tyres,
   aerodynamics, drivetrain, the standard manoeuvres and axis systems.
   *Blundell and Harty; Pacejka 2012; Popp and Schiehlen 2010; ISO 8855.*

## Part II. The first version and its review

4. **The first version, December 2025 to September 2026.** Its design, what it
   could do, and the solver error found in May. *J: before the review.*
5. **How to find out whether a solver is right.** Finite differences of
   position-level quantities, closed-form solutions, conservation laws; the
   bead on a spinning rod. *J: the review; Phase 1. D11.
   `documentation/review_2026-10-02/`.*
6. **The findings.** Four disagreeing copies of one formula (F1 to F4); the
   vehicle layer (F5 to F12); missing capabilities (F13 to F18); hygiene (F19
   to F21). *Master plan, section 3.*
7. **A plan to the end.** Findings, decisions, phases with "done when"
   conditions. *Master plan; D1 to D8.*

## Part III. Making the core trustworthy

8. **Builds that cannot hurt the machine.** One compiler, presets, pinned
   dependencies, the code-page failure, precompiled headers, CI with
   sanitizers. *J: Phase 0. D9, D10.*
9. **Invariant tests and the fixes.** Tests first; 315 of 845 assertions; the
   four fixes; brakes and the implicit wheel-spin step; derived tolerances.
   *J: Phase 1. D11.*

## Part IV. The kinematics kernel (Phase 2)

10. **Spatial algebra and joint models.** Motion and force vectors, frame
    changes, the joint interface (`X_J`, `S`, `c`, `G`), quaternions and why a
    Runge-Kutta step must not normalise its stages. *J: the kinematics kernel.
    D1, D2, D12. `docs/kernel.md`.*
11. **Recursive algorithms.** Forward kinematics, RNEA, CRBA, ABA; gravity as
    an accelerating ground; Model and Data. *J: the kinematics kernel.*
12. **Constraints.** Markers, five primitives, closures, Jacobian rows as
    wrenches. *J: constraints. D13.*
13. **The constrained solve.** Range-space method, redundancy, projection,
    Baumgarte. *J: constraints. D14.*
14. **The simulator and the vehicles on the kernel.** The force bridge,
    drivers, kinematic analysis with t as a parameter, the steering rack.
    *J: the simulator and the vehicles. D15, D16, D17.*
15. **One library, and checks that stay on.** Moving code by script, the
    anatomy of a C++ build, argument checks, `validate()`.
    *J: library and checks. D18, D19.*
16. **Performance.** The benchmark that misled, hybrid cores and pinning, one
    kinematics pass, `Z = L^-1 J^T`, proving "no allocation".
    *J: performance. D20, D21. `docs/performance.md`.*

## Part V. Solver completeness (Phase 3)

17. **Prescribed motion and kinematic analysis.** Drivers as constraints;
    velocities and accelerations of a motion (motion ratios). *J: prescribed
    motion and kinematic analysis. D15.*
18. **The force library.** Characteristics as monotone cubics; the
    spring-damper with stops; forces on joint coordinates and regularised
    friction; rotational springs and bushings; force laws with their
    derivatives. *J: the force library. D24.*
    *Outputs: J: output requests.* Joint reactions by recursive Newton-Euler
    with external forces; constraint forces per body from the rows that build
    the Jacobian; energies; the recorder.
19. **Assembling initial conditions.** The least correction subject to the
    constraints and the holds; its optimality conditions; why Gauss-Newton
    reaches the constraints but not the nearest point, and the iteration
    anchored at the given values that does; the choice of metric (a pendulum
    outside the loop under the full and the diagonal mass matrix); holds that
    over-determine a mechanism, found by comparing ranks. *J: initial-condition
    assembly. D25.*
20. **Static equilibrium.** Newton on the accelerations at rest; frozen
    drivers; a Jacobian by differences through projections; least-norm
    steps and the neutral directions of a vehicle on tyres; increment limits
    and relaxation on the static forces; what "not rank-revealing" means and
    how it showed; the folded suspension as a finding about the model.
    *J: static equilibrium. D26.*
21. **Linearisation.** Reduced coordinates on the constraints from the
    projector's pivoted columns; the state matrix; modes, damping ratios and
    shapes; the quarter car checked against its closed forms and its
    characteristic determinant; what the tyre's regularisation does to a
    vehicle's modes at rest. *J: linearisation. D27.*
22. **Contact.** Penalty contact with the IMPACT law; regularised friction
    with static and dynamic levels; the creep speed as a prediction; the
    friction angle as the test. *J: simple contact. D28.*
23. **Events.** Zero crossings located by root finding on the step length;
    just past the crossing; what endpoint detection cannot see (the bouncing
    ball near its end); the guard against endless chains. *J: events. D29.*

## Part VI. Time integration (Phase 4)

24. **States beyond the bodies.** First-order states owned by elements and
    integrated with the bodies in every stage; the order a lag shows when it
    is, and is not; the drivetrain's wheel spins as a hybrid system,
    continuous spins with discrete decisions (lock, gear) at the ends of
    steps; why a brake's hold is a lock and not a steep friction curve; the
    chatter of dry friction inside RK4's stages, and the lock on a wheel
    about to stop; the braking result that became independent of the step.
    *J: the integrator interface and state elements. D30.*
25. **Error control and dense output.** The Dormand-Prince pair; the
    stability polynomial as an exact test (an oscillator whose numerical
    solution is known to the last digit); tolerance proportionality and a
    global-error constant derived from the coefficients; output steps against
    integrator steps when inputs change between them; events on the
    continuous extension; a discontinuity that error control cannot cross,
    and the brake that keeps its direction through a step. *J: the
    Dormand-Prince pair. D31.*
26. **Stiffness and the implicit method.** Stability against accuracy;
    generalised-alpha and its one parameter; the recurrence as the test; the
    transient of a non-normal step matrix; Newton against rounding; a lock
    that must hold exactly whatever the tolerance; RK4 past its limit that
    does not blow up but chatters, bounded by a tyre's saturation, and the
    estimate on the wrong side of the limit that a test exposed. *J:
    generalised-alpha. D32 (and D30's correction).*

## Part VII onwards: one part per phase still to come

Validation (5), Python
(6), vehicle subsystems (7), the vehicle builder (8), vehicle analyses (9),
lap time (10), aerodynamics and CFD, including the in-house CFD solver (11),
real time on Windows and Linux (12), optimisation and AI (13), flexible bodies
and FMI (14), documentation and release (15). Each part follows its phase's
journal entries and retrospective.

## Part X. Engineering practice

- **Testing a numerical code.** Invariants, derived tolerances, tests that must
  be able to fail, hidden tests for special builds. *D11; working rules 2, 3.*
- **Measuring performance.** *D21; `docs/performance.md`.* Compile time as
  well as run time: the decomposition that cost 31 s per file (journal:
  initial-condition assembly).
- **Keeping a record.** This journal, the decision log, and how they fed this
  book. *D22.*
- **Help and diagnostics.** Messages written for the reader, codes and a
  catalogue, `validate()`; `describe()`, the state dump and the step trace,
  and the signature of conflicting constraints in a trace. *D19, D23; J:
  debugging aids.*
- **Working with an AI assistant.** The March 2026 rewrite from an AI proposal,
  the review, and the collaboration since: what worked, what had to be
  checked, and why every claim was tied to a measurement. *All journal entries.*

## Appendices

Conventions (`docs/conventions.md`); the equations of every element (the
theory manual, task 15.1); the test catalogue; benchmark tables
(`journal/data/`); glossary; bibliography (`references.bib`).
