# Troubleshooting

From symptom to cause to fix. Each entry names the journal entry where the
problem was first met, so the full story can be read there
(`documentation/journal/`). Messages carry a code (`MBD-K051` and so on); the
[message catalogue](messages.md) explains each.

## The tools, and when to use them

| Tool | What it tells you | When |
|---|---|---|
| `validate(system)`, `validate(system, q)` | Every problem it can find in a model, each naming the body, constraint or element; the degrees of freedom and redundant equations | Before simulating anything new |
| `describe(system)`, `describe(system, q)` | The model as text: counts, bodies and joints, constraints, forces | When the model is not what you think it is |
| `Simulator::trace_steps`, `write_trace(path)` | Per step: drift, residuals, projection iterations and convergence, rank, energies, events (CSV) | When a simulation fails, drifts or gains energy |
| `dump_state(sim)` | The state, each body's placement and velocity, the residuals | At the moment something looks wrong |
| `summary(model)` of the reports of `assemble`, `static_equilibrium`, `linearize` | What the analysis did, iteration by iteration, and its warnings | After every analysis whose result you will use |
| `init_logging()` and `diagnostic_sink()` | Where warnings go (stderr by default, or the logger, or your own function) | To collect or silence warnings |
| Finite differences and energy checks | Whether a new element's derivatives and potential agree with its force | Before building on a new joint, constraint or force (see the end) |

## Builds

**The PC restarts, or the build runs out of memory.** Each compiler needs
1.1 to 1.7 GB; several at once on a 16 GB laptop with a browser open is too
much. The build system runs one compiler at a time (`MBD_COMPILE_JOBS`,
default 1, decision D9), even for a plain `cmake --build`. Do not raise it,
and do not start a build while another runs. *(Phase 0.)*

**An edited header rebuilds nothing.** On a Visual Studio with a non-English
interface, the compiler's notes on included files are translated, and Ninja
recognises them by a prefix recorded at configure time. Configuring and
building under different console code pages makes the prefix stop matching,
and Ninja silently loses every header dependency. Build with
`scripts\build.ps1`, which pins the code page and reconfigures when it
changes. *(Phase 0.)*

**A build compiles against the old version of a header.** With precompiled
headers, a header edited while a build runs is not seen by the files that
build still has to compile. Do not edit headers or test sources during a
build. *(Phase 2: the simulator and the vehicles.)*

**One file takes far longer to compile than its neighbours.** Look at
`build/.ninja_log` (columns: start ms, end ms, mtime, output), or compile the
file alone with `/Bt+` (the command from `ninja -t commands <object>`) to
split parsing from code generation. Template-heavy Eigen decompositions cost
most: `CompleteOrthogonalDecomposition` added 31 s to every file that used
it. Prefer `RankRevealingLDLT`, or keep the heavy one in a single source file
(the general eigensolver lives only in `linearization.cpp`). *(Phase 3:
initial-condition assembly; Phase 2: library and checks.)*

**Build or run times vary by a factor of two or more.** The laptop changes
state (power plan, temperature): the same full rebuild took 5.7 and 28.8
minutes on one day, and the same benchmark 180 and 400 us. Compare only runs
made close together, alternately. *(Phase 2: performance; debugging aids.)*

**The Linux CI job fails on a missing declaration that Windows accepts.**
That job builds without precompiled headers, so a file that relies on the
header for an include fails there. Include what you use. *(Phase 2: library
and checks.)*

**Warnings appear in code that has existed for months.** A function that no
test calls may never have been compiled (an inline function in a header).
Moving it into a source file, or calling it from a test, brings out its
errors. *(Phase 2: library and checks.)*

**The message catalogue test fails.** Every code in the sources must have an
entry in `docs/help/messages.md`, and every entry must be used. The test scans
the source tree, so a new file counts even before it is compiled; every
statement that throws, warns or adds to a report must contain its code before
its semicolon. *(Phase 3: the force library; help and message codes.)*

**A single Catch2 test run by name runs nothing.** A comma in a test's name
separates names on Catch2's command line: write it as `\,`. *(Phase 3:
outputs.)*

**A test meant for a special build shows as a failure in `ctest`.** Catch2 3.6
cannot report a skip to CTest; such tests are hidden (`[.]` in their tags)
and run by name where they apply (the CI job `kernel-no-malloc`). *(Phase 2:
constraints.)*

**`git commit -F -` with a here-string fails in Windows PowerShell 5.1**
("did not match any file(s)"): the message is passed as a path. Write the
message to a file and use `git commit -F <file>`.

## Models

**`validate()` reports errors.** Each message names what is wrong; the M
codes in the catalogue say why it matters and what to do. Fix errors before
simulating; read warnings and notes.

**The counts from `validate()` or `describe()` look wrong.** By default they
are taken at the neutral configuration, which may be neither assembled nor
regular (a four-bar with every link along X has rank 1, not 2). Pass an
assembled configuration: `validate(system, q)`, `describe(system, q)`.
*(Debugging aids.)*

**Redundant constraints (MBD-K050, MBD-M051).** Some equations repeat others:
a planar loop closed by a 3D joint (a four-bar closed by a revolute: 5
equations of rank 2), or a joint closed twice. Harmless when they agree; the
multipliers of the repeated equations are set to zero.

**Redundant constraints that disagree.** Two drivers on one coordinate, or a
point held at two heights: one equation is kept, the other stays violated, and
the projection fails at every step (MBD-K051). The step trace shows it in every
row: rank below the number of equations, the projection not converged after
all its iterations, and a residual equal to the conflict (0.2 for targets at
0.1 and 0.3). Remove one of the two. *(Debugging aids; Phase 2: library and
checks.)*

## Initial conditions, statics and linearisation

**`assemble()` reports MBD-K063 (and MBD-K062).** The held coordinates fix
something the constraints already fix: holding a four-bar's crank and rocker,
for example. Hold at most as many coordinates as the mechanism has degrees of
freedom (`report.degrees_of_freedom`). *(Initial-condition assembly.)*

**`assemble()` changed a coordinate you expected to stay.** It changes only
coordinates that some constraint involves (the weights are diagonal), as
little as possible; a coordinate inside the loop moves with the loop. Hold it
if it must not move. *(Initial-condition assembly, D25.)*

**`static_equilibrium()` does not converge (MBD-K070).** A force with nothing
to resist it: a body on a slider with no spring, a wheel that never reaches
the ground. Newton cannot reduce such a force, relaxation lets the body fall
until its budget ends. Check that every loaded degree of freedom has
something to hold it. *(Static equilibrium.)*

**A template vehicle started with its wheels in the air settles folded.**
The template's springs have no rebound stops and its tyre's contact point is
fixed on the upright: a hanging wheel is pushed past the end of its travel,
the upright turns, and its "contact point" leaves the ground. Start with
every tyre loaded (from `set_vehicle_equilibrium`, or within a centimetre of
it) until the builders gain rebound stops (task 7.x). *(Static
equilibrium.)*

**Statics reports neutral directions (MBD-K072).** Directions with neither
force nor stiffness keep their given values: a vehicle on tyres at rest has
three, its position on the ground and its heading, because a tyre gives no
horizontal force without slip. *(Static equilibrium.)*

**A vehicle's modes at rest look heavily damped.** Below its `V_low` (1 m/s)
the tyre's slip force acts as a damper of tens of kN s/m, which overdamps
several modes. Vehicle modes belong at speed. *(Linearisation.)*

**`linearize()` warns MBD-K080 or MBD-K081.** The state was not on the
constraints, or not in equilibrium. Run `static_equilibrium(sim)` first.
*(Linearisation.)*

## Simulations

**The projection fails (MBD-K051).** Contradictory constraints (see
Models), a mechanism driven through a singular position, a driver whose
target jumps, or a step too long for the motion. Switch on the step trace:
the step where `projection_converged` first drops to 0, and the drift before
it, locate the cause; `dump_state(sim)` there shows the configuration.

**The constraints drift.** The trace's `drift` column is what the projection
corrects each step; if it grows, shorten the step. Without projection
(`project_constraints = false`), Baumgarte stabilisation
(`baumgarte_alpha`, `baumgarte_beta`) keeps the drift bounded.

**The energy grows.** With no damping the total energy (the trace's kinetic
and gravitational columns, plus each spring's `potential_energy()`) must stay
constant to the integrator's accuracy; RK4's error falls as the fourth power
of the step. Growth that does not fall with the step is a defect in a force:
check its potential against its force by finite differences. *(Phase 3:
outputs.)*

**A vehicle at low speed oscillates or blows up.** Below `V_low` a tyre's
lateral force acts as a damper of about 45 kN s/m; on an upright of 15 to 20
kg that decays at 2500 to 3000 per second, beyond what an explicit 1 ms step
can carry (it needs a rate below 2000). Use shorter steps at low speed; the
implicit integrators of Phase 4 remove the limit. *(Static equilibrium.)*

**Contact chatters or explodes.** The step must resolve the contact's
frequency: `omega dt` well below 1, with `omega = sqrt(k / m)` (a 1 kg block
on 4e5 N/m needs steps of about 1e-4 s). *(Simple contact.)*

**A block on a slope creeps instead of sticking.** Friction is regularised:
below `v_static` the contact creeps at the speed where the coefficient
balances the push, below `v_static`. A smaller `v_static` creeps less and
needs shorter steps. *(Simple contact.)*

**An event was missed.** Only the signs at the ends of each stretch of a step
are compared, so a function that goes out and back within one step (a bounce
shorter than the step) is not seen. Shorten the step, or give the action a
rest condition. With Dormand-Prince the stretch is one of its own steps,
which the error control may make longer than you expect where the motion is
smooth. *(Events; the Dormand-Prince pair.)*

**MBD-K052: too many events in one step.** An action that sends its function
back across zero while both directions are watched makes an endless chain.
Watch one direction. *(Events.)*

**Braking results differ from a hand calculation by about 0.2 %.** Since task
4.1 the wheel spins are integrated with the bodies and the result no longer
depends on the step (to nine digits between 1 and 0.25 ms); the difference
comes from the hand calculation, which takes the static rolling radius and no
tyre slip. Before task 4.1 the spins were advanced once per step, which added
an error of the same size. *(Phase 1; the integrator interface.)*

**A vehicle with a drivetrain becomes unstable at standstill.** Below the
tyre's `V_low` the wheel spin equation is stiff: the tyre's slip stiffness
over an undriven 40 kg wheel's inertia is about 2700 per second, just inside
RK4's limit at 1 ms (2.8 / dt) and outside explicit Euler's (2 / dt). Use RK4
at 1 ms or shorter steps; lighter wheels or stiffer tyres need the implicit
methods of Phase 4. *(The integrator interface.)*

**A braked wheel's spin chatters about zero.** Coulomb friction changes sign
when the spin does, and inside one step's stages that sign can flip back and
forth. The drivetrain keeps each brake's direction through a step, locks a
wheel that its brake will stop within the next step, and decides both at the
step's end; an element of your own with dry friction needs the same kind of
discrete decision in its `end_step`. *(The integrator interface; the
Dormand-Prince pair.)*

**Dormand-Prince stops with MBD-K055 or MBD-K056.** Its error control asked
for steps too short (K055) or too many within one `step(dt)` (K056). The
error estimate shrinks with the step only where the forces are smooth: a
force that jumps with the state (a friction sign taken at each evaluation, a
table read without interpolation) leaves an error of the order of the jump
in every step that straddles it, and the step shrinks without end. Make the
force continuous, or decide the jump at the end of a step in a state
element. A stiff model gives the same symptom more slowly: the steps settle
at the stability limit (about 3.3 divided by the largest decay rate), which
may be far shorter than `dt`. `integrator_stats()` gives the steps,
rejections and the last step length. *(The Dormand-Prince pair.)*

**Dormand-Prince is slower than RK4 at the same output step.** Each of its
steps costs six or seven evaluations against RK4's four, and its steps never
pass the end of `step(dt)`: with output steps as short as RK4's, it takes at
least one full step per call. It pays where the error control matters: a
tolerance to meet, a model whose time scales change, long output steps.
*(The Dormand-Prince pair.)*

**A step allocates memory.** `test_alloc` counts every `operator new`; the CI
job `kernel-no-malloc` also catches Eigen's own allocations. The usual cause
is an Eigen expression evaluated through a heap temporary: write
`x = y; x.noalias() -= A * b;` rather than `x = y - A * b` when `y` is a block
of a dynamic vector. *(Phase 2: constraints.)*

**A crash or garbage in a release build.** Eigen's size checks are off in
release builds. The kernel's entry points check their sizes in every build
(MBD-K001); a helper that writes into a vector must size it itself (an
unsized vector passed to one crashed the first linearisation). *(Phase 2:
library and checks; linearisation.)*

## Slow runs

**A benchmark varies by more than a factor of two.** The laptop's CPU mixes
performance and efficiency cores, and a single-threaded run lands on either.
Pin it to a performance core at high priority (PowerShell: start the program,
then set `ProcessorAffinity` to 4 and `PriorityClass` to `High`); pinned runs
agree within 1 %. *(Phase 2: performance, D21.)*

**A run is slower than the recorded figure.** Build the commit of the recorded
figure separately (`git archive -o x.tar <commit>`, `tar -xf`, configure with
the main build's dependency sources) and run both alternately, pinned. The
machine's own state can make everything 2.2 times slower. *(Phase 2:
performance; static equilibrium.)*

## Numerical traps

**A rank decision on a matrix with noise.** Eigen's `LDLT` is not
rank-revealing: it eliminates past the rank, divides by pivots made of
rounding error, and can misjudge the rank and spoil the solve. Use
`RankRevealingLDLT`, which stops at the numerical rank. *(Static
equilibrium.)*

**A test bound fails by a hair, or passes by a hair.** Derive the bound from
the method's error (working rule 3). Floating point does not obey decimal
arithmetic (`0.35 - 0.30` is not `0.05`), and a state placed exactly at a
crossing can round to a value with no sign. *(Phase 3: the force library;
events.)*

**A test that cannot fail.** A check that compares a quantity with itself
passes whatever the code does. Make every check able to fail; Phase 1's
invariant tests failed on the old code exactly where its defects were.
*(Phase 2: constraints; Phase 1.)*

**A check of published coefficients fails.** Check the reference first. A
scratch calculation of the Dormand-Prince extension reported first order
because its "exact" value was wrong (PowerShell built a complex number from
`New-Object Complex(0, $x)` in command mode); the endpoint, computed in the
same script and right to four digits, showed the stages were not at fault.
*(The Dormand-Prince pair.)*

**An integrator test that only looks at the order.** A ratio near 2^p shows
the order but not that the method is the one intended. For a linear test
problem every Runge-Kutta step multiplies the state by its stability
polynomial, known from the coefficients; compare the program's state with
it to rounding, as `tests/kernel/test_kernel_integrators.cpp` does.
*(The Dormand-Prince pair.)*

## Checking a new joint, constraint or force

Before building on it (working rule 2):

- **Derivatives:** a force element's `law()`, a constraint's Jacobian and
  right-hand sides, against central differences (the kernel's invariant
  tests in `tests/kernel/` show how).
- **Potential:** for a conservative element, its `potential_energy()`
  against its force, `-dV/dx` by differences, and an undamped run that keeps
  the total energy.
- **Closed forms:** one motion with a known answer (a period, a static
  deflection, a stopping distance), with the bound derived from the method's
  error.

## Coverage: every problem in the journal

| Journal entry | Problems, and where they are here |
|---|---|
| Before the review | Same kinematics written in several places (the reason for the kernel and for the checks above); restarts from parallel builds (Builds) |
| The review | Defects found by finite differences; a tolerance that hid an 80 % error (Numerical traps: tests that cannot fail, derived bounds) |
| Phase 0 | Code page and lost header dependencies; memory per compiler (Builds) |
| Phase 1 | RK4 truncation against real defects (Numerical traps); one quantity with three meanings (fixed in code); wheel spin outside the integrator (Simulations) |
| Phase 2: the kernel | (no problems recorded) |
| Phase 2: constraints | Heap temporary in ABA (Simulations: allocation); skipped tests and CTest (Builds); a tautological test (Numerical traps) |
| Phase 2: simulator and vehicles | Header edited during a build (Builds) |
| Phase 2: library and checks | Contradictory constraints now fail loudly (Models); untested inline code (Builds); measuring build time first (Builds); size checks (Simulations: crash) |
| Phase 2: performance | Hybrid cores, machine state, A/B runs (Slow runs) |
| Help and message codes | Codes in uncompiled files (Builds) |
| Phase 3: prescribed motion and kinematics | (no problems recorded) |
| Phase 3: the force library | `0.35 - 0.30` (Numerical traps); codes in uncompiled files (Builds) |
| Phase 3: outputs | Commas in test names (Builds); energy as a check of potentials (Simulations: energy) |
| Phase 3: initial-condition assembly | Slow compile of a decomposition (Builds); a thin-margin sanity check (Numerical traps: derived bounds); held coordinates that over-determine (Initial conditions) |
| Phase 3: static equilibrium | LDLT not rank-revealing (Numerical traps); neutral directions (Statics); the folded suspension (Statics); stiff tyres at low speed (Simulations); slow machine (Builds, Slow runs) |
| Phase 3: linearisation | Unsized vector and crash (Simulations: crash); modes at rest overdamped (Linearisation) |
| Phase 3: simple contact | Integrator's momentum error in bounds (Numerical traps: derived bounds); creep (Simulations) |
| Phase 3: events | Missed out-and-back crossings, exact zeros (Simulations, Numerical traps); endless chains (Simulations) |
| Debugging aids | Neutral configuration in `describe()` (Models); conflicting drivers in a trace (Models); slow machine (Builds) |
| Phase 4: the integrator interface | Braking result now step-independent; stiffness at standstill; chattering dry friction (Simulations); a test checking a decision against out-of-date state (fixed in the test) |
| Phase 4: the Dormand-Prince pair | A sign flip that would shrink the steps without end (Simulations: K055 and K056; chattering friction); cost at short output steps (Simulations); a wrong reference in a scratch check; testing more than the order (Numerical traps) |
