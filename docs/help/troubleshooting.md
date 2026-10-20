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
build still has to compile. Without them (Linux, `MBD_PCH=OFF`) the harm can
outlast the build: an object compiled against the old header but written
after the edit looks newer than the header, so Ninja never rebuilds it, and a
changed struct then reads as garbage (`lm.modes.size()` came out as
6148914691236517208). Do not edit headers or test sources during a build; if
it happened, touch the header and build again. *(Phase 2: the simulator and
the vehicles; Phase 5: analytic cases.)*

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
target jumps, or a step too long for the motion. The warning names a
smallest set of constraint equations that cannot all hold together, by
constraint index, kind and bodies (`assemble()` does the same in MBD-K062,
and lists their rows in `AssemblyReport::conflicting_equations`); if it says
the linearised equations can all hold, nothing contradicts and the cause is
one of the others. Switch on the step trace: the step where
`projection_converged` first drops to 0, and the drift before it, locate
the cause; `dump_state(sim)` there shows the configuration. *(Phase 5: the
IFToMM benchmarks.)*

**A closed mechanism stalls, reverses or switches branch at a singular
position (MBD-K055 there, or a motion that turns back).** At a singular
position (a parallelogram folding flat, a four-bar at a dead centre) the
loop's equations lose rank, and positions left off the constraints by even
1e-13 m make the model a slightly different mechanism: a parallelogram whose
coupler must turn at a rate growing as delta theta' / (L theta^2) as theta,
the distance from the singular position, falls. With the default
`constraint_tolerance` (1e-10) such residuals are never projected away. Hold
the constraints near rounding: 1e-14 for a model of metre size, 1e-12 for one
of tens of metres (smaller fails with MBD-K051 at every step). The IFToMM
N-four-bar, through 19 singular positions in 10 s, then follows its closed
form to 2e-8 rad. *(Phase 5: the IFToMM benchmarks.)*

**Results differ in the last digits from an earlier version, or a long run
ends slightly elsewhere.** Since task 4.8 the mass matrix is factorized
along the tree wherever `ConstraintSolver::mass_factorization` (Automatic by
default) chooses it, and densely otherwise. The two agree to rounding (1e-15
relative in one evaluation; the sedan's state after 500 steps of RK4 still
within 1e-14), but a sensitive model can grow such differences over a long
run, and a chaotic one grows them to the size of the motion: the four-bar
with a pendulum on its coupler, run at 1 ms both ways, agreed to six digits
for 20 s and then separated, ending 100 s with 53 % and 64 % of its energy
lost ([the integrators, verified](../integrators.md), section 3). Compare
such runs by their statistics, not point by point.
To reproduce results from before, set `sim.solver().mass_factorization =
ConstraintSolver::MassFactorization::Dense`; `tree_factorization()` says
which one the last solve used. *(The mass matrix along the tree.)*

**The constraints drift.** `Simulator::constraint_drift()` (the drift
channel; `add_drift_channels` puts it in a `Recorder`, and the trace has it
as the `drift` and `velocity_drift` columns) gives what each step left off
the constraints before the projection, and the residuals after it. The
residuals stay below `constraint_tolerance` (1e-10) as long as the
projection converges; the drift is the integrator's own error across the
constraints, and if it grows, shorten the step. Generalised-alpha with
`implicit_control.stabilized_index2` projects nothing: it solves each step
on the constraints, so its drift is its residual, held below the tolerance
by its Newton iterations. With `enforce_constraints = false` the state
drifts as the errors add up (linearly in time, faster with a first-order
method); Baumgarte stabilisation (`baumgarte_alpha`, `baumgarte_beta`) then
keeps the drift bounded. *(Constraints per integrator.)*

**Generalised-alpha drifts far more than the other methods, or its
accelerations and reactions on a closed mechanism look poor at long steps.**
Its acceleration-like variable lags the accelerations, so each of its steps
leaves the constraints by much more than its error along them (in velocity,
300 times RK4's drift at the same step on a four-bar); the projection
removes that from the state but not from the method's history. Held on the
constraints either way, its error on the four-bar came out about three
times that of the same method left free, which drifts off them (1.3e-3 in
20 s), still of second order. `implicit_control.stabilized_index2` solves
each step on the constraints instead: 1.5 to 1.1 times more accurate
positions and 1.7 to 1.2 times more accurate accelerations at steps of 4 to
1 ms there, but on the sedan each step costs about 1.5 times as much, and a
steering input that moves at every step costs it more Newton iterations
(each step must absorb the driver's jump). For a tolerance to meet, use
`RadauIIA`. Do not switch the constraints off. *(Constraints per
integrator.)*

**A spinning body's wobble dies away with semi-implicit Euler or the
linearly implicit step.** Both take the inertia's velocity products (the
Coriolis and gyroscopic terms: the coupling of a body's spin with its tilt)
linearly implicitly, as backward Euler does (D38), which damps a nutation
of rate W by 1 / sqrt(1 + (W h)^2) per step: on a flat body spinning at 30
rad/s, nutating at 28 rad/s, steps of 5 ms take 85 % of the wobble off in a
second. Shorten the step, or use RK4, Dormand-Prince or Radau IIA where a
spinning body's motion must be kept. Before D38 both methods took these
terms explicitly, which feeds energy in at every step: a top spinning at 30
rad/s, stepped at 5 ms, blew up within 100 s (then the projection failed at
almost every step, MBD-K051). *(Constraints per integrator; implicit
inertial terms.)*

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

**Wheel spins chatter at low speed, or a vehicle with a drivetrain becomes
unstable at standstill.** Below the tyre's `V_low` the wheel spin equation is
stiff: an undriven 40 kg wheel's spin relaxes to free rolling at 2800 to 3100
per second (measured on the simple vehicle at 0.5 m/s), just outside RK4's
limit at 1 ms (2.785 / dt) and far outside explicit Euler's (2 / dt). Beyond
the limit RK4 does not blow up: the spins chatter about free rolling, bounded
by the tyre's saturation, by 2.8 % of the speed at 1 ms and 14 % at 5 ms,
while the car's speed looks right. Use RK4 at 0.9 ms or less there, or
`Integrator::GeneralizedAlpha`, which is stable at any step (the car rolls at
walking pace in steps of 5 ms with its wheels within 1e-3 of free rolling). RK4
with `multirate` 6 at 3 ms steps the spins at 0.5 ms and the bodies at 3 ms.
`Integrator::LinearlyImplicitEuler` is stable at any step at a fixed cost.
*(The integrator interface; generalised-alpha.)*

**A stiff model needs tiny steps.** A very stiff spring, bushing or contact
limits an explicit method by stability, not accuracy: RK4 needs w h <= 2.8
for the fastest frequency w, even when nothing in the answer moves that
fast. `Integrator::GeneralizedAlpha` is stable at any step; its
`implicit_control.rho_infinity` (0.8) sets how fast the motions too quick for
the step are damped away. Each step costs one evaluation plus one per Newton
iteration (one or two at the default tolerances on the vehicles: three
evaluations a step on the sedan at 1 ms, against RK4's four), and an
occasional iteration matrix (2 nv + nx evaluations). For a tight tolerance
use `Integrator::RadauIIA` with error control: fifth order, it reaches 1e-7
m on the bushed pendulum for less than half the evaluations
generalised-alpha needs for 1e-6. For real time, where every step must cost
the same, `Integrator::LinearlyImplicitEuler`: one evaluation a step, stable
at any step, first order (but see the next entries for very stiff springs
between turning bodies). Measured costs for a given accuracy, method by
method: [the integrators, verified](../integrators.md). *(Generalised-alpha;
SDIRK4 and Radau IIA; linearly implicit Euler; the integrators verified.)*

**The real-time integrator blows up, or gains energy, on stiff bushings or
springs between bodies that turn.** Two different limits, both found by
the verification (task 4.9). Before D40 the linearly implicit step kept only
the symmetric part of the force derivatives, and a bushing that carries a
large rotation has a stiffness far from symmetric (k_r times half the
rotation): the bushed pendulum with 1e6 N m/rad diverged at 2, 1 and 0.5 ms.
It now takes them whole. What remains is a limit of linearizing once per
step: each attachment point moves along its tangent, the true motion along
an arc, and a very stiff spring turns their gap, about k L (w h)^2 / 2, into
a force that feeds energy in. On the pendulum with 1e8 N/m the energy rose
by 0.21 J at 1 ms and 8.6 J at 2 ms, and the swing ended 4 cm and 48 cm off;
at 0.5 ms and below it only lost energy. Keep k h^2 / m below about 25 for
such springs, make the hinge a joint, or, off line, use `GeneralizedAlpha`
or `RadauIIA`, whose Newton iterations do not have it. *(The integrators
verified, D40.)*

**The real-time integrator damps a fast motion away too soon.**
`LinearlyImplicitEuler` is first order and, like backward Euler, damps
every motion by about (w h)^2 / 2 per step on top of its physical damping:
a wheel hop of 15 Hz at 1 ms loses 0.44 % of its amplitude per step, half
of it in about 0.16 s. Slow motions (a body's roll and pitch, the car's
speed) are hardly touched. Keep the step short against the fastest motion
whose amplitude matters, or, off line, use `RadauIIA` or
`GeneralizedAlpha` with a high `rho_infinity`. *(Linearly implicit Euler.)*

**A model accelerates too slowly, or gains energy, with the linearly
implicit step after a change to a state element or a tyre.** The step is
only as good as the derivatives it is given (D35). A state whose rate
depends steeply on a body's velocity (a wheel's spin through the tyre's
slip) must be coupled to it in the step: without that coupling the sedan
accelerated a fifth too slowly at 1 ms. A dependence on the positions that
is not symmetric (the tyre's rolling radius changing with its deflection)
is left out on purpose: in the step's matrix (then symmetric, D35) it made
a coasting car gain speed. Check a new element against `Simulator::force_jacobian`
and a run with RK4 at a short step. *(Linearly implicit Euler.)*

**With multirate, a car drifts or creeps although single-rate RK4 at the
same fast step is fine.** The fast steps are too long for the states:
`multirate` leaves each part to its own stability, and a wheel spin that
chatters within the fast steps (at 1 ms, the sedan's front spins at walking
pace) is seen by the bodies' stages only at the middle and end of the step,
where the chatter does not average out as it does at a single rate; the
sedan then moved at 0.5 m/s two seconds after it had stopped. Raise
`multirate` (3 ms with 6 fast steps of 0.5 ms works there). If raising it
does not help, the step itself is too long for the bodies: at low speed
the tyres' slip also stiffens the hubs. *(Multirate.)*

**An implicit method stops with MBD-K058.** Newton could not solve a step,
even with a fresh iteration matrix: a force that is NaN or infinite, a force
that jumps within the step, or a step so long that the forces change their
character across it. Check the forces at that state, shorten the step there,
or make the jump continuous (or decide it at a step's end). With SDIRK4 or
Radau IIA at a fixed step, error control (`step_control.adaptive`, on by
default) shortens the step where Newton needs it instead. A Newton
tolerance far tighter than the step's accuracy makes it likelier: on the
stiff bushed pendulum generalised-alpha at 2 ms failed with the tolerances
at 1e-12 and ran at 1e-9 (its error there 2.7e-5 m); on the four-bar at 8
ms likewise. Its longest steps measured: 2 ms on that pendulum, 5 ms on the
sedan ([the integrators, verified](../integrators.md)). *(Generalised-alpha;
SDIRK4 and Radau IIA; the integrators verified.)*

**SDIRK4 is inaccurate on a model with stiff springs or bushings, and its
error control does not notice.** Its stage order is 1: as springs stiffen
towards joints its order falls from four towards one (on the bushed pendulum
about 3.6 at 1e6 N/m, 2.5 at 1e8, 1 at 1e10), and its third-order estimate
misses the loss (asked for 1e-4 there, it ended 7e-4 m out). This is a
property of the method, not a fault in the model. Use `RadauIIA` (stage order
3) or `GeneralizedAlpha` for stiff mechanical models; keep SDIRK4 for smooth
models and for stiffness in first-order states. *(SDIRK4 and Radau IIA.)*

**An implicit method builds many iteration matrices, or fails at long steps,
on a model with bushings.** The derivative of a bushing's forces can change
fast with the motion even when the forces do not: a bushing with stiffness
about two axes and none about the third (a hinge) measures its tilt about an
axis halfway between its two markers, so a bending deflection makes a
torsion moment of about k_r phi / 2 at a hinge angle phi, and on a link with
a small torsional inertia that coupling is stiff and turns with the swing.
A matrix built at a step's start then stops serving within a long step:
Newton slows, the matrix is rebuilt (`integrator_stats().jacobians`), and at
a fixed step it may fail (MBD-K058). Use error control, or shorter fixed
steps; generalised-alpha, which takes short steps for its accuracy anyway,
is less affected. *(SDIRK4 and Radau IIA.)*

**A locked or stuck state drifts slightly with the implicit method.**
Newton solves each step only to the tolerance, so a state that should stay
exactly fixed by a discrete decision (a wheel locked by its brake) can move
by about the tolerance. Enforce the decision in the element's `end_step`, as
the drivetrain does. *(Generalised-alpha.)*

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
machine's own state can make everything 2.2 to 2.3 times slower (one RK4
step of the sedan: 282 us one day, 123 the next). *(Phase 2: performance;
static equilibrium; the mass matrix along the tree.)*

**Single steps far longer than the median.** When single steps are timed
(for a real-time budget), the first ones size the integrator's work and warm
the caches: leave them out. After them, a step of a method with fixed work
can still take ten times the median now and then; if the longest steps
differ between runs, they are the operating system's (Windows is not a
real-time system), not the method's. A method whose longest step comes back
in every run (generalised-alpha rebuilding its iteration matrix) owns it.
*(The fixed-cost step.)*

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

**A bound from one step of a stable method.** A method that is stable can
still let a quantity grow for several steps before it decays, when its step
matrix is far from normal (generalised-alpha at very long steps: its
eigenvalues cluster near -rho_infinity, and the amplitude doubles first).
Compare with the method's own recurrence rather than a hand-made bound.
*(Generalised-alpha.)*

**A Newton tolerance below the rounding.** When an update adds terms much
larger than its result (a stiff spring at a long step: h^2 |a| far above the
displacement), the result carries their rounding, and Newton's corrections
cannot fall below it. The implicit method's convergence test has a floor at
that rounding; a test of your own should too. Stiff forces add a second
source: a 1e8 N/m bushing turns the positions' rounding (1e-16 m) into
forces of 1e-8 N, far above the rounding of the force itself; SDIRK4 and
Radau IIA put the state's rounding carried through their Jacobian, eps h
sum_j |J_ij| |y_j|, into the floor as well. *(Generalised-alpha; SDIRK4 and
Radau IIA.)*

**Newton's error is part of the global error.** An implicit method with
error control estimates the truncation error of each step; Newton's own
error in solving the step is not in that estimate, and it adds up from step
to step like any other. Stopped at a tenth of the tolerance, it made Radau
IIA's global error several times the tolerance on the bushed pendulum,
whose fifth-order truncation error is far below its third-order estimate.
SDIRK4 and Radau IIA stop Newton at min(0.03, sqrt(tol)) of the tolerance,
as RADAU5 does; if a global error is well above the tolerance, suspect the
solves before the steps. *(SDIRK4 and Radau IIA.)*

**An integrator test that only looks at the order.** A ratio near 2^p shows
the order but not that the method is the one intended. For a linear test
problem every Runge-Kutta step multiplies the state by its stability
polynomial, known from the coefficients; compare the program's state with
it to rounding, as `tests/kernel/test_kernel_integrators.cpp` does.
*(The Dormand-Prince pair.)*

**An exact reference, and errors past any order.** A motion driven by
forces fixed in time (a manufactured trajectory: forces from inverse
dynamics) has nothing pulling a deviation back, so an early error grows; on a
tree with a free body and a spherical root, a span of 1 s left the
first-order methods with errors as large as the motion. Measure an
integrator's order over a short span (0.5 s there), or over one step from the
exact state, which no growth touches (the local error falls by 2^(p+1) when
the step halves). *(Phase 5: analytic cases.)*

**Mode shapes that do not match, frequencies that do.** A shape scaled to its
largest entry is ambiguous when two entries are equally large (a symmetric
mode of a line of masses); two correct shapes can then differ by a sign or a
factor. Compare shapes up to a scalar: the part of one unit shape not along
the other. *(Phase 5: analytic cases.)*

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
| Phase 4: generalised-alpha | A bound guessed from one step; Newton against rounding (Numerical traps); a locked wheel that crept (Simulations); stiff models and K058 (Simulations) |
| Phase 4: SDIRK4 and Radau IIA | SDIRK4's order on stiff bushings (Simulations); a Jacobian that turns within a step (Simulations); Newton's error and Newton against amplified rounding (Numerical traps); fixed steps Newton cannot solve (Simulations: K058) |
| Phase 4: the fixed-cost step | The car that accelerated too slowly and the one that gained speed (Simulations: the linearly implicit step); damping of fast motions (Simulations); timing outliers from the first steps and the operating system (Slow runs: single steps far longer than the median) |
| Phase 4: multirate | Held states that biased braking, and the drift from fast steps beyond the spins' limit (Simulations: multirate) |
| Phase 4: constraints per integrator | The drift and its channel; generalised-alpha on a closed mechanism; a spinning body that blows up under explicit gyroscopic terms; a tolerance that is not positive (Simulations; MBD-K047) |
| Phase 4: the inertia's velocity products implicit | A spinning body's wobble damped by the two Euler steps (Simulations); a matrix not positive definite (MBD-K045) |
| Phase 4: the mass matrix along the tree | Results that differ in the last digits from before (Simulations); a machine 2.3 times slower from one day to the next (Slow runs) |
| Phase 4: the integrators verified | The real-time step diverging, then gaining energy, on stiff bushings between turning bodies (Simulations); generalised-alpha's Newton at long steps with a tight tolerance (Simulations: MBD-K058); a chaotic model's long runs depending on rounding (Simulations: results differ) |
| Phase 5: analytic cases and manufactured trajectories | A header edited during a build, leaving a stale object (Builds); forced motions that amplify errors past any order, and mode shapes compared by an ambiguous normalization (Numerical traps) |
| Phase 5: the IFToMM benchmarks | A mechanism stalling at a singular position with loose constraints (Simulations); contradictory constraints named (Simulations: MBD-K051, Initial conditions: MBD-K062); messages that wrote small numbers as 0.000000 (fixed) |
