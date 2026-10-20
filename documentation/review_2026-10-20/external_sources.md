# External sources reviewed, 20 October 2026

Eleven links and five documents the author found interesting, read with one
question: what would make this program better, and where in the plan would it
go? Each source is summarised, judged against what the program already has,
and given a verdict. The proposals are collected and ranked at the end. The
author adopted all of them the same day; they are decision D42 and are in the
plan in their phases. (Fast-F1 and REA, sections 15 and 16, were added after
the first reading.)

Local copies of every repository and web page are in `C:\dev\references\`
(outside the repository; its `README.md` lists each with its commit and
licence).

## How the review was done

- The five GitHub repositories were cloned (shallow) outside the project and
  read, not built or run: READMEs, licences, documentation and the central
  source files of each.
- The four other links were read on the web, with the blog posts they lead to
  where those carried the substance (MOVEdot).
- The five PDFs in `documentation/` were read in full.
- **Not read in full:** the MDPI article (*Vehicles* 8(9), 214). MDPI refuses
  automated downloads; its metadata and abstract came from Crossref. Dropping
  the PDF into `documentation/` would allow a full reading.
- Our own code was checked before proposing anything, so that nothing already
  present is proposed again (our SVG plots, for instance, already use the
  Okabe and Ito colour-blind-safe palette).
- **Licences:** TUM and JAOT are Apache 2.0, Yano's guide, ampl-agents,
  Fast-F1 and REA MIT, the numerical-methods course CC-BY 4.0 (text) and BSD-3
  (code). Every
  proposal below reimplements an idea from its published description and cites
  it; no code would be copied. (This repository has no licence file yet; that
  matters only when the program is published, Phase 15.)

## The sources

### 1. TUM, VehicleDynamicsPerformanceAnalysis (GitHub, Apache 2.0)

**What it is.** A Python tool from the TUM autonomous motorsport team that
turns logged vehicle data (CSV, MATLAB, ROS 2 MCAP, CAN dumps) into one
convention of channel names and units, resamples it, derives vehicle-dynamics
channels, slices it into laps, labels corners and their phases, estimates the
instantaneous cornering stiffness, and writes a PDF run report and an
interactive plot tool.

**What is useful to us.**
- **Channel names with their units** (`vx_mps`, `ay_mps2`, `yaw_rate_radps`,
  `delta_f_rad`): a convention that keeps units with every column of a CSV.
  Our recorder lets the user name channels freely.
- **A library of derived channels**, each computed only when its inputs exist:
  sideslip, turn radius and curvature, per-wheel slip angles and slip ratios,
  combined slip and slip velocities, axle lateral forces from the bicycle
  model, wheel loads with load transfer and aero, friction utilisation,
  understeer gradient, kinematic yaw-rate reference, excess steering, and
  `r cos phi` (the direction of the acceleration: -1 braking, 0 cornering, +1
  accelerating). These are the channels a vehicle report (plan 9.5) and a lap
  report (Phase 10) need, and computing them from simulation output with the
  same formulas used on measured data is what makes the two comparable.
- **Corner characterisation**: apexes from peaks of the smoothed curvature,
  corner limits from the sign changes and low values of lateral acceleration,
  phases (braking, entry, mid, exit) from `r cos phi`, and corners classed
  low, medium or high speed by apex speed. Useful for lap-time results.
- **Instantaneous cornering-stiffness estimation** from measured data
  (Sienel 1997; Haffner et al. 2008): windowed least-squares slopes of axle
  lateral force against slip angle. A tool for identifying tyre parameters
  from test data, not for simulation.
- The plan already names the TUM racetrack database format for the track
  (10.1).

**Verdict.** Worth taking, as ideas for 9.5 (reports), Phase 10 (lap results)
and correlation with measured data (proposal 7 below). The ROS, MCAP and CAN
ingestion is outside our scope.

### 2. Yano, AutonomousVehicleControlBeginnersGuide (GitHub, MIT)

**What it is.** Educational Python implementations of autonomous-driving
algorithms, each with a runnable simulation: localisation (Kalman and particle
filters), mapping, path planning (A*, RRT and others), and path tracking (pure
pursuit, adaptive pure pursuit, Stanley, rear-wheel feedback, LQR on the
lateral and heading error, MPC with do-mpc and CasADi, MPPI), with design
chapters for Stanley, MPC and MPPI.

**What is useful to us.** Plan 9.3 needs driver models (path following with
preview, speed control) and 10.5 needs the full car to track an optimal
trajectory. The controllers here are clear references, though all run on a
kinematic vehicle (position, heading, speed; inputs acceleration and yaw rate),
so their gains and assumptions would not carry over to a multibody car on
tyres as they are.

**Caveat.** 41 of its 55 test files run a simulation and check nothing; its
161 asserts are all in tests of small components (the state, the speed
profile, matrices), none in the controllers' tests. Our rule (tests assert
values derived independently) would need checks such as a pure-pursuit car
settling on a circle at the radius geometry predicts.

**Verdict.** A reference for 9.3 and 10.5 (proposal 9), nothing to take now.

### 3. headroom (GitHub, Apache 2.0)

**What it is.** A layer that compresses what an AI coding agent reads (tool
outputs, logs, files) before it reaches the model, as a library, a proxy or a
wrapper around agents including Claude Code. Originals are kept locally and
retrievable on demand (its "CCR"). `headroom learn` mines failed sessions and
writes corrections into instruction files such as `CLAUDE.md`.

**For our work.** It is a tool for the author's environment, not for the
program. Installing it would route every session through a local proxy, add an
MCP server and edit `~/.claude.json`; its compression is lossy by design, and
our work depends on reading exact build logs and numbers (a warning line, a
seventh digit). The sessions here are not short of context.

**One idea for the program.** When the program is driven by an agent (plan
13.8), a simulation should return a compact summary (metrics, warnings with
their codes, events) and a handle to retrieve the full channels on demand,
instead of dumping a run into the agent's context. That is headroom's "CCR"
pattern applied to our outputs (proposal 13).

**Verdict.** Do not install. Keep the summary-plus-handle idea for 13.8.

### 4. JAOT, Just Another Optimization Tool (GitHub, Apache 2.0)

**What it is.** A self-hosted optimisation platform: describe a decision
problem in words or JSON, get a model solved by one of five solvers (SCIP,
HiGHS, CBC, GLPK, its own), with exact post-solve analysis (binding
constraints, slacks), what-if answers measured by re-solving rather than read
off a relaxation, the minimal conflicting set of constraints when a model is
infeasible, and an MCP server so that agents can drive it.

**What is useful to us**, chiefly in how it is tested (`docs/TESTING.md`):
- **Every input must reach the model.** For each of its 102 templates, a test
  moves every number of the example input up and down, rebuilds the model and
  checks that it changed. A number that never reaches the model is a bug, unless
  it is listed, with its reason, as deliberately unused. A second gate checks
  that every key a template sets is one its generator reads (a misspelt key had
  silently reverted a mode in 23 of 43 cases). This is the failure the plan's
  model files (8.2, 8.3) are exposed to: a vehicle parameter that is read
  nowhere, or a misspelt key that falls back to a default, gives a model that
  looks right and is not (proposal 5).
- **A ratchet for known debt**: failing cases listed as expected failures that
  fail the build the moment they start passing, so the list can only shrink.
- **Mutation testing** of the critical modules: a mutated source line must make
  some test fail. Coverage shows a line ran; mutation shows a test would notice
  it being wrong.
- **What-if by re-solving, honestly reported**: a result limited by time is
  reported as a bound, never as exact. Our statics and assembly already report
  non-convergence this way.
- **The minimal conflicting set** for an infeasible model. Our analogue is the
  projection that cannot converge (MBD-K051) and the held coordinates that
  over-fix a loop (MBD-K063): naming the constraint equations that conflict,
  rather than only the residual, would help a user with a mechanism that
  will not close (the Bricard mechanism of 5.2 is the test case). A small
  addition worth considering with 5.2.

**Verdict.** Take the parameter-reach gate (proposal 5) and consider the
conflicting-set report with 5.2. The platform itself is unrelated.

### 5. marcos-dv, ampl-agents (GitHub, MIT)

**What it is.** Markdown "skills", "agents" and always-on "rules" that make
coding agents (Claude Code, Cursor, Codex) competent with the AMPL modelling
language, plus a library of verified documentation links. Its first rule:
never state a solve outcome without a solve actually run in the session; a
model not yet solved is labelled a draft.

**What is useful to us.**
- **Our own working rules live outside the repository.** One compiler at a
  time, the build script, the record kept with every task, derived values in
  tests, commit messages: these are in the author's local agent memory, not in
  the repository. A `CLAUDE.md` at the repository root (with an `AGENTS.md`
  for other tools) would version them, make them visible to anyone working on
  the code, and give the book its own source for them (proposal 4).
- **Skills for the program's users** (plan 13.8): the same packaging could
  ship with the program, teaching a user's agent to build models with our API
  from `docs/help/`, with a rule like AMPL's: never report a simulation result
  without having run it, and quote the message codes.

**Verdict.** Proposal 4 now; skills for users with 13.8.

### 6. numerical-mooc, Practical Numerical Methods (web and GitHub, CC-BY 4.0)

**What it is.** Lorena Barba's open course (2014, 11 000 learners), rebuilt in
2026 for "the agentic era": Jupyter notebooks from a physical model to a
method, its implementation, convergence and stability, and interpretation
(phugoid, convection and diffusion, traffic and shocks, elliptic problems). The
repository carries an `AGENTS.md` and a `DECISIONS.md` much like ours.

**What is useful to us.**
- **Manufactured solutions.** The course lists them among its verification
  methods. For a multibody code the idea becomes a manufactured trajectory:
  choose any smooth motion q(t) of any tree, compute the joint forces that
  produce it by inverse dynamics (RNEA, which we have), apply them, simulate,
  and require the motion to come back at the integrator's order. It needs no
  closed-form solution, works on random trees of any size, every joint type and
  closed loops (with the constraint forces added), and checks the forward
  dynamics, the constrained solve, the quaternion integration and every
  integrator against an exact reference. We test RNEA, CRBA and ABA against
  each other at single states (`test_kernel_invariants.cpp`), not along a
  trajectory. This belongs in 5.1 (proposal 1).
- **Its appendix on working with agents** gives exact vocabulary for what this
  project does: a task specification with acceptance evidence decided before
  the work, independent expectations derived without the agent ("an agent can
  encode the same misunderstanding in both code and tests"), inspection of
  evidence rather than assurances, and a record of material agent
  contributions. Citable in the book (CC-BY).

**Verdict.** Proposal 1, and a reference for the book.

### 7. anyplot.ai (web)

**What it is.** An open catalogue of 325 library-independent plot
specifications, each implemented by AI in 15 plotting libraries, with light and
dark previews in a colour-blind-safe palette, and an MCP server.

**For us.** Our plotter (`bench/svg_plot.hpp`) already uses the Okabe and Ito
colour-blind-safe palette. The specification-first idea fits the standard
vehicle-dynamics plots of 9.5 (g-g diagram, understeer curve, step-steer
response, frequency response, track map coloured by a channel), but that is
how 9.5 would be written anyway.

**Verdict.** Nothing to take.

### 8. MOVEdot (web)

**What it is.** A commercial platform of AI agents that operate existing
vehicle-engineering tools (Adams, VI-CarRealTime, MATLAB, Canopy) and data,
used by racing teams (Arrow McLaren, WRT BMW). Its blog describes a
sim-to-test correlation agent: find and crop the manoeuvres in measured data,
rerun them in simulation, overlay the channels, flag disagreements and propose
parameter changes. With Canopy, a full correlation campaign was run in hours:
straight line first (powertrain, drag, downforce, ride height, springs,
brakes), then skidpads at three radii (grip, aero balance, roll, steering),
then a full lap, adjusting one parameter at a time, each confirmed before the
next phase, over six channels (speed, brake, throttle, steering, lateral and
longitudinal acceleration). "Every parameter that moved is tied to the
manoeuvre and the data that justified moving it."

**What is useful to us.**
- **Correlation with measured data is a workflow the plan lacks.** 9.6 compares
  the car with other models (bicycle, quarter car, Chrono HMMWV); nothing
  compares it with a real car's data. The steps above, plus TUM's channel
  library, define such a task (proposal 7).
- **What an agent needs from a simulation tool**: submit jobs, retrieve
  results, update configurations directly, many runs in parallel, every result
  tracked. Our Phase 6 (Python), 8 (model files) and 13.1 (batch runner and
  result store) provide exactly this; MOVEdot is evidence that these phases
  are what makes the program usable by agents, not extras.

**Caveat.** Marketing material: no method details, no error figures beyond the
quoted claims.

**Verdict.** Proposal 7 (correlation); a confirmation of Phases 6, 8 and 13.1.

### 9. MDPI, *Vehicles* 8(9), 214 (2026): Dobre, Burduș, Mihăilă, Durigon, "Excitation Parameters Effect on Damper Performance Through Experimental Analysis"

**What it is** (abstract only, see above). A test rig measures a shock
absorber's force against stroke under sinusoidal excitation at 0.3 to 1 Hz and
several amplitudes, standard or adaptive dampers; the force-stroke diagrams
show the energy dissipated and hysteresis; an AMESim model is compared with the
measurements.

**What is useful to us.** Our dampers are static force-against-velocity curves
(different in bump and rebound). Real dampers show hysteresis from oil and gas
compressibility and from the top mount's compliance, which a stiffness in
series with the damper (a Maxwell element) reproduces, at the cost of one state.
Our state elements (4.1) make this a small addition (proposal 8), and a virtual
damper dyno (sinusoidal stroke, force against stroke and velocity) is the test
that shows it.

**Verdict.** Proposal 8; the full paper would supply measured loops to compare
with.

### 10. `document1.pdf`: Seddighi, genetic-algorithm optimisation of an electric motor's mount bushings

**What it is.** A 68.5 kg motor on three bushings: the 6x6 mass and stiffness
matrices in closed form, natural frequencies and modal energy decoupling from
eig(K, M), design targets (frequencies outside forbidden bands, at least 2 Hz
between modes, 90 % of each mode's energy in one direction, bushing deflections
within limits under 23 load cases, with a nonlinear bushing curve), and a
genetic algorithm (population 5000, 30 generations) over six variables. The
result meets every target.

**What is useful to us.**
- **An analytic test case** for 5.1: a rigid body on three bushings has its
  stiffness matrix in closed form, so `linearize()` must return the six
  frequencies of eig(K, M) to rounding. It checks the bushing's derivatives,
  the free joint and the linearisation together (proposal 2).
- **Modal energy decoupling**, the fraction of a mode's kinetic energy in each
  coordinate, is the standard metric for mounts and for a car's ride modes
  (how pure its bounce, pitch and roll are). Our modes have frequency, damping
  and shape but not this; it is a few lines on top of `linearize()`
  (proposal 2). (The slides use only the diagonal of M, which ignores the
  products of inertia; the exact fraction for coordinate k is
  phi_k (M phi)_k / phi^T M phi.)
- **A benchmark for 13.3** (optimisation): small, physical, with constraints
  and a known feasible optimum.

**Verdict.** Proposal 2; the problem as an example for 13.3.

### 11. `document-2.pdf`: Chindamo, Magri, Gadola, "Towards Rapid Suspension Development: An Integrated Framework for Elastokinematic Design through Automatic Optimisation"

**What it is.** A two-page abstract of a quasi-static elastokinematic solver
(equilibrium equations, five-link, double-wishbone and McPherson layouts,
nonlinear joint compliance) that reads Adams/Car suspension databases, makes
each bushing curve a function of a few parameters so that its shape becomes an
optimisation variable (resampled back into Adams/Car tables afterwards), and
runs sensitivity analysis and automatic optimisation over hardpoints and
compliances together. It builds on two 2024 papers, one a double-wishbone case
study in the ASME *Journal of Computational and Nonlinear Dynamics*
(doi:10.1115/1.4066092).

**What is useful to us.**
- **A validation reference for 9.1** (compliance under lateral, longitudinal
  and aligning loads): the 2024 double-wishbone case study, if it publishes
  its results. Worth obtaining.
- **Sensitivities of kinematic and compliance responses to hardpoints.** Our
  kinematics (`Kinematics`) solves phi(q, p) = 0; the derivative of every
  response with respect to a hardpoint p follows exactly from the constraint
  Jacobians (dq/dp = -phi_q^+ phi_p), without finite differences. That gives
  sensitivity tables (9.1, 13.2) and gradients for hardpoint optimisation
  (13.3) (proposal 10).
- **A parametric bushing curve**: a curve family with a few shape parameters
  (linear range, progressive end, stop), so that optimisation can move the
  shape; our curves are tables or straight lines (proposal 10).
- **Reading Adams/Car property files** (springs, dampers, bushings in the
  TeimOrbit format) would let users bring existing data; an interoperability
  item for 8.2 or 14 (proposal 10).

**Verdict.** Proposal 10, and the 2024 paper for 9.1.

### 12. `technical-brief-discovering-missing-physics-sciml-dyad.pdf`: JuliaHub, SciML and Dyad Agent

**What it is.** Two case studies of universal differential equations: a
physical model with a small neural network inside the differential equations,
trained through the solver. A 4-DOF quarter truck (tyre, cabin, seat, driver)
with linear elements learns, with 68 parameters, the forces of a cubic tyre
spring, suspension Coulomb friction and a viscoelastic seat; trained on one
30 mm bump, it is validated on ISO 8608 random roads of classes A and D. An EV
driveline isolator with a Maxwell element and Bouc-Wen hysteresis learns two
latent states (60 parameters) and is validated on a tip-in with 80 Hz torque
ripple.

**What is useful to us.**
- **ISO 8608 random roads**: road classes A to H from the displacement power
  spectral density, falling with slope -2 against spatial frequency. A standard
  input for ride studies and the four-post rig (9.4), and absent from the
  road task (7.2) (proposal 3).
- **Hysteresis as state elements**: Bouc-Wen for rubber bushings and isolators,
  Maxwell for viscoelastic seats and damper top mounts. Our state-element
  interface makes each a small element (proposal 8).
- **Hybrid models** (a learned residual force inside the equations): possible
  later through the Python layer, but training through the solver needs
  derivatives of the simulation with respect to parameters, which the program
  does not compute (proposal 12).

**Caveats.** The "ground truth" is a more detailed simulation, not
measurement. The claim that the optimiser "had already located the absolute
mathematical minimum" cannot be known from three LBFGS steps. On the RMS
summary (Table 1) the hybrid model is closer than the linear one in all four
rows, but by 20 to 30 N against 191 to 1310 N of tyre force; RMS values do not
show phase errors.

**Verdict.** Proposals 3 and 8; hybrid models deferred to Phase 13 (proposal
12).

### 13. `siemens-ai-surrogate-modeling-for-industrial-cae-white-paper.pdf`: Siemens, "AI surrogate modeling for industrial CAE"

**What it is.** A vendor overview of surrogates that predict whole simulation
fields on unstructured meshes: machine-learning basics, the industrial
constraints (10 to 100 training simulations, meshes of millions of cells), and
the methods (graph networks such as MeshGraphNet; operator learning; geometry
encodings; transformers such as GNOT, AB-UPT, Transolver and GeoTransolver), with
the verdict that transformers on latent "physics tokens" lead today, and that
physics-informed networks (PINNs) and diffusion models are not yet usable at
industrial scale.

**For us.** The plan's surrogates (13.5) are of low-dimensional maps (tyre
forces, suspension maps, an aero map over ride heights and attitude), for
which Gaussian processes and small networks are the right tools; field
surrogates would matter only with our own CFD (Phases 11 and 12). Two points
carry over: the discipline of training, validation and test sets, and
**knowing when to trust a surrogate**: Siemens scores a query's similarity to
the training geometries. For our maps the analogue is a check that a query
lies inside the sampled range, reported with a message code rather than
extrapolated silently (proposal 11). The PINN verdict supports leaving them out
of the plan.

**Verdict.** Proposal 11.

### 14. `document2.pdf`: Díaz Rivera, "Implementation-Aware Power Estimation for Architectural Simulation" (master's thesis, UPC and Openchip, 2026)

**What it is.** Linear power models of a RISC-V core (CVA6) calibrated
against RTL simulation, restricted to the six hardware counters the core can
read at once, chosen by nested cross-validation and an exhaustive search over
all 1716 six-counter subsets, then deployed in the gem5 simulator. On
integration, one counter turned out to have no gem5 equivalent, which made two
of four models uncomputable; another counter was silently absent from gem5's
output for short runs; two counters identical in every run made the "best"
subset an arbitrary tie-break between statistics of different reliability.

**What is useful to us.** Not the subject but the method, which is close to
ours and states its own weaknesses carefully. For any fitted model we build
(13.5 surrogates, tyre or damper parameters identified from data):
- **a held-out test set used once**: the thesis reused its seven validation
  benchmarks for later choices and says plainly that they then stopped being
  independent;
- **selection bias**: choosing the best of many candidates by a cross-validated
  score and reporting that same score overstates it;
- **a constant baseline**: is the model better than a constant, by a margin
  fixed in advance?
- **errors in physical units beside the range of the target**: a 1.8 % error
  that is two thirds of the target's whole spread; R squared unstable when the
  target barely varies;
- **every input a model needs must exist where it will run**, checked before
  the model is chosen, not discovered at deployment; and outputs that go
  missing without an error. The same lesson as JAOT's, from the other end.

**For the book.** The thesis opens with a section on its use of AI tools: what
was drafted with an assistant, what was verified, and what remained the
author's. The book will need one (see the end).

**Verdict.** Its lessons go into proposal 11 and the surrogate tasks of
Phase 13.

### 15. theOehrly, Fast-F1 (GitHub, MIT)

**What it is.** A Python package that downloads Formula 1 results, timing
and telemetry from the sport's live-timing service and the Ergast-compatible
jolpica API, as pandas tables, with caching and plotting. Per car and lap:
speed, engine speed, gear, throttle, brake (on or off), DRS at about 4 to 5 Hz,
and position (x, y, z) on a separate time base; lap and sector times to the
millisecond; and, for each circuit, corner positions and numbers (entered by
hand, approximate).

**What is useful to us.**
- **Real laps to compare with.** The plan's risk "no measured vehicle data"
  stands: no car's full data are available. These are measured laps of real
  race cars, public, on real circuits: speed traces, gear and throttle use,
  braking points, minimum corner speeds and lap times. They cannot validate
  a model of a car whose parameters are unknown, but they give the
  formula-style reference car (8.6) and the lap simulation (10.7) a check of
  realism (speeds, where the car brakes, lap time within a margin), and the
  correlation tooling (9.7) its first real data to work on (alignment by
  distance, cropping, overlays).
- **Its author's notes on accurate calculation**
  (`docs/data_reference/howto_accurate_calculations.rst`) state the
  discipline correlation needs with sparse, jittery data: validate a derived
  quantity against an independent accurate one (computed lap starts against
  the measured lap times; the car at the finish line at every lap start),
  compute on the original time bases and resample only results, and do not
  build on integrated quantities, whose errors accumulate.
- **Elevation**: the position data carry a height, from which a circuit's
  elevation profile can be estimated for the three-dimensional track (10.1),
  to be checked against a published profile before use.

**Caveats.** The data are unofficial (the package states it is not
associated with Formula 1) and coarse: 4 to 5 Hz, braking only on or off,
no accelerations, forces or steering. The package fetches them from a
service the project does not control, so a check that depends on them must
cache what it used (with the session, date and package version) and cannot
run in CI. The raw data should not be redistributed in the repository.

**Verdict.** A data source for 9.7, 10.1 and 10.7 through the Python layer
(Phase 6), as an optional dependency (proposal 14).

### 16. morluto, REA, "Reverse Engineer Anything" (GitHub, MIT)

**What it is.** An MCP server and command-line tool that lets a coding agent
inspect software without its source: native binaries through Hopper, Ghidra
or IDA, JavaScript and Electron applications, .NET assemblies, Android
packages, websites and network captures. It returns findings with their
evidence and limitations, "so that the agent can explain a feature and build
a version of it". Its disclaimer excludes circumventing licence checks and
copying proprietary software, and leaves the legality of each use to the user.

**For the program.** Nothing to use it on. The commercial
vehicle-dynamics tools whose features one might want to understand this way
(Adams, CarSim, VI-grade) forbid reverse engineering in their licences, and
the plan builds every model from the published literature, which is also what
the book will cite. The files we need to read from such tools for
interoperability (Adams/Car property files, `.tir` tyre files) are documented
text formats and need no reverse engineering.

**What is useful to us: how it talks to agents** (`AGENTS.md`,
`docs/mcp-contracts.md`, `docs/error-contract.schema.json`), for the agent
interface (13.8):
- every error is a structured record: a stable code, a category, a message,
  whether retrying can help, and the action that fixes it. Our message
  catalogue already has the code, the meaning, the causes and what to do;
  exported as data, it would give the Python layer and agents the same
  records;
- results separate what was observed, what was derived, what was inferred and
  what is unknown, and state their coverage ("complete" or "unknown");
  something unknown stays unknown instead of being reported as absent (the
  thesis's silent absences, from the other side);
- the smallest sufficient answer first, the full evidence retained and
  retrievable by an identifier: headroom's pattern again;
- configuration changes made by its installer are shown, approved, additive
  and backed up.

**Verdict.** Not to be installed or used. Its contracts inform proposal 13.

## What several sources point to

1. **Inputs that silently do not reach the model, outputs that silently go
   missing** (JAOT, the thesis). For us: model files that reject unknown keys,
   a test that every parameter changes the model, and checks that requested
   outputs exist.
2. **Derivatives of results with respect to parameters** are what correlation
   (MOVEdot), hardpoint optimisation (Chindamo), mount optimisation
   (Seddighi), hybrid models (SciML) and sensitivity tables (13.2) all need.
   For kinematics they follow exactly from the constraint Jacobians; for a
   simulation they are forward sensitivities, a larger piece of work.
3. **Correlation with measured data as a workflow** (MOVEdot, TUM, SciML's
   out-of-sample validation). The plan validates the solver against analytic
   cases and other codes, but never the car against a real car.
4. **Hysteresis and frequency dependence** (SciML, the damper article, rubber
   bushings), cheap to add now that state elements exist.
5. **Discipline for fitted models** (the thesis, Siemens): held-out data used
   once, a constant baseline, errors with the range, and a flag outside the
   training domain.
6. **Programs that agents can drive** (MOVEdot, JAOT's MCP server,
   ampl-agents, headroom, REA, the numerical-methods course): headless runs,
   batches with a result store, compact results with handles, structured
   errors, and instructions for agents kept in the repository.
7. **Measured data, used carefully** (Fast-F1, TUM, MOVEdot): public race
   telemetry exists, coarse but real; using it means validating every derived
   quantity against an independent accurate one.

## Proposals

Ranked by value against cost. Each names where it would go in the plan.

### Now, with Phase 5 or before it

1. **Manufactured trajectories** (5.1). Prescribe a smooth motion of a random
   tree or a closed mechanism, compute its driving forces by inverse dynamics
   (and the constraint forces for loops), simulate, and require the motion back
   at each integrator's order. An exact reference for any model, every joint
   type and every integrator. *Small; strongest single check available.*
2. **A rigid body on three bushings** (5.1) against eig(K, M) in closed form,
   and **modal energy distribution** added to the modes of `linearize()` (the
   fraction of each mode's kinetic energy in each coordinate). *Small.*
3. **ISO 8608 random roads** (7.2, the road task): road classes A to H,
   generated from the standard's spectrum with a seed, as a road for ride
   studies and the four-post rig (9.4). *Small; Phase 7.*
4. **The working rules in the repository**: a `CLAUDE.md` (and `AGENTS.md`)
   at the root with the rules now held in local agent memory: one compiler,
   the build script, the record kept with every task, derived values in tests,
   the commit format. *Small; independent of the plan.*

### Later phases: amendments to the plan

5. **Strict model files and a parameter-reach test** (8.2, 8.3): unknown keys
   rejected with a message naming them; a test that moves every parameter of
   each reference car up and down and requires the built model to change,
   exceptions listed with their reasons. Mutation testing of the kernel as an
   optional Linux CI job.
6. **Derived vehicle channels and lap characterisation** (9.5, Phase 10):
   channel names carrying their units, the TUM set of derived channels computed
   from simulation output, lap slicing, corners and their phases.
7. **Correlation with measured data** (a new task 9.7): import measured
   channels, resample them to the simulation's times, crop the manoeuvres,
   overlay, compute per-channel errors normalised by each channel's spread,
   and identify parameters one group at a time in a set sequence (straight
   line, then steady cornering, then a lap), with a log tying each change to
   its manoeuvre and data.
8. **Hysteretic elements as state elements** (7.6, 7.8): a damper with a
   series stiffness (Maxwell: top mount, oil compressibility), a Bouc-Wen
   bushing and driveline isolator; checked on a virtual damper dyno (sinusoidal
   stroke, force against stroke and velocity).
9. **Driver models** (9.3): pure pursuit or Stanley steering with preview and
   a PI speed loop first, then LQR on the lateral-error dynamics; MPC with
   10.5. Tested by derived results (the settled radius on a circle, the
   decay of an offset on a straight), not by running.
10. **Suspension design aids** (9.1, 13.2, 13.3): exact sensitivities of
    kinematic and compliance responses to hardpoints from the constraint
    Jacobians; a parametric bushing-curve family; a reader for Adams/Car
    property files (8.2 or 14). Obtain the 2024 double-wishbone paper for
    9.1's validation.
11. **Discipline for fitted models** (13.5): a test set held out and used
    once, a constant baseline, errors with the target's range, and a warning
    with a message code when a surrogate is queried outside its sampled range.
12. **Derivatives of a simulation with respect to its parameters** (Phase 13):
    forward sensitivities, the enabler for gradient-based correlation and
    hybrid (learned-residual) models. Large; to decide when Phase 13 is
    planned in detail.
13. **The agent interface** (13.8): runs returned as a compact summary
    (metrics, warnings with their codes, events) with a handle to the full
    channels; errors as structured records (code, category, whether a retry
    can help, the remedy), from the message catalogue exported as data;
    results that separate what was computed from what is unknown; an MCP
    server over the Python API; skills and rules for users' agents built from
    `docs/help/`.
14. **Real race telemetry** (9.7, 10.1, 10.7): Fast-F1 as an optional Python
    dependency, its sessions cached with their provenance, for the
    correlation tooling's first real data, a circuit's elevation profile, and
    a realism check of the lap simulation and the formula-style car.

### Noted, not proposed

- Installing headroom (reasons in section 3).
- Using REA, on any program (reasons in section 16).
- Field surrogates on meshes and PINNs (Siemens): only relevant with our own
  CFD, and PINNs not proven at scale.
- anyplot: our palette already serves its purpose.
- Data ingestion from ROS, MCAP and CAN (TUM), and path planning (Yano).

## For the book

- **A statement on the use of AI**, as the thesis has: which parts were
  written with an assistant, how they were specified and checked, and what
  remained the author's. The journal, the decisions and the co-author lines on
  every commit already hold the evidence; the outline needs a place for it
  (Part X, engineering practice).
- **The numerical-methods course's appendix on working with agents** as a
  citable reference for the specification, the independent expectations and
  the evidence that this project's practice rests on.
- **JAOT's ratchet** and **the thesis's self-criticism** as examples of keeping
  known weaknesses visible rather than quiet.
