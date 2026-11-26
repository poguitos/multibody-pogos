# Tyres

Plan tasks 7.4 and 7.5 (decisions D63, D64, D65): the Magic Formula tyre,
from a tyre property file (.tir), in its two versions MF 6.1 and MF 5.2; the
tyre on a wheel, its slips lagging behind the motion over its relaxation
lengths, valid down to rest; and three simple tyres beside it. This page says what it computes, from which equations,
how it was checked, and what it leaves out.

| What | Where |
|---|---|
| The tyre: steady-state forces and moments | `mbd/forces/magic_formula.hpp` |
| The .tir reader | `mbd/io/tir_file.hpp` |
| The tyre on a wheel: contact, lag, mirroring | `mbd/forces/tire.hpp` (`FullTireForce`) |
| Any tyre's law: the interface | `mbd/forces/tyre_model.hpp` (`TyreModel`) |
| Simple tyres: linear, Fiala, brush | `mbd/forces/simple_tyres.hpp`; tests `tests/forces/test_simple_tyres.cpp` |
| Tests: identities at the nominal state | `tests/forces/test_magic_formula.cpp` |
| Tests: the tyre on a wheel | `tests/forces/test_full_tire.cpp`; the car on a slope, `tests/test_drivetrain.cpp` |
| Tests: against MFeval, an independent implementation | `tests/forces/test_magic_formula_mfeval.cpp` |
| Test tyres and MFeval's curves | `tests/forces/data/*.tir`, `tests/forces/data/mfeval/*.csv` |

Every vehicle's tyre is this one, Pacejka's Appendix 3 tyre (a 205/60R15)
unless another is given (`VehicleParams::tyre`, `AxleConfig::tyre`). The
simpler formula the library had before the plan (`mbd/forces/pacejka.hpp`)
is left to the bicycle model, which reduces a vehicle's tyre to it.

## Using it

```cpp
#include "mbd/forces/magic_formula.hpp"
#include "mbd/io/tir_file.hpp"

const mbd::io::TirFile file = mbd::io::read_tir("my_tyre.tir");
// file.ignored lists the keys the tyre does not use (SWIFT, contact patch, ...)
const mbd::MagicFormulaTyre tyre(file.parameters);

mbd::MagicFormulaInput in;
in.Fz = 4000.0;      // N
in.kappa = 0.05;     // longitudinal slip, positive in traction
in.alpha = 0.03;     // slip angle, rad
in.gamma = 0.0;      // inclination, rad
in.Vcx = 20.0;       // forward speed of the contact centre, m/s
const mbd::MagicFormulaOutput out = tyre.evaluate(in);
// out.Fx, out.Fy, out.Mx, out.My, out.Mz; out.Kxk, out.Kya, out.trail,
// out.sigma_kappa, out.sigma_alpha, ...
```

## Axes and signs

Those of a .tir file: ISO, the tyre's W axes. x is along the wheel's heading
in the road plane, y to its left, z up along the road's normal. The inputs:

| Input | Definition |
|---|---|
| `Fz` | Normal load, N; at zero or below every output is zero |
| `kappa` | Longitudinal slip, -Vsx / \|Vcx\|: positive in traction, -1 for a locked wheel rolling forwards |
| `alpha` | Slip angle, tan(alpha) = Vcy / \|Vcx\|: positive when the contact centre moves to the left. A car tyre's cornering stiffness `Kya` is then negative, and so is its side force at a positive slip angle |
| `gamma` | Inclination angle: positive when the wheel's top leans to the right (to -y) |
| `Vcx` | Forward speed of the contact centre: its sign, the rolling resistance's speed terms, and the slip speed of LMUV |
| `pressure` | Inflation pressure, Pa (MF 6.1); below zero, the file's INFLPRES |

The outputs are the forces and moments of the road on the tyre at the
contact point, in the same axes. The rolling resistance moment `My` is
negative when rolling forwards: it opposes the wheel's spin, which is about
+y.

## The equations

The tyre evaluates TNO's MF-Tyre 6.1 as Besselink, Schmeitz and Pacejka list
it in the appendix of "An improved Magic Formula/Swift tyre model that can
handle inflation pressure changes" (Vehicle System Dynamics 48, 2010),
equations (33) to (90). Pacejka's Tire and Vehicle Dynamics, 3rd edition
(2012), section 4.3.2, equations (4.E1) to (4.E78), gives the same in most
places, and the source comments carry both numbers. Turn slip is not
modelled (every zeta factor is 1).

Where the book and the paper differ, the paper is followed. .tir files are
fitted with TNO's tools, so their parameters mean what TNO's equations make
of them. MFeval's author compared his implementation with TNO's solver
(MF-Tyre/MF-Swift 6.1.2.1) and found the paper's forms match it where the
book's do not. The differences, and what each would change on the test
tyres: each book form was put back alone and compared with MFeval, and the
last column gives the largest change of a channel as a fraction of that
channel's largest magnitude over the sweep (the Appendix 3 tyre, or the
scaled one where marked):

| Point | Book | Here (paper, TNO) | Change if the book's form were used |
|---|---|---|---|
| Slip angle | alpha* = tan(alpha) sgn(Vcx) (4.E3); gamma* = sin(gamma) (4.E4) | alpha and gamma as they are | Mz 1.6 %, Fx 0.7 %, Fy 0.4 % (alpha* alone) |
| cos(alpha) in the residual torque | in Dr (4.E47) and again in Mzr (4.E36, 4.E75) | once, in Mzr (88); not in Dr (90) | Mzr 0.5 %, Mz 0.09 % |
| Camber in the trail | (1 + qBz5 \|gamma*\| + qBz6 gamma*^2) (4.E40); (1 + qDz3 \|gamma*\| + qDz4 gamma*^2) (4.E43) | (1 + qBz4 gamma + qBz5 \|gamma\|) (84); (1 + qDz3 gamma + qDz4 gamma^2) (86) | t 5.9 %, Mz 4.5 % |
| Equivalent slips, combined | sqrt(alpha_t^2 + (Kxk / Kya)^2 kappa^2) (4.E77, 4.E78) | atan(sqrt(tan^2 alpha_t + (Kxk / Kya)^2 kappa^2)) (80, 81) | t 2.4 %, Mz 1.3 % |
| Rolling resistance | Fz R0 (...) (Fz / Fz0)^qsy7 ..., positive (4.E70) | -R0 Fz0 (...) (Fz / Fz0)^qsy7 ... (50) | My 75 % (at 7000 N), and its sign |
| Degressive friction factor | lambda' = A_mu lambda* / (1 + (A_mu - 1) lambda*), A_mu = 10 (4.E8) | lambda' = lambda* (A_mu = 1; (42), (65) have lambda_mu) | Mz 0.4 %, Fy 0.2 % (only with friction scaling other than 1) |
| Peak trail | Fz R0 / Fz0' (4.E42) | Fz R0 / Fz0 (86) | t 9.1 %, Mz 8.9 % (only with LFZO other than 1) |
| Overturning couple | (4.E69), without qsx12 to qsx14 | (49), with them | none on the test tyres (their qsx12 to qsx14 are 0) |
| Side force in Mz, combined | Gyk at zero inclination (4.E74; the paper's note to (75) too) | Gyk at the inclination, as MFeval found TNO's solver has it | none on the test tyres (their rBy4 is 0); not checked independently |
| qBz6 | in (4.E40) | not a parameter: no .tir file has it | |

Elsewhere the book is followed where the paper is silent: the small
quantities epsilon (1e-6, with their neighbour's sign) that keep
denominators from zero, the friction scaling with slip speed `LMUV` (4.E7,
default 0, which TNO's files do not use), and the sign of the forward speed
in the trail and in the residual torque (4.E42, 4.E47).

### MF 5.2

A file's FITTYP chooses the version: 6 or 21 is MF 5.2, 61 or 62 is MF 6.1
(MF 6.2's steady-state forces are 6.1's). A file without FITTYP may name the
version as Adams does, `PROPERTY_FILE_FORMAT = 'PAC2002'` or `'MF_05'` (MF
5.2), `'MF_61'` or `'MF_62'` (MF 6.1).

MF 5.2 takes the same equations, with 5.2's forms where the paper says 6.1
changed them:

- camber shifts the side force curve in proportion, `SHy = (pHy1 + pHy2 dfz)
  lHy + pHy3 gamma_y`, rather than through 6.1's camber stiffness (pKy6,
  pKy7);
- the aligning torque's arm takes the side force at the inclination, less
  its slip-induced part: `Mz = -t (Fy - SVyk) + Mzr + s Fx`; 6.1 takes the
  side force at zero inclination;
- the rolling resistance is `My = -R0 Fz lMy (qsy1 + qsy2 Fx / Fz0 + qsy3
  |Vx / V0| + qsy4 (Vx / V0)^4)`, without 6.1's camber, load exponent and
  pressure terms;
- the relaxation lengths come from their own coefficients: `sigma_kappa =
  Fz (pTx1 + pTx2 dfz) exp(-pTx3 dfz) (R0 / Fz0) lsgkp`, `sigma_alpha = pTy1
  sin(2 atan(Fz / (pTy2 Fz0'))) (1 - pKy3 |gamma_y|) R0 lFz0 lsgal`; 6.1
  divides the slip stiffnesses by the carcass's stiffnesses;
- there is no inflation pressure;
- the inclination is scaled for each use, by LGAX (the longitudinal force),
  LGAY (the side force) and LGAZ (the aligning torque); 6.1 scales the
  camber stiffnesses instead (LKYC, LKZC).

No written MF 5.2 equation manual was within reach. The forms of the rolling
resistance and of the relaxation lengths are those MFeval gives as TNO's 5.2
manual's: for them MFeval is the source as well as the check. The camber
scaling factors LGAX, LGAY and LGAZ come from older descriptions of MF 5.2
(PAC2002); MFeval does not take them, so they are not compared.

## On a wheel: FullTireForce

`FullTireForce` (`mbd/forces/tire.hpp`, D64) is the tyre on a wheel body of
task 7.3: the contact from the wheel plane and the road, and the Magic
Formula at the contact point, in its W axes. A parameter set gives it
everything: the free radius (UNLOADED_RADIUS), the vertical stiffness and
damping of the contact (VERTICAL_STIFFNESS, VERTICAL_DAMPING), VXLOW, and the
forces and moments.

```cpp
const mbd::io::TirFile file = mbd::io::read_tir("my_tyre.tir");
mbd::VehicleParams vp;                // the simple car
vp.tyre = file.parameters;            // its four tyres
// or, on a template: tmpl.front_axle.tyre = file.parameters;
```

**The slips lag** behind the motion (Pacejka, 3rd ed., section 8.6, the
semi-non-linear model). Each tyre has two states, its tread's deflections
u and v:

    du/dt = -Vsx - |Vx| u / sigma_kappa,    kappa' = u / sigma_kappa
    dv/dt =  Vsy - |Vx| v / sigma_alpha,    alpha' = atan(v / sigma_alpha)

with Vsx = Vx - Omega R_eff and Vsy = Vy the contact's slip speeds and sigma
the Magic Formula's relaxation lengths at the present load (at least 0.02 m).
The Magic Formula takes kappa' and alpha'. In a steady state they are the
steady-state slips, -Vsx / |Vx| and atan(Vsy / |Vx|), so the forces are those
of the steady-state tyre; after a step in slip, the deflection reaches 1 - 1/e
of its new value in one relaxation length of travel; at rest the deflections
follow the contact's motion and the tyre holds like a spring of its
carcass's stiffness (MF 6.1: LONGITUDINAL_STIFFNESS and LATERAL_STIFFNESS of
[STRUCTURAL]). A simulator made from a system holding the tyre integrates
its two states (the tyre is a state element; it is attached when the
simulator is made).

Below VXLOW three things keep the tyre sane at rest:

- a damping k_V_low = k_V_low0 (1 + cos(pi |Vx| / VXLOW)) / 2, k_V_low0 = 770
  N s/m (the book's Table 8.4), added to the slips as -k_V_low Vs / K (8.127,
  8.128): it damps the tyre's spring at rest, lightly (3 % of critical on a
  quarter of a car);
- the deflection limit of (8.112): past the slip of full sliding, 3 D / K, a
  deflection is not let grow;
- TNO's low-speed model, as MFeval has it: the Magic Formula's shifts (SHx,
  SVx, SHy, SVy, SVyk), its trail and its residual torque fade out by (1 -
  cos(pi |Vcx| / VXLOW)) / 2, so that a tyre at rest, undeflected, pushes
  nothing. The rolling resistance moment acts in full once the wheel rolls
  at `V_roll` = 0.01 m/s, in proportion below it (dry friction, smoothed):
  it opposes the spin, and a wheel at rest has none.

Off the road the deflections relax over the nominal relaxation lengths, at
|Vx| + VXLOW. With `lag = false` (set before the simulator is made) the
tyre has no states and takes the steady-state slips over max(|Vx|, VXLOW),
as the tyre before task 7.4 did: at rest a damper, not a spring. A car held
on its brakes on a 10 % slope then creeps down at about 4 mm/s; with the
lag it rolls back about a millimetre onto its tyres' springs and stays
(`tests/test_drivetrain.cpp`).

**Mirroring.** A parameter set describes a tyre of one side of a car (the
file's TYRESIDE, left by default). A tyre mounted on the other side is
evaluated mirrored, as TNO's MF-Tyre does: the slip angle and the
inclination negated going in, the side force, the overturning and the
aligning moments negated coming out. The vehicle builders mount every
corner (`FullTireForce::mount(right_side)`), so a car's left and right tyres
from one file push against each other at zero slip (their ply steer and
conicity) instead of steering the car.

## Simple tyres

Task 7.5 (D65), `mbd/forces/simple_tyres.hpp`: three tyres of a few
parameters and closed forms, kept for real-time use and for teaching. Each is
a `TyreModel` (`mbd/forces/tyre_model.hpp`), the interface the Magic Formula
implements too, so the tyre on a wheel, its lag, and the vehicles take any of
them:

```cpp
#include "mbd/forces/simple_tyres.hpp"

mbd::VehicleParams vp;                                          // the simple car
vp.tyre_model = std::make_shared<mbd::FialaTyre>(mbd::FialaTyreParams{});
// or, on a template: tmpl.front_axle.tyre_model = ...;
// or on one wheel: FullTireForce(wheel, std::make_shared<mbd::BrushTyre>(p));
```

| Tyre | Forces | Aligning torque | Source |
|---|---|---|---|
| `LinearTyre` | Fx = Kxk kappa, Fy = Kya alpha, the stiffnesses in proportion to the load; the force brought back onto the friction circle mu Fz | -t Fy, a constant trail t | |
| `FialaTyre` | Fy the brush model's at pure side slip (3.11), in tan(alpha), with C_alpha; Fx = C_kappa kappa up to kappa_c = mu Fz / (2 C_kappa), then mu Fz - (mu Fz)^2 / (4 \|kappa\| C_kappa); the stiffnesses constant; mu from mu_max at no slip to mu_min at a combined slip sqrt(kappa^2 + tan^2 alpha) of 1 | the brush model's (3.12), the tyre's width for the half contact length | MSC Adams's Fiala tyre, as Project Chrono's `ChFialaTire` (BSD-3) has it |
| `BrushTyre` | the isotropic brush model with a parabolic pressure distribution: the theoretical slip s = (kappa, -tan alpha) / (1 + kappa), the force mu Fz (3 theta s - 3 (theta s)^2 + (theta s)^3) along it up to full sliding at theta s = 1, theta = C / (3 mu Fz), C = 2 c_p a^2 in proportion to the load and a growing as its square root | -t Fy with the trail of (3.13) at theta s | Pacejka, 3rd ed., section 3.2, (3.11) to (3.14), (3.34), (3.46) to (3.50) |

Common to the three: the rolling resistance My = -f_r R0 Fz against the
rolling; no overturning moment, no camber, no pressure, no shifts; the
relaxation lengths the slip stiffnesses over the carcass's stiffnesses, as
MF 6.1's, so that at rest each holds by the carcass's spring. The defaults
describe roughly the Appendix 3 tyre at its nominal load (4000 N): its slip
stiffnesses (86748 N, 53356 N/rad; the brush model's one stiffness is the
cornering stiffness), friction 1, a trail of 0.028 m (the brush model's a / 3
with a = 0.075 m), f_r = 0.007, the carcass's stiffnesses 358066 and 102673
N/m, the radius, vertical stiffness and damping.

What each leaves out, by its design: the linear tyre has no peak and no
interaction but the circle; the Fiala tyre's two forces do not reduce each
other, so their sum can pass mu Fz at combined slip (as Adams's has it), and
its stiffnesses do not follow the load; the brush model has one stiffness in
both directions, while a real tyre's longitudinal one is often half as large
again, and its aligning torque never changes sign. Chrono's Fiala takes its
rolling resistance as a length times the load; here it is a coefficient
times the radius, as for the other two. The book's own "Fiala" model (1954)
has a flexible carcass and no combined slip; the industry's Fiala tyre is
the one here.

**How they were checked** (`tests/forces/test_simple_tyres.cpp`): each
against its closed forms, written out in the test from the source: the
linear tyre's stiffnesses, circle and trail; Fiala's longitudinal force on
both branches and continuous at kappa_c, its side force and torque equal to
the brush model's (3.11), (3.12) with the width (the torque's peak 27 mu Fz
w / 256 at theta s = 1/4), full sliding, the friction falling with the
combined slip; the brush model's (3.11) to (3.14) at four slip angles, its
torque's peak 27 mu Fz a / 256, isotropy in the theoretical slip (the
book's Figure 3.8), combined slip (3.48) to (3.50), the locked wheel's mu Fz
cos(alpha) and sin(alpha), the scaling with load; all to rounding. On a
wheel each model's lag settles to its steady state, and a car on each, held
on its brakes on a 10 % slope, stays put and rolls back by the hand
calculation shared with the Magic Formula (`tests/test_drivetrain.cpp`).

**Their cost** (`build\bench\mbd_bench.exe`, pinned to a performance core,
26 November 2026, on battery power: compare the rows with each other): one
evaluation at a combined slip, the Magic Formula 1.35 us, the linear tyre
0.034, Fiala 0.100, the brush model 0.067, 13 to 40 times less; one RK4 step
of 1 ms of the double-wishbone sedan, 325 us on the Magic Formula and 289 to
291 us on a simple tyre, 11 % less: the Magic Formula's 32 evaluations a
step (four tyres, four stages, two each: its characteristics, then its
forces) are what the simple tyres save; the rest of the step is the
vehicle's. `docs/performance.md` has the table.

## Reading .tir files

A .tir file is text in sections, `[SECTION]`, of `KEY = value` lines. `$`
begins a comment, as does a line beginning with `!`; strings are quoted;
lines without `=` (the rows of a table such as `[SHAPE]`) are skipped.

- `[UNITS]` must be SI: meter, newton, radians, kg, second, pascal
  (MBD-I021 otherwise). The reader does not convert units.
- FITTYP (or PROPERTY_FILE_FORMAT), FNOMIN and UNLOADED_RADIUS are required
  (MBD-I023). Every other parameter takes its default when the file leaves
  it out: coefficients 0, scaling factors 1, except LMUV 0 and PKY4 2.
- Keys are matched by name, whatever their section. A key the tyre does not
  use (a SWIFT parameter, the contact patch's, the header's) is listed in
  `TirFile::ignored`, not refused.
- The ranges of a file (`[LONG_SLIP_RANGE]`, `[VERTICAL_FORCE_RANGE]` and so
  on) are read and not used: the tyre evaluates the formulas at the inputs
  it is given, as MFeval's mode 211 does.

The messages are MBD-I020 to MBD-I024 and MBD-F053 (`docs/help/messages.md`).

## How it was checked

**Identities** (`test_magic_formula.cpp`), at the nominal state of the
book's Appendix 3 tyre (205/60R15, its Table A3.1 written out as
`tests/forces/data/pacejka_a3_205_60R15.tir`): the slip stiffnesses are the
slopes of the curves at their centres (central differences, within 1e-8);
the peaks are mu Fz lifted by the vertical shifts (a sweep every 2e-5 of
slip, within 1e-5); combined slip leaves each force alone where the other
slip is zero (exactly, both versions); the rolling resistance and the
relaxation lengths are their closed forms.

**Against MFeval** (`test_magic_formula_mfeval.cpp`). MFeval 4.3.1 (Marco
Furlan, BSD-3 licence, MATLAB File Exchange 63618) is an independent
implementation of MF 5.2, 6.1 and 6.2, written from the book and the paper
and compared by its author with TNO's solver. It is kept outside the
repository; a MATLAB script kept with it, `mfeval_curves.m`, evaluates the
three test tyres in its mode 211 (no limit checks, no low-speed reduction,
no alpha*, no turn slip) and writes the CSV files in
`tests/forces/data/mfeval/`. Their headers say how they were made; the
journal entry of 25 November 2026 quotes the script.

| Tyre | Version | Points |
|---|---|---|
| Appendix 3 (`pacejka_a3_205_60R15.tir`) | MF 6.1 | 618 |
| Appendix 3 with every scaling factor other than 1 (`..._scaled.tir`) | MF 6.1 | 618 |
| A synthetic MF 5.2 tyre with scaling factors (`mf52_test_tyre.tir`) | MF 5.2 | 576 |

The points sweep the longitudinal slip from -0.9 to 0.9 and the slip angle
from -0.35 to 0.35 rad at loads of 1500, 4000 and 7000 N and three
inclinations, combine the two at 3000 and 6000 N, vary the forward speed
from 2 to 40 m/s and (MF 6.1) the pressure from 180 to 260 kPa. Thirteen
channels are compared: Fx, Fy, Mx, My, Mz, the trail, the residual torque,
both friction coefficients, both slip stiffnesses and both relaxation
lengths. Each must be within 1e-10 of its largest magnitude in the file, the
trail and Mz within 1e-6: MFeval multiplies the trail by Vcx / (Vc + 1e-6)
where the paper has cos(alpha), a difference of 1e-6 / Vc. Seen: every
channel but those two within 3.5e-12 (the CSV's 12 significant digits), the
trail within 2.1e-7 and Mz within 1.1e-7, both at the 2 m/s point.

MFeval fills a parameter a file leaves out with values of a sample tyre
(QSX4 = 5, for instance); the script gives them this tyre's defaults
instead, and each CSV lists which.

**Pacejka's chapter 4 curves.** The plan asked that the curves match the
reference curves of the book's chapter 4. Those figures are fits of measured
tyres whose parameters the book does not give, so they cannot be redrawn.
The book's own parameter set is Appendix 3's, and that is what the
identities and the comparison with MFeval use (D63).

## What is not modelled

- Turn slip and the zeta factors (every one 1).
- The tyre's vertical behaviour from the file beyond its free radius,
  stiffness and damping: the loaded radius's speed and force terms, the
  effective rolling radius (BREFF, DREFF, FREFF), the contact patch. The
  wheels' contact is task 7.3's (`mbd/forces/tire.hpp`), linear in the
  depth, its rolling radius R - depth / 2.
- Limit checks: the file's ranges are not applied, and loads, slips and
  inclinations outside them are evaluated as they are.
- The transient models beyond the semi-non-linear one: the carcass's own
  compliance and the contact patch's mass, rigid-ring dynamics (SWIFT).
- Rolling backwards was not compared with any reference: MFeval's
  conventions for it differ (its slip is -Vsx / Vx, signed, and it reverses
  the longitudinal force). Here the deflections' equations use |Vx| and give
  slips that oppose the sliding, the trail and the residual torque change
  sign with Vcx, and the rolling resistance opposes the spin.
