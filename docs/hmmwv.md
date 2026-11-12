# The HMMWV benchmark

Plan task 5.4 (decisions D47, D48): the fourteen-body model of the Army's
high mobility multipurpose wheeled vehicle from Haug, Negrut, Serban and
Solis, "Numerical methods for high speed vehicle dynamic simulation" (1999),
driven at 10 mph over a half-cylinder bump, with joints and with the paper's
stiff bushings, run by every integrator of the kernel. What each method's
result is worth and what it costs on a vehicle model, the second with a
stiffness ratio of a million.

The model is `tests/kernel/hmmwv.hpp`; its tests,
`tests/kernel/test_kernel_hmmwv.cpp`; the tables below come from one program:

```powershell
scripts\build.ps1 -Target mbd_hmmwv
build\bench\mbd_hmmwv.exe              # the tables: about 45 minutes, most of it two explicit runs on the bushed model
build\bench\mbd_hmmwv.exe quick        # without those
build\bench\mbd_hmmwv.exe modes        # the modes at rest
build\bench\mbd_hmmwv.exe calibrate    # the springs' free lengths
build\bench\mbd_hmmwv.exe record <folder> [bushed]   # a run for playback (below)
```

**Playback** (task 6.5): `record` runs the joint model (or with `bushed` the
bushed one) for 5 s with the tyres' events, by Radau IIA at a tolerance of
1e-6, and writes the model as a model file (`hmmwv.json`, without its tyres,
which are forces written as code) and the coordinates every 10 ms
(`hmmwv_run.csv`). `python/examples/replay_hmmwv.py <folder>` plays them back
in Rerun (`docs/python.md`, "Viewing").

The times are wall-clock times of the stepping alone, from one run pinned
to one performance core at high priority (D21) on the development laptop
(i7-13700H, Windows 11, MSVC, RelWithDebInfo); compare them with each other,
not with another machine or another day. Raw output:
`documentation/journal/data/2026-11-07_hmmwv-benchmark.md`.

## The model

| | |
|---|---|
| Bodies | 14: chassis, steering rack, and at each corner a lower arm, an upper arm and a spindle body (upright, spindle, wheel and tyre in one; no wheel spin, as the paper's model) |
| Joints | chassis free; rack prismatic, held by a driver (the steering input); arms revolute to the chassis; spindle spherical to the lower arm |
| Closures | at each corner the upper ball joint (spherical) and the tie rod (a distance to the rack in front, to the chassis behind) |
| Degrees of freedom | 10 (27 velocities, 17 equations); the paper's count |
| Suspension | progressive coil springs and linear shocks, chassis to lower arm |
| Tyres | one spring-damper each, 296,325 N/m and 3,502 N s/m (the paper's), from the wheel centre to the nearest point of the road, pushing only, no friction |
| Road | flat, with a half cylinder of radius 0.05 m across it |
| Run | from rest on the tyres, every body moving at 10 mph (4.4704 m/s); the front tyres turn to the bump at 0.5 s, the rear ones at about 1.26 s; 5 s |
| With bushings | each upper arm on a spherical joint at its front point, with two spring-dampers of 2e7 N/m and 2e6 N s/m across its axis at its back point (the paper's figures): 18 degrees of freedom |

The paper gives no geometry, masses or suspension laws. They come from
Project Chrono's HMMWV (BSD-3-Clause), a descendant of the same group's
model: its chassis rests 0.709 m above the ground at the design position,
the paper's "approximately 0.71 m". Two of its values were corrected (D48):
the control arms' moments of inertia, which no body can have (0.4 + 0.4 <
0.8938 kg m^2), taken as flat arms; and the springs' free lengths, with
which the vehicle rested 0.085 m high on its rebound stops, set to rest the
arms at their design angles (0.2947 and 0.3330 m for Chrono's 0.339 and
0.382).

At rest the chassis's centre of mass is 0.6875 m up (the design height less
the tyres' deflection under load); the static equilibrium takes 3 Newton
iterations with joints, 4 with bushings, which first needed the static
solver's step changed (D47). After both bumps the chassis settles within
about 3.5 s; the paper says about 4 s and 0.71 m.

The modes at rest set the explicit methods' steps:

| | Fastest modes [s^-1] | RK4's step limit (2.785 / \|lambda\|) |
|---|---|---|
| With joints | -66.6, -56.8 +- 21.0i, -36.8 +- 47.9i | 42 ms |
| With bushings | -2.06e6, -1.44e6 (each twice), overdamped: a bushing's damping on about 1 kg of upper arm | 1.35e-6 s |

The paper gives -2.6e5 for its bushed model; its bushings' geometry is not
given, and these are eight times faster.

## Results

Each run is compared with Radau IIA at a tolerance of 1e-10, on every
body's origin every 10 ms over the 5 s ("any body"; it includes the position
along the road, so it carries any difference in the speed the bump takes
away) and on the chassis's height (the paper's z1). Adaptive methods are
given equal absolute and relative tolerances, as in the paper.

The bump is the hard part. Where a tyre turns from the ground to the bump,
its force turns by about 26 degrees and its damper's rate jumps by about
2 m/s: 7 kN at once. A step that straddles such a jump loses its method's
order there. The runs go as the paper's did, without events; a second table
repeats some with events at the tyres' switches (`Hmmwv::add_events`), which
cut the steps there.

### With joints

Reference: Radau IIA at tol 1e-10, 1.4 s, 2746 steps; z1 from 0.679145 to 0.706081 m; speed at 5 s 4.250279 m/s.

#### Without events

| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] | steps (rejected) | evaluations | Newton iterations | Jacobians |
|---|---|---|---|---|---|---|---|---|
| RK4 | h 4 ms | 4.26e+00 | 2.37e-02 | 0.207 | 1500 (0) | 6000 | 0 | 0 |
| RK4 | h 2 ms | 4.75e-03 | 5.07e-06 | 0.342 | 2500 (0) | 10000 | 0 | 0 |
| RK4 | h 1 ms | 6.03e-03 | 4.15e-05 | 0.682 | 5000 (0) | 20000 | 0 | 0 |
| RK4 | h 0.5 ms | 1.31e-03 | 1.39e-05 | 2.24 | 10000 (0) | 40000 | 0 | 0 |
| RK4 | h 0.25 ms | 1.04e-04 | 1.14e-05 | 4.77 | 20000 (0) | 80000 | 0 | 0 |
| Dormand-Prince | tol 1e-02 | 3.93e-02 | 2.71e-04 | 0.197 | 500 (0) | 3500 | 0 | 0 |
| Dormand-Prince | tol 1e-03 | 1.50e-02 | 1.16e-04 | 0.201 | 505 (3) | 3553 | 0 | 0 |
| Dormand-Prince | tol 1e-04 | 1.20e-02 | 1.11e-04 | 0.204 | 513 (6) | 3627 | 0 | 0 |
| Dormand-Prince | tol 1e-05 | 4.33e-05 | 6.24e-06 | 0.229 | 554 (39) | 4112 | 0 | 0 |
| Dormand-Prince | tol 1e-06 | 9.35e-06 | 6.72e-07 | 0.253 | 608 (52) | 4568 | 0 | 0 |
| Dormand-Prince | tol 1e-08 | 2.22e-07 | 5.70e-09 | 0.353 | 843 (91) | 6447 | 0 | 0 |
| generalised-alpha | h 4 ms | 4.25e+00 | 2.37e-02 | 0.298 | 1500 (0) | 3871 | 2317 | 1 |
| generalised-alpha | h 2 ms | 1.81e-02 | 7.03e-05 | 0.488 | 2500 (0) | 6357 | 3803 | 1 |
| generalised-alpha | h 1 ms | 9.05e-03 | 5.74e-05 | 0.916 | 5000 (0) | 11911 | 6857 | 1 |
| generalised-alpha | h 0.5 ms | 8.27e-04 | 5.86e-05 | 1.74 | 10000 (0) | 22540 | 12486 | 1 |
| generalised-alpha | h 0.25 ms | 2.68e-03 | 1.24e-05 | 3.31 | 20000 (0) | 42361 | 22307 | 1 |
| SDIRK4 | tol 1e-02 | 9.64e-03 | 1.13e-04 | 0.195 | 513 (13) | 2931 | 2871 | 1 |
| SDIRK4 | tol 1e-03 | 9.80e-04 | 1.53e-04 | 0.244 | 529 (22) | 3703 | 3583 | 2 |
| SDIRK4 | tol 1e-04 | 2.94e-03 | 5.18e-05 | 0.311 | 557 (25) | 4964 | 4904 | 1 |
| SDIRK4 | tol 1e-05 | 2.63e-03 | 2.70e-05 | 0.551 | 610 (33) | 9234 | 9114 | 2 |
| SDIRK4 | tol 1e-06 | 2.66e-03 | 2.14e-05 | 0.594 | 717 (44) | 9781 | 9721 | 1 |
| SDIRK4 | tol 1e-08 | 7.96e-04 | 1.05e-05 | 1.4 | 1308 (49) | 22965 | 22905 | 1 |
| Radau IIA | tol 1e-02 | 6.47e-04 | 6.81e-05 | 0.168 | 508 (9) | 2359 | 576 | 2 |
| Radau IIA | tol 1e-03 | 5.24e-04 | 1.38e-05 | 0.192 | 520 (21) | 2668 | 654 | 3 |
| Radau IIA | tol 1e-04 | 1.51e-05 | 9.98e-07 | 0.235 | 552 (42) | 3159 | 842 | 1 |
| Radau IIA | tol 1e-05 | 5.36e-06 | 4.97e-08 | 0.329 | 618 (68) | 4307 | 1201 | 1 |
| Radau IIA | tol 1e-06 | 1.99e-07 | 1.01e-08 | 0.457 | 703 (79) | 5777 | 1663 | 1 |
| Radau IIA | tol 1e-08 | 6.35e-10 | 1.75e-10 | 0.957 | 1175 (101) | 11440 | 3392 | 1 |
| linearly implicit Euler | h 1 ms | 4.38e-02 | 5.31e-04 | 1.38 | 5000 (0) | 5000 | 0 | 5000 |
| linearly implicit Euler | h 0.5 ms | 1.95e-02 | 2.45e-04 | 2.71 | 10000 (0) | 10000 | 0 | 10000 |
| linearly implicit Euler | h 0.25 ms | 1.05e-02 | 1.33e-04 | 5.39 | 20000 (0) | 20000 | 0 | 20000 |
| linearly implicit Euler | h 0.125 ms | 5.95e-03 | 6.90e-05 | 10.7 | 40000 (0) | 40000 | 0 | 40000 |

The reference with the tyres' events (Radau IIA at tol 1e-10, 6.98 s, 6952 steps, 19 events) differs from it by 1.64e-11 m (any body), 1.55e-12 m (chassis height).

#### With the tyres' events

| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] | steps (rejected) | evaluations | Newton iterations | Jacobians |
|---|---|---|---|---|---|---|---|---|
| RK4 | h 2 ms | 7.52e-04 | 6.31e-06 | 0.685 | 2765 (0) | 11060 | 0 | 0 |
| RK4 | h 1 ms | 1.99e-03 | 8.26e-06 | 1.31 | 5283 (0) | 21132 | 0 | 0 |
| RK4 | h 0.5 ms | 1.09e-03 | 6.56e-06 | 2.55 | 10270 (0) | 41080 | 0 | 0 |
| RK4 | h 0.25 ms | 5.13e-04 | 3.00e-06 | 5.01 | 20268 (0) | 81072 | 0 | 0 |
| Dormand-Prince | tol 1e-03 | 1.42e-02 | 1.03e-04 | 0.217 | 521 (3) | 3665 | 0 | 0 |
| Dormand-Prince | tol 1e-06 | 3.99e-06 | 2.02e-07 | 0.272 | 622 (50) | 4654 | 0 | 0 |
| generalised-alpha | h 1 ms | failed: MBD-K058: kernel::Simulator: the generalised-alpha step of 0.000362 s from t = 0.606000 s | | 0.124 | 746 | | | |
| generalised-alpha | h 0.25 ms | failed: MBD-K058: kernel::Simulator: the generalised-alpha step of 0.000077 s from t = 0.606250 s | | 0.443 | 2558 | | | |
| SDIRK4 | tol 1e-03 | 1.82e-04 | 6.27e-06 | 1.11 | 1321 (199) | 17486 | 10826 | 111 |
| SDIRK4 | tol 1e-06 | 2.12e-07 | 8.30e-09 | 3.27 | 2426 (533) | 49831 | 38311 | 192 |
| Radau IIA | tol 1e-03 | 5.76e-05 | 5.65e-07 | 0.835 | 1056 (113) | 10312 | 2366 | 36 |
| Radau IIA | tol 1e-06 | 1.52e-07 | 1.31e-09 | 2.16 | 2253 (631) | 25981 | 6671 | 61 |
| linearly implicit Euler | h 1 ms | 2.18e-02 | 3.76e-04 | 1.5 | 5281 (0) | 5281 | 0 | 5281 |
| linearly implicit Euler | h 0.5 ms | 1.08e-02 | 1.92e-04 | 2.87 | 10245 (0) | 10245 | 0 | 10245 |

### With bushings

Reference: Radau IIA at tol 1e-10, 4.24 s, 2936 steps; z1 from 0.679123 to 0.706047 m; speed at 5 s 4.250286 m/s.

#### Without events

| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] | steps (rejected) | evaluations | Newton iterations | Jacobians |
|---|---|---|---|---|---|---|---|---|
| RK4 | h 1.25e-6 s | 3.55e-06 | 3.42e-08 | 1.06e+03 | 4000000 (0) | 16000000 | 0 | 0 |
| Dormand-Prince | tol 1e-06 | 2.03e-06 | 9.17e-09 | 1.52e+03 | 3113941 (515854) | 24892711 | 0 | 0 |
| generalised-alpha | h 4 ms | 4.25e+00 | 2.37e-02 | 0.349 | 1500 (0) | 3933 | 2363 | 1 |
| generalised-alpha | h 2 ms | 1.79e-02 | 6.82e-05 | 0.568 | 2500 (0) | 6348 | 3778 | 1 |
| generalised-alpha | h 1 ms | 9.28e-03 | 5.55e-05 | 1.07 | 5000 (0) | 11845 | 6775 | 1 |
| generalised-alpha | h 0.5 ms | 1.05e-03 | 5.66e-05 | 2.09 | 10000 (0) | 22526 | 12456 | 1 |
| generalised-alpha | h 0.25 ms | 2.90e-03 | 2.09e-05 | 3.88 | 20000 (0) | 42288 | 22218 | 1 |
| SDIRK4 | tol 1e-02 | 9.80e-04 | 1.58e-04 | 0.485 | 514 (13) | 6761 | 6521 | 3 |
| SDIRK4 | tol 1e-03 | 1.85e-03 | 1.39e-04 | 0.28 | 528 (19) | 3694 | 3614 | 1 |
| SDIRK4 | tol 1e-04 | 2.26e-04 | 4.21e-05 | 0.423 | 559 (29) | 5095 | 5015 | 1 |
| SDIRK4 | tol 1e-05 | 5.69e-06 | 2.55e-07 | 0.576 | 652 (62) | 7879 | 7799 | 1 |
| SDIRK4 | tol 1e-06 | 1.61e-04 | 1.36e-06 | 0.814 | 761 (67) | 11149 | 11069 | 1 |
| SDIRK4 | tol 1e-08 | 4.41e-07 | 3.74e-09 | 2.14 | 1495 (71) | 29087 | 29007 | 1 |
| Radau IIA | tol 1e-02 | 9.76e-03 | 7.75e-05 | 0.287 | 514 (8) | 3396 | 776 | 7 |
| Radau IIA | tol 1e-03 | 7.82e-04 | 1.33e-05 | 0.342 | 531 (29) | 3912 | 886 | 9 |
| Radau IIA | tol 1e-04 | 2.06e-05 | 9.61e-07 | 0.298 | 543 (34) | 3157 | 839 | 1 |
| Radau IIA | tol 1e-05 | 7.20e-06 | 2.32e-07 | 0.514 | 615 (57) | 5403 | 1300 | 11 |
| Radau IIA | tol 1e-06 | 2.41e-07 | 8.10e-09 | 0.78 | 706 (76) | 7116 | 1838 | 11 |
| Radau IIA | tol 1e-08 | 2.23e-09 | 1.44e-10 | 1.67 | 1228 (102) | 13829 | 3899 | 11 |
| linearly implicit Euler | h 1 ms | 4.36e-02 | 5.27e-04 | 1.92 | 5000 (0) | 5000 | 0 | 5000 |
| linearly implicit Euler | h 0.5 ms | 1.93e-02 | 2.43e-04 | 3.94 | 10000 (0) | 10000 | 0 | 10000 |
| linearly implicit Euler | h 0.25 ms | 1.02e-02 | 1.23e-04 | 7.74 | 20000 (0) | 20000 | 0 | 20000 |
| linearly implicit Euler | h 0.125 ms | 5.72e-03 | 6.79e-05 | 14.9 | 40000 (0) | 40000 | 0 | 40000 |

## What the tables show

- **Explicit and implicit agree** (the task's condition). With joints,
  Dormand-Prince at 1e-8 is within 2.2e-7 m of the reference, Radau IIA at
  1e-8 within 6.4e-10 m. With bushings, RK4 at 1.25e-6 s and Dormand-Prince
  at 1e-6 are within 3.6e-6 and 2.0e-6 m, Radau IIA at 1e-6 and 1e-8 within
  2.4e-7 and 2.2e-9 m. The reference is itself checked: run again with the
  tyres' events, so with entirely different steps, it moves by 1.6e-11 m.
- **The bushings cost an explicit method a thousandfold.** On the bushed
  model RK4 needs 4,000,000 steps (1,060 s) and Dormand-Prince 3,113,941
  accepted and 515,854 rejected (1,520 s), both held at the stability limit
  of the bushings' -2.06e6 s^-1 whatever the accuracy asked. Radau IIA
  comes as close in 0.51 s (7.2e-6 m at a tolerance of 1e-5) and does
  better than either in 0.78 s (2.4e-7 m at 1e-6). The
  implicit methods barely notice the bushings: Radau IIA at 1e-6 takes 706
  steps on the bushed model and 703 on the one with joints. That is the
  paper's point, measured.
- **Radau IIA is the method for this model.** At every accuracy it is the
  cheapest or close to it, on both variants, and it alone converges
  regularly across the bump without help: 6.5e-4, 5.2e-4, 1.5e-5, 5.4e-6,
  2.0e-7 m from tolerances 1e-2 to 1e-6 with joints.
- **The bump's jumps defeat fixed steps.** RK4, generalised-alpha and the
  linearly implicit Euler step converge irregularly: RK4 4.8e-3, 6.0e-3,
  1.3e-3, 1.0e-4 m from 2 to 0.25 ms. At 4 ms RK4 and generalised-alpha end
  4.3 m from the reference along the road (not looked into: a step of 4 ms
  carries the tyres 1.8 cm across a jump of 7 kN). With events at the
  tyres' switches RK4 does better at the longer steps, worse at 0.25 ms
  (5.1e-4 m), and no more regularly; the
  chassis's height, the paper's z1, stays within 3e-6 to 8e-6 m. Something
  else not smooth remains, not found here (a tyre grazing loss of contact
  near the crest is a candidate).
- **SDIRK4 does not converge across the bump without events** (2.7e-3 m at
  1e-6, 8e-4 m at 1e-8 with joints): its error estimate misses the jumps,
  its stage order being 1 (D33). With events it does (2.1e-7 m at 1e-6),
  at five times the cost.
- **Dormand-Prince needs a tight tolerance here.** From 1e-2 to 1e-4 it
  stays 1.2e-2 to 3.9e-2 m away; from 1e-5 it converges, and at 1e-8 it is
  the cheapest run within 3e-7 m (0.35 s against Radau IIA's 0.46 s for
  2.0e-7 m). Events do not help it: it locates them on its continuous
  extension, which assumes the dynamics smooth across the crossing.
- **Generalised-alpha fails with events** (MBD-K058 on the remainder of a
  step cut at an event, at 0.606 s); without them it runs at every step.
- **The linearly implicit Euler step is first order and steady**: 4.4e-2 m
  at 1 ms, halving with the step, the same on both variants (its steps do
  not depend on the stiffness). It reports MBD-K045 where the tyres turn to
  the bump: there the tyre force's turning gives the stiffness a negative
  part.
- **Against the paper.** Its Tables 1 and 2 give SDIRK and the trapezoidal
  rule on the bushed model 4 s for 84 to 500 and 100 to 1568 CPU seconds
  (tolerances 1e-2 to 1e-5) on a 200 MHz SGI Onyx of 1999. The runs here
  take 0.3 to 0.6 s for 5 s; the machines are not comparable, and neither
  are the models' details (its fastest mode is eight times slower). The shape
  agrees: on the stiff model the implicit methods' cost grows modestly with
  the accuracy asked, and an explicit method's is fixed by stability.
