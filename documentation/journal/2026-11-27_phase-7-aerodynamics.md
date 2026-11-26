# Phase 7: the aerodynamic element

- **Dates:** 27 November 2026
- **Plan:** task 7.10; finding F12; D71
- **Commits:** see git log
- **Written:** at the time

## Goal

Task 7.10: forces and moments from a table over front and rear ride height,
pitch, roll and yaw, applied in the body frame, with wind; the present
constant model a special case. Done when a synthetic map is reproduced
exactly.

## Starting point

`AerodynamicForce` (`mbd/forces/aerodynamics.hpp`): a drag area acting
against the chassis's velocity along the ground, a downforce area along the
ground normal growing linearly below a ride height, both at one centre of
pressure. Finding F12: no lift, side force, yaw dependence, balance or wind,
and no way to take CFD results. Of the regression scenarios only
`sports-car-turn` installs it.

## What was done

1. `AeroMap`: six coefficients (force and moment over the dynamic pressure,
   in the chassis's axes) over five variables, multilinear between nodes,
   held beyond them. D71 gives the reasons.
2. `AerodynamicForce` reads the variables from the chassis's state (ride
   heights at two chassis points above a road or the ground plane, pitch and
   roll to the ground plane, yaw of the velocity relative to the air), the
   wind a function of position. Without a map it uses the constant model,
   now in the chassis's axes; `AeroMap::from` gives that model as a map.
3. The template's `ChassisConfig` takes a map and the three points.
4. Tests (`tests/forces/test_aerodynamics.cpp`, 5 new cases, 1 changed);
   `sports-car-turn` written again; the records; `docs/aerodynamics.md`;
   MBD-F060.

## Decisions

- D71: the map in the chassis's axes, as the plan asks, with the constant
  model moved there too, so that a one-node map is the constant model; a
  multilinear map held at its edges.

## What went wrong, and how it was found

- **A model's meaning changed, not only its code.** Making the constant
  model a map in the chassis's axes changes what it does whenever the car
  is not level and straight: the drag no longer turns with the velocity, the
  downforce tilts with the body. One existing test said the opposite ("drag
  opposes lateral velocity too"); it now states the new behaviour, with the
  reason in its comment. The regression scenario moved for the same reason;
  its differences were checked by phase (below) before the reference was
  written again.
- **A missing include.** The refusal checks used Catch2's
  `ContainsSubstring` without its header; the first build failed there.
- **A stray term.** The synthetic function first held a term `0.1 b^2 0.0`,
  zero but not multilinear in form; found on reading the test before the
  build, removed.
- **A catalogue entry out of order.** MBD-F060 was first inserted before
  F056; moved after F059.
- **A reference rewritten by rounding.** Writing the references again also
  rewrote `wishbone-sweep`, by 1.3e-15, as in task 7.6; restored.

## Verification

- The plan's test: a map of 288 nodes holding a function multilinear in the
  five variables, read through the element for a chassis turned (yaw 0.7
  rad), pitched (0.012), rolled (-0.017), with wind, a vertical velocity and
  a turning rate: the force and moment equal the function's coefficients
  times q, computed by hand from the pose, to 4.6e-13 N and 6.1e-14 N m
  on a force of 946 N, with and without a road (allowed 1e-12 of the
  largest component).
- At the nodes, arbitrary values bitwise; beyond the ends of each variable,
  the end node's values bitwise.
- The constant model equals its two-node map to 1e-12, above and below
  h_ref, level and tilted; level and straight on, the old closed form.
- `sports-car-turn` against its old reference: at rest (0 to 0.3 s, q zero)
  1.1e-15 of each channel's largest value; accelerating straight (0.3 to 1
  s) up to 2.2e-5; turning (1 to 2 s) up to 2.35e-4 (a front camber). The
  speed: 3.3e-7 m/s higher at 0.6 s, 3.7e-6 at 1 s, 7.6e-5 at 2 s, growing
  with q. Estimate (the record holds no attitude): a squat of a few mrad
  tilts about 30 N of downforce forward by 30 N x 0.005 = 0.15 N, about
  1e-4 m/s over the run for about 1300 kg (the 1200 kg body and its
  corners); the sign (faster) fits. The sensitivity
  probe gives 0.0108 and 1.08e4 of the tolerance, as before.
- The long regression scenarios (`hmmwv-*-5s`, `sedan-slalom`) install no
  aerodynamics and were not run.
- The suite: see the data file.

## Measurements

In `data/2026-11-27_phase-7-aerodynamics.md`: the tests' printed values, the
regression comparison by phase, the sensitivity probe, and the suite's run.

## Lessons

- When a plan asks for an old model to become a special case of a new one,
  check that the new one's conventions (here, its axes) contain the old
  model, and say plainly where they do not.
- A multilinear test function makes an interpolating table's test exact,
  and so able to catch an off-by-one in its indexing.

## Open ends

- The lap model reads `CdA` and `ClA`, not a map.
- No steering or unsteady variables; ride heights on a road are vertical,
  not along its normal.
- Not in model files or Python.

## For the book

The aerodynamic map as the interface to CFD; multilinear interpolation and
the exact test it allows; the constant model brought into the body's axes,
and the cost of that choice measured on a scenario.
