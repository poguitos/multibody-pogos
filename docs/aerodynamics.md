# Aerodynamics

Plan task 7.10 (decision D71): the aerodynamic element takes its forces and
moments from a map over the ride heights and the chassis's attitude, as a
CFD study or a wind tunnel gives them, with wind. The constant model it had
before (a drag area, a downforce area, one centre of pressure) is now a map
with one node. This page says what it computes, how it was checked, and
what it leaves out.

| What | Where |
|---|---|
| The element, its settings and the map | `mbd/forces/aerodynamics.hpp` (`AerodynamicForce`, `AeroParams`, `AeroMap`) |
| On a template car | `ChassisConfig`'s `CdA`, `ClA`, `aero_*` and `aero_map`; `VehicleHandle::install_aerodynamics` |
| Tests | `tests/forces/test_aerodynamics.cpp` |

## What is modelled

- **The map.** Six coefficients over five variables. The variables are the
  front ride height, the rear ride height, the pitch, the roll and the yaw.
  The coefficients are the force over q in the chassis's axes (A_x, A_y,
  A_z, in m^2) and the moment over q about a reference point (B_x, B_y, B_z,
  in m^3). The force is q A and the moment q B, with q = rho V^2 / 2.
  Signs follow the chassis's axes (ISO 8855: x forward, y left, z up): a
  drag area of 0.7 m^2 is A_x = -0.7, a downforce area of 1.2 m^2 is
  A_z = -1.2.
- **Between and beyond the nodes.** Between nodes the coefficients are
  multilinear (linear along each variable in turn). Beyond a variable's
  first or last node they are held at that node's values. A variable with
  one node is one the coefficients do not depend on.
- **The variables.**
  - The ride heights are the heights of two points fixed in the chassis
    (`front_point`, `rear_point`), above the road under them when the
    element has one (`road`), or else above the plane through the world's
    origin along the ground normal.
  - The pitch is asin(-x_B . n), positive nose down; the roll is
    atan2(y_B . n, z_B . n), positive left side up (ISO 8855's angles to the
    ground plane).
  - The yaw is atan2(V . y_B, V . x_B): the angle of the velocity relative
    to the air from the chassis's x axis, positive when the air comes from
    the left.
- **The velocity.** V is the reference point's velocity minus the wind's,
  with its component along the ground normal removed (as the constant model
  did). The wind (`wind`) is a function of the world position, so a gust or
  a crosswind section can be placed along the road.
- **The constant model** is used when no map is given: drag area `CdA`,
  downforce area `ClA` growing by `dClA_dh` below the ride height `h_ref`,
  at the centre of pressure `cop_offset_chassis`. `AeroMap::from` gives it
  as a map (one node, or two along the front ride height when `dClA_dh` is
  not zero).

## Using it

```cpp
auto map = std::make_shared<mbd::AeroMap>();
map->nodes[0] = {0.03, 0.06, 0.09};          // front ride height [m]
map->nodes[1] = {0.05, 0.09, 0.13};          // rear ride height [m]
map->nodes[4] = {-0.1, 0.0, 0.1};            // yaw [rad]; pitch and roll: one node each
map->values = /* 27 sets of six, the front ride height's index changing fastest */;
tmpl.chassis.aero_map = map;
tmpl.chassis.aero_front_point = mbd::Vec3(1.4, 0.0, -0.25);
tmpl.chassis.aero_rear_point = mbd::Vec3(-1.5, 0.0, -0.25);
mbd::AerodynamicForce* aero = vehicle.install_aerodynamics(sys);
aero->params.wind = [](const mbd::Vec3& p) { return mbd::Vec3(0.0, p.x() > 200.0 ? -8.0 : 0.0, 0.0); };
```

The element checks the map when it is made (MBD-F060). A map set on
`params` afterwards is not checked.

## How it was checked

- **A synthetic map is reproduced exactly** (the plan's test). A map of 288
  nodes holds a function that is multilinear in its five variables, so its
  interpolation is that function everywhere between the nodes. A chassis is
  turned, pitched and rolled, moved with wind and with a vertical velocity,
  so that every variable lies between nodes. Its force and moment equal the
  function's coefficients times q, computed from the pose by hand, to
  4.6e-13 N on a force of 946 N, with and without a road.
- At its nodes, a map of arbitrary values gives each node's values bitwise.
- Beyond the first and last nodes of each variable the coefficients are
  those of the end node, bitwise.
- The constant model and its map give the same forces and moments to
  1e-12, above and below `h_ref`, level and tilted; level and straight on,
  the closed form of the drag, the downforce and the moment of the centre of
  pressure.

## What changed for the constant model

Before task 7.10 the constant model's drag acted against the velocity along
the ground, and its downforce along the ground normal. Now both act in the
chassis's axes, as a map's do: the drag along -x, the downforce along -z.
The two agree for a level car running straight. They differ when the car
pitches, rolls, or runs at a yaw angle: at a yaw of 37 degrees the old drag
had a sideways part, the new one has none (a constant model has no yaw
dependence; give the map a yaw axis for that). The regression scenario
`sports-car-turn`, the only one with aerodynamics, moved by this (see the
journal entry of 27 November 2026).

## What it leaves out

- No steering or wheel-speed variables; no unsteady aerodynamics (the
  coefficients depend on the present state only).
- The dynamic pressure ignores the velocity along the ground normal.
- With a road, the ride heights are vertical (world Z) distances to the
  road under each point, not distances along the road's normal.
- The lap model (`mbd/analysis/lap_*`) still uses `CdA` and `ClA`, not the
  map.
- Not in model files or Python.
