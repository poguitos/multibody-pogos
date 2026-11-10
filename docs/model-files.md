# Model files

A model (a `kernel::System`: the tree of bodies, the constraints, the force
elements and the joint forces) as a JSON file, and back (plan task 6.4,
decision D56). This page describes the format, version 1.

```cpp
#include "mbd/io/model_file.hpp"

mbd::io::save_model(system, "four_bar.json");
mbd::kernel::System copy = mbd::io::load_model("four_bar.json");
std::string text = mbd::io::model_to_json(system);      // and model_from_json(text)
```

```python
mbd.save_model(system, "four_bar.json")      # a str or a pathlib.Path
copy = mbd.load_model("four_bar.json")
text = mbd.model_to_json(system)             # and mbd.model_from_json(text)
```

**A model survives a round trip unchanged.** Every number is written in the
shortest form that reads back as the same double, and a model read from a
file writes the same text again, character for character. A simulation of
the model read back gives the same numbers, bit for bit, as one of the
original (`tests/io/test_model_files.cpp` checks both on a model with every
kind of part).

**Nothing is left out without a word.** What a file cannot hold is refused
when writing (MBD-I008): a force written as code (`UserForce`), a time
function given as code (only a constant can be written), the vehicle layer's
elements, and any element, constraint or joint written in C++ that the
format does not know. Reading checks every key: an unknown one, usually a
misspelling, is an error (MBD-I004), as is a key given twice. Every error
names its place in the file (`docs/help/messages.md`, area I).

## An example

A pendulum: one body hinged about Z, its joint frame 1 m above it, gravity
along -Y. This is the text `model_to_json` writes (the test of the format
compares it to this):

```json
{
  "format": "multibody model",
  "version": 1,
  "units": "SI",
  "gravity": [0.0, -9.81, 0.0],
  "bodies": [
    {
      "name": "bob",
      "parent": 0,
      "joint": "revolute",
      "x_pj": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
      "x_cj": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 1.0, 0.0]},
      "inertia": {
        "mass": 1.0,
        "com_B": [0.0, 0.0, 0.0],
        "I_com_B": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      }
    }
  ],
  "constraints": [],
  "force_elements": [],
  "joint_forces": []
}
```

Written by hand, the keys with defaults may be left out; this is the same
model:

```json
{"format": "multibody model", "version": 1,
 "gravity": [0, -9.81, 0],
 "bodies": [{"name": "bob", "parent": 0, "joint": "revolute",
             "x_cj": {"translation": [0, 1, 0]}, "inertia": {"mass": 1}}]}
```

## The file

| Key | Value | Default |
|---|---|---|
| `format` | `"multibody model"` | required |
| `version` | `1` | required |
| `units` | `"SI"`: m, kg, s, rad, N | `"SI"` |
| `gravity` | 3 numbers [m/s^2] | `[0, 0, -9.81]` |
| `bodies` | the bodies, in order: the first is body 1 (0 is the ground) | none |
| `constraints` | see below | none |
| `force_elements` | see below | none |
| `joint_forces` | see below | none |

The names of keys and types are those of the C++ and Python classes and
their arguments (D52): what a file says is what the code would say.

**Numbers** may be written with or without a decimal point. An infinite
value (a stop's clearance, a joint's limit) is the text `"inf"` or `"-inf"`;
those are the only places a text stands for a number.

**A transform** (a frame on a body, in the body's frame) is
`{"quaternion": [w, x, y, z], "translation": [x, y, z]}`, each part
defaulting to the identity. A quaternion must be within 1e-6 of unit
length; one further than 1e-12 from it, typed by hand, is normalised, and one
the program wrote is kept exactly.

**A marker** (a frame fixed on a body) is `{"body": b, "X_BM": transform}`.

**A curve** is a line `{"slope": k, "offset": c}` (each defaulting to 0, so
`{}` is the zero curve) or a table `{"x": [...], "y": [...]}`, x strictly
increasing, at least two points (`Curve.linear`, `Curve.table`).

## Bodies

| Key | Value | Default |
|---|---|---|
| `name` | text | `""` |
| `parent` | the parent's index, smaller than the body's own | required |
| `joint` | `revolute`, `prismatic`, `fixed`, `universal`, `cylindrical`, `planar`, `spherical` or `free` | required |
| `x_pj` | the joint frame on the parent: a transform | identity |
| `x_cj` | the joint frame on the body: a transform | identity |
| `inertia` | `{"mass": m, "com_B": [x, y, z], "I_com_B": 3 rows of 3}` | `mass` required, the others zero |

## Constraints

Each is an object with its `type`:

| `type` | Keys | Defaults |
|---|---|---|
| `PointCoincidence` | `i`, `j` (markers) | |
| `Dot1` | `i`, `axis_i`, `j`, `axis_j` (0 = X, 1 = Y, 2 = Z), `s` | `s` 0 |
| `Dot2` | `i`, `axis_i`, `j`, `s` | `s` 0 |
| `Distance` | `i`, `j`, `length`, `nominal_length` | `nominal_length` = `length` |
| `NoTwist` | `i`, `j` | |
| `JointDriver` | `body` (a revolute or prismatic joint's body), `s` | `s` 0 |
| `point_on_line`, `point_on_plane`, `spherical_closure`, `revolute_closure`, `universal_closure`, `cylindrical_closure`, `prismatic_closure`, `constant_velocity_closure` | `i`, `j` (markers) | |
| `ConstraintSet` | `name`, `parts` (a list of constraints) | |

`s` and `length` are constants: a time function given as code cannot be
written. A closure is written as the call that makes it when the set is
exactly what that call makes; any other set, made by hand in C++, is
written part by part as a `ConstraintSet`.

## Force elements

| `type` | Keys | Defaults |
|---|---|---|
| `LinearSpringDamper` | `body1`, `body2`, `anchor1`, `anchor2` (points in the bodies' frames), `stiffness`, `damping`, `rest_length` | `damping` 0 |
| `SpringDamper` | `body1`, `body2`, `anchor1`, `anchor2`, `params` | |
| `RotationalSpringDamper` | `body1`, `X_1M`, `body2`, `X_2M` (transforms), `params` | markers identity |
| `Bushing` | `body1`, `X_1M`, `body2`, `X_2M`, `params` | markers identity |
| `PlaneContact` | `body`, `centres` (a list of points), `radius`, `plane_body`, `plane_X`, `params` | `radius` 0, `plane_X` identity |

The `params` of each are its settings class, every setting optional with the
C++ default:

- `SpringDamper`: `free_length`, `spring`, `preload`, `damper`,
  `bump_clearance`, `bump_stop`, `rebound_clearance`, `rebound_stop` (the
  clearances `"inf"` by default: no stop);
- `RotationalSpringDamper`: `reference`, `spring`, `preload`, `damper`;
- `Bushing`: `rotational_stiffness`, `stiffness`, `rotational_damping`,
  `damping`, each a list of three curves (about or along X, Y, Z of marker 1);
- `PlaneContact`: `stiffness`, `exponent`, `damping`, `damping_depth`,
  `mu_static`, `mu_dynamic`, `v_static`, `v_dynamic`.

## Joint forces

| `type` | Keys |
|---|---|
| `JointCoordinateForce` | `body`, `params`: `reference`, `spring`, `preload`, `damper`, `friction`, `friction_velocity`, `lower_limit`, `upper_limit`, `limit_stiffness`, `limit_damping` (the limits `"-inf"` and `"inf"` by default) |

## What comes later

Phase 8 (task 8.2) extends the format to the vehicle, with curve files and
a schema; this version is the kernel's model, as Python and C++ build it.
