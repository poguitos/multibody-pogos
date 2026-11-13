# How to use the engine from Python

Build a model, run it, find its rest and its modes, add a force written in
Python, save a model and its results, play a run back, and catch the
engine's errors (Phase 6, tasks 6.1 to 6.6, the package `multibody`). Every
example is run and its result checked in `python/tests/test_howto.py`, and
this page quotes that code exactly (task H.6). [The Python
interface](../../python.md) says how to install the package and lists what
is bound; [the notebooks](../../../python/notebooks/) are longer worked
examples.

The examples use `import multibody as mbd`. Classes and functions keep their
C++ names, so the other guides apply with Python's spelling.

## Build a model and run it

A block of 2 kg sliding along Z, hung from a spring:

<!-- example: python-build -->
```python
system = mbd.System()
system.model.gravity = [0.0, 0.0, -9.81]
block = system.model.add_body("ground", mbd.PrismaticJointModel(), name="block",      # slides along Z
                              inertia=mbd.RigidBodyInertia.from_solid_box(2.0, [0.05, 0.05, 0.05]))
# A spring of 50 N/m with a free length of 1 m, from 1 m above the block's origin to it:
system.add_force_element(mbd.LinearSpringDamper(0, block, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 50.0, 0.0, 1.0))
```

`add_body` takes the parent by its name (`"ground"` for body 0) and returns
the new body's index; the joint's frames default to the identity, and three
numbers stand for a point wherever a frame or a point is expected. The
settings classes take their settings by keyword, and a misspelt one is
refused.

<!-- example: python-run -->
```python
sim = mbd.Simulator(system)
sim.method = mbd.Integrator.DormandPrince     # to a tolerance; the default is RK4 at a fixed step
sim.initialize()                              # from the state as set: here q = 0 and v = 0
run = sim.run_recorded(2.0, 0.01)             # NumPy arrays: run["time"], run["q"], run["v"]
```

`sim.q` and `sim.v` are the engine's own state: write `sim.q[0] = 0.3`
before `initialize()` to start elsewhere. `run_recorded` returns the state
at the start and after every step. Released at the spring's free length,
the block swings about its rest at 5 rad/s, as the test checks against
q(t) = -(m g / k)(1 - cos 5 t).

## Find the rest and the modes

<!-- example: python-rest-and-modes -->
```python
sim = mbd.Simulator(system)
rest = mbd.static_equilibrium(sim)            # moves the state to rest; print(rest.summary(system.model))
linear = mbd.linearize(sim)                   # about the state the simulator holds
for mode in linear.modes:
    print(f"{mode.natural_frequency:.4f} rad/s, damping ratio {mode.damping_ratio:.3f}")
```

The block comes to rest 0.392 m lower (m g / k), and its one mode is at
5 rad/s, undamped. Each report prints a line of what matters; its
`summary(model)` gives the whole text.

## Write a force in Python

A damper of 4 N s/m written in Python. The function gets the bodies' world
states and adds forces through the handle it is given, valid during the
call only:

<!-- example: python-user-force -->
```python
def drag(states, forces):                     # called at every evaluation of the forces
    body = states[block]                      # the bodies' world states; index 0 is the ground
    forces.add_force(block, -4.0 * body.v_WB, body.p_WB)     # [N], world axes, at the body's origin

system.add_force_element(mbd.UserForce("drag", [block], drag))
```

The linearised mode then has the damping ratio c / (2 sqrt(k m)) = 0.2.
Each call goes through Python, so use the C++ elements where one exists. A
force on joint coordinates goes through `sim.force_callback`, which must
read the state being evaluated from `sim.states()`, not from `sim.q`
([the Python interface](../../python.md#python-code-that-the-engine-calls)).
A model with a force written in Python cannot be saved to a file
(MBD-I008).

## Save a model and a run

Model files are JSON ([the format](../../model-files.md)); results are
pandas tables, saved as Parquet or CSV (pandas and pyarrow needed):

<!-- example: python-files -->
```python
mbd.save_model(system, "block.json")          # JSON, as docs/model-files.md describes
system = mbd.load_model("block.json")
table = mbd.dataframe(run, sim)               # indexed by time; table["q"]["block"], table[("v", "block")]
mbd.write_results(table, "run.parquet")       # or "run.csv"; both keep every number exactly
table = mbd.read_results("run.parquet")
```

The model read back runs bit for bit as the original, and the table read
back holds the run's numbers exactly. A misspelt key in a model file is an
error (MBD-I004), and each error names its place in the file.

## Play a run back

<!-- example: python-replay -->
```python
mbd.replay(system, run, name="block", path="block.rrd")     # without path: the viewer itself
```

Without `path`, the Rerun viewer is shown: below the cell in a notebook
(with `rerun-notebook` installed), in a window of its own elsewhere or with
`spawn=True`. The body is drawn as the box of its inertia, the spring as a
line; q and v are plotted beside the view, on one time line. `channels=`
adds a table's columns to the plots, and `mbd.plot(table)` plots a table
alone ([viewing](../../python.md#viewing-playback-and-plots)).

## Catch the engine's errors and warnings

<!-- example: python-errors -->
```python
try:
    options = mbd.StaticOptions(max_iteration=50)          # misspelt: max_iterations
except mbd.MbdError as error:
    code = error.code                     # "MBD-P010"; docs/help/messages.md explains each code
warnings.simplefilter("error", mbd.MbdWarning)    # the engine's warnings raised as errors from here on
```

Every error of the engine is an `mbd.MbdError` with its code in `.code`, and
every warning an `mbd.MbdWarning`, so Python's warning filters apply: with
the filter above, the warning MBD-K050 (redundant constraint equations) of
a closed planar loop stops its run. [The message catalogue](../messages.md)
explains each code.
