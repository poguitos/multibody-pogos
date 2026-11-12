"""multibody: multibody dynamics for vehicles, from Python (plan Phase 6).

The classes and functions of the C++ engine, with the same names. A minimal
pendulum::

    import multibody as mbd

    system = mbd.System()
    system.model.gravity = [0.0, -9.81, 0.0]    # swinging in the XY plane, about Z
    bob = system.model.add_body(
        0, mbd.RevoluteJointModel(), x_cj=[0.0, 1.0, 0.0],   # the hinge 1 m above the bob
        inertia=mbd.RigidBodyInertia.from_solid_box(1.0, [0.02, 0.02, 0.02]), name="bob")
    sim = mbd.Simulator(system)
    sim.q[0] = 0.5
    sim.initialize()
    sim.run(1.0, 1e-3)

Joint frames default to the identity, and a list [x, y, z] stands for a
translation wherever a frame is expected. Settings classes take their
settings by keyword (``mbd.StaticOptions(max_iterations=50)``), and every
object shows what it holds when printed.

Errors of the engine raise MbdError and its warnings are MbdWarning, both
with the message's code in ``.code``. Arrays are NumPy arrays: the state
(``sim.q``, ``sim.v``, ``sim.x``, ``sim.tau``) is a view of the simulator's
own memory, so writing an element writes the simulator's state.
"""

from . import _display, _mbd, _settings
from ._errors import MbdError, MbdWarning
from ._mbd import *  # noqa: F401,F403
from ._mbd import __version__  # noqa: F401
from ._results import coordinate_names, dataframe, read_results, write_results  # noqa: F401
from ._viewer import plot, replay  # noqa: F401


def _install(module):
    """The reprs and keyword constructors, on the classes of the compiled
    module. An editable install reads these Python files from the repository
    but its compiled module from its last install: after a change to the
    bindings the files may name a class the module lacks."""
    try:
        _display.install(module)
        _settings.install(module)
    except AttributeError as missing:
        raise ImportError(f"MBD-P011: multibody's compiled module ({getattr(module, '__file__', '?')}) is older than "
                          f"its Python files ({missing}); install the package again (docs/python.md, \"After a "
                          f"change\")") from missing


_install(_mbd)
