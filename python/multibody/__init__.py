"""multibody: multibody dynamics for vehicles, from Python (plan Phase 6).

The classes and functions of the C++ engine, with the same names. A minimal
pendulum::

    import numpy as np
    import multibody as mbd

    system = mbd.System()
    bob = system.model.add_body(
        0, mbd.RevoluteJointModel(), mbd.Transform3(),
        mbd.Transform3.from_translation([0.0, 0.0, -1.0]),
        mbd.RigidBodyInertia(1.0, np.zeros(3), np.zeros((3, 3))), "bob")
    sim = mbd.Simulator(system)
    sim.q[0] = 0.5
    sim.initialize()
    sim.run(1.0, 1e-3)

Errors of the engine raise MbdError and its warnings are MbdWarning, both
with the message's code in ``.code``. Arrays are NumPy arrays: the state
(``sim.q``, ``sim.v``, ``sim.x``, ``sim.tau``) is a view of the simulator's
own memory, so writing an element writes the simulator's state.
"""

from ._errors import MbdError, MbdWarning
from ._mbd import *  # noqa: F401,F403
from ._mbd import __version__  # noqa: F401
