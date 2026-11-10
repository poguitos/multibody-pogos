"""Settings classes built by keyword, and shown as they would be built (D54).

A settings class holds what an analysis or an element reads: tolerances,
iteration limits, a spring's curves. ``mbd.StaticOptions(max_iterations=50)``
starts from the defaults and sets each keyword through the attribute of that
name, so the attribute's own checks apply; a name that is not one of the
class's settings raises MBD-P010, where an attribute set by hand would simply
be refused. The repr lists every setting, as the call that builds it.
"""

from ._display import short
from ._errors import MbdError


def settings(cls):
    """The names of cls's settings: its writable properties, in the order
    they were bound."""
    return [name for klass in reversed(cls.__mro__) for name, value in vars(klass).items()
            if isinstance(value, property) and value.fset is not None and not name.startswith("_")]


def with_keywords(cls):
    """Give cls a constructor taking its settings by keyword, and a repr that
    lists them."""
    base_init = cls.__init__
    names = settings(cls)

    def __init__(self, **keywords):
        base_init(self)
        for name, value in keywords.items():
            if name not in names:
                raise MbdError(f"MBD-P010: {cls.__name__} has no setting '{name}'; its settings are "
                               + ", ".join(names))
            setattr(self, name, value)

    def __repr__(self):
        return f"{cls.__name__}(" + ", ".join(f"{name}={short(getattr(self, name))}" for name in names) + ")"

    __init__.__doc__ = (f"{cls.__name__}(**settings): the defaults, with each keyword setting the one of that "
                        f"name (MBD-P010 for any other). Its settings: {', '.join(names)}.")
    cls.__init__ = __init__
    cls.__repr__ = __repr__


def install(m):
    """Keyword construction for the settings classes of module m
    (multibody._mbd)."""
    for cls in (m.AssemblyOptions, m.StaticOptions, m.LinearizationOptions, m.StepControl, m.ImplicitControl,
                m.SpringDamperParams, m.RotationalSpringDamperParams, m.BushingParams, m.ContactParams,
                m.JointCoordinateForceParams):
        with_keywords(cls)
