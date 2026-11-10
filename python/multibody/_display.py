"""How the engine's objects show at the prompt (D54).

Plain records (an inertia, a body's state, the integrator's counts) show as a
call listing their fields; reports and objects that hold a model or a state
show as ``<Name: what matters in it>``. Each function here is installed as
its class's ``__repr__`` by ``install()``, which ``multibody/__init__.py``
calls once.
"""

import numpy as np


def short(value):
    """A value as it reads in a repr: arrays on one line, their numbers to six
    digits without padding, abridged past eight entries; everything else as
    its own repr (a float's reads back exactly)."""
    if isinstance(value, np.ndarray):
        text = np.array2string(value, separator=", ", threshold=8, edgeitems=3, max_line_width=10**9,
                               formatter={"float_kind": lambda x: f"{x:.6g}"})
        return text.replace("\n", "")
    return repr(value)


def plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


def fields(obj, names):
    return f"{type(obj).__name__}(" + ", ".join(f"{n}={short(getattr(obj, n))}" for n in names) + ")"


def record(*names):
    """A repr listing the named fields, as a call."""
    return lambda self: fields(self, names)


def names_list(names, limit=8):
    shown = ", ".join(names[:limit])
    return shown + (f", ... ({len(names) - limit} more)" if len(names) > limit else "")


# --- Objects that hold a model or a state --------------------------------------

def model_repr(model):
    n = model.nbodies() - 1
    names = [name or f"body {b}" for b, name in enumerate(model.name) if b > 0]
    listed = f" ({names_list(names)})" if names else ""
    return (f"<Model: {plural(n, 'body', 'bodies')} and the ground{listed}; nq = {model.nq}, nv = {model.nv}; "
            f"gravity {short(model.gravity)}>")


def system_repr(system):
    n = system.model.nbodies() - 1
    return (f"<System: {plural(n, 'body', 'bodies')} and the ground, {plural(len(system.constraints), 'constraint')}, "
            f"{plural(len(system.force_elements), 'force element')}, {plural(len(system.joint_forces), 'joint force')}>")


def simulator_repr(sim):
    return (f"<Simulator: t = {sim.time:.9g} s, {sim.method.name}; {plural(len(sim.q), 'coordinate')}, "
            f"{plural(len(sim.v), 'velocity', 'velocities')}, {plural(len(sim.x), 'state')}>")


def kinematics_repr(k):
    return f"<Kinematics: t = {k.t:.9g}, q = {short(k.q)}>"


def recorder_repr(r):
    names = list(r.names())
    return f"<Recorder: {plural(len(names), 'channel')} ({names_list(names)}), {plural(r.samples(), 'sample')}>"


def element_repr(element):
    """A force element or a joint force: its kind, its name, its bodies."""
    return f"<{type(element).__name__}: {element.name()}, bodies {list(element.bodies())}>"


def state_element_repr(element):
    # name() and size() may be Python methods of a subclass, possibly missing.
    try:
        return f"<{type(element).__name__}: {element.name()}, {plural(element.size(), 'state')}>"
    except Exception:
        return f"<{type(element).__name__}>"


# --- Reports -------------------------------------------------------------------

def assembly_repr(r):
    parts = ["ok" if r.ok() else "not ok"]
    if r.positions_assembled:
        parts.append(f"position residual {r.position_residual:.3g} (given {r.position_residual_given:.3g})")
    else:
        parts.append("positions not assembled")
        if len(r.conflicting_equations):
            parts.append(f"conflicting equations {list(r.conflicting_equations)}")
    parts.append(f"velocity residual {r.velocity_residual:.3g} (given {r.velocity_residual_given:.3g})"
                 if r.velocities_assembled else "velocities not assembled")
    parts.append(plural(r.degrees_of_freedom, "degree of freedom", "degrees of freedom"))
    parts.append(plural(len(r.warnings), "warning"))
    return "<AssemblyReport: " + "; ".join(parts) + ">"


def static_repr(r):
    head = (f"converged in {plural(r.iterations, 'iteration')}" if r.converged
            else f"not converged after {plural(r.iterations, 'iteration')}")
    return (f"<StaticReport: {head}, residual {r.residual:.3g} (given {r.residual_given:.3g}); "
            f"{plural(r.degrees_of_freedom, 'degree of freedom', 'degrees of freedom')}; "
            f"{plural(len(r.warnings), 'warning')}>")


def mode_text(mode):
    if mode.oscillatory():
        return (f"{mode.natural_frequency:.4g} rad/s ({mode.natural_frequency / (2 * np.pi):.4g} Hz), "
                f"damping ratio {mode.damping_ratio:.3g}")
    return f"real eigenvalue {mode.eigenvalue.real:.4g} 1/s"


def mode_repr(mode):
    return f"<Mode: {mode_text(mode)}>"


def linear_repr(lm):
    modes = list(lm.modes)
    listed = "; ".join(mode_text(mode) for mode in modes[:4]) + ("; ..." if len(modes) > 4 else "")
    return (f"<LinearModel: {plural(lm.degrees_of_freedom, 'degree of freedom', 'degrees of freedom')} at "
            f"t = {lm.time:.9g} s; {plural(len(modes), 'mode')}{': ' + listed if modes else ''}; "
            f"{plural(len(lm.warnings), 'warning')}>")


def validation_repr(r):
    return (f"<ValidationReport: {plural(len(r.errors), 'error')}, {plural(len(r.warnings), 'warning')}, "
            f"{plural(len(r.notes), 'note')}; {plural(r.bodies, 'body', 'bodies')}, "
            f"{plural(r.coordinates, 'coordinate')}, {plural(r.constraint_equations, 'constraint equation')} "
            f"({r.independent_equations} independent, {r.redundant_equations} redundant), "
            f"{plural(r.degrees_of_freedom, 'degree of freedom', 'degrees of freedom')}>")


def loads_repr(loads):
    return (f"<Loads: v_dot = {short(loads.v_dot)}, lambda_ = {short(loads.lambda_)}; constraint wrenches and "
            f"joint reactions of {plural(len(loads.joint_reaction), 'body', 'bodies')}>")


def curve_repr(curve):
    # A curve's points are not readable from Python yet (task 6.4 adds them,
    # for files): its value and slope at 0 say what can be said.
    if curve.is_zero():
        return "<Curve: zero>"
    return f"<Curve: f(0) = {curve.value(0.0):.6g}, f'(0) = {curve.slope(0.0):.6g}>"


def install(m):
    """Install the reprs on the classes of module m (multibody._mbd)."""
    reprs = {
        m.RigidBodyInertia: record("mass", "com_B", "I_com_B"),
        m.RigidBodyState: record("p_WB", "q_WB", "v_WB", "w_WB"),
        m.Marker: record("body", "X_BM"),
        m.LawValue: record("force", "d_position", "d_rate"),
        m.IntegratorStats: record("steps", "rejected", "evaluations", "step_size", "error", "newton_iterations",
                                  "jacobians", "state_evaluations"),
        m.ConstraintDrift: record("position", "velocity", "position_residual", "velocity_residual",
                                  "max_position", "max_velocity", "max_position_residual", "max_velocity_residual"),
        m.Simulator.FiredEvent: record("event", "time"),
        m.Simulator.StepRecord: record("time", "dt", "drift", "velocity_drift", "position_residual",
                                       "velocity_residual", "projection_iterations", "projection_converged",
                                       "equations", "rank", "kinetic_energy", "potential_energy", "events"),
        m.Model: model_repr,
        m.System: system_repr,
        m.Simulator: simulator_repr,
        m.Kinematics: kinematics_repr,
        m.Recorder: recorder_repr,
        m.ForceElement: element_repr,
        m.JointForce: element_repr,
        m.StateElement: state_element_repr,
        m.AssemblyReport: assembly_repr,
        m.StaticReport: static_repr,
        m.Mode: mode_repr,
        m.LinearModel: linear_repr,
        m.ValidationReport: validation_repr,
        m.Loads: loads_repr,
        m.Curve: curve_repr,
        m.TimeFunction: lambda self: "<TimeFunction: s(t) with its rate and acceleration>",
        m.BodyForces: lambda self: "<BodyForces: valid during its UserForce's call>",
    }
    for cls, function in reprs.items():
        cls.__repr__ = function
