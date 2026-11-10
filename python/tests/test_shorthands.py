"""Shorthands for writing models and reading results: the interface as tuned
with the author (D54)."""

import enum
import math
import re

import numpy as np
import pytest

import multibody as mbd
from models import pendulum, slider, vertical_spring


def box(mass=1.0):
    return mbd.RigidBodyInertia.from_solid_box(mass, [0.05, 0.05, 0.05])


def test_joint_frames_default_to_the_identity_and_three_numbers_make_a_translation():
    # The same chain built with explicit transforms and with the shorthands
    # has the same frames, to the bit: the shorthands call the same
    # constructors (the identity, Transform3.from_translation).
    explicit = mbd.Model()
    a = explicit.add_body(0, mbd.RevoluteJointModel(), mbd.Transform3(), mbd.Transform3(), box(), "a")
    explicit.add_body(a, mbd.RevoluteJointModel(), mbd.Transform3.from_translation([0.3, 0.0, 0.0]),
                      mbd.Transform3.from_translation([0.0, 0.1, 0.0]), box(), "b")
    short = mbd.Model()
    a = short.add_body(0, mbd.RevoluteJointModel(), inertia=box(), name="a")
    short.add_body(a, mbd.RevoluteJointModel(), x_pj=[0.3, 0.0, 0.0], x_cj=(0.0, 0.1, 0.0), inertia=box(),
                   name="b")
    for X, Y in zip(explicit.X_PJ + explicit.X_CJ, short.X_PJ + short.X_CJ):
        assert np.array_equal(X.translation, Y.translation)
        assert np.array_equal(X.quaternion, Y.quaternion)


def test_three_numbers_are_a_frame_wherever_one_is_expected():
    marker = mbd.Marker(2, [1.0, 2.0, 3.0])
    assert np.array_equal(marker.X_BM.translation, [1.0, 2.0, 3.0])
    assert np.array_equal(marker.X_BM.quaternion, [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(mbd.Transform3([1.0, 2.0, 3.0]).translation, [1.0, 2.0, 3.0])
    # Anything else is refused, as before: two numbers, or a matrix.
    with pytest.raises(TypeError):
        mbd.Marker(2, [1.0, 2.0])
    with pytest.raises(TypeError):
        mbd.Marker(2, np.eye(3))


def test_bodies_are_found_by_name():
    model = mbd.Model()
    crank = model.add_body("ground", mbd.RevoluteJointModel(), inertia=box(), name="crank")
    coupler = model.add_body("crank", mbd.RevoluteJointModel(), x_pj=[0.3, 0.0, 0.0], inertia=box(),
                             name="coupler")
    assert (crank, coupler) == (1, 2)
    assert model.parent[coupler] == crank
    assert (model.body("ground"), model.body("coupler")) == (0, coupler)
    with pytest.raises(mbd.MbdError) as caught:
        model.body("rocker")
    assert caught.value.code == "MBD-P008"
    assert "'ground', 'crank', 'coupler'" in str(caught.value)
    # A name that two bodies share says nothing: refused, with both indices.
    model.add_body(0, mbd.RevoluteJointModel(), inertia=box(), name="crank")
    with pytest.raises(mbd.MbdError) as caught:
        model.add_body("crank", mbd.RevoluteJointModel(), inertia=box())
    assert caught.value.code == "MBD-P009"
    assert "(1, 3)" in str(caught.value)


def test_settings_are_given_by_keyword_and_the_rest_keep_their_defaults():
    default = mbd.StaticOptions()
    options = mbd.StaticOptions(max_iterations=7, tolerance=1e-9)
    assert (options.max_iterations, options.tolerance) == (7, 1e-9)
    for name in ("perturbation", "max_translation", "max_rotation", "relaxation_step", "max_relaxation_steps",
                 "relaxation_tolerance"):
        assert getattr(options, name) == getattr(default, name)
    # A misspelt name would leave the default in place without a word.
    with pytest.raises(mbd.MbdError) as caught:
        mbd.StaticOptions(max_iteration=7)
    assert caught.value.code == "MBD-P010"
    assert "max_iterations" in str(caught.value)


def test_a_keyword_goes_through_the_settings_own_checks():
    spring = mbd.SpringDamperParams(free_length=0.3, spring=mbd.Curve.linear(3.0e4))
    assert spring.free_length == 0.3
    assert spring.spring.slope(0.1) == 3.0e4
    # A bushing's curves come in threes; two are refused by the setter.
    with pytest.raises(TypeError):
        mbd.BushingParams(stiffness=[mbd.Curve.linear(1.0), mbd.Curve.linear(2.0)])


def test_a_settings_object_shows_the_call_that_builds_it():
    # Python writes a float's repr with the fewest digits that read back
    # exactly, so the rebuilt object equals the original to the bit.
    options = mbd.StepControl(adaptive=True, relative_tolerance=1e-8, max_steps=500)
    rebuilt = eval(repr(options), {"StepControl": mbd.StepControl})
    for name in ("adaptive", "relative_tolerance", "absolute_tolerance", "min_step", "max_steps"):
        assert getattr(rebuilt, name) == getattr(options, name)


def test_objects_show_what_they_hold():
    system = pendulum()
    sim = mbd.Simulator(system)
    assert "1 body and the ground (bob)" in repr(system.model)
    assert "nq = 1, nv = 1" in repr(system.model)
    assert "0 constraints" in repr(system)
    assert f"{sim.method.name}; 1 coordinate, 1 velocity, 0 states" in repr(sim)
    sim.q[0] = 0.3
    report = mbd.static_equilibrium(sim)
    assert report.converged
    assert f"converged in {report.iterations} iteration" in repr(report)
    # A curve shows the call that makes it, its numbers in full (task 6.4).
    assert repr(mbd.Curve()) == "Curve()"
    assert repr(mbd.Curve.linear(3.0e4)) == "Curve.linear(30000.0)"
    assert repr(mbd.Curve.linear(2.0, -0.1)) == "Curve.linear(2.0, offset=-0.1)"
    table = mbd.Curve.table([-0.1, 0.0, 0.3], [-50.0, 0.0, 0.1 + 0.2])
    assert repr(table) == "Curve.table([-0.1, 0.0, 0.3], [-50.0, 0.0, 0.30000000000000004])"
    rebuilt = eval(repr(table), {"Curve": mbd.Curve})
    assert rebuilt.table_x() == table.table_x() and rebuilt.table_y() == table.table_y()


def test_a_mode_shows_its_frequency_and_damping():
    # 2 kg on a spring of 400 N/m and a damper of 10 N s/m: omega =
    # sqrt(k / m) = 14.142 rad/s (2.2508 Hz), zeta = c / (2 sqrt(k m)) =
    # 0.17678. The repr rounds to four and three digits, far coarser than the
    # linearisation's differences.
    mass, k, c = 2.0, 400.0, 10.0
    system = slider(mass)
    vertical_spring(system, k, c)
    mode = mbd.linearize(mbd.Simulator(system)).modes[0]
    omega, zeta = math.sqrt(k / mass), c / (2.0 * math.sqrt(k * mass))
    assert repr(mode) == f"<Mode: {omega:.4g} rad/s ({omega / (2 * math.pi):.4g} Hz), damping ratio {zeta:.3g}>"


def public_classes():
    classes = [value for value in vars(mbd).values()
               if isinstance(value, type) and value.__module__.startswith("multibody")
               and not issubclass(value, BaseException)]
    return classes + [mbd.Simulator.FiredEvent, mbd.Simulator.StepRecord]


def test_every_class_shows_more_than_its_address():
    bare = [cls.__name__ for cls in public_classes()
            if not any("__repr__" in vars(k) for k in cls.__mro__
                       if k.__name__ not in ("object", "pybind11_object"))]
    assert bare == []


def documented(function):
    # pybind11 opens a docstring with the signature (one per overload): a
    # function documented in its binding has text besides.
    signature = re.compile(r"^\s*(\d+\. )?\w+\(.*\) ->")
    lines = (function.__doc__ or "").splitlines()
    return any(line.strip() and not signature.match(line) and line != "Overloaded function." for line in lines)


def test_every_function_says_what_it_does():
    functions = [(name, f) for name, f in vars(mbd).items() if callable(f) and not isinstance(f, type)
                 and not name.startswith("_")]
    for cls in public_classes():
        if issubclass(cls, enum.Enum):
            continue
        functions += [(f"{cls.__name__}.{name}", f) for name, f in vars(cls).items()
                      if callable(f) and not isinstance(f, type) and (name == "__init__" or not name.startswith("_"))]
    assert [name for name, f in functions if not documented(f)] == []
