"""Constraints, drivers and force elements from Python (task 6.1, D52)."""

import math

import numpy as np
import pytest

import multibody as mbd
import models
from models import parallelogram

M, K = 2.0, 50.0            # the oscillator's mass [kg] and stiffness [N/m]
W = math.sqrt(K / M)        # its frequency: 5 rad/s
A = 0.1                     # its amplitude [m]


def slider():
    """The slider of models.py, of mass M, without gravity."""
    return models.slider(M)


def linear_spring(system):
    models.vertical_spring(system, K)


def joint_spring(system):
    params = mbd.JointCoordinateForceParams()
    params.spring = mbd.Curve.linear(K)
    system.add_joint_force(mbd.JointCoordinateForce(system.model, 1, params))


def python_spring(system):
    def spring(states, forces):
        forces.add_force(1, [0.0, 0.0, -K * states[1].p_WB[2]])
    system.add_force_element(mbd.UserForce("python spring", [1], spring))


@pytest.mark.parametrize("add_spring", [linear_spring, joint_spring, python_spring])
def test_each_spring_makes_the_same_oscillator(add_spring):
    # Released at rest from A, x(t) = A cos(W t). At tolerances of 1e-12
    # over 1.3 s (a turn and a twentieth), Dormand-Prince ended 8.4e-13 m and
    # 7.6e-13 m/s from it, the same for the three springs to two digits;
    # allowed 1e-11.
    system = slider()
    add_spring(system)
    sim = mbd.Simulator(system)
    sim.method = mbd.Integrator.DormandPrince
    sim.step_control.relative_tolerance = 1e-12
    sim.step_control.absolute_tolerance = 1e-12
    sim.q[0] = A
    sim.initialize()
    sim.step(1.3)
    assert abs(sim.q[0] - A * math.cos(W * 1.3)) < 1e-11
    assert abs(sim.v[0] + A * W * math.sin(W * 1.3)) < 1e-11


def test_a_parallelogram_stays_a_parallelogram():
    # With the crank at theta, the closed loop has the coupler level (its
    # joint at -theta) and the rocker parallel to the crank (its joint at
    # +theta), whatever the motion: q = (theta, -theta, theta). Released at
    # 1 rad above the horizontal and run for 1 s, it swings down and up to
    # -(pi + 1), symmetric about the bottom (a range of 5.14 rad).
    # How far it may leave that branch follows from the projection's
    # tolerance on the loop, r = 1e-10 m. With phi the coupler's angle
    # (q0 + q1) and psi the rocker's (q0 + q1 + q2), the loop's Jacobian in
    # (phi, psi) on the branch is [[0, a sin theta], [d, -a cos theta]], of
    # determinant -a d sin theta, singular at the change points (theta = 0
    # and -pi, the crank along the ground link):
    # - where |sin theta| >= 0.5 its smaller singular value is at least
    #   0.14 m, so r bounds (phi, psi) at 7e-10 rad and q2 - q0 = psi - phi -
    #   theta at 1.0e-9 rad; allowed 3e-9;
    # - through a change point a departure phi costs only
    #   d (d - a) / (2 a) phi^2 of closure, so r bounds phi at
    #   sqrt(2 a r / (d (d - a))) = 1.2e-5 rad and q2 - q0 = (d / a - 1) phi
    #   at 2.0e-5 rad; allowed 5e-5. The antiparallelogram meets this branch
    #   there; a switch to it would show away from them, as a departure of
    #   order 1.
    # (Measured: on Windows 1.9e-15 rad at theta = -0.009 and 2.2e-16 where
    # |sin theta| >= 0.5, the loop closed to 3.9e-16 m; on Linux 4.0e-9 rad,
    # which failed a bound of 3e-9 on the whole run, derived without the
    # change points.)
    # A planar loop closed by a spatial hinge has redundant equations, which
    # the engine reports as warning MBD-K050, here a Python warning.
    sim = mbd.Simulator(parallelogram())
    sim.q[:] = [1.0, -1.0, 1.0]
    with pytest.warns(mbd.MbdWarning) as caught:
        sim.initialize()
        result = sim.run_recorded(1.0, 1e-3)
    assert any(w.message.code == "MBD-K050" for w in caught)
    q = result["q"]
    far = np.abs(np.sin(q[:, 0])) >= 0.5
    for departure in (np.abs(q[:, 0] + q[:, 1]), np.abs(q[:, 2] - q[:, 0])):
        worst = int(np.argmax(departure))
        assert departure[worst] < 5e-5, f"{departure[worst]:.3e} at theta = {q[worst, 0]:.4f}"
        assert np.max(departure[far]) < 3e-9
    assert abs(np.ptp(q[:, 0]) - (math.pi + 2.0)) < 1e-3   # from 1 to -(pi + 1)
    assert sim.constraint_drift().max_position_residual <= 1e-10


def test_a_joint_driven_by_python_functions_follows_them():
    # q(t) = 0.5 sin 2t through a JointDriver whose TimeFunction is three
    # Python callables. The driver is a constraint, held to 1e-10 by the
    # projection after every step; allowed 1e-9.
    system = mbd.System()
    system.model.add_body(0, mbd.RevoluteJointModel(), mbd.Transform3(), mbd.Transform3(),
                          mbd.RigidBodyInertia.from_solid_box(1.0, [0.1, 0.1, 0.1]), "driven")
    s = mbd.TimeFunction(lambda t: 0.5 * math.sin(2.0 * t), lambda t: math.cos(2.0 * t),
                         lambda t: -2.0 * math.sin(2.0 * t))
    system.add_constraint(mbd.JointDriver(system.model, 1, s))
    sim = mbd.Simulator(system)
    sim.initialize()
    sim.run(1.0, 1e-3)
    assert abs(sim.q[0] - 0.5 * math.sin(2.0)) < 1e-9
    assert abs(sim.v[0] - math.cos(2.0)) < 1e-9


def test_lists_are_read_as_tuples_and_extended_by_add():
    system = parallelogram()
    assert len(system.constraints) == 1 and system.constraints[0].size() == 5
    with pytest.raises(TypeError):
        system.constraints[0] = None
    with pytest.raises(TypeError):
        system.add_constraint(None)
    params = mbd.BushingParams()
    params.stiffness = [mbd.Curve.linear(1e5)] * 3
    assert params.stiffness[2](0.001) == pytest.approx(100.0)
    with pytest.raises(TypeError):
        params.stiffness[0] = mbd.Curve()           # a tuple: no silent change of a copy


def test_a_user_forces_handle_is_valid_during_its_call_only():
    kept = []

    def keep(states, forces):
        kept.append(forces)

    system = slider()
    system.add_force_element(mbd.UserForce("keeper", [1], keep))
    sim = mbd.Simulator(system)
    sim.initialize()
    sim.step(1e-3)
    with pytest.raises(mbd.MbdError) as caught:
        kept[0].add_force(1, [1.0, 0.0, 0.0])
    assert caught.value.code == "MBD-P005"

    def wrong_body(states, forces):
        forces.add_force(7, [1.0, 0.0, 0.0])

    system = slider()
    system.add_force_element(mbd.UserForce("wrong", [1], wrong_body))
    sim = mbd.Simulator(system)
    with pytest.raises(mbd.MbdError) as caught:
        sim.initialize()
        sim.step(1e-3)
    assert caught.value.code == "MBD-P006"


def test_an_error_in_a_python_force_reaches_the_caller():
    def broken(states, forces):
        raise ValueError("broken force")

    system = slider()
    system.add_force_element(mbd.UserForce("broken", [1], broken))
    sim = mbd.Simulator(system)
    with pytest.raises(ValueError, match="broken force"):
        sim.initialize()
        sim.step(1e-3)
