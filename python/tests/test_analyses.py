"""The analyses from Python: initial conditions, statics, linearisation,
kinematic analysis, events, the recorder, the force callback, loads,
energies and checks (task 6.1, D52)."""

import math
import warnings

import numpy as np
import pytest

import multibody as mbd
from models import G, parallelogram, pendulum, slider, vertical_spring

M, K = 2.0, 50.0            # a slider's mass [kg] and its spring [N/m]


@pytest.fixture(autouse=True)
def quiet_redundancy():
    # The parallelogram's loop is planar under a spatial hinge: its redundant
    # equations give MBD-K050 at every solve (test_constraints_and_forces.py
    # checks that warning).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MBD-K050", category=mbd.MbdWarning)
        yield


def test_assembly_closes_the_loop_with_the_crank_held():
    # Given an open loop, crank 1.0, coupler -0.9, rocker 1.1, and the crank
    # held: the parallelogram nearest closes at (1, -1, 1) (see models.py),
    # with one degree of freedom, to the assembly's tolerance of 1e-10 m.
    system = parallelogram()
    sim = mbd.Simulator(system)
    sim.q[:] = [1.0, -0.9, 1.1]
    options = mbd.AssemblyOptions().hold_joint(system.model, 1)
    report = mbd.assemble(sim, options)
    assert report.ok() and report.degrees_of_freedom == 1
    assert report.held_positions == [0]
    assert np.allclose(sim.q, [1.0, -1.0, 1.0], rtol=0, atol=1e-9)
    assert report.position_residual <= 1e-10
    assert "crank" in report.summary(system.model)


def test_statics_finds_where_the_spring_carries_the_weight():
    # A mass M on a spring K under gravity rests at q = -M g / K = -0.3924 m.
    # Statics stops when the acceleration at rest is below 1e-6 m/s^2, that
    # is within 1e-6 / (K / M) = 4e-8 m of it; allowed 1e-7. (Measured:
    # 5e-13 m, after 20 Newton steps, each limited to max_translation = 0.02
    # m on the way to 0.392 m.)
    system = slider(M, gravity=G)
    vertical_spring(system, K)
    sim = mbd.Simulator(system)
    report = mbd.static_equilibrium(sim)
    assert report.converged
    assert abs(sim.q[0] + M * G / K) < 1e-7
    assert report.degrees_of_freedom == 1 and report.supported == 1


def test_linearisation_gives_the_oscillators_frequency_and_damping():
    # Mass M, spring K, damper C: w = sqrt(K / M) = 5 rad/s and a damping
    # ratio C / (2 sqrt(K M)) = 0.2. The model is linear, so the central
    # differences of the linearisation are exact but for rounding: allowed
    # 1e-8 relative.
    C = 4.0
    system = slider(M)
    vertical_spring(system, K, C)
    linear = mbd.linearize(mbd.Simulator(system))
    assert linear.degrees_of_freedom == 1 and len(linear.modes) == 1
    mode = linear.modes[0]
    assert mode.oscillatory()
    assert mode.natural_frequency == pytest.approx(5.0, rel=1e-8)
    assert mode.damping_ratio == pytest.approx(0.2, rel=1e-8)
    assert mode.eigenvalue.real == pytest.approx(-1.0, rel=1e-8)     # -zeta w
    assert linear.A.shape == (2, 2)
    assert mode.energy_fraction[0] == pytest.approx(1.0)


def test_kinematic_analysis_follows_a_driven_point():
    # A driver holds the slider's origin at height 0 + t: q = t, dq/dt = 1,
    # d2q/dt2 = 0, at the solve's tolerance of 1e-10.
    system = slider(M)
    system.add_constraint(mbd.point_height_driver(1, [0.0, 0.0, 0.0], 2, 0.0))
    kinematics = mbd.Kinematics(system)
    kinematics.t = 0.3
    assert kinematics.solve()
    assert abs(kinematics.q[0] - 0.3) < 1e-10
    assert kinematics.velocities() == pytest.approx([1.0], abs=1e-12)
    assert kinematics.accelerations() == pytest.approx([0.0], abs=1e-9)
    assert kinematics.state(1).p_WB[2] == pytest.approx(0.3, abs=1e-10)


def test_an_event_bounces_a_falling_mass_and_the_recorder_keeps_its_height():
    # Dropped from 1 m, the mass reaches the ground at t* = sqrt(2 / g) =
    # 0.45152 s; the event (falling crossings of the height) reverses its
    # velocity there, so it rises with g t*. RK4 is exact on the parabola, so
    # the crossing is found to the event tolerance, 1e-10 s; allowed 1e-9 s.
    # At 0.6 s the height is g t* (t - t*) - g (t - t*)^2 / 2.
    sim = mbd.Simulator(slider(M, gravity=G))
    sim.q[0] = 1.0

    def bounce(s):
        s.v[0] = -s.v[0]

    sim.add_event("ground", lambda s: s.q[0], bounce, direction=-1)
    recorder = mbd.Recorder()
    recorder.add("height", lambda: sim.q[0])
    sim.initialize()
    recorder.sample(sim.time)
    for _ in range(600):
        sim.step(1e-3)
        recorder.sample(sim.time)
    t_star = math.sqrt(2.0 / G)
    log = sim.event_log()
    assert len(log) == 1 and log[0].event == 0
    assert abs(log[0].time - t_star) < 1e-9
    dt = 0.6 - t_star
    assert abs(sim.q[0] - (G * t_star * dt - 0.5 * G * dt * dt)) < 1e-8
    data = recorder.as_dict()
    assert data["time"].shape == data["height"].shape == (601,)
    assert data["height"].min() > 0.0         # never below the ground at a sample
    assert recorder.names() == ["height"] and sim.event_names() == ["ground"]


def test_the_force_callback_adds_to_tau_in_place():
    # tau += -K q through the callback is the spring of K: the oscillator of
    # test_constraints_and_forces.py, x(t) = A cos(w t), to the same 1e-11.
    # The callback reads the state being evaluated from the bodies' states:
    # sim.q holds the step's start, not the integrator's stage (reading it
    # gave a constant force here, the first draft of this test).
    sim = mbd.Simulator(slider(M))

    def spring(s, t, tau):
        tau[0] += -K * s.states()[1].p_WB[2]

    sim.force_callback = spring
    assert sim.force_callback
    sim.method = mbd.Integrator.DormandPrince
    sim.step_control.relative_tolerance = 1e-12
    sim.step_control.absolute_tolerance = 1e-12
    sim.q[0] = 0.1
    sim.initialize()
    sim.step(1.3)
    w = math.sqrt(K / M)
    assert abs(sim.q[0] - 0.1 * math.cos(w * 1.3)) < 1e-11
    sim.force_callback = None
    assert not sim.force_callback


def test_energies_and_the_joint_reaction_of_a_pendulum():
    # At angle q and rate w the bob (1 kg, 0.8 m) has kinetic energy
    # (0.8 w)^2 / 2 and potential energy g y = -g 0.8 cos q. Hanging at rest,
    # its hinge holds it up: the joint passes it a force of g upwards and no
    # moment, in the joint frame (aligned with the world at q = 0).
    sim = mbd.Simulator(pendulum())
    sim.q[0], sim.v[0] = 0.3, 2.0
    sim.refresh()
    assert mbd.kinetic_energy(sim) == pytest.approx(0.5 * (0.8 * 2.0) ** 2, rel=1e-14)
    assert mbd.potential_energy(sim) == pytest.approx(-G * 0.8 * math.cos(0.3), rel=1e-14)
    sim.q[0], sim.v[0] = 0.0, 0.0
    sim.refresh()
    loads = mbd.compute_loads(sim)
    assert np.allclose(loads.v_dot, 0.0, atol=1e-12)
    assert np.allclose(loads.joint_reaction[1], [0.0, 0.0, 0.0, 0.0, G, 0.0], atol=1e-12)


def test_the_checks_count_the_parallelograms_freedom():
    # Three hinges, one loop of 5 equations of which 2 are independent in the
    # plane: 3 - 2 = 1 degree of freedom, 3 redundant equations.
    system = parallelogram()
    report = mbd.validate(system, np.array([1.0, -1.0, 1.0]))
    assert report.errors == []
    assert report.constraint_equations == 5 and report.independent_equations == 2
    assert report.redundant_equations == 3 and report.degrees_of_freedom == 1
    text = mbd.describe(system)
    assert "crank" in text and "revolute" in text
    sim = mbd.Simulator(system)
    sim.q[:] = [1.0, -1.0, 1.0]
    sim.initialize()
    assert "rocker" in mbd.dump_state(sim)
