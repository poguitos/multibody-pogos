"""A pendulum built and simulated from Python (plan task 6.1's condition)."""

import math

import numpy as np

import multibody as mbd

from models import G, complete_elliptic_k, pendulum

L = 0.8        # the pendulum's length [m] (models.pendulum's default)


def test_the_period_at_a_large_amplitude_is_the_elliptic_integral():
    # Released at rest from theta0 = 1 rad, the pendulum's period is
    # T = 4 K(k) / w0 with k = sin(theta0 / 2) and w0 = sqrt(g / L): 1.0663
    # times the small-angle period here. A quarter period later it passes
    # through the bottom, half a period later it is at -theta0, and after a
    # whole one back at theta0. Dormand-Prince ends its own steps exactly at
    # each call's end, so step(T / 4) samples those instants. At tolerances of
    # 1e-12 the angle there was measured within 2.9e-12 rad of the closed form
    # and the speed within 1.7e-12 m/s; at 1e-10, within 2.9e-10 rad and
    # 1.7e-10 m/s (the error follows the tolerance). Allowed 3e-11: ten times
    # the measurement, and ten times below what tolerances of 1e-10 give, so
    # the check also shows that the tolerances set here reach the integrator.
    # A small-angle period would miss by 0.0663 theta0.
    theta0 = 1.0
    w0 = math.sqrt(G / L)
    T = 4.0 * complete_elliptic_k(math.sin(theta0 / 2.0)) / w0
    sim = mbd.Simulator(pendulum())
    sim.method = mbd.Integrator.DormandPrince
    sim.step_control.relative_tolerance = 1e-12
    sim.step_control.absolute_tolerance = 1e-12
    sim.q[0] = theta0
    sim.initialize()
    expected = [0.0, -theta0, 0.0, theta0]
    for k in range(4):
        sim.step(T / 4.0)
        assert abs(sim.q[0] - expected[k]) < 3e-11, (k, sim.q[0])
    assert abs(sim.time - T) < 1e-12 * T
    # Through the bottom the speed is the energy's: v^2 = 2 g L (1 - cos theta0).
    sim.q[0] = theta0
    sim.v[0] = 0.0
    sim.time = 0.0
    sim.step(T / 4.0)
    speed = abs(sim.v[0]) * L
    assert abs(speed - math.sqrt(2.0 * G * L * (1.0 - math.cos(theta0)))) < 3e-11


def test_the_bob_is_where_the_angle_puts_it():
    # The bob at angle q about Z from hanging: (L sin q, -L cos q, 0).
    sim = mbd.Simulator(pendulum())
    sim.q[0] = 0.3
    sim.refresh()
    bob = sim.states()[1]
    assert np.allclose(bob.p_WB, [L * math.sin(0.3), -L * math.cos(0.3), 0.0], rtol=0, atol=1e-15)


def test_run_recorded_takes_the_same_steps_as_run():
    # run_recorded is run() with the states kept: the same calls in the same
    # order, so its last row is the state run() reaches, to the last bit.
    a = mbd.Simulator(pendulum())
    b = mbd.Simulator(pendulum())
    for sim in (a, b):
        sim.q[0] = 0.4
        sim.initialize()
    steps = a.run(0.5, 1e-3)
    result = b.run_recorded(0.5, 1e-3)
    assert steps == 500
    assert result["time"].shape == (501,)
    assert result["q"].shape == (501, 1) and result["v"].shape == (501, 1) and result["x"].shape == (501, 0)
    assert result["time"][0] == 0.0 and result["q"][0, 0] == 0.4
    assert result["q"][-1, 0] == a.q[0] and result["v"][-1, 0] == a.v[0] and result["time"][-1] == a.time
