"""State elements written in Python, and the step trace (task 6.1, D52)."""

import math

import pytest

import multibody as mbd
from models import G, pendulum, slider

TAU = 0.2           # the lag's time constant [s]


class Lag(mbd.StateElement):
    """x' = (u - x) / TAU, from x = 0: a first-order lag towards u."""

    def __init__(self, u, cap=None):
        super().__init__()
        self.u, self.cap, self.steps = u, cap, 0

    def name(self):
        return "lag"

    def size(self):
        return 1

    def initial_states(self, x):
        x[0] = 0.0

    def rates(self, sim, t, x, x_dot):
        x_dot[0] = (self.u - x[0]) / TAU

    def end_step(self, sim, t, x):
        self.steps += 1
        if self.cap is not None and x[0] > self.cap:
            x[0] = self.cap


def test_a_lag_written_in_python_follows_its_exponential():
    # x(t) = u (1 - e^(-t / TAU)). RK4 at 1 ms, h / TAU = 0.005: its global
    # error is of the order (h / TAU)^4 / 120 = 5e-12 of x per time constant,
    # at most about 3e-11 here (2.5 time constants, x up to 2); measured
    # 2.1e-12 at 0.5 s; allowed 1e-10.
    lag = Lag(u=2.0)
    sim = mbd.Simulator(slider(1.0))
    sim.attach(lag)
    assert sim.x.shape == (1,) and sim.x[0] == 0.0
    sim.initialize()
    sim.run(0.5, 1e-3)
    assert abs(sim.x[0] - 2.0 * (1.0 - math.exp(-0.5 / TAU))) < 1e-10
    assert lag.steps == 500


def test_end_step_may_change_the_states():
    # Capped at 0.5 after every step: the lag towards 2 reaches 0.5 where
    # 2 (1 - e^(-t / TAU)) = 0.5, at t = TAU ln(4 / 3) = 0.0575 s, between the
    # samples at 57 and 58 ms, and stays there, each step's overshoot cut
    # back. (A first draft put the crossing at TAU ln 4; this test caught it.)
    sim = mbd.Simulator(slider(1.0))
    sim.attach(Lag(u=2.0, cap=0.5))
    sim.initialize()
    result = sim.run_recorded(0.5, 1e-3)
    x = result["x"][:, 0]
    assert x.max() == 0.5
    assert x[57] < 0.5 and x[58] == 0.5 and x[-1] == 0.5


def test_a_state_element_missing_a_method_is_named():
    class Incomplete(mbd.StateElement):
        def __init__(self):
            super().__init__()

        def name(self):
            return "incomplete"

        def size(self):
            return 1

        def initial_states(self, x):
            x[0] = 0.0

    sim = mbd.Simulator(slider(1.0))
    sim.attach(Incomplete())
    with pytest.raises(mbd.MbdError) as caught:
        sim.initialize()
        sim.step(1e-3)
    assert caught.value.code == "MBD-P007" and "rates()" in str(caught.value)


def test_the_trace_keeps_the_pendulums_energy():
    # The pendulum of models.py released at 1 rad from rest has energy
    # -g L cos 1 throughout (L = 0.8 m, 1 kg). RK4 at 1 ms, w h = 0.0035 with
    # w = sqrt(g / L), keeps it to the order of (w h)^4 = 1.5e-10 of its 4.2
    # J, 6e-10 J; measured 1.0e-12 J over 0.5 s; allowed 1e-9 J.
    sim = mbd.Simulator(pendulum())
    sim.trace_steps = True
    sim.q[0] = 1.0
    sim.initialize()
    sim.run(0.5, 1e-3)
    trace = sim.trace()
    assert len(trace) == 500
    energy = -G * 0.8 * math.cos(1.0)
    assert max(abs(r.kinetic_energy + r.potential_energy - energy) for r in trace) < 1e-9
    assert all(r.dt == pytest.approx(1e-3) for r in trace)
