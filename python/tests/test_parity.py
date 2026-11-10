"""Parity (plan task 6.3, D55): ten C++ tests reproduced through Python, the
numbers identical.

Each test below runs its C++ test case (tests/support/parity.hpp says what
that case records), builds the same model through the package, runs the same
steps, records the same numbers, and requires every one to be the same
double. The C++ case checks its numbers against what it derives (a closed
form, a conservation law, an independent method); identical numbers carry
those checks over to the package. The checks are not repeated here.

The setups are copied expression for expression, since an input one
rounding apart would make every number after it differ: Eigen's norm of a
short vector is the square root of the sum of squares in order, a vector
divided by a scalar is divided entry by entry, and Transform3.from_axis_angle
builds the quaternion as the C++ Quat(AngleAxisd(angle, axis)) does for a
unit axis. Only numbers the engine computes are compared, not the C++ tests'
own arithmetic on them (a speed along a slope, a spring's length).

The C++ test executables are those of the suite's build; where there is
none (an installed package) these tests are skipped."""

import math
import warnings

import numpy as np
import pytest

import multibody as mbd
from parity import Record, assert_identical, cpp_record

G = mbd.g_accel
T3 = mbd.Transform3


@pytest.fixture(autouse=True)
def quiet_redundancy():
    # The four-bar's loop is planar under a spatial hinge: its redundant
    # equations give MBD-K050 at every solve (as in test_analyses.py).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MBD-K050", category=mbd.MbdWarning)
        yield


class FourBar:
    """tests/kernel/four_bar.hpp: a planar four-bar in the XY plane, hinges
    along Z, gravity along -Y; crank AB, coupler BC, rocker CD, ground AD;
    a tree of three revolute joints, closed at D."""

    a, b, c, d = 0.3, 0.8, 0.6, 0.7

    def __init__(self):
        self.system = mbd.System()
        model = self.system.model
        model.gravity = [0.0, -G, 0.0]
        hinge = mbd.RevoluteJointModel()
        I3 = T3()
        self.crank = model.add_body(0, hinge, I3, I3, self.link(self.a, 1.0), "crank")
        self.coupler = model.add_body(self.crank, hinge, T3.from_translation([self.a, 0.0, 0.0]), I3,
                                      self.link(self.b, 2.0), "coupler")
        self.rocker = model.add_body(self.coupler, hinge, T3.from_translation([self.b, 0.0, 0.0]), I3,
                                     self.link(self.c, 1.5), "rocker")
        self.system.add_constraint(mbd.revolute_closure(mbd.Marker(0, T3.from_translation([self.d, 0.0, 0.0])),
                                                        mbd.Marker(self.rocker, T3.from_translation([self.c, 0.0, 0.0]))))

    @staticmethod
    def link(length, mass):
        inertia = mbd.RigidBodyInertia.from_solid_box(mass, [0.5 * length, 0.02, 0.02])
        inertia.com_B = [0.5 * length, 0.0, 0.0]
        return inertia

    @classmethod
    def configuration(cls, th1):
        """The closed loop at crank angle th1: C where the circles about B
        (radius b) and D (radius c) meet, on the upper side."""
        a, b, c, d = cls.a, cls.b, cls.c, cls.d
        Bx, By = a * math.cos(th1), a * math.sin(th1)
        Dx, Dy = d, 0.0
        dist = math.sqrt((Dx - Bx) * (Dx - Bx) + (Dy - By) * (Dy - By))
        along = (b * b - c * c + dist * dist) / (2.0 * dist)
        h = math.sqrt(b * b - along * along)
        ex, ey = (Dx - Bx) / dist, (Dy - By) / dist
        Cx = Bx + along * ex + h * -ey
        Cy = By + along * ey + h * ex
        phi2 = math.atan2(Cy - By, Cx - Bx)
        phi3 = math.atan2(Dy - Cy, Dx - Cx)
        return np.array([th1, phi2 - th1, phi3 - phi2])


def bob():
    """tests/kernel/test_kernel_outputs.cpp: 2 kg, its centre of mass 0.9 m
    down its Z axis."""
    inertia = mbd.RigidBodyInertia.from_solid_box(2.0, [0.05, 0.04, 0.03])
    inertia.com_B = [0.0, 0.0, -0.9]
    return inertia


def z_along_y():
    """Frames whose Z axis is the world's Y."""
    return T3.from_axis_angle([1.0, 0.0, 0.0], -0.5 * mbd.pi)


def test_assembly_of_a_four_bar_given_inconsistent_values():
    cpp = cpp_record("test_kernel", "Assembly: a four-bar given inconsistent values is assembled and reported")
    fb = FourBar()
    model = fb.system.model
    sim = mbd.Simulator(fb.system)
    closed = FourBar.configuration(1.0)
    sim.q[:] = closed + np.array([0.0, 0.05, -0.04])
    sim.v[:] = [2.0, 0.3, -0.5]
    options = mbd.AssemblyOptions()
    options.hold_joint(model, fb.crank)
    r = mbd.assemble(sim, options)

    py = Record()
    py("q", sim.q)
    py("v", sim.v)
    py("smallest_correction", r.smallest_correction)
    py("projection_iterations", r.projection_iterations)
    py("refinement_iterations", r.refinement_iterations)
    py("position_residual_given", r.position_residual_given)
    py("position_residual", r.position_residual)
    py("velocity_residual_given", r.velocity_residual_given)
    py("velocity_residual", r.velocity_residual)
    py("acceleration_residual", r.acceleration_residual)
    py("equations", r.equations)
    py("rank", r.rank)
    py("degrees_of_freedom", r.degrees_of_freedom)
    py("position_hold_rank", r.position_hold_rank)
    py("velocity_hold_rank", r.velocity_hold_rank)
    py("v_dot", r.loads.v_dot)
    py("lambda", r.loads.lambda_)
    assert_identical(py, cpp)


def test_a_limit_stop_holds_a_falling_slider():
    cpp = cpp_record("test_forces", "Force library: a limit stop holds a falling slider at m g / k past the limit")
    m, k = 2.0, 1e5
    system = mbd.System()
    # The C++ slider along Z has frames Quat::FromTwoVectors(Z, Z): the
    # identity exactly (cross product 0, s = 2, w = s / 2 = 1).
    b = system.model.add_body(0, mbd.PrismaticJointModel(), T3(), T3(),
                              mbd.RigidBodyInertia.from_solid_box(m, [0.05, 0.05, 0.05]), "slider")
    p = mbd.JointCoordinateForceParams(lower_limit=-0.1, limit_stiffness=k, limit_damping=200.0)
    system.add_joint_force(mbd.JointCoordinateForce(system.model, b, p))
    sim = mbd.Simulator(system)
    sim.initialize()
    sim.run(2.0, 1e-3)

    py = Record()
    py("q", sim.q)
    py("v", sim.v)
    py("time", sim.time)
    assert_identical(py, cpp)


def test_a_four_bar_on_a_spring_linearised_at_its_equilibrium():
    cpp = cpp_record("test_kernel", "Linearisation: a four-bar on a spring oscillates at sqrt(V'' / M) along its motion")
    fb = FourBar()
    fb.system.add_force_element(mbd.LinearSpringDamper(0, fb.coupler, [0.35, 0.9, 0.0], [0.4, 0.0, 0.0],
                                                       300.0, 0.0, 0.2))
    sim = mbd.Simulator(fb.system)
    sim.q[:] = FourBar.configuration(1.63)
    assert mbd.static_equilibrium(sim).converged
    lm = mbd.linearize(sim)
    mode = lm.modes[0]

    py = Record()
    py("q", sim.q)
    py("residual", lm.residual)
    py("degrees_of_freedom", lm.degrees_of_freedom)
    py("basis", lm.basis)
    py("A", lm.A)
    py("eigenvalue", mode.eigenvalue)
    py("natural_frequency", mode.natural_frequency)
    py("damping_ratio", mode.damping_ratio)
    py("shape", mode.shape)
    py("energy_fraction", mode.energy_fraction)
    assert_identical(py, cpp)


def test_a_bouncing_ball_and_its_impact_events():
    cpp = cpp_record("test_kernel", "Events: a bouncing ball's impact times match the closed form")
    h, e, level = 1.0, 0.8, 0.0
    system = mbd.System()
    system.model.add_body(0, mbd.PrismaticJointModel(), T3(), T3(),
                          mbd.RigidBodyInertia.from_solid_box(1.0, [0.05, 0.05, 0.05]), "ball")
    sim = mbd.Simulator(system)
    sim.q[0] = h
    impacts = []

    def ground(s):
        impacts.append(s.time)
        s.v[0] = -e * s.v[0]

    sim.add_event("ground", lambda s: s.q[0] - level, ground, direction=-1)
    sim.run(3.0, 0.01)

    py = Record()
    for t in impacts:
        py("impact", t)
    for fired in sim.event_log():
        py("event", fired.event)
        py("event_time", fired.time)
    py("q", sim.q)
    py("v", sim.v)
    py("time", sim.time)
    assert_identical(py, cpp)


class Lag(mbd.StateElement):
    """tests/kernel/test_kernel_states.cpp: a force decaying as f' = -f / tau
    from f0, its value at each evaluation kept for the force callback."""

    def __init__(self, f0, tau):
        super().__init__()
        self.f0, self.tau, self.force = f0, tau, 0.0

    def name(self):
        return "lag"

    def size(self):
        return 1

    def initial_states(self, x):
        x[0] = self.f0

    def begin_evaluation(self, sim, t, x):
        self.force = x[0]

    def rates(self, sim, t, x, x_dot):
        x_dot[0] = -x[0] / self.tau


def test_a_lag_written_in_python_pushing_a_mass():
    cpp = cpp_record("test_kernel", "State elements: a lag pushing a mass converges at the integrator's order")
    # PushedBlock: a 2 kg block on a slider along Z, no gravity, pushed by a
    # lag of 3 N decaying in 0.2 s; one lag for every run, as there.
    m, f0, tau = 2.0, 3.0, 0.2
    system = mbd.System()
    system.model.gravity = [0.0, 0.0, 0.0]
    system.model.add_body(0, mbd.PrismaticJointModel(), T3(), T3(),
                          mbd.RigidBodyInertia.from_solid_box(m, [0.1, 0.1, 0.1]), "block")
    lag = Lag(f0, tau)
    py = Record()

    def position(method, dt, T):
        sim = mbd.Simulator(system)
        sim.method = method
        sim.multirate = 1
        sim.attach(lag)

        def push(s, t, generalized):
            generalized[0] += lag.force

        sim.force_callback = push
        sim.run(T, dt)
        py("q", sim.q)
        py("v", sim.v)
        py("x", sim.x)
        py("evaluations", sim.integrator_stats().evaluations)

    position(mbd.Integrator.RK4, 0.02, 1.0)
    position(mbd.Integrator.RK4, 0.01, 1.0)
    position(mbd.Integrator.SemiImplicitEuler, 1e-3, 1.0)
    position(mbd.Integrator.SemiImplicitEuler, 5e-4, 1.0)
    assert_identical(py, cpp)


def test_a_pendulum_held_by_a_distance_constraint():
    cpp = cpp_record("test_kernel", "Kernel simulator: a constrained pendulum keeps its energy and its length")
    system = mbd.System()
    system.model.gravity = [0.0, 0.0, -G]
    system.model.add_body(0, mbd.FreeJointModel(), T3(), T3(), mbd.RigidBodyInertia.from_solid_box(1.0, [0.1, 0.1, 0.1]))
    system.add_constraint(mbd.Distance(mbd.Marker(0, T3()), mbd.Marker(1, T3()), 1.0))
    sim = mbd.Simulator(system)
    sim.q[0:3] = [math.sin(1.0), 0.0, -math.cos(1.0)]
    sim.q[3:7] = [0.0, 0.0, 0.0, 1.0]          # Quat::Identity().coeffs(): x, y, z, w
    sim.initialize()

    py = Record()
    py("q", sim.q)
    py("kinetic_energy", mbd.kinetic_energy(sim))
    py("potential_energy", mbd.potential_energy(sim))
    sim.run(2.0, 1e-3)
    py("q", sim.q)
    py("v", sim.v)
    py("time", sim.time)
    py("kinetic_energy", mbd.kinetic_energy(sim))
    py("potential_energy", mbd.potential_energy(sim))
    py("projection_failures", sim.projection_failures())
    last = sim.last_projection()
    py("last_projection.iterations", last.iterations)
    py("last_projection.initial_position_residual", last.initial_position_residual)
    py("last_projection.position_residual", last.position_residual)
    py("last_projection.initial_velocity_residual", last.initial_velocity_residual)
    py("last_projection.velocity_residual", last.velocity_residual)
    py("last_projection.converged", last.converged)
    assert_identical(py, cpp)


def test_a_pendulums_hinge_reaction():
    cpp = cpp_record("test_kernel", "Outputs: a pendulum's hinge reaction is m a - m g, with no moment")
    system = mbd.System()
    system.model.add_body(0, mbd.RevoluteJointModel(), z_along_y(), z_along_y(), bob(), "bob")
    sim = mbd.Simulator(system)
    sim.q[0] = 1.0
    sim.initialize()

    py = Record()
    for _ in range(6):
        sim.run(0.137, 1e-3)
        loads = mbd.compute_loads(sim)
        py("q", sim.q)
        py("v", sim.v)
        py("v_dot", loads.v_dot)
        py("joint_reaction", loads.joint_reaction[1])
    assert_identical(py, cpp)


def test_a_block_on_an_incline_creeps_and_slides():
    cpp = cpp_record("test_forces", "Contact: a block on an incline creeps below the friction angle and slides above it")
    m, mu_s, mu_d, v_s, v_d, k = 1.0, 0.5, 0.4, 1e-3, 1e-2, 1e5
    friction_angle = math.atan(mu_s)
    py = Record()

    def slide(th, duration, dt_sample):
        system = mbd.System()
        block = system.model.add_body(0, mbd.FreeJointModel(), T3(), T3(),
                                      mbd.RigidBodyInertia.from_solid_box(m, [0.1, 0.1, 0.05]), "block")
        p = mbd.ContactParams(stiffness=k, damping=200.0, damping_depth=1e-5, mu_static=mu_s, mu_dynamic=mu_d,
                              v_static=v_s, v_dynamic=v_d)
        R = T3.from_axis_angle([0.0, 1.0, 0.0], th)
        corners = [[x, y, -0.05] for x in (-0.1, 0.1) for y in (-0.1, 0.1)]
        system.add_force_element(mbd.PlaneContact(block, corners, 0.0, 0, R, p))
        sim = mbd.Simulator(system)
        n = R.rotate([0.0, 0.0, 1.0])
        delta0 = m * G * math.cos(th) / (4.0 * k)
        iq = system.model.idx_q[block]
        sim.q[iq:iq + 3] = (0.05 - delta0) * n
        w, x, y, z = R.quaternion
        sim.q[iq + 3:iq + 7] = [x, y, z, w]       # Eigen's coefficients: x, y, z, w
        py("q_start", sim.q)
        sim.initialize()
        steps = round(duration / 1e-4)
        every = round(dt_sample / 1e-4)
        for i in range(1, steps + 1):
            sim.step(1e-4)
            if i % every == 0:
                py("v_WB", sim.states()[block].v_WB)
        py("q", sim.q)
        py("v", sim.v)

    # The C++ test's two sections, in their order: 1 degree below the
    # friction angle, sampled every 10 ms; 1 degree above, every 0.1 s.
    slide(friction_angle - mbd.pi / 180.0, 0.5, 0.01)
    slide(friction_angle + mbd.pi / 180.0, 0.5, 0.1)
    assert_identical(py, cpp)


def test_springs_with_preloads_on_a_swinging_bob():
    cpp = cpp_record("test_kernel", "Outputs: energy is conserved with springs and preloads")
    system = mbd.System()
    b = system.model.add_body(0, mbd.RevoluteJointModel(), z_along_y(), z_along_y(), bob(), "bob")
    jf = mbd.JointCoordinateForce(system.model, b, mbd.JointCoordinateForceParams(spring=mbd.Curve.linear(25.0),
                                                                                  preload=3.0))
    system.add_joint_force(jf)
    sp = mbd.SpringDamperParams(free_length=1.2, spring=mbd.Curve.linear(150.0), preload=20.0)
    system.add_force_element(mbd.SpringDamper(0, b, [0.8, 0.0, -1.0], [0.0, 0.0, -0.9], sp))
    sim = mbd.Simulator(system)
    sim.q[0] = 0.6
    sim.initialize()
    for _ in range(20000):
        sim.step(1e-4)

    py = Record()
    py("q", sim.q)
    py("v", sim.v)
    py("time", sim.time)
    py("kinetic_energy", mbd.kinetic_energy(sim))
    py("potential_energy", mbd.potential_energy(sim))
    py("joint_spring_energy", jf.potential_energy(sim.q[0]))
    py("p_WB", sim.states()[1].p_WB)
    assert_identical(py, cpp)


class BushedPendulum:
    """tests/kernel/bushed_pendulum.hpp: two 1 kg links of 1 m on free
    joints, held at the hinges by bushings stiff in translation and about X
    and Z, free about Y, damped at a ratio of 0.2."""

    def __init__(self, k=1e8, k_r=1e6):
        self.system = mbd.System()
        model = self.system.model

        def link():
            inertia = mbd.RigidBodyInertia.from_solid_box(1.0, [0.5, 0.025, 0.025])
            inertia.com_B = [0.5, 0.0, 0.0]
            return inertia

        free = mbd.FreeJointModel()
        self.link1 = model.add_body(0, free, T3(), T3(), link(), "link 1")
        self.link2 = model.add_body(0, free, T3(), T3(), link(), "link 2")
        zeta = 0.2
        p = mbd.BushingParams()
        p.stiffness = [mbd.Curve.linear(k)] * 3
        p.damping = [mbd.Curve.linear(2.0 * zeta * math.sqrt(k * 1.0))] * 3
        bending = mbd.Curve.linear(k_r)
        bending_damping = mbd.Curve.linear(2.0 * zeta * math.sqrt(k_r / 3.0))
        p.rotational_stiffness = [bending, mbd.Curve(), bending]          # about X and Z; Y is the hinge
        p.rotational_damping = [bending_damping, mbd.Curve(), bending_damping]
        self.system.add_force_element(mbd.Bushing(0, T3(), self.link1, T3(), p))
        self.system.add_force_element(mbd.Bushing(self.link1, T3.from_translation([1.0, 0.0, 0.0]), self.link2,
                                                  T3(), p))

    def release(self, sim):
        """Both links level along X, at rest."""
        model = self.system.model
        sim.q[:] = model.neutral_configuration()
        sim.q[model.idx_q[self.link2]] = 1.0
        sim.v[:] = 0.0
        sim.refresh()

    def tip(self, sim):
        return sim.states()[self.link2].pose_WB().apply([1.0, 0.0, 0.0])


def test_radau_and_generalised_alpha_on_the_bushed_pendulum():
    cpp = cpp_record("test_kernel", "Radau IIA beats generalised-alpha on the bushed pendulum at tight tolerance")
    bp = BushedPendulum()
    T, target = 1.0, 1e-6
    py = Record()

    def run(method, h, tol, adaptive):
        sim = mbd.Simulator(bp.system)
        sim.method = method
        bp.release(sim)
        sim.step_control.adaptive = adaptive
        sim.step_control.relative_tolerance = tol
        sim.step_control.absolute_tolerance = tol
        sim.run(T, h)
        stats = sim.integrator_stats()
        py("q", sim.q)
        py("v", sim.v)
        py("steps", stats.steps)
        py("rejected", stats.rejected)
        py("evaluations", stats.evaluations)
        py("newton_iterations", stats.newton_iterations)
        py("jacobians", stats.jacobians)
        return bp.tip(sim)

    # The C++ test's sequence: the reference, generalised-alpha's steps
    # halved until it meets the target (its error far from the target at
    # each, so that the choice of runs cannot hinge on a rounding), Radau IIA.
    reference = run(mbd.Integrator.DormandPrince, 0.01, 1e-12, True)
    for h in (5e-4, 2.5e-4, 1.25e-4):
        if np.linalg.norm(run(mbd.Integrator.GeneralizedAlpha, h, 1e-9, False) - reference) <= target:
            break
    run(mbd.Integrator.RadauIIA, 0.01, target, True)
    assert_identical(py, cpp)
