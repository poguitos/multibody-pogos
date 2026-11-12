"""The four-bar of the kernel's tests, held up by a spring, set swinging and
played back in Rerun (plan task 6.5).

    python python/examples/replay_four_bar.py              # opens the viewer
    python python/examples/replay_four_bar.py four_bar.rrd # or saves a recording

A planar four-bar in the XY plane (crank 0.3 m, coupler 0.8 m, rocker 0.6 m,
ground 0.7 m; hinges along Z, gravity along -Y), as tests/kernel/four_bar.hpp
builds it, with a spring from a point above it to the coupler's middle. It
rests on the spring, is given a turn of the crank at 4 rad/s, and swings for
3 s.
"""

import math
import sys
import warnings

import multibody as mbd

A, B, C, D = 0.3, 0.8, 0.6, 0.7


def link(length, mass):
    inertia = mbd.RigidBodyInertia.from_solid_box(mass, [0.5 * length, 0.02, 0.02])
    inertia.com_B = [0.5 * length, 0.0, 0.0]
    return inertia


def four_bar():
    system = mbd.System()
    model = system.model
    model.gravity = [0.0, -mbd.g_accel, 0.0]
    hinge = mbd.RevoluteJointModel()
    model.add_body(0, hinge, inertia=link(A, 1.0), name="crank")
    model.add_body("crank", hinge, x_pj=[A, 0.0, 0.0], inertia=link(B, 2.0), name="coupler")
    rocker = model.add_body("coupler", hinge, x_pj=[B, 0.0, 0.0], inertia=link(C, 1.5), name="rocker")
    system.add_constraint(mbd.revolute_closure(mbd.Marker(0, [D, 0.0, 0.0]), mbd.Marker(rocker, [C, 0.0, 0.0])))
    system.add_force_element(mbd.LinearSpringDamper(0, model.body("coupler"), [0.35, 0.9, 0.0], [0.4, 0.0, 0.0],
                                                    300.0, 0.0, 0.2))
    return system


def configuration(th1):
    """The closed loop at crank angle th1, C on the upper side."""
    bx, by = A * math.cos(th1), A * math.sin(th1)
    dist = math.hypot(D - bx, -by)
    along = (B * B - C * C + dist * dist) / (2.0 * dist)
    h = math.sqrt(B * B - along * along)
    ex, ey = (D - bx) / dist, -by / dist
    cx, cy = bx + along * ex - h * ey, by + along * ey + h * ex
    phi2 = math.atan2(cy - by, cx - bx)
    phi3 = math.atan2(-cy, D - cx)
    return [th1, phi2 - th1, phi3 - phi2]


def main(path=None):
    warnings.filterwarnings("ignore", message="MBD-K050")    # the planar loop's redundant equations
    system = four_bar()
    sim = mbd.Simulator(system)
    sim.q[:] = configuration(1.63)
    mbd.static_equilibrium(sim)                              # resting on its spring
    sim.v[0] = 4.0                                           # the crank turned at 4 rad/s, the rest following
    mbd.assemble(sim, mbd.AssemblyOptions().hold_joint(system.model, 1))
    run = sim.run_recorded(3.0, 1e-3)
    mbd.replay(system, run, name="four-bar", path=path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
