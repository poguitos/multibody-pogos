"""Small models shared by the Python tests, each with the closed forms it is
checked against described where it is built."""

import math

import numpy as np

import multibody as mbd

G = 9.81


def pendulum(length=0.8, gravity=G):
    """A point mass of 1 kg on a massless rod of `length`, hinged at the
    world origin about Z, swinging in the XY plane under gravity -Y. The bob's
    frame is at the mass; its joint frame is `length` above it, so at q = 0 the
    bob hangs at (0, -length, 0) and q is the angle from hanging."""
    system = mbd.System()
    system.model.gravity = [0.0, -gravity, 0.0]
    system.model.add_body(0, mbd.RevoluteJointModel(), mbd.Transform3(),
                          mbd.Transform3.from_translation([0.0, length, 0.0]),
                          mbd.RigidBodyInertia(1.0, np.zeros(3), np.zeros((3, 3))), "bob")
    return system


def slider(mass, gravity=0.0):
    """A body of `mass` on a prismatic joint along world Z, under gravity
    -Z of `gravity`: q is its height."""
    system = mbd.System()
    system.model.gravity = [0.0, 0.0, -gravity]
    system.model.add_body(0, mbd.PrismaticJointModel(), mbd.Transform3(), mbd.Transform3(),
                          mbd.RigidBodyInertia.from_solid_box(mass, [0.05, 0.05, 0.05]), "mass")
    return system


def vertical_spring(system, stiffness, damping=0.0):
    """A linear spring and damper from (0, 0, -1) on the ground to the
    slider's origin, of rest length 1: it pushes back with -stiffness q."""
    system.add_force_element(mbd.LinearSpringDamper(0, 1, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0],
                                                    stiffness, damping, 1.0))


def parallelogram(a=0.3, d=0.8):
    """A four-bar whose crank and rocker have the same length a and whose
    coupler is as long as the ground link d. The crank turns about the world
    origin, the rocker hangs from the coupler's end down its -X axis, and a
    revolute closure joins the rocker's lower end to the ground at (d, 0, 0).
    Gravity -Y, in the plane of the links. Closed, q = (theta, -theta, theta):
    the coupler level and the rocker parallel to the crank."""
    system = mbd.System()
    model = system.model
    model.gravity = [0.0, -G, 0.0]
    hinge = mbd.RevoluteJointModel()

    def link(mass, length, direction):
        inertia = mbd.RigidBodyInertia.from_solid_box(mass, [0.5 * length, 0.02, 0.02])
        inertia.com_B[0] = 0.5 * length * direction          # the centre halfway along the link
        return inertia

    crank = model.add_body(0, hinge, mbd.Transform3(), mbd.Transform3(), link(1.0, a, 1.0), "crank")
    coupler = model.add_body(crank, hinge, mbd.Transform3.from_translation([a, 0.0, 0.0]), mbd.Transform3(),
                             link(2.0, d, 1.0), "coupler")
    rocker = model.add_body(coupler, hinge, mbd.Transform3.from_translation([d, 0.0, 0.0]), mbd.Transform3(),
                            link(1.0, a, -1.0), "rocker")
    system.add_constraint(mbd.revolute_closure(mbd.Marker(0, mbd.Transform3.from_translation([d, 0.0, 0.0])),
                                               mbd.Marker(rocker, mbd.Transform3.from_translation([-a, 0.0, 0.0]))))
    return system


def complete_elliptic_k(k):
    """K(k) = pi / (2 AGM(1, sqrt(1 - k^2))), the arithmetic-geometric mean
    converging quadratically (Abramowitz and Stegun 17.6)."""
    a, b = 1.0, math.sqrt(1.0 - k * k)
    while abs(a - b) > 1e-15 * a:
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)
