"""Joints added out of depth-first order: CRBA and related algorithms return
wrong results without any error in a release build.

Three revolute joints: A and B on the world, C on A. Added in the order
A, C, B (depth first) everything agrees; added A, B, C the mass matrix of
crba differs from the one recursive Newton-Euler gives, as do
computeMinverse and the Cholesky solve, while aba and rnea stay right.
model.check(data) returns False for the second model, but nothing calls it.

    python pinocchio_depth_first.py      (Pinocchio and NumPy only)
"""

import numpy as np
import pinocchio as pin


def three_joints(order):
    rng = np.random.default_rng(0)
    placement = {k: pin.SE3(pin.exp3(rng.normal(size=3)), rng.normal(size=3)) for k in "ABC"}
    inertia = {k: pin.Inertia(1.0 + rng.random(), 0.3 * rng.normal(size=3), np.diag(0.1 + rng.random(3)))
               for k in "ABC"}
    model, ids = pin.Model(), {}
    for name in order:
        parent = ids["A"] if name == "C" else 0
        ids[name] = model.addJoint(parent, pin.JointModelRZ(), placement[name], name)
        model.appendBodyToJoint(ids[name], inertia[name], pin.SE3.Identity())
    return model


def symmetric(M):
    return np.triu(M) + np.triu(M, 1).T


print("Pinocchio", pin.__version__)
q, v = np.array([0.3, -0.7, 1.1]), np.array([0.5, -1.0, 2.0])
for order in ("ACB", "ABC"):
    model = three_joints(order)
    # The mass matrix from rnea, column by column (gravity off, v = 0).
    no_gravity = pin.Model(model)
    no_gravity.gravity = pin.Motion.Zero()
    M_rnea = np.column_stack([pin.rnea(no_gravity, no_gravity.createData(), q, np.zeros(3), e)
                              for e in np.eye(3)])
    M_crba = symmetric(pin.crba(model, model.createData(), q))
    Minv = symmetric(pin.computeMinverse(model, model.createData(), q))
    data = model.createData()
    pin.crba(model, data, q)
    pin.cholesky.decompose(model, data)
    x = pin.cholesky.solve(model, data, np.array([1.0, 2.0, 3.0]))
    a = np.array([1.0, -2.0, 0.5])
    tau = pin.rnea(model, model.createData(), q, v, a)
    print(f"order {order}, parents {list(model.parents)}, model.check(data) = {model.check(model.createData())}")
    print(f"  crba - mass matrix from rnea:        {np.abs(M_crba - M_rnea).max():.2e}")
    print(f"  computeMinverse - inverse of it:     {np.abs(Minv - np.linalg.inv(M_rnea)).max():.2e}")
    print(f"  cholesky.solve - numpy solve:        {np.abs(x - np.linalg.solve(M_rnea, [1.0, 2.0, 3.0])).max():.2e}")
    print(f"  aba(rnea(a)) - a:                    {np.abs(pin.aba(model, model.createData(), q, v, tau) - a).max():.2e}")
