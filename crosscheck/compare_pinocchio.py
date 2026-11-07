"""Compare this library's tree dynamics with Pinocchio's (plan task 5.3).

Reads the random trees written by mbd_crosscheck (export_trees.cpp), builds
each one in Pinocchio, and compares, state by state:

- the accelerations of the articulated-body algorithm (ABA),
- the generalized forces of recursive Newton-Euler (RNEA),
- the mass matrix of the composite-rigid-body algorithm (CRBA),
- every body's placement in the world (forward kinematics).

A difference passes when its largest entry is within the tolerance times
(1 + the largest entry of Pinocchio's value): relative for large values,
absolute for small ones. The plan asks for 1e-8.

    python compare_pinocchio.py trees.json [--tolerance 1e-8]

Conventions, ours against Pinocchio's:

- Frames. Each of our bodies has its own frame; its joint connects a frame
  fixed in the parent (X_PJ, in the parent's body frame) to one fixed in the
  child (X_CJ, in the child's body frame). Pinocchio's body frame is the
  joint's child-side frame, and its joint placement is given in the parent
  joint's frame. So a joint's placement in Pinocchio is T(parent) * X_PJ,
  where T(b) is the pose of our body frame b in its Pinocchio joint frame:
  inverse(X_CJ(b)) for a moving joint, and for a fixed joint, merged into its
  parent, T(parent) * X_PJ * inverse(X_CJ). Inertias go in with T(b).
- Joints. Revolute and prismatic are Pinocchio's RZ and PZ; universal (Z,
  then the rotated X), cylindrical (Z rotation and translation) and planar
  (X and Y in the parent-side frame, then Z rotation) are composites of
  those; spherical and free are Pinocchio's spherical and free flyer, with
  the same quaternion order (x, y, z, w). The coordinates are the same in
  both libraries. Velocities too, except the free joint's: ours is
  (angular, linear), Pinocchio's (linear, angular), both in the child frame.
- Transforms. Both are (rotation from local to parent axes, origin in the
  parent), so they compose the same way.
- Numbering. Pinocchio's CRBA assumes the joints are numbered depth first,
  each joint's subtree taking the ids just after it (it writes each row of
  the mass matrix as one block over the subtree's velocities); addJoint does
  not check it, and on other numberings the matrix comes out wrong while ABA
  and RNEA stay right. Our trees are numbered in the order the bodies were
  added, so the script adds the joints to Pinocchio depth first and maps the
  coordinates between the two orders.
"""

import argparse
import json
import sys

import numpy as np
import pinocchio as pin


def rotation(quat_xyzw):
    """The rotation matrix of a unit quaternion stored (x, y, z, w)."""
    x, y, z, w = quat_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def se3(t):
    """A transform written by the exporter, as a Pinocchio SE3."""
    return pin.SE3(rotation(t["q"]), np.array(t["p"]))


def composite(*joints):
    jc = pin.JointModelComposite()
    for j in joints:
        jc.addJoint(j)
    return jc


def joint_model(kind):
    if kind == "revolute":
        return pin.JointModelRZ()
    if kind == "prismatic":
        return pin.JointModelPZ()
    if kind == "universal":
        return composite(pin.JointModelRZ(), pin.JointModelRX())
    if kind == "cylindrical":
        return composite(pin.JointModelRZ(), pin.JointModelPZ())
    if kind == "planar":
        return composite(pin.JointModelPX(), pin.JointModelPY(), pin.JointModelRZ())
    if kind == "spherical":
        return pin.JointModelSpherical()
    if kind == "free":
        return pin.JointModelFreeFlyer()
    raise ValueError(f"no Pinocchio joint for '{kind}'")


def depth_first(tree):
    """Our bodies in depth-first order, children in the order they were added."""
    children = {0: []}
    for b in tree["bodies"]:
        children[b["body"]] = []
        children[b["parent"]].append(b)
    order, stack = [], list(reversed(children[0]))
    while stack:
        b = stack.pop()
        order.append(b)
        stack.extend(reversed(children[b["body"]]))
    return order


def build(tree):
    """The tree in Pinocchio, its joints added depth first; for each of our
    bodies, its Pinocchio joint and the pose of our body frame in that joint's
    frame; and, for each Pinocchio coordinate and velocity, the index of ours
    (q_pinocchio = q[qmap], v_pinocchio = v[vmap])."""
    model = pin.Model()
    model.gravity = pin.Motion(np.array(tree["gravity"]), np.zeros(3))
    joint_of = {0: 0}            # our body -> Pinocchio joint id (0: the world)
    frame_of = {0: pin.SE3.Identity()}
    qmap = np.full(tree["nq"], -1)
    vmap = np.full(tree["nv"], -1)
    for b in depth_first(tree):
        body, parent = b["body"], b["parent"]
        X_PJ, X_JC = se3(b["X_PJ"]), se3(b["X_CJ"]).inverse()
        inertia = pin.Inertia(b["mass"], np.array(b["com"]), np.array(b["I_com"]))
        if b["joint"] == "fixed":
            joint_of[body] = joint_of[parent]
            frame_of[body] = frame_of[parent] * X_PJ * X_JC
        else:
            jid = model.addJoint(joint_of[parent], joint_model(b["joint"]), frame_of[parent] * X_PJ,
                                 f"joint_{body}")
            if model.nqs[jid] != b["nq"] or model.nvs[jid] != b["nv"]:
                raise RuntimeError(f"body {body}: {model.nqs[jid]}/{model.nvs[jid]} coordinates in Pinocchio, "
                                   f"{b['nq']}/{b['nv']} here")
            iq, iv = model.idx_qs[jid], model.idx_vs[jid]
            qmap[iq:iq + b["nq"]] = b["idx_q"] + np.arange(b["nq"])
            within = [3, 4, 5, 0, 1, 2] if b["joint"] == "free" else range(b["nv"])
            vmap[iv:iv + b["nv"]] = b["idx_v"] + np.array(within, dtype=int)
            joint_of[body] = jid
            frame_of[body] = X_JC
        model.appendBodyToJoint(joint_of[body], inertia, frame_of[body])
    if (qmap < 0).any() or (vmap < 0).any():
        raise RuntimeError(f"tree {tree['tree']}: coordinates left unmapped")
    return model, joint_of, frame_of, qmap, vmap


def relative(ours, theirs):
    return float(np.max(np.abs(ours - theirs)) / (1.0 + np.max(np.abs(theirs)))) if np.size(theirs) else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trees", help="JSON written by mbd_crosscheck")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    with open(args.trees, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Pinocchio {pin.__version__}; {len(data['trees'])} trees from {data['program']} (seed {data['seed']})")

    worst = {"aba": 0.0, "rnea": 0.0, "crba": 0.0, "placements": 0.0}
    failures = []
    joints_seen = {}
    for tree in data["trees"]:
        model, joint_of, frame_of, qmap, vmap = build(tree)
        pdata = model.createData()
        for b in tree["bodies"]:
            joints_seen[b["joint"]] = joints_seen.get(b["joint"], 0) + 1
        for s, st in enumerate(tree["states"]):
            q = np.array(st["q"])[qmap]
            v, tau, a = (np.array(st[k])[vmap] for k in ("v", "tau", "a"))
            # Compared in Pinocchio's order: ours[vmap].
            errors = {
                "aba": relative(np.array(st["ddq"])[vmap], pin.aba(model, pdata, q, v, tau)),
                "rnea": relative(np.array(st["tau_rnea"])[vmap], pin.rnea(model, pdata, q, v, a)),
            }
            M = pin.crba(model, pdata, q)
            M = np.triu(M) + np.triu(M, 1).T      # in case only the upper triangle is filled
            errors["crba"] = relative(np.array(st["M"])[np.ix_(vmap, vmap)], M)
            pin.forwardKinematics(model, pdata, q)
            placement = 0.0
            for b, pose in zip(tree["bodies"], st["oMi"]):
                theirs = pdata.oMi[joint_of[b["body"]]] * frame_of[b["body"]]
                placement = max(placement, relative(rotation(pose["q"]), theirs.rotation),
                                relative(np.array(pose["p"]), theirs.translation))
            errors["placements"] = placement
            for k, e in errors.items():
                worst[k] = max(worst[k], e)
                if not e <= args.tolerance:
                    failures.append(f"tree {tree['tree']} state {s}: {k} differs by {e:.3e}")

    bodies = sum(len(t["bodies"]) for t in data["trees"])
    print(f"{bodies} bodies; joints: " + ", ".join(f"{k} {n}" for k, n in sorted(joints_seen.items())))
    for k, e in worst.items():
        print(f"  {k:<11} largest relative difference {e:.3e}")
    if failures:
        print(f"FAILED: {len(failures)} differences above {args.tolerance:g}:")
        for line in failures[:20]:
            print("  " + line)
        return 1
    print(f"All within {args.tolerance:g}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
