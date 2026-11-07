"""Compare this library's closed mechanisms with Project Chrono's (plan task 5.3).

Reads the random mechanisms written by `mbd_crosscheck mechanisms`
(export_mechanisms.cpp): trees on joints of every kind, closed by spherical,
revolute, universal, cylindrical, prismatic and distance constraints. Builds
each one in PyChrono, every body on its own, every joint and closure a link,
and compares:

- At rest, at the configuration written: every body's angular acceleration
  and the acceleration of its centre of mass; the force and moment of the
  closures on each body; each tree joint's reaction on its child. Relative
  differences, as in compare_pinocchio.py: the largest entry of the
  difference over 1 + the largest entry of Chrono's value. Tolerance 1e-10:
  the matrices come from files with 13 significant digits, which puts the
  floor near 1e-12 (the largest difference seen, 1.4e-12), and an inertia
  changed by one part in a million moved the answers by 2e-9.
- The motion from rest under gravity: every body's placement at the times
  written, from Chrono at two steps, h and h/2.

What Chrono 10.0 computes, and how far it can be taken at its word, shaped
the method (measured on a pendulum; see the journal entry of 7 November
2026):

- Its accelerations come from its assembly at the acceleration level
  (ChSystem.DoAssembly), which takes a velocity-level step of 1e-6 s and
  differentiates. The step holds the positions on the constraints through
  a right-hand side -C / dt, so a residual C of one rounding error, 1e-16 m,
  moves the accelerations by 1e-16 / (1e-6)^2 = 1e-4: a pendulum's joint
  opened by 1e-15 m changed its angular acceleration by 5e-4 rad/s^2, and
  every configuration rebuilt in Chrono's frames has such residuals. (With
  the bodies moving it also leaves out the velocity-product terms: a
  pendulum turning at 3 rad/s lacked its centripetal acceleration
  entirely.) Its time steppers do no better for accelerations and reaction
  forces: with alpha = 0 they oscillate, with alpha = -0.2 they converge at
  first order to about 1e-4.
  So the script takes from Chrono what its assembly sets up, its mass
  matrix, constraint Jacobian and forces in its own coordinates (the matrix
  blocks of its system descriptor, written with 13 significant digits),
  sets the constraints' right-hand side to its exact value at rest, zero,
  and solves those equations itself. Each link's rows of the constraint
  forces give its wrench on its two bodies.
- The assembly leaves the velocities advanced by its step (a * 1e-6 s), and
  the implicit integrators start from the accelerations stored in the
  bodies, zero unless assembled: without the assembly, HHT converged at first
  order, and with the velocities left advanced it stopped converging below
  1e-6. For the motion the script assembles, then puts the velocities back
  to rest.

The motion uses HHT with alpha = 0 (the trapezoidal rule, second order).
Chrono's error falls as h^2 and its expansion has even powers only, so if
our motion is right, Chrono's differences from it fall about fourfold when
the step halves, and Richardson's extrapolation of Chrono's two runs,
x(h/2) + (x(h/2) - x(h)) / 3, comes far closer to ours than either run. The
check: the ratio of the two differences is within RATIO, and the
extrapolated difference is below EXTRAPOLATED times the difference at h/2,
or below the floor set by our own motion's error estimate and Chrono's
Newton tolerance.

    python compare_chrono.py mechanisms.json [--step 2e-4]

Conventions, ours against Chrono's:

- Bodies. Chrono's body is ChBodyAuxRef: its reference frame is our body
  frame (oMi), its centre-of-mass frame is offset by com and parallel to it,
  and the inertia about the centre of mass is in those axes. Chrono's
  SetInertiaXY takes the off-diagonal entries of the matrix as they stand.
- Joints of the tree: a link from the parent (body 1) to the child (body 2),
  placed at the child-side joint frame in the world: revolute, prismatic,
  spherical, cylindrical, planar and fixed are ChLinkLockRevolute,
  ChLinkLockPrismatic, ChLinkLockSpherical, ChLinkLockCylindrical,
  ChLinkLockPlanar and ChLinkLockLock, all about Z as ours. Ours turns about
  the parent-side Z and then the child-side X, which stay perpendicular:
  ChLinkUniversal holds the X axis of its frame 1 perpendicular to the Y
  axis of its frame 2, so its frame has X along the parent's Z and Y along
  the child's X. A free joint has no link.
- Closures: a link from marker i's body (body 1) to marker j's (body 2) at
  marker i, the same classes; the universal's frame has X along Z_i and Y
  along Z_j; the distance is ChLinkDistance between the two origins.
- Chrono's coordinates. Each body that moves has six velocities: its centre
  of mass's, in world axes, and its angular velocity in its own axes; the
  bodies in the order added, the links' equations likewise.
- Loads. Ours: the closures' wrench on each body in world axes about the
  world origin, and each joint's reaction on its child in the child-side
  joint frame about its origin. Chrono's, from a link's rows: a force in
  world axes and a moment about the centre of mass in body axes, on each of
  its bodies. Both are compared in world axes.
"""

import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import pychrono as ch

RATIO = (3.0, 5.0)        # the difference at h over the difference at h/2
EXTRAPOLATED = 0.1        # the extrapolated difference over the difference at h/2
NEWTON_TOLERANCE = 1e-12  # HHT's absolute tolerances, and its relative one


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------

def rotation(quat_xyzw):
    """The rotation matrix of a unit quaternion stored (x, y, z, w)."""
    x, y, z, w = quat_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform(t):
    """A transform written by the exporter, as (R, p)."""
    return rotation(t["q"]), np.array(t["p"])


def compose(a, b):
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def quaternion(R):
    """A unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    t = np.trace(R)
    k = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    if k == 0:
        s = 2.0 * np.sqrt(1.0 + t)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i, j, l = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][k - 1]
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[l, l])
    q = np.zeros(4)
    q[0] = (R[l, j] - R[j, l]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + l] = (R[l, i] + R[i, l]) / s
    return q


def ch_vector(v):
    return ch.ChVector3d(float(v[0]), float(v[1]), float(v[2]))


def ch_frame(T):
    w, x, y, z = quaternion(T[0])
    return ch.ChFramed(ch_vector(T[1]), ch.ChQuaterniond(w, x, y, z))


def vec(v):
    return np.array([v.x, v.y, v.z])


def matrix(frame):
    M = frame.GetRotMat()
    return np.array([[M[i, j] for j in range(3)] for i in range(3)])


def cross_frame(a, b, origin):
    """A frame at `origin` with X along a and Y along b (perpendicular unit vectors)."""
    return np.column_stack([a, b, np.cross(a, b)]), origin


# ----------------------------------------------------------------------------
# The mechanism in Chrono
# ----------------------------------------------------------------------------

TREE_LINKS = {
    "revolute": ch.ChLinkLockRevolute,
    "prismatic": ch.ChLinkLockPrismatic,
    "spherical": ch.ChLinkLockSpherical,
    "cylindrical": ch.ChLinkLockCylindrical,
    "planar": ch.ChLinkLockPlanar,
    "fixed": ch.ChLinkLockLock,
}
CLOSURE_LINKS = {
    "spherical": ch.ChLinkLockSpherical,
    "revolute": ch.ChLinkLockRevolute,
    "cylindrical": ch.ChLinkLockCylindrical,
    "prismatic": ch.ChLinkLockPrismatic,
}
JOINT_EQUATIONS = {"revolute": 5, "prismatic": 5, "spherical": 3, "cylindrical": 4, "planar": 3, "fixed": 6,
                   "universal": 4, "free": 0}


def build(mech, h=None):
    """The mechanism in a new Chrono system at rest, with HHT at step h if given.
    Returns the system, the bodies (ours -> Chrono's), and the links of the
    tree (child body -> link) and of the closures, in order."""
    system = ch.ChSystemNSC()
    system.SetGravitationalAcceleration(ch_vector(mech["gravity"]))
    system.SetSolver(ch.ChSolverSparseQR())
    if h is not None:
        system.SetTimestepperType(ch.ChTimestepper.Type_HHT)
        stepper = ch.CastToChTimestepperHHT(system.GetTimestepper())
        stepper.SetAlpha(0.0)
        stepper.SetMaxIters(50)
        stepper.SetAbsTolerances(NEWTON_TOLERANCE, NEWTON_TOLERANCE)
        stepper.SetRelTolerance(NEWTON_TOLERANCE)
        stepper.SetStepControl(False)

    ground = ch.ChBody()
    ground.SetFixed(True)
    system.AddBody(ground)
    bodies = {0: ground}
    pose = {0: (np.eye(3), np.zeros(3))}
    for b, X in zip(mech["bodies"], mech["oMi"]):
        body = ch.ChBodyAuxRef()
        body.SetMass(b["mass"])
        I = np.array(b["I_com"])
        body.SetInertiaXX(ch_vector(np.diag(I)))
        body.SetInertiaXY(ch_vector([I[0, 1], I[0, 2], I[1, 2]]))
        body.SetFrameCOMToRef(ch.ChFramed(ch_vector(b["com"]), ch.QUNIT))
        pose[b["body"]] = transform(X)
        body.SetFrameRefToAbs(ch_frame(pose[b["body"]]))
        system.AddBody(body)
        bodies[b["body"]] = body

    tree_links = {}
    for b in mech["bodies"]:
        child, parent, kind = b["body"], b["parent"], b["joint"]
        if kind == "free":
            continue
        Jc = compose(pose[child], transform(b["X_CJ"]))
        if kind == "universal":
            Jp = compose(pose[parent], transform(b["X_PJ"]))
            link = ch.ChLinkUniversal()
            link.Initialize(bodies[parent], bodies[child], ch_frame(cross_frame(Jp[0][:, 2], Jc[0][:, 0], Jc[1])))
        else:
            link = TREE_LINKS[kind]()
            link.Initialize(bodies[parent], bodies[child], ch_frame(Jc))
        system.AddLink(link)
        tree_links[child] = link

    closure_links = []
    for c in mech["closures"]:
        bi, bj = c["i"]["body"], c["j"]["body"]
        Wi = compose(pose[bi], transform(c["i"]["X_BM"]))
        Wj = compose(pose[bj], transform(c["j"]["X_BM"]))
        if c["kind"] == "distance":
            link = ch.ChLinkDistance()
            link.Initialize(bodies[bi], bodies[bj], False, ch_vector(Wi[1]), ch_vector(Wj[1]), True)
        elif c["kind"] == "universal":
            link = ch.ChLinkUniversal()
            link.Initialize(bodies[bi], bodies[bj], ch_frame(cross_frame(Wi[0][:, 2], Wj[0][:, 2], Wi[1])))
        else:
            link = CLOSURE_LINKS[c["kind"]]()
            link.Initialize(bodies[bi], bodies[bj], ch_frame(Wi))
        system.AddLink(link)
        closure_links.append((bi, bj, link))
    return system, bodies, tree_links, closure_links


def assemble_at_rest(system, bodies):
    """Chrono's accelerations at rest, the velocities put back to rest after."""
    system.DoAssembly(ch.ACCELERATION)
    for b, body in bodies.items():
        if b:
            body.SetLinVel(ch.ChVector3d(0, 0, 0))
            body.SetAngVelParent(ch.ChVector3d(0, 0, 0))


def read_blocks(directory, prefix):
    """The blocks written by ChSystemDescriptor.WriteMatrixBlocks: H, Cq and E
    as (row, column, value) triplets counted from 1, f and b as columns."""
    def column(name):
        with open(os.path.join(directory, f"{prefix}_{name}.dat"), encoding="utf-8") as f:
            return np.array([float(x) for x in f.read().split()])

    def triplets(name, shape):
        A = np.zeros(shape)
        with open(os.path.join(directory, f"{prefix}_{name}.dat"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    i, j, value = line.split()
                    A[int(i) - 1, int(j) - 1] += float(value)
        return A

    f, b = column("f"), column("b")
    n, m = len(f), len(b)
    return triplets("H", (n, n)), triplets("Cq", (m, n)), triplets("E", (m, m)), f


def chrono_at_rest(mech):
    """Chrono's accelerations and link forces at rest, from its own equations
    with their right-hand side at rest (see the top of the file). Returns, for
    our bodies in order, the angular acceleration and the acceleration of the
    centre of mass (world axes), and, for each link in the order added, its
    bodies and its wrench on each, as (moment about the centre of mass, force),
    both in world axes."""
    system, bodies, tree_links, closure_links = build(mech)
    ids = [b["body"] for b in mech["bodies"]]
    links = [(b["parent"], b["body"], tree_links[b["body"]]) for b in mech["bodies"]
             if b["body"] in tree_links] + closure_links
    system.DoAssembly(ch.ACCELERATION)          # sets up the blocks at rest
    with tempfile.TemporaryDirectory() as tmp:
        system.GetSystemDescriptor().WriteMatrixBlocks(tmp, "rest")
        H, Cq, E, f = read_blocks(tmp, "rest")

    rows = [link.GetNumConstraintsBilateral() for _, _, link in links]
    expected = sum(JOINT_EQUATIONS[b["joint"]] for b in mech["bodies"]) + mech["equations"]
    if H.shape[0] != 6 * len(ids) or Cq.shape[0] != sum(rows) or sum(rows) != expected or np.any(E):
        raise RuntimeError(f"mechanism {mech['mechanism']}: Chrono's blocks are {H.shape[0]} velocities and "
                           f"{Cq.shape[0]} equations ({sum(rows)} from its links); expected {6 * len(ids)} and "
                           f"{expected}, without compliance")

    # f is the impulse of the assembly's step: the step times gravity's forces.
    weights = np.concatenate([np.concatenate([b["mass"] * np.array(mech["gravity"]), np.zeros(3)])
                              for b in mech["bodies"]])
    dt = float(f @ weights / (weights @ weights))
    if np.max(np.abs(f - dt * weights)) > 1e-11 * np.max(np.abs(f)):
        raise RuntimeError(f"mechanism {mech['mechanism']}: Chrono's forces at rest are not gravity's")

    # H a + Cq^T y = gravity's forces, Cq a = 0: the constraint forces are -Cq^T y.
    n, m = H.shape[0], Cq.shape[0]
    solution = np.linalg.solve(np.block([[H, Cq.T], [Cq, np.zeros((m, m))]]),
                               np.concatenate([f / dt, np.zeros(m)]))
    acc, y = solution[:n], solution[n:]
    R = {b: matrix(bodies[b].GetFrameRefToAbs()) for b in ids}     # the centre of mass's axes
    alpha = [R[b] @ acc[6 * k + 3:6 * k + 6] for k, b in enumerate(ids)]
    a_com = [acc[6 * k:6 * k + 3] for k, b in enumerate(ids)]

    column_of = {b: 6 * k for k, b in enumerate(ids)}
    wrenches = []
    start = 0
    for (b1, b2, _), count in zip(links, rows):
        forces = -Cq[start:start + count].T @ y[start:start + count]
        start += count
        on = {}
        for b in (b1, b2):
            if b:
                c = column_of[b]
                on[b] = (R[b] @ forces[c + 3:c + 6], forces[c:c + 3].copy())
                forces[c:c + 6] = 0.0
        if np.max(np.abs(forces)) > 1e-9 * (1.0 + np.max(np.abs(y))):
            raise RuntimeError(f"mechanism {mech['mechanism']}: a link's equations reach other bodies "
                               "(the equations are not in the order the links were added)")
        wrenches.append((b1, b2, on))
    return alpha, a_com, wrenches, {b: vec(bodies[b].GetPos()) for b in ids}


def moment_about(wrench, centre, point):
    """(moment about the centre of mass, force) as [moment about `point`; force]."""
    moment, force = wrench
    return np.concatenate([moment + np.cross(centre - point, force), force])


def relative(ours, theirs):
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    return float(np.max(np.abs(ours - theirs)) / (1.0 + np.max(np.abs(theirs))))


# ----------------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------------

def compare_at_rest(mech):
    alpha, a_com, wrenches, centre = chrono_at_rest(mech)
    ours = mech["at_rest"]
    ids = [b["body"] for b in mech["bodies"]]
    tree = sum(1 for b in mech["bodies"] if b["joint"] != "free")

    # The closures' wrenches on each body, about the world origin.
    W = {b: np.zeros(6) for b in ids}
    for _, _, on in wrenches[tree:]:
        for b, wrench in on.items():
            W[b] += moment_about(wrench, centre[b], np.zeros(3))

    # Each joint's reaction on its child, in world axes about the joint's origin.
    reaction_of = {b2: on[b2] for _, b2, on in wrenches[:tree]}
    ours_joint, theirs_joint = [], []
    for b, X, r in zip(mech["bodies"], mech["oMi"], ours["joint_reaction"]):
        Jc = compose(transform(X), transform(b["X_CJ"]))
        ours_joint.append(np.concatenate([Jc[0] @ np.array(r[:3]), Jc[0] @ np.array(r[3:])]))
        if b["body"] in reaction_of:
            theirs_joint.append(moment_about(reaction_of[b["body"]], centre[b["body"]], Jc[1]))
        else:
            theirs_joint.append(np.zeros(6))     # a free joint transmits nothing
    return {
        "angular acceleration": relative(ours["alpha"], alpha),
        "centre-of-mass acceleration": relative(ours["a_com"], a_com),
        "closure loads": relative(ours["constraint_W"], [W[b] for b in ids]),
        "joint reactions": relative(ours_joint, theirs_joint),
    }


def chrono_motion(mech, h, interval, samples):
    """Every body's placement (R, p) at each sample time, by HHT at step h."""
    system, bodies, _, _ = build(mech, h)
    assemble_at_rest(system, bodies)
    steps = round(interval / h)
    out = []
    for _ in range(samples):
        for _ in range(steps):
            system.DoStepDynamics(h)
        out.append([(matrix(bodies[b["body"]].GetFrameRefToAbs()), vec(bodies[b["body"]].GetFrameRefToAbs().GetPos()))
                    for b in mech["bodies"]])
    return out


def compare_motion(mech, h, duration, samples):
    ours = [[transform(X) for X in poses] for poses in mech["motion"]["oMi"]]
    coarse = chrono_motion(mech, h, duration / samples, samples)
    fine = chrono_motion(mech, h / 2, duration / samples, samples)

    def largest(theirs):
        """Largest position [m] and rotation-matrix entry differences."""
        dp = max(np.max(np.abs(o[1] - t[1])) for os, ts in zip(ours, theirs) for o, t in zip(os, ts))
        dR = max(np.max(np.abs(o[0] - t[0])) for os, ts in zip(ours, theirs) for o, t in zip(os, ts))
        return dp, dR

    extrapolated = [[(f[0] + (f[0] - c[0]) / 3, f[1] + (f[1] - c[1]) / 3) for f, c in zip(fs, cs)]
                    for fs, cs in zip(fine, coarse)]
    return largest(coarse), largest(fine), largest(extrapolated)


def chrono_version():
    """PyChrono's version, from conda's record of the package (the module has none)."""
    for path in glob.glob(os.path.join(sys.prefix, "conda-meta", "pychrono-*.json")):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mechanisms", help="JSON written by mbd_crosscheck mechanisms")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="at rest")
    parser.add_argument("--step", type=float, default=2e-4, help="Chrono's larger step h [s]")
    args = parser.parse_args()

    with open(args.mechanisms, encoding="utf-8") as f:
        data = json.load(f)
    duration, samples = data["duration"], data["samples"]
    print(f"PyChrono {chrono_version()}; {len(data['mechanisms'])} mechanisms from {data['program']} "
          f"(seed {data['seed']})")

    failures = []
    worst = {}
    for mech in data["mechanisms"]:
        m = mech["mechanism"]
        for k, e in compare_at_rest(mech).items():
            worst[k] = max(worst.get(k, 0.0), e)
            if not e <= args.tolerance:
                failures.append(f"mechanism {m}: {k} at rest differs by {e:.3e}")

        (p1, r1), (p2, r2), (pe, re) = compare_motion(mech, args.step, duration, samples)
        floor = 10.0 * (mech["motion"]["rk4_change"] + NEWTON_TOLERANCE)
        print(f"  mechanism {m:2d} ({len(mech['bodies'])} bodies, {mech['dof']:2d} dof, "
              f"{'+'.join(c['kind'] for c in mech['closures'])}): motion differs by "
              f"{p1:.2e} / {p2:.2e} / {pe:.2e} m and {r1:.2e} / {r2:.2e} / {re:.2e} in rotation "
              f"(h, h/2, extrapolated)")
        for what, d1, d2, de in (("positions", p1, p2, pe), ("rotations", r1, r2, re)):
            if de <= floor:
                continue
            ratio = d1 / d2 if d2 > 0 else float("inf")
            if not RATIO[0] <= ratio <= RATIO[1]:
                failures.append(f"mechanism {m}: {what}: Chrono's difference falls by {ratio:.2f} when its step halves")
            if not de <= EXTRAPOLATED * d2:
                failures.append(f"mechanism {m}: {what}: extrapolated difference {de:.2e} against {d2:.2e} at h/2")

    for k, e in worst.items():
        print(f"  at rest, {k:<28} largest relative difference {e:.3e}")
    if failures:
        print(f"FAILED: {len(failures)} differences:")
        for line in failures[:20]:
            print("  " + line)
        return 1
    print(f"All within tolerance (at rest {args.tolerance:g}; motion converging to ours at second order).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
