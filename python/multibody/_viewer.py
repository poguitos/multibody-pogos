"""Viewing (plan task 6.5, D57): a motion played back in three dimensions,
and channels as plots, with Rerun (rerun.io).

``replay(system, motion)`` draws each body from its inertia, as the uniform
box with the body's mass and moments (a point for a point mass), with lines
from its centre of mass to its joints; the force elements and distances
between two points as lines; contact spheres and the markers of closures as
points. The model has no geometry of its own, so these stand for it. The
motion is a run (``sim.run_recorded``), a table (``dataframe``,
``read_results``, or a CSV with columns ``q[0]`` ... as a C++ program writes
it), or ``(time, q)``; its coordinates and velocities are plotted beside the
view, with any other channels given.

The scene is computed apart from the drawing (``_scene``), so that it can be
checked against the engine without a viewer. rerun-sdk is needed here only.
"""

import numpy as np

from ._errors import MbdError
from ._results import body_labels, coordinate_names


def _rerun():
    try:
        import rerun
        import rerun.blueprint
    except ImportError as missing:
        raise ImportError("MBD-P015: viewing needs Rerun (pip install rerun-sdk, or conda install rerun-sdk; "
                          "or the package's extra, pip install -e \".[view]\")") from missing
    return rerun, rerun.blueprint


# --- The motion ----------------------------------------------------------------

def _motion(motion, model):
    """(time, q, v or None) from the forms replay() takes."""
    v = None
    if isinstance(motion, tuple) and len(motion) == 2:
        time, q = motion
    elif isinstance(motion, dict) and {"time", "q"} <= motion.keys():
        time, q, v = motion["time"], motion["q"], motion.get("v")
    elif hasattr(motion, "columns") and hasattr(motion, "index"):
        time = motion.index.to_numpy()
        if getattr(motion.columns, "nlevels", 1) == 2:
            q = motion["q"].to_numpy()
            v = motion["v"].to_numpy() if "v" in motion.columns.get_level_values(0) else None
        else:
            names = [f"q[{k}]" for k in range(model.nq)]
            missing = [n for n in names if n not in motion.columns]
            if missing:
                raise MbdError(f"MBD-P013: the table has no column {missing[0]}; a table of one level needs the "
                               f"columns q[0] to q[{model.nq - 1}]")
            q = motion[names].to_numpy()
    else:
        raise TypeError("a motion is a run (sim.run_recorded), a table (dataframe, read_results) or (time, q)")
    time = np.asarray(time, dtype=float)
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[1] != model.nq or len(time) != q.shape[0]:
        raise MbdError(f"MBD-P013: the motion has {q.shape[1]} values of q per sample ({q.shape[0]} samples, "
                       f"{len(time)} times); the model {model.nq}: it is not this model's motion")
    if v is not None:
        v = np.atleast_2d(np.asarray(v, dtype=float))
        if v.shape != (len(time), model.nv):
            v = None
    return time, q, v


def _frames(time, fps):
    """The samples shown: the first at or after each 1 / fps from the start,
    and the last; every sample when fps is None."""
    if fps is None or len(time) < 2:
        return np.arange(len(time))
    targets = np.arange(time[0], time[-1], 1.0 / fps)
    picked = np.searchsorted(time, targets - 1e-9 * max(1.0, abs(time[-1])))
    return np.unique(np.append(picked[picked < len(time)], len(time) - 1))


# --- The scene -----------------------------------------------------------------

def _rotate(q, v):
    """Vectors v turned by unit quaternions q = (w, x, y, z), broadcast."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _inertia_box(inertia):
    """(half sizes, centre, principal axes as columns) of the uniform box
    with the body's mass and moments; None for a point mass or none."""
    m = inertia.mass
    if not m > 0.0:
        return None
    moments, axes = np.linalg.eigh(np.asarray(inertia.I_com_B, dtype=float))
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]
    # A box of half sizes a has I_k = m (a_i^2 + a_j^2) / 3 about its axis k,
    # so a_k^2 = 3 (I_i + I_j - I_k) / (2 m).
    total = moments.sum()
    half = np.sqrt(np.maximum(3.0 * (total - 2.0 * moments) / (2.0 * m), 0.0))
    if not half.max() > 0.0:
        return None
    return half, np.asarray(inertia.com_B, dtype=float), axes


def _markers(constraint):
    """The markers of a constraint and of the parts of a set."""
    import multibody as mbd
    if isinstance(constraint, mbd.ConstraintSet):
        return [m for part in constraint.parts() for m in _markers(part)]
    if isinstance(constraint, (mbd.PointCoincidence, mbd.Dot1, mbd.Dot2, mbd.NoTwist, mbd.Distance)):
        return [constraint.i(), constraint.j()]
    return []


def _scene(system, motion, fps=60.0):
    """What replay() draws, as arrays: the frames' times, each body's pose at
    each frame, the lines' ends in world axes, the static drawings on each
    body, and the coordinates to plot."""
    import multibody as mbd
    model = system.model
    time, q, v = _motion(motion, model)
    frames = _frames(time, fps)
    n = model.nbodies()

    sim = mbd.Simulator(system)
    positions = np.zeros((len(frames), n, 3))
    quaternions = np.zeros((len(frames), n, 4))
    quaternions[..., 0] = 1.0
    for k, i in enumerate(frames):
        sim.q[:] = q[i]
        sim.refresh()
        states = sim.states()
        for b in range(1, n):
            positions[k, b] = states[b].p_WB
            quaternions[k, b] = states[b].q_WB

    def world(body, point):
        point = np.asarray(point, dtype=float)
        return positions[:, body] + _rotate(quaternions[:, body], np.broadcast_to(point, (len(frames), 3)))

    labels = body_labels(model)
    bodies = [{"label": labels[b], "box": _inertia_box(model.inertia[b]),
               "centre": np.asarray(model.inertia[b].com_B, dtype=float), "links": [], "points": []}
              for b in range(n)]
    for b in range(1, n):
        bodies[b]["links"].append(model.X_CJ[b].translation)
        if model.parent[b] > 0:
            bodies[model.parent[b]]["links"].append(model.X_PJ[b].translation)

    lines = []
    for k, e in enumerate(system.force_elements):
        label = f"{k} {e.name()}"
        if isinstance(e, mbd.LinearSpringDamper):
            ends = (e.body1_idx, e.anchor1_B), (e.body2_idx, e.anchor2_B)
        elif isinstance(e, mbd.SpringDamper):
            b1, b2 = e.bodies()
            ends = (b1, e.anchor1()), (b2, e.anchor2())
        elif isinstance(e, (mbd.Bushing, mbd.RotationalSpringDamper)):
            b1, b2 = e.bodies()
            ends = (b1, e.X_1M().translation), (b2, e.X_2M().translation)
        elif isinstance(e, mbd.PlaneContact):
            bodies[e.bodies()[0]]["points"].append((label, np.asarray(e.centres(), dtype=float), e.radius()))
            continue
        else:
            continue                                  # a UserForce, or an element with no points to show
        lines.append({"label": label, "ends": np.stack([world(*ends[0]), world(*ends[1])], axis=1)})
    for k, c in enumerate(system.constraints):
        if isinstance(c, mbd.Distance):
            lines.append({"label": f"constraint {k} distance",
                          "ends": np.stack([world(c.i().body, c.i().X_BM.translation),
                                            world(c.j().body, c.j().X_BM.translation)], axis=1)})
        for m in _markers(c):
            bodies[m.body]["points"].append((f"constraint {k} {c.name()}",
                                             np.asarray([m.X_BM.translation], dtype=float), 0.0))

    gravity = np.asarray(model.gravity, dtype=float)
    axis = int(np.argmax(np.abs(gravity))) if np.any(gravity) else 2
    up = f"RIGHT_HAND_{'XYZ'[axis]}_{'UP' if not np.any(gravity) or gravity[axis] < 0.0 else 'DOWN'}"
    extent = np.ptp(positions[:, 1:].reshape(-1, 3), axis=0).max() if n > 1 else 1.0
    for b in bodies[1:]:
        if b["box"] is not None:
            extent = max(extent, 2.0 * b["box"][0].max())
    return {"time": time, "frames": frames, "frame_time": time[frames], "positions": positions,
            "quaternions": quaternions, "bodies": bodies, "lines": lines, "q": q, "v": v,
            "q_names": coordinate_names(model, "q"), "v_names": coordinate_names(model, "v"),
            "up": up, "size": max(extent, 1e-3) / 200.0}


# --- Drawing -------------------------------------------------------------------

def _stream(rr, name, path, spawn, blueprint):
    recording = rr.RecordingStream("multibody", recording_id=name)
    if path is not None:
        recording.save(str(path), default_blueprint=blueprint)
    elif spawn:
        recording.spawn(default_blueprint=blueprint)
    else:
        raise ValueError("give a path to save the recording to, or spawn=True to open the viewer")
    return recording


def _plot(rr, recording, prefix, time, values, names):
    for k, name in enumerate(names):
        recording.send_columns(f"{prefix}/{rr.escape_entity_path_part(str(name))}",
                               indexes=[rr.TimeColumn("time", duration=time)],
                               columns=rr.Scalars.columns(scalars=np.asarray(values[:, k], dtype=float)))


def replay(system, motion, *, name="multibody", fps=60.0, path=None, spawn=None, channels=None):
    """Play a motion of `system` back in Rerun: the view on the left, its
    coordinates and velocities (and the columns of `channels`, a table such
    as dataframe(recorder)) plotted on the right, on one time line.

    `fps` frames per second of motion are drawn (None: every sample); the
    plots take every sample. With `path` the recording is saved to that .rrd
    file (open it with `rerun <file>`); otherwise, or with spawn=True, the
    Rerun viewer opens. Returns the recording stream."""
    rr, rrb = _rerun()
    scene = _scene(system, motion, fps)
    plots = [rrb.TimeSeriesView(origin="plots/q", name="q"), rrb.TimeSeriesView(origin="plots/v", name="v")]
    if scene["v"] is None:
        plots.pop()
    if channels is not None:
        plots.append(rrb.TimeSeriesView(origin="plots/channels", name="channels"))
    blueprint = rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin="world", name=name), rrb.Vertical(*plots),
                                             column_shares=[3, 2]))
    recording = _stream(rr, name, path, path is None if spawn is None else spawn, blueprint)
    esc = rr.escape_entity_path_part
    size = scene["size"]
    frame_time = [rr.TimeColumn("time", duration=scene["frame_time"])]

    recording.log("world", getattr(rr.ViewCoordinates, scene["up"]), static=True)
    for b, body in enumerate(scene["bodies"]):
        path_b = "world/ground" if b == 0 else f"world/{esc(body['label'])}"
        if b > 0:
            recording.send_columns(path_b, indexes=frame_time, columns=rr.Transform3D.columns(
                translation=scene["positions"][:, b],
                quaternion=scene["quaternions"][:, b][:, [1, 2, 3, 0]]))      # Rerun's order: x, y, z, w
            if body["box"] is not None:
                half, centre, axes = body["box"]
                recording.log(f"{path_b}/inertia", rr.Transform3D(translation=centre, mat3x3=axes), static=True)
                recording.log(f"{path_b}/inertia", rr.Boxes3D(half_sizes=[half]), static=True)
            else:
                recording.log(f"{path_b}/inertia", rr.Points3D([body["centre"]], radii=2.0 * size), static=True)
            if body["links"]:
                recording.log(f"{path_b}/links", rr.LineStrips3D(
                    [[body["centre"], np.asarray(p, dtype=float)] for p in body["links"]], radii=0.5 * size),
                    static=True)
        for k, (label, points, radius) in enumerate(body["points"]):
            recording.log(f"{path_b}/points/{esc(f'{k} {label}')}",
                          rr.Points3D(points, radii=radius if radius > 0.0 else 1.5 * size), static=True)
    for line in scene["lines"]:
        recording.send_columns(f"world/elements/{esc(line['label'])}", indexes=frame_time,
                               columns=rr.LineStrips3D.columns(strips=line["ends"]).partition(
                                   [1] * len(scene["frame_time"])))

    _plot(rr, recording, "plots/q", scene["time"], scene["q"], scene["q_names"])
    if scene["v"] is not None:
        _plot(rr, recording, "plots/v", scene["time"], scene["v"], scene["v_names"])
    if channels is not None:
        _plot(rr, recording, "plots/channels", channels.index.to_numpy(), channels.to_numpy(),
              [" ".join(map(str, c)) if isinstance(c, tuple) else c for c in channels.columns])
    recording.flush()
    return recording


def plot(table, *, name="multibody", path=None, spawn=None):
    """Plot every column of a table indexed by time (dataframe(run, sim),
    dataframe(recorder), read_results(...)) in Rerun, one time line. Saved
    to `path` (.rrd) if given, otherwise shown in the viewer. Returns the
    recording stream."""
    rr, rrb = _rerun()
    blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin="plots", name=name))
    recording = _stream(rr, name, path, path is None if spawn is None else spawn, blueprint)
    _plot(rr, recording, "plots", table.index.to_numpy(), table.to_numpy(),
          [" ".join(map(str, c)) if isinstance(c, tuple) else c for c in table.columns])
    recording.flush()
    return recording
