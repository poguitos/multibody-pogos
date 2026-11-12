"""Viewing (plan task 6.5, D57): what replay() draws, checked against closed
forms without a viewer, and recordings written for the four-bar and a car.

The scene (multibody._viewer._scene) holds every number replay() sends to
Rerun; Rerun's own drawing of them is not tested here."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import multibody as mbd
from multibody import _viewer
from models import parallelogram, pendulum, slider, vertical_spring


@pytest.fixture(autouse=True)
def quiet_redundancy():
    # The parallelogram's planar loop under a spatial hinge: MBD-K050 at
    # every solve (test_constraints_and_forces.py checks that warning).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MBD-K050", category=mbd.MbdWarning)
        yield


def test_a_pendulums_poses_are_its_closed_form():
    # The bob of models.py hangs L = 0.8 m below the origin at q = 0 and
    # swings about Z: at angle q its frame is at (L sin q, -L cos q, 0), turned
    # by q about Z, the quaternion (cos q/2, 0, 0, sin q/2). Rounding of the
    # products alone: allowed 1e-15.
    system = pendulum()
    time = np.linspace(0.0, 1.0, 11)
    q = 0.7 * np.sin(3.0 * time)[:, None]
    scene = _viewer._scene(system, (time, q), fps=None)
    p, quat = scene["positions"][:, 1], scene["quaternions"][:, 1]
    assert np.allclose(p, np.column_stack([0.8 * np.sin(q[:, 0]), -0.8 * np.cos(q[:, 0]), 0 * q[:, 0]]),
                       rtol=0, atol=1e-15)
    assert np.allclose(quat, np.column_stack([np.cos(q[:, 0] / 2), 0 * q[:, 0], 0 * q[:, 0], np.sin(q[:, 0] / 2)]),
                       rtol=0, atol=1e-15)
    assert scene["up"] == "RIGHT_HAND_Y_UP"            # gravity along -Y
    assert scene["q_names"] == ["bob"]


def test_frames_are_taken_at_the_rate_asked():
    # Samples every 1 ms for 1 s at 50 frames per second: the samples at 0,
    # 20, 40, ... ms, the last (1 s) included: 51 frames.
    time = np.arange(1001) * 1e-3
    frames = _viewer._frames(time, 50.0)
    assert list(frames) == list(range(0, 1001, 20))
    assert list(_viewer._frames(time, None)) == list(range(1001))


def test_a_springs_line_runs_between_its_anchors():
    # The slider's spring of models.py runs from (0, 0, -1) on the ground to
    # the slider's origin at (0, 0, q).
    system = slider(1.0)
    vertical_spring(system, 50.0)
    time = np.array([0.0, 0.5, 1.0])
    q = np.array([[0.0], [0.25], [-0.1]])
    line = _viewer._scene(system, (time, q), fps=None)["lines"][0]
    assert line["label"] == "0 spring-damper"
    expected = np.stack([np.tile([0.0, 0.0, -1.0], (3, 1)), np.column_stack([0 * q[:, 0], 0 * q[:, 0], q[:, 0]])],
                        axis=1)
    assert np.array_equal(line["ends"], expected)


def test_a_body_is_drawn_as_the_box_of_its_inertia():
    # A uniform box of half sizes (0.3, 0.2, 0.1) has those half sizes back
    # from its moments; the axes may come in another order.
    inertia = mbd.RigidBodyInertia.from_solid_box(5.0, [0.3, 0.2, 0.1])
    half, centre, axes = _viewer._inertia_box(inertia)
    assert sorted(half) == pytest.approx([0.1, 0.2, 0.3], rel=1e-12)
    assert np.allclose(np.abs(axes) @ half, [0.3, 0.2, 0.1], rtol=1e-12)
    assert np.linalg.det(axes) == pytest.approx(1.0)
    assert _viewer._inertia_box(mbd.RigidBodyInertia(1.0, [0, 0, 0], np.zeros((3, 3)))) is None   # a point mass


def test_the_motion_may_be_a_run_a_table_or_a_csv(tmp_path):
    pd = pytest.importorskip("pandas")
    sim = mbd.Simulator(parallelogram())
    sim.q[:] = [1.0, -1.0, 1.0]
    sim.v[:] = [2.0, -2.0, 2.0]
    sim.initialize()
    run = sim.run_recorded(0.2, 1e-3)
    from_run = _viewer._scene(sim.system, run, fps=25)
    from_table = _viewer._scene(sim.system, mbd.dataframe(run, sim), fps=25)
    # As a C++ program writes it: one level, q[0] ...
    csv = tmp_path / "run.csv"
    flat = pd.DataFrame(run["q"], index=pd.Index(run["time"], name="time"),
                        columns=[f"q[{k}]" for k in range(3)])
    flat.to_csv(csv)
    from_csv = _viewer._scene(sim.system, mbd.read_results(csv), fps=25)
    for scene in (from_table, from_csv):
        assert np.array_equal(scene["positions"], from_run["positions"])
        assert np.array_equal(scene["frame_time"], from_run["frame_time"])
    assert len(from_run["frame_time"]) == 6               # 0, 40, ..., 200 ms
    assert from_run["v"] is not None and from_csv["v"] is None


def toy_car():
    """A chassis on a free joint, four wheels on vertical sliders under it,
    each held by a spring and touching the ground through a sphere: a car
    in the kernel's parts."""
    system = mbd.System()
    model = system.model
    model.add_body(0, mbd.FreeJointModel(), inertia=mbd.RigidBodyInertia.from_solid_box(800.0, [2.0, 0.8, 0.3]),
                   name="chassis")
    contact = mbd.ContactParams(stiffness=2e5, damping=2e3)
    for x in (1.4, -1.4):
        for y in (0.8, -0.8):
            wheel = model.add_body("chassis", mbd.PrismaticJointModel(), x_pj=[x, y, -0.3],
                                   inertia=mbd.RigidBodyInertia.from_solid_box(20.0, [0.3, 0.1, 0.3]),
                                   name="wheel")
            system.add_force_element(mbd.LinearSpringDamper(1, wheel, [x, y, 0.0], [0.0, 0.0, 0.0],
                                                            4e4, 3e3, 0.3))
            system.add_force_element(mbd.PlaneContact(wheel, [[0.0, 0.0, 0.0]], 0.3, 0, mbd.Transform3(), contact))
    return system


def test_the_four_bar_and_a_car_are_written_to_recordings(tmp_path, capfd):
    pytest.importorskip("rerun")
    sim = mbd.Simulator(parallelogram())
    sim.q[:] = [1.0, -1.0, 1.0]
    sim.v[:] = [2.0, -2.0, 2.0]
    sim.initialize()
    path = tmp_path / "four_bar.rrd"
    mbd.replay(sim.system, sim.run_recorded(1.0, 1e-3), name="four-bar", path=path)
    assert path.stat().st_size > 10_000

    car = toy_car()
    sim = mbd.Simulator(car)
    sim.q[2] = 0.9                                          # the chassis above its rest
    sim.initialize()
    run = sim.run_recorded(0.5, 1e-3)
    scene = _viewer._scene(car, run, fps=30)
    assert [b["label"] for b in scene["bodies"]] == ["ground", "chassis", "wheel#2", "wheel#3", "wheel#4", "wheel#5"]
    assert len(scene["lines"]) == 4                         # the springs; the contacts as spheres on the wheels
    assert all(len(b["points"]) == 1 for b in scene["bodies"][2:])
    assert scene["up"] == "RIGHT_HAND_Z_UP"
    path = tmp_path / "car.rrd"
    pd = pytest.importorskip("pandas")
    channels = pd.DataFrame({"chassis height": run["q"][:, 2]}, index=pd.Index(run["time"], name="time"))
    mbd.replay(car, run, name="car", path=path, channels=channels)
    assert path.stat().st_size > 10_000
    mbd.plot(channels, path=tmp_path / "channels.rrd")
    assert (tmp_path / "channels.rrd").stat().st_size > 0
    # Rerun warns, on the process's own stderr, about an entity path it had to
    # read by guessing (a first version left a space unescaped in the
    # closures' and contacts' points); none may be left.
    assert "WARN" not in capfd.readouterr().err


def test_every_replay_is_a_recording_of_its_own(tmp_path):
    # Rerun merges what arrives under one recording id into one recording:
    # with the id taken from the name (6.5), a notebook cell run twice sent
    # both runs into one recording in the viewer. Two replays under one name
    # are now two recordings, under the application of that name (task 6.6).
    pytest.importorskip("rerun")
    sim = mbd.Simulator(pendulum())
    sim.q[0] = 0.5
    sim.initialize()
    run = sim.run_recorded(0.1, 1e-2)
    first = mbd.replay(sim.system, run, name="pendulum", path=tmp_path / "1.rrd")
    second = mbd.replay(sim.system, run, name="pendulum", path=tmp_path / "2.rrd")
    assert first.get_recording_id() != second.get_recording_id()
    assert first.get_application_id() == second.get_application_id() == "pendulum"
    # The viewer is started from the package's own copy, found where Rerun's
    # launcher (rerun_cli/__main__.py) looks: the PATH of a notebook's kernel
    # may not reach the environment's scripts.
    executable = _viewer._viewer_executable()
    assert executable is not None
    import rerun_cli
    assert Path(executable).parent == Path(rerun_cli.__file__).parent and Path(executable).is_file()


def test_viewing_says_what_is_wrong(monkeypatch):
    system = parallelogram()
    with pytest.raises(mbd.MbdError) as caught:
        _viewer._scene(system, (np.zeros(3), np.zeros((3, 2))))
    assert caught.value.code == "MBD-P013"
    with pytest.raises(TypeError):
        _viewer._scene(system, [1, 2, 3])
    monkeypatch.setitem(sys.modules, "rerun", None)          # as if Rerun were not installed
    with pytest.raises(ImportError) as missing:
        mbd.replay(system, (np.zeros(1), np.zeros((1, 3))), path="x.rrd")
    assert str(missing.value).startswith("MBD-P015: ")
