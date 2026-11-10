"""Model files and results as tables, from Python (plan task 6.4, D56).

The format itself is tested in C++ (tests/io/test_model_files.cpp: every kind
of part, the documented text, every error); here, that the package reaches
it, and the tables of results."""

import sys
import warnings

import numpy as np
import pytest

import multibody as mbd
from models import parallelogram


@pytest.fixture(autouse=True)
def quiet_redundancy():
    # The parallelogram's planar loop under a spatial hinge: MBD-K050 at
    # every solve (test_constraints_and_forces.py checks that warning).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MBD-K050", category=mbd.MbdWarning)
        yield


def bits(a):
    """An array's numbers as their bit patterns: equal only if identical."""
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


def test_a_model_built_in_python_runs_the_same_after_a_file(tmp_path):
    # Written, read back and written again, the text is the same; the model
    # read back runs 0.5 s bit for bit as the original (the C++ test shows the
    # same for every kind of part).
    system = parallelogram()
    system.add_force_element(mbd.LinearSpringDamper(0, 2, [0.4, 0.5, 0.0], [0.4, 0.0, 0.0], 300.0, 2.0, 0.2))
    spring = mbd.SpringDamperParams(free_length=0.3, preload=5.0,
                                    spring=mbd.Curve.table([-0.1, 0.0, 0.1], [-50.0, 0.0, 80.0]))
    system.add_force_element(mbd.SpringDamper(0, 3, [0.8, -0.5, 0.0], [-0.3, 0.0, 0.0], spring))
    system.add_joint_force(mbd.JointCoordinateForce(system.model, 1,
                                                    mbd.JointCoordinateForceParams(damper=mbd.Curve.linear(0.5))))
    path = tmp_path / "parallelogram.json"
    mbd.save_model(system, path)                     # a pathlib.Path, or a str
    loaded = mbd.load_model(str(path))
    assert mbd.model_to_json(loaded) == mbd.model_to_json(system) == path.read_text()
    assert mbd.describe(loaded) == mbd.describe(system)

    runs = []
    for s in (system, loaded):
        sim = mbd.Simulator(s)
        sim.q[:] = [1.0, -1.0, 1.0]
        sim.v[:] = [2.0, -2.0, 2.0]
        sim.initialize()
        runs.append(sim.run_recorded(0.5, 1e-3))
    for key in ("q", "v"):
        assert (bits(runs[0][key]) == bits(runs[1][key])).all()
    assert runs[0]["q"][-1, 0] != 1.0                # it moved


def test_errors_in_a_file_name_their_place():
    text = '{"format": "multibody model", "version": 1, "gravity": [0, -9.81]}'
    with pytest.raises(mbd.MbdError) as caught:
        mbd.model_from_json(text)
    assert caught.value.code == "MBD-I006"
    assert str(caught.value).startswith("MBD-I006: gravity: expected 3 numbers")
    with pytest.raises(mbd.MbdError) as caught:
        mbd.model_from_json('{"format": "multibody model", "version": 1, "gravty": [0, 0, -9.81]}')
    assert caught.value.code == "MBD-I004" and "unknown key 'gravty'" in str(caught.value)


def test_a_force_written_in_python_is_refused_by_name():
    system = parallelogram()
    system.add_force_element(mbd.UserForce("push", [1], lambda states, forces: None))
    with pytest.raises(mbd.MbdError) as caught:
        mbd.model_to_json(system)
    assert caught.value.code == "MBD-I008"
    assert "force_elements[0]: a force written as code (UserForce 'push')" in str(caught.value)


def test_a_curve_shows_what_it_was_made_from():
    table = mbd.Curve.table([0.0, 0.01, 0.02], [0.0, 2000.0, 8000.0])
    assert table.is_table() and table.table_x() == [0.0, 0.01, 0.02] and table.table_y() == [0.0, 2000.0, 8000.0]
    line = mbd.Curve.linear(150.0, -3.0)
    assert not line.is_table() and line.line_slope() == 150.0 and line.line_offset() == -3.0


class Pair(mbd.StateElement):
    """Two states that do nothing but grow: x' = (1, 2)."""

    def __init__(self):
        super().__init__()

    def name(self):
        return "pair"

    def size(self):
        return 2

    def initial_states(self, x):
        x[:] = 0.0

    def rates(self, sim, t, x, x_dot):
        x_dot[:] = [1.0, 2.0]


def five_bodies():
    """A body on each kind of naming: one coordinate (arm), several (a free
    block), no name (body 3), and a name two bodies share (twin)."""
    system = mbd.System()
    model = system.model
    inertia = mbd.RigidBodyInertia.from_solid_box(1.0, [0.1, 0.1, 0.1])
    model.add_body(0, mbd.RevoluteJointModel(), inertia=inertia, name="arm")
    model.add_body(1, mbd.FreeJointModel(), inertia=inertia, name="block")
    model.add_body(0, mbd.PrismaticJointModel(), inertia=inertia)
    model.add_body(0, mbd.PrismaticJointModel(), inertia=inertia, name="twin")
    model.add_body(0, mbd.PrismaticJointModel(), inertia=inertia, name="twin")
    return system


def test_a_run_as_a_table_names_its_columns_after_the_bodies():
    pytest.importorskip("pandas")
    sim = mbd.Simulator(five_bodies())
    sim.attach(Pair())
    sim.initialize()
    run = sim.run_recorded(0.01, 1e-3)
    df = mbd.dataframe(run, sim)

    block_q = [f"block[{k}]" for k in range(7)]       # x, y, z, then the quaternion's x, y, z, w
    block_v = [f"block[{k}]" for k in range(6)]
    assert list(df.columns.names) == ["vector", "coordinate"]
    assert list(df["q"].columns) == ["arm", *block_q, "body3", "twin#4", "twin#5"]
    assert list(df["v"].columns) == ["arm", *block_v, "body3", "twin#4", "twin#5"]
    assert list(df["x"].columns) == ["0", "1"]
    assert df.index.name == "time" and (bits(df.index.to_numpy()) == bits(run["time"])).all()
    for key in ("q", "v", "x"):
        assert (bits(df[key].to_numpy()) == bits(run[key])).all()
    assert df["q"]["arm"].iloc[-1] == run["q"][-1, 0] and df[("v", "twin#5")].iloc[-1] == run["v"][-1, -1]
    assert mbd.coordinate_names(sim.system.model, "v") == list(df["v"].columns)


@pytest.mark.parametrize("ending", [".csv", ".parquet"])
def test_a_table_survives_a_file_to_the_last_bit(tmp_path, ending):
    pd = pytest.importorskip("pandas")
    if ending == ".parquet":
        pytest.importorskip("pyarrow")
    sim = mbd.Simulator(five_bodies())
    sim.attach(Pair())
    sim.q[0] = 0.1 + 0.2                            # 0.30000000000000004, which a short form would lose
    sim.v[0] = -0.0
    sim.initialize()
    df = mbd.dataframe(sim.run_recorded(0.01, 1e-3), sim)
    path = tmp_path / ("run" + ending)
    mbd.write_results(df, path)
    back = mbd.read_results(path)
    pd.testing.assert_frame_equal(back, df, check_exact=True)
    assert (bits(back.to_numpy()) == bits(df.to_numpy())).all()       # -0.0 kept as well
    assert (bits(back.index.to_numpy()) == bits(df.index.to_numpy())).all()


def test_a_recorders_channels_as_a_table(tmp_path):
    pd = pytest.importorskip("pandas")
    recorder = mbd.Recorder()
    x = [0.0]
    recorder.add("x", lambda: x[0])
    recorder.add("x_squared", lambda: x[0] * x[0])
    for k in range(4):
        x[0] = 0.5 * k
        recorder.sample(0.1 * k)
    df = mbd.dataframe(recorder)
    assert list(df.columns) == ["x", "x_squared"] and df.index.name == "time"
    assert df["x_squared"].iloc[3] == 2.25
    # The recorder's own CSV (17 digits) reads back as the same table.
    recorder.write_csv(str(tmp_path / "channels.csv"))
    pd.testing.assert_frame_equal(mbd.read_results(tmp_path / "channels.csv"), df, check_exact=True)


def test_tables_say_what_is_wrong(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    sim = mbd.Simulator(five_bodies())
    run = sim.run_recorded(0.002, 1e-3)
    other = mbd.Simulator(parallelogram())
    with pytest.raises(mbd.MbdError) as caught:
        mbd.dataframe(run, other)
    assert caught.value.code == "MBD-P013"
    with pytest.raises(TypeError):
        mbd.dataframe(run)                           # the names need the simulator
    with pytest.raises(mbd.MbdError) as caught:
        mbd.write_results(mbd.dataframe(run, sim), tmp_path / "run.xlsx")
    assert caught.value.code == "MBD-P014"
    monkeypatch.setitem(sys.modules, "pandas", None)  # as if pandas were not installed
    with pytest.raises(ImportError) as missing:
        mbd.dataframe(run, sim)
    assert str(missing.value).startswith("MBD-P012: ")
