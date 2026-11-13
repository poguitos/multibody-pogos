"""The examples of the how-to guide docs/help/howto/python.md (task H.6 for
Phase 6), each run and checked against a value derived from the physics. The
guide quotes the code between the example markers exactly
(tests/howto/test_howto_pages.cpp checks it).

The model of the examples: a block of M = 2 kg sliding along Z, hung from a
spring of K = 50 N/m whose free length, 1 m, is the distance from its upper
end to the block's origin at q = 0. Gravity -Z. At rest it hangs M G / K =
0.3924 m lower; it swings about that at w = sqrt(K / M) = 5 rad/s."""

import math
import warnings

import numpy as np
import pytest

import multibody as mbd
from models import parallelogram

G, M, K = 9.81, 2.0, 50.0


def block_on_a_spring():
    # [example: python-build]
    system = mbd.System()
    system.model.gravity = [0.0, 0.0, -9.81]
    block = system.model.add_body("ground", mbd.PrismaticJointModel(), name="block",      # slides along Z
                                  inertia=mbd.RigidBodyInertia.from_solid_box(2.0, [0.05, 0.05, 0.05]))
    # A spring of 50 N/m with a free length of 1 m, from 1 m above the block's origin to it:
    system.add_force_element(mbd.LinearSpringDamper(0, block, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 50.0, 0.0, 1.0))
    # [end example]
    return system, block


def test_a_model_is_built_and_run():
    # Released at rest at the spring's free length, the block falls and
    # swings: q(t) = -(M G / K)(1 - cos w t). Dormand-Prince's default
    # tolerance keeps each step's error below about 1e-6; over 200 steps of
    # 10 ms, on an oscillator that neither amplifies nor damps an error, the
    # sum is below 2e-4 m. (Measured: 3.3e-10 m.)
    system, block = block_on_a_spring()
    # [example: python-run]
    sim = mbd.Simulator(system)
    sim.method = mbd.Integrator.DormandPrince     # to a tolerance; the default is RK4 at a fixed step
    sim.initialize()                              # from the state as set: here q = 0 and v = 0
    run = sim.run_recorded(2.0, 0.01)             # NumPy arrays: run["time"], run["q"], run["v"]
    # [end example]
    w = math.sqrt(K / M)
    expected = -(M * G / K) * (1.0 - np.cos(w * run["time"]))
    assert run["q"].shape == (201, 1)
    assert np.max(np.abs(run["q"][:, 0] - expected)) < 2e-4


def test_the_rest_and_the_modes_are_found():
    # At rest q = -M G / K; statics stops when the acceleration is below
    # 1e-6 m/s^2, within 1e-6 / w^2 = 4e-8 m of it: allowed 1e-7 m. The one
    # mode is at w = 5 rad/s, undamped; the model is linear, so the
    # linearisation's central differences are exact but for rounding:
    # allowed 1e-8 relative.
    system, block = block_on_a_spring()
    # [example: python-rest-and-modes]
    sim = mbd.Simulator(system)
    rest = mbd.static_equilibrium(sim)            # moves the state to rest; print(rest.summary(system.model))
    linear = mbd.linearize(sim)                   # about the state the simulator holds
    for mode in linear.modes:
        print(f"{mode.natural_frequency:.4f} rad/s, damping ratio {mode.damping_ratio:.3f}")
    # [end example]
    assert rest.converged
    assert abs(sim.q[0] + M * G / K) < 1e-7
    assert len(linear.modes) == 1
    assert linear.modes[0].natural_frequency == pytest.approx(math.sqrt(K / M), rel=1e-8)
    assert abs(linear.modes[0].damping_ratio) < 1e-8


def test_a_force_written_in_python_acts():
    # A damper written in Python, F = -c v with c = 4 N s/m: the linearised
    # mode then has the damping ratio c / (2 sqrt(K M)) = 0.2 at the same
    # natural frequency, 5 rad/s. The force is linear in v, so the central
    # differences are exact but for rounding: allowed 1e-8 relative.
    system, block = block_on_a_spring()
    # [example: python-user-force]
    def drag(states, forces):                     # called at every evaluation of the forces
        body = states[block]                      # the bodies' world states; index 0 is the ground
        forces.add_force(block, -4.0 * body.v_WB, body.p_WB)     # [N], world axes, at the body's origin

    system.add_force_element(mbd.UserForce("drag", [block], drag))
    # [end example]
    sim = mbd.Simulator(system)
    mbd.static_equilibrium(sim)
    mode = mbd.linearize(sim).modes[0]
    assert mode.damping_ratio == pytest.approx(4.0 / (2.0 * math.sqrt(K * M)), rel=1e-8)
    assert mode.natural_frequency == pytest.approx(math.sqrt(K / M), rel=1e-8)


def test_a_model_and_a_run_are_saved_and_read_back(tmp_path, monkeypatch):
    # A model file and a results file both keep every number (D56): the
    # model read back runs bit for bit as the original, and the table read
    # back holds the run's numbers exactly.
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    monkeypatch.chdir(tmp_path)
    system, block = block_on_a_spring()
    sim = mbd.Simulator(system)
    sim.initialize()
    run = sim.run_recorded(0.5, 0.01)
    # [example: python-files]
    mbd.save_model(system, "block.json")          # JSON, as docs/model-files.md describes
    system = mbd.load_model("block.json")
    table = mbd.dataframe(run, sim)               # indexed by time; table["q"]["block"], table[("v", "block")]
    mbd.write_results(table, "run.parquet")       # or "run.csv"; both keep every number exactly
    table = mbd.read_results("run.parquet")
    # [end example]
    again = mbd.Simulator(system)
    again.initialize()
    assert np.array_equal(again.run_recorded(0.5, 0.01)["q"], run["q"])
    assert np.array_equal(table["q"]["block"].to_numpy(), run["q"][:, 0])
    assert np.array_equal(table.index.to_numpy(), run["time"])


def test_a_run_is_played_back(tmp_path, monkeypatch, capfd):
    # What the view draws is checked against closed forms in test_viewer.py;
    # here, that the example writes its recording and Rerun has nothing to
    # warn about.
    pytest.importorskip("rerun")
    monkeypatch.chdir(tmp_path)
    system, block = block_on_a_spring()
    sim = mbd.Simulator(system)
    sim.initialize()
    run = sim.run_recorded(1.0, 0.01)
    # [example: python-replay]
    mbd.replay(system, run, name="block", path="block.rrd")     # without path: the viewer itself
    # [end example]
    assert (tmp_path / "block.rrd").stat().st_size > 10_000
    assert "WARN" not in capfd.readouterr().err


def test_the_engines_errors_and_warnings_are_caught():
    # A misspelt setting is refused with MBD-P010 (D54). A planar loop under
    # a spatial hinge has redundant equations, which the engine reports with
    # the warning MBD-K050 as it runs; with the filter, that warning is
    # raised and stops the run.
    with warnings.catch_warnings():
        # [example: python-errors]
        try:
            options = mbd.StaticOptions(max_iteration=50)          # misspelt: max_iterations
        except mbd.MbdError as error:
            code = error.code                     # "MBD-P010"; docs/help/messages.md explains each code
        warnings.simplefilter("error", mbd.MbdWarning)    # the engine's warnings raised as errors from here on
        # [end example]
        assert code == "MBD-P010" and "options" not in locals()
        sim = mbd.Simulator(parallelogram())
        sim.q[:] = [1.0, -1.0, 1.0]
        with pytest.raises(mbd.MbdWarning) as raised:
            sim.initialize()
            sim.run_recorded(1.0, 1e-3)
        assert raised.value.code == "MBD-K050"
