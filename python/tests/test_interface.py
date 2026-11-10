"""How the Python interface behaves: errors, arrays, frames, lifetimes (task 6.1, D52)."""

import gc
import importlib.metadata
import math
import types

import numpy as np
import pytest

import multibody as mbd


def one_body_system():
    system = mbd.System()
    system.model.add_body(0, mbd.RevoluteJointModel(), mbd.Transform3(),
                          mbd.Transform3.from_translation([0.0, 1.0, 0.0]),
                          mbd.RigidBodyInertia.from_solid_box(2.0, [0.1, 0.1, 0.1]), "link")
    return system


def test_the_version_is_the_last_release():
    assert mbd.__version__ == "0.6.0"


def test_an_installed_package_states_the_modules_version():
    # pip's metadata takes the version from CMakeLists.txt's project() line
    # (pyproject.toml), the module from the build's PROJECT_VERSION: the two
    # must agree. In the suite's build tree nothing is installed.
    try:
        installed = importlib.metadata.version("multibody")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("multibody is not installed with pip")
    assert installed == mbd.__version__


def test_a_compiled_module_older_than_the_python_files_is_named():
    # An editable install after a change to the bindings: the Python files
    # name a class (here ProjectionInfo, bound for task 6.3) that the module
    # of the last install lacks. The import says what to do (MBD-P011)
    # instead of failing on the missing attribute. The reprs are installed
    # only once every class is found, so the real classes are left as they
    # were.
    old = types.SimpleNamespace(**{name: getattr(mbd._mbd, name) for name in dir(mbd._mbd)
                                   if name != "ProjectionInfo"})
    with pytest.raises(ImportError) as caught:
        mbd._install(old)
    assert str(caught.value).startswith("MBD-P011: ")
    assert "ProjectionInfo" in str(caught.value) and "install the package again" in str(caught.value)
    assert repr(mbd.StaticOptions()).startswith("StaticOptions(")


def test_an_engine_error_is_an_mbd_error_with_its_code():
    # Model::add_body refuses a parent that does not exist (MBD-K010).
    model = mbd.System().model
    with pytest.raises(mbd.MbdError) as caught:
        model.add_body(5, mbd.RevoluteJointModel(), mbd.Transform3(), mbd.Transform3(), mbd.RigidBodyInertia())
    assert caught.value.code == "MBD-K010"
    assert str(caught.value).startswith("MBD-K010: ")
    assert isinstance(caught.value, RuntimeError)


def test_the_state_is_a_view_of_the_simulators_own_vector():
    sim = mbd.Simulator(one_body_system())
    q = sim.q
    q[0] = 0.25                       # through a view kept from before
    assert sim.q[0] == 0.25
    sim.v[:] = [3.0]                  # slice assignment writes in place too
    assert sim.v[0] == 3.0
    sim.q = np.array([0.5])           # assignment of a whole vector
    assert sim.q[0] == 0.5 and q[0] == 0.5


def test_a_state_of_the_wrong_size_is_refused():
    sim = mbd.Simulator(one_body_system())
    with pytest.raises(mbd.MbdError) as caught:
        sim.q = [1.0, 2.0]
    assert caught.value.code == "MBD-P001"
    assert sim.q.shape == (1,)
    with pytest.raises(mbd.MbdError) as caught:
        sim.acceleration(np.zeros(2), np.zeros(1), 0.0)
    assert caught.value.code == "MBD-P001"


def test_the_simulator_keeps_its_system_alive():
    sim = mbd.Simulator(one_body_system())   # no other reference to the system
    gc.collect()
    sim.q[0] = 0.1
    sim.initialize()
    assert sim.run(0.01, 1e-3) == 10
    assert sim.system.model.nbodies() == 2


def test_the_integrators_are_an_enum():
    sim = mbd.Simulator(one_body_system())
    assert sim.method == mbd.Integrator.RK4
    sim.method = mbd.Integrator.RadauIIA
    assert sim.method == mbd.Integrator.RadauIIA
    assert {m.name for m in mbd.Integrator} == {
        "RK4", "SemiImplicitEuler", "DormandPrince", "GeneralizedAlpha", "SDIRK4", "RadauIIA",
        "LinearlyImplicitEuler"}


def test_settings_are_written_through():
    sim = mbd.Simulator(one_body_system())
    sim.step_control.relative_tolerance = 1e-9
    sim.implicit_control.rho_infinity = 0.5
    assert sim.step_control.relative_tolerance == 1e-9
    assert sim.implicit_control.rho_infinity == 0.5
    sim.system.model.gravity[2] = -1.62
    assert sim.system.model.gravity[2] == -1.62
    with pytest.raises(TypeError):
        sim.system.model.gravity = [0.0, -9.81]      # a Vec3 has three entries
    inertia = mbd.RigidBodyInertia.from_solid_box(1.0, [0.5, 0.5, 0.5])
    inertia.com_B[0] = 0.1
    inertia.I_com_B[2, 2] = 0.3
    assert inertia.com_B[0] == 0.1 and inertia.I_com_B[2, 2] == 0.3


def test_transforms_compose_invert_and_check_their_rotations():
    # A quarter turn about Z maps X to Y; then a translation.
    X = mbd.Transform3.from_axis_angle([0.0, 0.0, 2.0], math.pi / 2.0, [1.0, 2.0, 3.0])
    assert np.allclose(X.apply([1.0, 0.0, 0.0]), [1.0, 3.0, 3.0], atol=1e-15)
    assert np.allclose(X.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)
    I = X * X.inverse()
    assert np.allclose(I.rotation, np.eye(3), atol=1e-15) and np.allclose(I.translation, 0.0, atol=1e-15)
    # The quaternion of a quarter turn about Z: [cos 45, 0, 0, sin 45].
    h = math.sqrt(0.5)
    assert np.allclose(X.quaternion, [h, 0.0, 0.0, h], atol=1e-15)
    with pytest.raises(mbd.MbdError) as caught:
        mbd.Transform3(np.diag([1.0, 1.0, -1.0]), np.zeros(3))   # a reflection
    assert caught.value.code == "MBD-P002"
    with pytest.raises(mbd.MbdError) as caught:
        mbd.Transform3.from_quaternion([2.0, 0.0, 0.0, 0.0])
    assert caught.value.code == "MBD-P003"
    with pytest.raises(mbd.MbdError) as caught:
        mbd.Transform3.from_axis_angle([0.0, 0.0, 0.0], 1.0)
    assert caught.value.code == "MBD-P004"


def test_the_model_reports_its_tree():
    model = one_body_system().model
    assert model.nbodies() == 2 and model.nq == 1 and model.nv == 1
    assert model.parent == [-1, 0]
    assert model.name[1] == "link"
    assert model.joint[0] is None and model.joint[1].name() == "revolute"
    assert model.inertia[1].mass == 2.0
    assert np.allclose(model.X_CJ[1].translation, [0.0, 1.0, 0.0])
