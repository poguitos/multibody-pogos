// Frames, inertias and body states (plan task 6.1, D52).

#include <cmath>
#include <sstream>

#include <Eigen/Geometry>

#include "bindings.hpp"
#include "mbd/model/rigid_body.hpp"

namespace mbd::python {
namespace {

/// A rotation given from Python, checked: orthonormal to 1e-9 and proper
/// (determinant +1), MBD-P002 otherwise. The C++ constructors trust their
/// callers; a matrix typed by hand may be neither.
Quat checked_rotation(const Mat3& R)
{
    const Real orthonormality = (R.transpose() * R - Mat3::Identity()).norm();
    MBD_THROW_IF(!(orthonormality <= 1e-9) || !(R.determinant() > 0.0),
                 "MBD-P002: a rotation matrix must be orthonormal with determinant +1; |R^T R - I| = "
                     + std::to_string(orthonormality) + ", det R = " + std::to_string(R.determinant()));
    return Quat(R).normalized();
}

/// A quaternion [w, x, y, z] given from Python, of unit length to 1e-6
/// (MBD-P003 otherwise), then normalized to rounding.
Quat checked_quaternion(const QuatArray& a)
{
    const Real n = a.norm();
    MBD_THROW_IF(!(std::abs(n - 1.0) <= 1e-6),
                 "MBD-P003: a quaternion [w, x, y, z] must have unit length; its length is " + std::to_string(n));
    return to_quat(a / n);
}

std::string repr(const Transform3& X)
{
    std::ostringstream s;
    const QuatArray q = to_array(X.q);
    s << "Transform3(quaternion=[" << q(0) << ", " << q(1) << ", " << q(2) << ", " << q(3) << "], translation=["
      << X.p.x() << ", " << X.p.y() << ", " << X.p.z() << "])";
    return s.str();
}

} // namespace

void bind_core(py::module_& m)
{
    using namespace pybind11::literals;

    m.attr("g_accel") = g_accel;
    m.attr("pi") = pi;

    py::class_<Transform3>(m, "Transform3",
                           "A rigid transform from a local frame to a parent frame: x_parent = R x_local + p.")
        .def(py::init<>(), "The identity.")
        .def(py::init([](const Mat3& R, const Vec3& p) { return Transform3(checked_rotation(R), p); }),
             "rotation"_a, "translation"_a, "From a rotation matrix (checked, MBD-P002) and a translation.")
        .def(py::init(&Transform3::FromTranslation), "translation"_a,
             "A translation only: the local origin at `translation` in the parent frame. Wherever a "
             "Transform3 is expected, a list [x, y, z] stands for this.")
        .def_static("identity", &Transform3::Identity)
        .def_static("from_translation", &Transform3::FromTranslation, "translation"_a)
        .def_static("from_rotation", [](const Mat3& R) { return Transform3::FromRotation(checked_rotation(R)); },
                    "rotation"_a)
        .def_static("from_quaternion",
                    [](const QuatArray& q, const Vec3& p) { return Transform3(checked_quaternion(q), p); },
                    "quaternion"_a, "translation"_a = Vec3::Zero(),
                    "From a unit quaternion [w, x, y, z] (checked, MBD-P003) and a translation.")
        .def_static("from_axis_angle",
                    [](const Vec3& axis, Real angle, const Vec3& p) {
                        MBD_THROW_IF(!(axis.norm() > 0.0), "MBD-P004: a rotation axis must not be zero");
                        return Transform3(Quat(Eigen::AngleAxisd(angle, axis.normalized())), p);
                    },
                    "axis"_a, "angle"_a, "translation"_a = Vec3::Zero(),
                    "A rotation by `angle` [rad] about `axis` (normalized here), then a translation.")
        .def_property(
            "rotation", [](const Transform3& X) { return X.rotation_matrix(); },
            [](Transform3& X, const Mat3& R) { X.q = checked_rotation(R); }, "The rotation matrix R.")
        .def_property(
            "quaternion", [](const Transform3& X) { return to_array(X.q); },
            [](Transform3& X, const QuatArray& q) { X.q = checked_quaternion(q); }, "The rotation as [w, x, y, z].")
        .def_property(
            "translation", [](const Transform3& X) { return X.p; }, [](Transform3& X, const Vec3& p) { X.p = p; },
            "The translation p: the local origin in the parent frame.")
        .def("apply", &Transform3::apply, "point"_a, "A point of the local frame in the parent frame: R x + p.")
        .def("rotate", &Transform3::rotate, "vector"_a, "A vector of the local frame in the parent frame: R v.")
        .def("inverse", &Transform3::inverse, "The transform the other way, from the parent frame to the local one.")
        .def("__mul__", [](const Transform3& a, const Transform3& b) { return a * b; },
             "Composition: if a maps B to A and b maps C to B, a * b maps C to A.")
        .def("__repr__", &repr);
    // A sequence where a Transform3 is expected goes through the constructors
    // above: three numbers make a translation (task 6.2's tuning, D54); any
    // other sequence fails to convert, and the call reports the types it takes.
    py::implicitly_convertible<py::sequence, Transform3>();

    py::class_<RigidBodyInertia> inertia(m, "RigidBodyInertia",
                                         "Mass, centre of mass and rotational inertia, in a body's frame.");
    eigen_member(inertia, "com_B", &RigidBodyInertia::com_B, "The centre of mass [m].");
    eigen_member(inertia, "I_com_B", &RigidBodyInertia::I_com_B, "The inertia about the centre of mass [kg m^2].");
    inertia.def(py::init<>())
        .def(py::init<Real, const Vec3&, const Mat3&>(), "mass"_a, "com"_a, "inertia"_a,
             "mass [kg], centre of mass [m] and inertia about the centre of mass [kg m^2], all in the body frame.")
        .def_static("from_solid_box", &RigidBodyInertia::from_solid_box, "mass"_a, "half_extents"_a,
                    "A uniform box centred on the body origin, its edges along the body axes.")
        .def_readwrite("mass", &RigidBodyInertia::mass);

    py::class_<RigidBodyState>(m, "RigidBodyState", "A body's pose and velocity in world axes.")
        .def_readonly("p_WB", &RigidBodyState::p_WB, "The body origin [m].")
        .def_property_readonly("q_WB", [](const RigidBodyState& s) { return to_array(s.q_WB); },
                               "The orientation as a quaternion [w, x, y, z].")
        .def_property_readonly("R_WB", [](const RigidBodyState& s) { return Mat3(s.q_WB.toRotationMatrix()); },
                               "The orientation as a rotation matrix.")
        .def_readonly("v_WB", &RigidBodyState::v_WB, "The body origin's velocity [m/s].")
        .def_readonly("w_WB", &RigidBodyState::w_WB, "The angular velocity [rad/s].")
        .def("pose_WB", &RigidBodyState::pose_WB,
             "The pose as a Transform3 from the body frame to the world: x_W = R_WB x_B + p_WB.");
}

} // namespace mbd::python
