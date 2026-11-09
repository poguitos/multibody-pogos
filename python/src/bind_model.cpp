// Joint models and the model tree (plan task 6.1, D52).

#include <vector>

#include "bindings.hpp"
#include "mbd/kernel/joint_model.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::python {

void bind_model(py::module_& m)
{
    using namespace pybind11::literals;
    using namespace mbd::kernel;

    // Joint models hold no state; one instance may serve several bodies. The
    // smart holder lets a model keep them as shared_ptr<const JointModel>.
    py::classh<JointModel>(m, "JointModel", "A joint's kinematics; the joint axis is the joint frame's Z axis.")
        .def("name", &JointModel::name)
        .def("nq", &JointModel::nq, "Number of coordinates.")
        .def("nv", &JointModel::nv, "Number of velocities (degrees of freedom).")
        .def("__repr__", [](const JointModel& j) { return std::string(j.name()) + " joint"; });

    py::classh<RevoluteJointModel, JointModel>(m, "RevoluteJointModel", "Rotation about Z: q the angle [rad].")
        .def(py::init<>());
    py::classh<PrismaticJointModel, JointModel>(m, "PrismaticJointModel", "Translation along Z: q the distance [m].")
        .def(py::init<>());
    py::classh<FixedJointModel, JointModel>(m, "FixedJointModel", "No motion: no coordinates.").def(py::init<>());
    py::classh<UniversalJointModel, JointModel>(m, "UniversalJointModel", "Two rotations, about X then Y.")
        .def(py::init<>());
    py::classh<CylindricalJointModel, JointModel>(m, "CylindricalJointModel",
                                                   "Rotation about and translation along Z.")
        .def(py::init<>());
    py::classh<PlanarJointModel, JointModel>(m, "PlanarJointModel",
                                              "Translation in the XY plane and rotation about Z.")
        .def(py::init<>());
    py::classh<SphericalJointModel, JointModel>(m, "SphericalJointModel",
                                                 "Any rotation: q a quaternion [w, x, y, z], v the angular velocity.")
        .def(py::init<>());
    py::classh<FreeJointModel, JointModel>(
        m, "FreeJointModel", "No constraint: q the position and a quaternion [w, x, y, z], v the twist.")
        .def(py::init<>());

    py::class_<Model> model(m, "Model", "The tree of bodies and joints. Body 0 is the ground.");
    eigen_member(model, "gravity", &Model::gravity, "Gravitational acceleration in world axes [m/s^2].");
    model.def(py::init<>(), "Ground only.")
        .def("add_body", &Model::add_body, "parent"_a, "joint"_a, "x_pj"_a, "x_cj"_a, "inertia"_a, "name"_a = "",
             "Add a body attached to `parent` (0 for ground) by `joint`; x_pj is the parent-side joint frame "
             "in the parent body's frame, x_cj the child-side one in the new body's frame, `inertia` in the new "
             "body's frame. Returns the new body's index.")
        .def("nbodies", &Model::nbodies, "Number of bodies, ground included.")
        .def_readonly("nq", &Model::nq, "Total number of coordinates.")
        .def_readonly("nv", &Model::nv, "Total number of velocities.")
        .def("neutral_configuration", &Model::neutral_configuration,
             "Every joint at its neutral coordinates (identity quaternions, not zeros).")
        // The per-body tables, as copies: change the tree only through add_body.
        .def_property_readonly("parent", [](const Model& model) { return model.parent; })
        .def_property_readonly("name", [](const Model& model) { return model.name; })
        .def_property_readonly("joint", [](const Model& model) { return model.joint; })
        .def_property_readonly("inertia", [](const Model& model) { return model.inertia; })
        .def_property_readonly("idx_q", [](const Model& model) { return model.idx_q; },
                               "Each body's first coordinate.")
        .def_property_readonly("idx_v", [](const Model& model) { return model.idx_v; },
                               "Each body's first velocity.")
        .def_property_readonly("nqs", [](const Model& model) { return model.nqs; })
        .def_property_readonly("nvs", [](const Model& model) { return model.nvs; })
        .def_property_readonly("X_PJ", [](const Model& model) { return model.X_PJ; })
        .def_property_readonly("X_CJ", [](const Model& model) { return model.X_CJ; });
}

} // namespace mbd::python
