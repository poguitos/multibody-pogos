// Force laws and the force elements: springs, dampers, bushings, contact,
// forces written in Python, and forces on joint coordinates (plan task 6.1,
// D52). The tyre, aerodynamic and anti-roll-bar elements belong to the
// vehicle layer and are bound with it (Phases 7 and 8 rebuild them).

#include <array>
#include <memory>
#include <optional>

#include "bindings.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/contact.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/user_force.hpp"
#include "mbd/kernel/joint_forces.hpp"

namespace mbd::python {
namespace {

/// What a force written in Python adds to the bodies, valid during its call
/// only: afterwards a kept reference raises MBD-P005 rather than writing into
/// forces that are gone. Forces and torques are in world axes; a force at a
/// point adds its moment about the body's origin, where the dynamics reads
/// the loads.
class BodyForces {
public:
    BodyForces(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces)
        : states_(&states), forces_(&forces)
    {}

    void add_force(int body, const Vec3& f_W, const std::optional<Vec3>& point_W)
    {
        check(body);
        (*forces_)[static_cast<std::size_t>(body)].f_W += f_W;
        if (point_W) {
            const Vec3 r = *point_W - (*states_)[static_cast<std::size_t>(body)].p_WB;
            (*forces_)[static_cast<std::size_t>(body)].tau_W += r.cross(f_W);
        }
    }

    void add_torque(int body, const Vec3& tau_W)
    {
        check(body);
        (*forces_)[static_cast<std::size_t>(body)].tau_W += tau_W;
    }

    void close()
    {
        states_ = nullptr;
        forces_ = nullptr;
    }

private:
    void check(int body) const
    {
        MBD_THROW_IF(!forces_, "MBD-P005: BodyForces used after its user force returned; add the forces "
                               "during the call");
        MBD_THROW_IF(body < 0 || body >= static_cast<int>(forces_->size()),
                     "MBD-P006: BodyForces: no body " + std::to_string(body) + "; the model has "
                         + std::to_string(forces_->size()) + " bodies, ground included");
    }

    const std::vector<RigidBodyState>* states_;
    std::vector<RigidBodyForces>* forces_;
};

/// A curve triple as a Python tuple (immutable: an item assignment would
/// change a copy), and back from any sequence of three.
py::tuple curves(const std::array<Curve, 3>& c) { return py::make_tuple(c[0], c[1], c[2]); }

} // namespace

void bind_forces(py::module_& m)
{
    using namespace pybind11::literals;
    using namespace mbd::kernel;

    py::class_<Curve>(m, "Curve", "A force law: zero, linear, or a smooth curve through tabulated points.")
        .def(py::init<>(), "The zero function.")
        .def_static("linear", &Curve::linear, "slope"_a, "offset"_a = 0.0, "f(x) = offset + slope x.")
        .def_static("table", &Curve::table, "x"_a, "y"_a,
                    "Through the points (x[k], y[k]); x strictly increasing, at least two points.")
        .def("value", &Curve::value, "x"_a, "f(x).")
        .def("slope", &Curve::slope, "x"_a, "df/dx at x.")
        .def("integral", &Curve::integral, "a"_a, "b"_a, "The integral from a to b: a spring's stored energy.")
        .def("is_zero", &Curve::is_zero, "True for the zero function.")
        .def("is_table", &Curve::is_table, "True for a curve through tabulated points (Curve.table).")
        .def("table_x", &Curve::table_x, "A table's x values, as given (empty for a line).")
        .def("table_y", &Curve::table_y, "A table's y values, as given (empty for a line).")
        .def("line_slope", &Curve::line_slope, "A line's slope (Curve.linear); 0 for a table.")
        .def("line_offset", &Curve::line_offset, "A line's offset (Curve.linear); 0 for a table.")
        .def("__call__", &Curve::value, "x"_a);

    py::class_<LawValue>(m, "LawValue", "A force law's value at a coordinate and rate, with its derivatives.")
        .def_readonly("force", &LawValue::force)
        .def_readonly("d_position", &LawValue::d_position)
        .def_readonly("d_rate", &LawValue::d_rate);

    py::classh<ForceElement>(m, "ForceElement", "A force between bodies, read from their states.")
        .def("name", &ForceElement::name, "The element's kind, in words.")
        .def("bodies", &ForceElement::bodies, "The bodies it acts on.");

    py::classh<LinearSpringDamper, ForceElement>(m, "LinearSpringDamper",
                                                 "A linear spring and damper between two points.")
        .def(py::init<BodyIndex, BodyIndex, const Vec3&, const Vec3&, Real, Real, Real>(), "body1"_a, "body2"_a,
             "anchor1"_a, "anchor2"_a, "stiffness"_a, "damping"_a, "rest_length"_a,
             "Between point anchor1 on body1 and anchor2 on body2, each in its body's frame.")
        .def_readonly("body1_idx", &LinearSpringDamper::body1_idx)
        .def_readonly("body2_idx", &LinearSpringDamper::body2_idx)
        .def_readonly("anchor1_B", &LinearSpringDamper::anchor1_B, "The point on body1, in its frame [m].")
        .def_readonly("anchor2_B", &LinearSpringDamper::anchor2_B, "The point on body2, in its frame [m].")
        .def_readonly("k", &LinearSpringDamper::k, "Stiffness [N/m].")
        .def_readonly("c", &LinearSpringDamper::c, "Damping [N s/m].")
        .def_readonly("rest_length", &LinearSpringDamper::rest_length, "[m]");

    py::class_<SpringDamperParams>(m, "SpringDamperParams")
        .def(py::init<>())
        .def_readwrite("free_length", &SpringDamperParams::free_length, "[m]")
        .def_readwrite("spring", &SpringDamperParams::spring, "Pushing apart, against free_length - L.")
        .def_readwrite("preload", &SpringDamperParams::preload, "[N], pushing apart.")
        .def_readwrite("damper", &SpringDamperParams::damper, "Opposing the rate of extension.")
        .def_readwrite("bump_clearance", &SpringDamperParams::bump_clearance, "[m]")
        .def_readwrite("bump_stop", &SpringDamperParams::bump_stop)
        .def_readwrite("rebound_clearance", &SpringDamperParams::rebound_clearance, "[m]")
        .def_readwrite("rebound_stop", &SpringDamperParams::rebound_stop);
    py::classh<SpringDamper, ForceElement>(m, "SpringDamper", "A nonlinear spring, damper and stops between two points.")
        .def(py::init<BodyIndex, BodyIndex, const Vec3&, const Vec3&, SpringDamperParams>(), "body1"_a, "body2"_a,
             "anchor1"_a, "anchor2"_a, "params"_a,
             "Between point anchor1 on body1 and anchor2 on body2, each in its body's frame.")
        .def("law", &SpringDamper::law, "length"_a, "rate"_a,
             "The force pushing the points apart at a length and its rate, with its derivatives.")
        .def("potential_energy", &SpringDamper::potential_energy, "length"_a,
             "The energy stored at a length by the spring, the preload and the stops, zero at the free "
             "length [J].")
        .def("anchor1", &SpringDamper::anchor1, "The point on body1, in its frame [m].")
        .def("anchor2", &SpringDamper::anchor2, "The point on body2, in its frame [m].")
        .def("params", &SpringDamper::params, "The settings (a copy).");

    py::class_<RotationalSpringDamperParams>(m, "RotationalSpringDamperParams")
        .def(py::init<>())
        .def_readwrite("reference", &RotationalSpringDamperParams::reference, "[rad]")
        .def_readwrite("spring", &RotationalSpringDamperParams::spring)
        .def_readwrite("preload", &RotationalSpringDamperParams::preload, "[N m]")
        .def_readwrite("damper", &RotationalSpringDamperParams::damper);
    py::classh<RotationalSpringDamper, ForceElement>(m, "RotationalSpringDamper",
                                                     "A torsion spring and damper about marker 1's Z axis.")
        .def(py::init<BodyIndex, const Transform3&, BodyIndex, const Transform3&, RotationalSpringDamperParams>(),
             "body1"_a, "X_1M"_a, "body2"_a, "X_2M"_a, "params"_a,
             "Between marker X_1M on body1 and marker X_2M on body2, each in its body's frame; the axis is "
             "the first marker's Z axis.")
        .def("law", &RotationalSpringDamper::law, "angle"_a, "rate"_a,
             "The torque on body2 about the axis at a twist and its rate, with its derivatives.")
        .def("potential_energy", &RotationalSpringDamper::potential_energy, "angle"_a,
             "The energy stored at a twist by the spring and the preload, zero at the reference [J].")
        .def("X_1M", &RotationalSpringDamper::X_1M, "Marker 1, in body1's frame.")
        .def("X_2M", &RotationalSpringDamper::X_2M, "Marker 2, in body2's frame.")
        .def("params", &RotationalSpringDamper::params, "The settings (a copy).");

    py::class_<BushingParams>(m, "BushingParams",
                              "Per axis of marker 1 (X, Y, Z): curves of stiffness and damping, each set given "
                              "and read as three curves.")
        .def(py::init<>())
        .def_property(
            "rotational_stiffness", [](const BushingParams& p) { return curves(p.rotational_stiffness); },
            [](BushingParams& p, const std::array<Curve, 3>& c) { p.rotational_stiffness = c; })
        .def_property(
            "stiffness", [](const BushingParams& p) { return curves(p.stiffness); },
            [](BushingParams& p, const std::array<Curve, 3>& c) { p.stiffness = c; })
        .def_property(
            "rotational_damping", [](const BushingParams& p) { return curves(p.rotational_damping); },
            [](BushingParams& p, const std::array<Curve, 3>& c) { p.rotational_damping = c; })
        .def_property(
            "damping", [](const BushingParams& p) { return curves(p.damping); },
            [](BushingParams& p, const std::array<Curve, 3>& c) { p.damping = c; });
    py::classh<Bushing, ForceElement>(m, "Bushing", "Six springs and dampers between two markers.")
        .def(py::init<BodyIndex, const Transform3&, BodyIndex, const Transform3&, BushingParams>(), "body1"_a,
             "X_1M"_a, "body2"_a, "X_2M"_a, "params"_a, "At rest the markers coincide.")
        .def("X_1M", &Bushing::X_1M, "Marker 1, in body1's frame.")
        .def("X_2M", &Bushing::X_2M, "Marker 2, in body2's frame.")
        .def("params", &Bushing::params, "The settings (a copy).");

    py::class_<ContactParams>(m, "ContactParams", "Penalty contact with regularised friction.")
        .def(py::init<>())
        .def_readwrite("stiffness", &ContactParams::stiffness, "k [N/m^e]")
        .def_readwrite("exponent", &ContactParams::exponent, "e >= 1")
        .def_readwrite("damping", &ContactParams::damping, "c_max [N s/m]")
        .def_readwrite("damping_depth", &ContactParams::damping_depth, "d_max [m]")
        .def_readwrite("mu_static", &ContactParams::mu_static)
        .def_readwrite("mu_dynamic", &ContactParams::mu_dynamic)
        .def_readwrite("v_static", &ContactParams::v_static, "[m/s]")
        .def_readwrite("v_dynamic", &ContactParams::v_dynamic, "[m/s]");
    py::classh<PlaneContact, ForceElement>(m, "PlaneContact", "Spheres on a body against a plane on another.")
        .def(py::init<BodyIndex, std::vector<Vec3>, Real, BodyIndex, const Transform3&, ContactParams>(), "body"_a,
             "centres"_a, "radius"_a, "plane_body"_a, "plane_X"_a, "params"_a,
             "Spheres of `radius` centred at `centres` (body frame) against the XY plane of frame plane_X on "
             "plane_body, its Z axis out of the solid side.")
        .def("normal_law", &PlaneContact::normal_law, "penetration"_a, "rate"_a,
             "The normal force at a penetration and its rate, with its derivatives.")
        .def("friction", &PlaneContact::friction, "speed"_a,
             "The friction coefficient at a sliding speed, in .force, with its derivative in .d_rate.")
        .def("penetrations", &PlaneContact::penetrations, "states"_a,
             "Each sphere's penetration at the bodies' states (sim.states()); negative when apart [m].")
        .def("centres", &PlaneContact::centres, "The spheres' centres, in the body's frame [m].")
        .def("radius", &PlaneContact::radius, "The spheres' radius [m].")
        .def("plane_frame", &PlaneContact::plane_frame, "The plane's frame on plane_body: its XY plane, Z out.")
        .def("params", &PlaneContact::params, "The settings (a copy).");

    py::classh<BodyForces>(m, "BodyForces", "What a UserForce adds to the bodies, during its call.")
        .def("add_force", &BodyForces::add_force, "body"_a, "force"_a, "point"_a = py::none(),
             "Add a force [N] in world axes; at `point` (world) if given, else at the body's origin.")
        .def("add_torque", &BodyForces::add_torque, "body"_a, "torque"_a, "Add a torque [N m] in world axes.");
    py::classh<UserForce, ForceElement>(m, "UserForce", "A force written in Python.")
        .def(py::init([](std::string name, std::vector<BodyIndex> bodies, py::function function) {
                 return UserForce(std::move(name), std::move(bodies),
                                  [function](const std::vector<RigidBodyState>& states,
                                             std::vector<RigidBodyForces>& forces) {
                                      py::gil_scoped_acquire gil;
                                      auto out = std::make_shared<BodyForces>(states, forces);
                                      try {
                                          function(states, out);
                                      } catch (...) {
                                          out->close();
                                          throw;
                                      }
                                      out->close();
                                  });
             }),
             "name"_a, "bodies"_a, "function"_a,
             "function(states, forces) is called at every evaluation: `states` lists the bodies' world states "
             "(index 0 the ground), and forces.add_force / add_torque add loads. `bodies` lists the bodies "
             "it acts on, for the checks. Each call goes through Python: slow beside the C++ elements.");

    py::classh<JointForce>(m, "JointForce", "A force on joint coordinates directly.")
        .def("name", &JointForce::name, "The force's kind, in words.")
        .def("bodies", &JointForce::bodies, "The bodies whose joints it acts on.");
    py::class_<JointCoordinateForceParams>(m, "JointCoordinateForceParams")
        .def(py::init<>())
        .def_readwrite("reference", &JointCoordinateForceParams::reference, "[rad or m]")
        .def_readwrite("spring", &JointCoordinateForceParams::spring)
        .def_readwrite("preload", &JointCoordinateForceParams::preload)
        .def_readwrite("damper", &JointCoordinateForceParams::damper)
        .def_readwrite("friction", &JointCoordinateForceParams::friction)
        .def_readwrite("friction_velocity", &JointCoordinateForceParams::friction_velocity)
        .def_readwrite("lower_limit", &JointCoordinateForceParams::lower_limit)
        .def_readwrite("upper_limit", &JointCoordinateForceParams::upper_limit)
        .def_readwrite("limit_stiffness", &JointCoordinateForceParams::limit_stiffness)
        .def_readwrite("limit_damping", &JointCoordinateForceParams::limit_damping);
    py::classh<JointCoordinateForce, JointForce>(m, "JointCoordinateForce",
                                                 "Spring, preload, damper, friction and stops on the coordinate "
                                                 "of `body`'s joint (one coordinate).")
        .def(py::init<const Model&, int, JointCoordinateForceParams>(), "model"_a, "body"_a, "params"_a,
             "On the joint of `body`, which must have one coordinate (revolute or prismatic).")
        .def("law", &JointCoordinateForce::law, "q"_a, "v"_a,
             "The generalized force at a coordinate and its rate, with its derivatives.")
        .def("potential_energy", &JointCoordinateForce::potential_energy, "q"_a,
             "The energy stored at a coordinate by the spring, the preload and the stops, zero at the "
             "reference [J]; damping and friction store none.")
        .def("body", &JointCoordinateForce::body, "The body whose joint it acts on.")
        .def("params", &JointCoordinateForce::params, "The settings (a copy).");
}

} // namespace mbd::python
