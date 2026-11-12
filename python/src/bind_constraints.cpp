// Markers, prescribed functions of time, and the constraints that close loops
// or drive joints (plan task 6.1, D52).

#include <memory>

#include "bindings.hpp"
#include "mbd/kernel/constraints.hpp"

namespace mbd::python {

void bind_constraints(py::module_& m)
{
    using namespace pybind11::literals;
    using namespace mbd::kernel;

    py::class_<Marker>(m, "Marker", "A frame fixed on a body: the body (0 for ground) and its placement there.")
        .def(py::init([](int body, const Transform3& X_BM) { return Marker{body, X_BM}; }), "body"_a,
             "X_BM"_a = Transform3(), "The frame X_BM, in the body's frame, on `body`.")
        .def_readwrite("body", &Marker::body)
        .def_readwrite("X_BM", &Marker::X_BM);

    py::class_<TimeFunction>(m, "TimeFunction", "A prescribed function of time with its first two derivatives.")
        .def(py::init<>(), "The constant zero.")
        .def(py::init([](py::function s, py::function ds, py::function dds) {
                 // Called from C++ at every evaluation, with the interpreter
                 // lock held (D52): s(t), ds/dt and d2s/dt2 as Python floats.
                 auto wrap = [](py::function f) {
                     return std::function<Real(Real)>([f](Real t) { return f(t).cast<Real>(); });
                 };
                 return TimeFunction(wrap(std::move(s)), wrap(std::move(ds)), wrap(std::move(dds)));
             }),
             "s"_a, "ds"_a, "dds"_a,
             "Any function, given as three callables of t: the value, its rate and its acceleration. The "
             "three must agree; a driver's motion is only as consistent as they are.")
        .def_static("constant", &TimeFunction::constant, "value"_a)
        .def("value", &TimeFunction::value, "t"_a, "s(t).")
        .def("rate", &TimeFunction::rate, "t"_a, "ds/dt at t.")
        .def("acceleration", &TimeFunction::acceleration, "t"_a, "d2s/dt2 at t.")
        .def("is_constant", &TimeFunction::is_constant, "True for a constant (made by constant() or by default).")
        .def("constant_value", &TimeFunction::constant_value, "A constant's value.");

    py::classh<ConstraintModel>(m, "ConstraintModel", "A constraint: equations phi(q, t) = 0 between bodies.")
        .def("name", &ConstraintModel::name, "The constraint's kind, in words.")
        .def("size", &ConstraintModel::size, "Number of equations.")
        .def("bodies", &ConstraintModel::bodies, "The bodies it connects.")
        .def("__repr__", [](const ConstraintModel& c) {
            return std::string(c.name()) + " constraint (" + std::to_string(c.size()) + " equations)";
        });

    py::classh<PointCoincidence, ConstraintModel>(m, "PointCoincidence",
                                                   "The origins of two markers coincide: 3 equations.")
        .def(py::init<Marker, Marker>(), "i"_a, "j"_a, "Marker j's origin held on marker i's.")
        .def("i", &PointCoincidence::i, "Marker i.")
        .def("j", &PointCoincidence::j, "Marker j.");
    py::classh<Dot1, ConstraintModel>(m, "Dot1",
                                      "Axis axis_i of marker i perpendicular to axis axis_j of marker j "
                                      "(axes 0, 1, 2 for X, Y, Z), or their dot product s(t).")
        .def(py::init<Marker, int, Marker, int, TimeFunction>(), "i"_a, "axis_i"_a, "j"_a, "axis_j"_a,
             "s"_a = TimeFunction(), "a_i . a_j = s(t), the axes' dot product; s = 0 keeps them perpendicular.")
        .def("i", &Dot1::i, "Marker i.")
        .def("axis_i", &Dot1::axis_i, "Marker i's axis (0, 1, 2 for X, Y, Z).")
        .def("j", &Dot1::j, "Marker j.")
        .def("axis_j", &Dot1::axis_j, "Marker j's axis.")
        .def("s", &Dot1::s, "The dot product's prescribed value.");
    py::classh<Dot2, ConstraintModel>(m, "Dot2",
                                      "Marker j's origin at signed distance s(t) (0 by default) from the plane "
                                      "through marker i's origin normal to its axis axis_i.")
        .def(py::init<Marker, int, Marker, TimeFunction>(), "i"_a, "axis_i"_a, "j"_a, "s"_a = TimeFunction(),
             "a_i . (p_j - p_i) = s(t), with a_i axis axis_i of marker i (0, 1, 2 for X, Y, Z).")
        .def("i", &Dot2::i, "Marker i.")
        .def("axis_i", &Dot2::axis_i, "Marker i's axis (0, 1, 2 for X, Y, Z).")
        .def("j", &Dot2::j, "Marker j.")
        .def("s", &Dot2::s, "The prescribed distance from the plane.");
    py::classh<Distance, ConstraintModel>(m, "Distance", "The markers' origins a fixed or prescribed distance apart.")
        .def(py::init<Marker, Marker, Real>(), "i"_a, "j"_a, "length"_a, "A fixed length [m].")
        .def(py::init<Marker, Marker, TimeFunction, Real>(), "i"_a, "j"_a, "length"_a, "nominal_length"_a,
             "A prescribed length; nominal_length scales the equation.")
        .def("i", &Distance::i, "Marker i.")
        .def("j", &Distance::j, "Marker j.")
        .def("length", &Distance::length, "The length, a function of time (a constant for a fixed length).")
        .def("nominal_length", &Distance::nominal_length, "The length that scales the equation [m].");
    py::classh<NoTwist, ConstraintModel>(m, "NoTwist",
                                         "The markers' relative rotation a pure bend, about an axis across their "
                                         "Z axes: the condition of a constant-velocity joint.")
        .def(py::init<Marker, Marker>(), "i"_a, "j"_a,
             "x_i . y_j - y_i . x_j = 0, for markers whose Z axes are close to aligned.")
        .def("i", &NoTwist::i, "Marker i.")
        .def("j", &NoTwist::j, "Marker j.");
    py::classh<JointDriver, ConstraintModel>(m, "JointDriver",
                                             "Drives the coordinate of `body`'s joint (one coordinate) along s(t).")
        .def(py::init<const Model&, int, TimeFunction>(), "model"_a, "body"_a, "s"_a,
             "The coordinate follows s(t); an angle is read on the branch nearest s(t).")
        .def("body", &JointDriver::body, "The body whose joint it drives.")
        .def("s", &JointDriver::s, "The coordinate's prescribed value.");
    // Made by the closures below, or read from a model file (task 6.4).
    py::classh<ConstraintSet, ConstraintModel>(m, "ConstraintSet", "Several constraints as one.")
        .def("parts", &ConstraintSet::parts, "The constraints it stacks, in order.");

    m.def("point_on_line", &point_on_line, "i"_a, "j"_a, "Marker j's origin on marker i's Z axis (2 equations).");
    m.def("point_on_plane", &point_on_plane, "i"_a, "j"_a,
          "Marker j's origin on marker i's XY plane (1 equation).");
    m.def("spherical_closure", &spherical_closure, "i"_a, "j"_a, "Origins coincide (3 equations).");
    m.def("revolute_closure", &revolute_closure, "i"_a, "j"_a,
          "Origins coincide and the Z axes stay aligned (5 equations).");
    m.def("universal_closure", &universal_closure, "i"_a, "j"_a,
          "Origins coincide and the Z axes stay perpendicular, the cross of a Cardan joint (4 equations).");
    m.def("cylindrical_closure", &cylindrical_closure, "i"_a, "j"_a,
          "Marker j's origin on marker i's Z axis and the Z axes aligned (4 equations).");
    m.def("prismatic_closure", &prismatic_closure, "i"_a, "j"_a,
          "As cylindrical, without rotation about Z (5 equations).");
    m.def("constant_velocity_closure", &constant_velocity_closure, "i"_a, "j"_a,
          "Origins coincide and the shafts, along the Z axes, turn at the same rate (4 equations).");
}

} // namespace mbd::python
