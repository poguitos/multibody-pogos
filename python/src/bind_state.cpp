// State elements written in Python, and the step trace (plan task 6.1, D52).

#include <string>

#include "bindings.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"

namespace mbd::python {
namespace {

using kernel::ConstVecRef;
using kernel::Simulator;
using kernel::StateElement;
using kernel::VecRef;

/// A NumPy view of an element's states, in the integrator's own memory:
/// writable where the element writes them, read-only where it reads.
py::array view(const Real* data, Index size, Index stride, bool writable)
{
    py::array a(py::dtype::of<Real>(), {size}, {stride * static_cast<Index>(sizeof(Real))}, data,
                py::capsule(data, [](void*) {}));   // a base that owns nothing: no copy
    if (!writable) a.attr("setflags")(py::arg("write") = false);
    return a;
}
py::array view(VecRef x) { return view(x.data(), x.size(), x.innerStride(), true); }
py::array view(ConstVecRef x) { return view(x.data(), x.size(), x.innerStride(), false); }

py::function required(const StateElement* self, const char* method)
{
    py::function f = py::get_override(self, method);
    MBD_THROW_IF(!f, std::string("MBD-P007: a StateElement written in Python must define ") + method + "()");
    return f;
}

/// The trampoline: each virtual call goes to the Python subclass's method,
/// with the interpreter lock taken. name() keeps the string it returns.
class PyStateElement : public StateElement, public py::trampoline_self_life_support {
public:
    const char* name() const override
    {
        py::gil_scoped_acquire gil;
        name_ = required(this, "name")().cast<std::string>();
        return name_.c_str();
    }
    int size() const override
    {
        py::gil_scoped_acquire gil;
        return required(this, "size")().cast<int>();
    }
    void initial_states(VecRef x) const override
    {
        py::gil_scoped_acquire gil;
        required(this, "initial_states")(view(x));
    }
    void begin_evaluation(const Simulator& sim, Real t, ConstVecRef x) override
    {
        py::gil_scoped_acquire gil;
        if (py::function f = py::get_override(this, "begin_evaluation")) {
            f(py::cast(sim, py::return_value_policy::reference), t, view(x));
        }
    }
    void rates(const Simulator& sim, Real t, ConstVecRef x, VecRef x_dot) override
    {
        py::gil_scoped_acquire gil;
        required(this, "rates")(py::cast(sim, py::return_value_policy::reference), t, view(x), view(x_dot));
    }
    void end_step(const Simulator& sim, Real t, VecRef x) override
    {
        py::gil_scoped_acquire gil;
        if (py::function f = py::get_override(this, "end_step")) {
            f(py::cast(sim, py::return_value_policy::reference), t, view(x));
        }
    }

private:
    mutable std::string name_;
};

} // namespace

void bind_state_elements(py::module_& m, py::class_<Simulator>& sim)
{
    using namespace pybind11::literals;

    py::classh<StateElement, PyStateElement>(
        m, "StateElement",
        "States integrated with the bodies (a lag, a wheel's spin, a damper's internal state). Subclass it in "
        "Python, call super().__init__(), and define name(), size(), initial_states(x), rates(sim, t, x, "
        "x_dot), and optionally begin_evaluation(sim, t, x) and end_step(sim, t, x); x_dot and the x of "
        "initial_states and end_step are written in place. Each call goes through Python.")
        .def(py::init<>(), "Call it from the subclass's __init__: super().__init__().");

    sim.def("attach", &Simulator::attach, "element"_a, py::keep_alive<1, 2>(),
            "Integrate `element`'s states with the bodies from now on (in sim.x, after those attached before); "
            "attach before stepping. The simulator keeps the element alive.");

    py::class_<kernel::StepRecord>(sim, "StepRecord", "One step of the trace.")
        .def_readonly("time", &kernel::StepRecord::time)
        .def_readonly("dt", &kernel::StepRecord::dt)
        .def_readonly("drift", &kernel::StepRecord::drift)
        .def_readonly("velocity_drift", &kernel::StepRecord::velocity_drift)
        .def_readonly("position_residual", &kernel::StepRecord::position_residual)
        .def_readonly("velocity_residual", &kernel::StepRecord::velocity_residual)
        .def_readonly("projection_iterations", &kernel::StepRecord::projection_iterations)
        .def_readonly("projection_converged", &kernel::StepRecord::projection_converged)
        .def_readonly("equations", &kernel::StepRecord::equations)
        .def_readonly("rank", &kernel::StepRecord::rank)
        .def_readonly("kinetic_energy", &kernel::StepRecord::kinetic_energy)
        .def_readonly("potential_energy", &kernel::StepRecord::potential_energy)
        .def_readonly("events", &kernel::StepRecord::events);
    sim.def("trace", [](const Simulator& s) { return s.trace(); },
            "The steps recorded while trace_steps is set (a copy).")
        .def("write_trace", &Simulator::write_trace, "path"_a, "The trace as CSV, one row per step.");
}

} // namespace mbd::python
