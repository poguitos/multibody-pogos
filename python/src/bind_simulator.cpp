// The system, the integrators' settings and the simulator (plan task 6.1, D52).

#include <cmath>

#include "bindings.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::python {
namespace {

using RowMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// A property of `Cls` that reads as a NumPy view of the simulator's own
/// vector (writing an element writes the simulator's state) and is assigned
/// only with a vector of the same size (MBD-P001).
template <class Cls>
void state_vector(Cls& cls, const char* name, VecX kernel::Simulator::*member, const char* doc)
{
    cls.def_property(
        name,
        py::cpp_function([member](kernel::Simulator& s) -> VecX& { return s.*member; },
                         py::return_value_policy::reference_internal),
        [member, name](kernel::Simulator& s, const VecX& value) {
            assign_same_size(s.*member, value, (std::string("Simulator.") + name).c_str());
        },
        doc);
}

} // namespace

void bind_simulator(py::module_& m)
{
    using namespace pybind11::literals;
    using namespace mbd::kernel;

    py::class_<System>(m, "System", "A model with what acts on it: constraints and force elements.")
        .def(py::init<>(), "The ground alone, with nothing acting on it.")
        .def_property_readonly(
            "model", [](System& s) -> Model& { return s.model; }, py::return_value_policy::reference_internal,
            "The tree of bodies; build it with system.model.add_body(...).")
        // The lists as tuples (an append to a copied list would be lost) and
        // add_* to extend them, before a simulator is made from the system.
        .def(
            "add_constraint",
            [](System& s, std::shared_ptr<ConstraintModel> c) { s.constraints.push_back(std::move(c)); },
            py::arg("constraint").none(false), "Add a constraint; before a simulator is made from the system.")
        .def(
            "add_force_element",
            [](System& s, std::shared_ptr<ForceElement> f) { s.force_elements.push_back(std::move(f)); },
            py::arg("element").none(false), "Add a force element; before a simulator is made from the system.")
        .def(
            "add_joint_force",
            [](System& s, std::shared_ptr<JointForce> f) { s.joint_forces.push_back(std::move(f)); },
            py::arg("force").none(false), "Add a joint force; before a simulator is made from the system.")
        .def_property_readonly("constraints", [](const System& s) { return py::tuple(py::cast(s.constraints)); })
        .def_property_readonly("force_elements",
                               [](const System& s) { return py::tuple(py::cast(s.force_elements)); })
        .def_property_readonly("joint_forces", [](const System& s) { return py::tuple(py::cast(s.joint_forces)); });

    py::native_enum<Integrator>(m, "Integrator", "enum.Enum", "The integration methods (docs/integrators.md).")
        .value("RK4", Integrator::RK4)
        .value("SemiImplicitEuler", Integrator::SemiImplicitEuler)
        .value("DormandPrince", Integrator::DormandPrince)
        .value("GeneralizedAlpha", Integrator::GeneralizedAlpha)
        .value("SDIRK4", Integrator::SDIRK4)
        .value("RadauIIA", Integrator::RadauIIA)
        .value("LinearlyImplicitEuler", Integrator::LinearlyImplicitEuler)
        .finalize();

    py::class_<StepControl>(m, "StepControl", "How the methods with error control choose their steps.")
        .def(py::init<>())
        .def_readwrite("adaptive", &StepControl::adaptive)
        .def_readwrite("relative_tolerance", &StepControl::relative_tolerance)
        .def_readwrite("absolute_tolerance", &StepControl::absolute_tolerance)
        .def_readwrite("min_step", &StepControl::min_step)
        .def_readwrite("max_steps", &StepControl::max_steps);

    py::class_<ImplicitControl>(m, "ImplicitControl", "Settings of the implicit methods.")
        .def(py::init<>())
        .def_readwrite("rho_infinity", &ImplicitControl::rho_infinity)
        .def_readwrite("max_iterations", &ImplicitControl::max_iterations)
        .def_readwrite("stabilized_index2", &ImplicitControl::stabilized_index2);

    py::class_<IntegratorStats>(m, "IntegratorStats", "What the integrator has done so far.")
        .def_readonly("steps", &IntegratorStats::steps)
        .def_readonly("rejected", &IntegratorStats::rejected)
        .def_readonly("evaluations", &IntegratorStats::evaluations)
        .def_readonly("step_size", &IntegratorStats::step_size)
        .def_readonly("error", &IntegratorStats::error)
        .def_readonly("newton_iterations", &IntegratorStats::newton_iterations)
        .def_readonly("jacobians", &IntegratorStats::jacobians)
        .def_readonly("state_evaluations", &IntegratorStats::state_evaluations);

    py::class_<ProjectionInfo>(m, "ProjectionInfo", "How a projection onto the constraints went.")
        .def_readonly("iterations", &ProjectionInfo::iterations, "Gauss-Newton steps taken.")
        .def_readonly("initial_position_residual", &ProjectionInfo::initial_position_residual,
                      "|phi| before: the drift it corrected.")
        .def_readonly("position_residual", &ProjectionInfo::position_residual, "|phi| at the end.")
        .def_readonly("initial_velocity_residual", &ProjectionInfo::initial_velocity_residual,
                      "|J v - nu| before the velocities were moved, at the final positions.")
        .def_readonly("velocity_residual", &ProjectionInfo::velocity_residual, "|J v - nu| at the end.")
        .def_readonly("converged", &ProjectionInfo::converged);

    py::class_<ConstraintDrift>(m, "ConstraintDrift", "How far the state is from the loop constraints.")
        .def_readonly("position", &ConstraintDrift::position)
        .def_readonly("velocity", &ConstraintDrift::velocity)
        .def_readonly("position_residual", &ConstraintDrift::position_residual)
        .def_readonly("velocity_residual", &ConstraintDrift::velocity_residual)
        .def_readonly("max_position", &ConstraintDrift::max_position)
        .def_readonly("max_velocity", &ConstraintDrift::max_velocity)
        .def_readonly("max_position_residual", &ConstraintDrift::max_position_residual)
        .def_readonly("max_velocity_residual", &ConstraintDrift::max_velocity_residual);

    py::class_<Simulator> sim(m, "Simulator",
                              "Integrates a system in time. The system must not gain bodies or constraints "
                              "afterwards; the simulator keeps it alive.");
    sim.def(py::init<System&>(), "system"_a, py::keep_alive<1, 2>(),
            "The state starts at the neutral configuration, at rest, at time 0.")
        .def_property_readonly(
            "system", [](Simulator& s) -> System& { return s.system; }, py::return_value_policy::reference_internal)
        .def_readwrite("time", &Simulator::time, "[s]")
        .def_readwrite("method", &Simulator::method)
        .def_readwrite("step_control", &Simulator::step_control)
        .def_readwrite("implicit_control", &Simulator::implicit_control)
        .def_readwrite("enforce_constraints", &Simulator::enforce_constraints)
        .def_readwrite("constraint_tolerance", &Simulator::constraint_tolerance, "[m, rad]")
        .def_readwrite("multirate", &Simulator::multirate)
        .def_readwrite("event_tolerance", &Simulator::event_tolerance, "[s]")
        .def_readwrite("max_events_per_step", &Simulator::max_events_per_step)
        .def_readwrite("trace_steps", &Simulator::trace_steps);
    state_vector(sim, "q", &Simulator::q, "The coordinates: a view of the simulator's own vector.");
    state_vector(sim, "v", &Simulator::v, "The velocities: a view of the simulator's own vector.");
    state_vector(sim, "x", &Simulator::x, "The state elements' states: a view of the simulator's own vector.");
    state_vector(sim, "tau", &Simulator::tau,
                 "Generalized forces for the next step, cleared after it: a view of the simulator's own vector.");
    sim.def("initialize", &Simulator::initialize, "Project the state onto the constraints and refresh the kinematics.")
        .def("step", &Simulator::step, "dt"_a, "Advance by dt [s].")
        .def("run", &Simulator::run, "duration"_a, "dt"_a, "Advance by round(duration / dt) steps; returns their number.")
        .def(
            "run_recorded",
            [](Simulator& s, Real duration, Real dt) {
                // run(), with the state kept at its start and after every step.
                const int steps = static_cast<int>(std::round(duration / dt));
                VecX time(steps + 1);
                RowMatrix q(steps + 1, s.q.size()), v(steps + 1, s.v.size()), x(steps + 1, s.x.size());
                for (int k = 0; k <= steps; ++k) {
                    if (k > 0) s.step(dt);
                    time(k) = s.time;
                    q.row(k) = s.q.transpose();
                    v.row(k) = s.v.transpose();
                    x.row(k) = s.x.transpose();
                }
                return py::dict("time"_a = time, "q"_a = q, "v"_a = v, "x"_a = x);
            },
            "duration"_a, "dt"_a,
            "As run(), and returns the states at the start and after every step: a dict of NumPy arrays "
            "'time' (steps + 1), 'q', 'v' and 'x' (one row per sample).")
        .def("refresh", &Simulator::refresh, "Kinematics and body states at the current (q, v).")
        .def(
            "acceleration",
            [](Simulator& s, const VecX& q, const VecX& v, Real t) -> VecX {
                MBD_THROW_IF(q.size() != s.q.size() || v.size() != s.v.size(),
                             "MBD-P001: Simulator.acceleration: q and v must have the model's sizes, "
                                 + std::to_string(s.q.size()) + " and " + std::to_string(s.v.size()));
                return s.acceleration(q, v, t);
            },
            "q"_a, "v"_a, "t"_a, "The accelerations at (q, v, t) with everything applied.")
        .def("states", [](const Simulator& s) { return s.states(); },
             "The bodies' world states at the current state (a copy; index 0 is the ground).")
        .def("integrator_stats", [](const Simulator& s) { return s.integrator_stats(); },
             "What the integrator has done so far: steps, rejections, evaluations, Newton iterations.")
        .def("constraint_drift", [](const Simulator& s) { return s.constraint_drift(); },
             "How far the last step's result and the state are from the constraints, and the largest values "
             "so far.")
        .def("projection_failures", &Simulator::projection_failures,
             "How many steps' projections onto the constraints did not converge (the first is reported as "
             "MBD-K051).")
        .def("last_projection", [](const Simulator& s) { return s.last_projection(); },
             "How the last projection onto the constraints went (a copy): iterations, residuals before and "
             "after, converged.")
        .def(
            "interpolate",
            [](const Simulator& s, Real t) {
                VecX q, v, x;
                s.interpolate(t, q, v, x);
                return py::make_tuple(q, v, x);
            },
            "t"_a, "DormandPrince: the state (q, v, x) at time t within its last step.");
    bind_simulator_callbacks(sim);
    bind_state_elements(m, sim);
}

} // namespace mbd::python
