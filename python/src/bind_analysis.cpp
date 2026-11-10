// The analyses: initial conditions, static equilibrium, linearisation and
// kinematic analysis; loads and energies; checks and descriptions; the
// recorder; events and the force callback (plan task 6.1, D52).

#include <memory>

#include "bindings.hpp"
#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/analysis/recorder.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/describe.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/validate.hpp"

namespace mbd::python {
namespace {

/// A list of indices as a tuple (an item assignment or an append would change
/// a copy), set as a whole from any sequence.
template <class C>
void index_list(py::class_<C>& cls, const char* name, std::vector<int> C::*member, const char* doc)
{
    cls.def_property(
        name, [member](const C& c) { return py::tuple(py::cast(c.*member)); },
        [member](C& c, std::vector<int> value) { c.*member = std::move(value); }, doc);
}

/// The simulator as Python sees it, for a callback: the same Python object
/// that owns it, so its views and settings are the ones the user holds.
py::object as_python(const kernel::Simulator& sim)
{
    return py::cast(sim, py::return_value_policy::reference);
}

} // namespace

void bind_analysis(py::module_& m)
{
    using namespace pybind11::literals;
    using namespace mbd::kernel;

    // --- Loads and energies -----------------------------------------------------
    py::class_<Loads>(m, "Loads", "Accelerations, constraint forces and joint reactions at a state.")
        .def_readonly("v_dot", &Loads::v_dot)
        .def_readonly("lambda_", &Loads::lambda, "The constraint multipliers (lambda is a Python keyword).")
        .def_readonly("constraint_W", &Loads::constraint_W,
                      "Per body: the constraint wrench [moment about the world origin; force], world axes.")
        .def_readonly("joint_reaction", &Loads::joint_reaction,
                      "Per body: what its joint transmits from the parent, [moment; force] in the child-side "
                      "joint frame.");
    m.def("compute_loads", &compute_loads, "sim"_a,
          "Accelerations, constraint forces and joint reactions at the simulator's state.");
    m.def(
        "kinetic_energy", [](const Simulator& sim) { return kinetic_energy(sim.system.model, sim.data()); }, "sim"_a,
        "Kinetic energy at the simulator's current state (after initialize, step or refresh) [J].");
    m.def(
        "potential_energy", [](const Simulator& sim) { return potential_energy(sim.system.model, sim.data()); },
        "sim"_a, "Gravity's potential energy at the simulator's current state [J].");

    // --- Initial conditions -----------------------------------------------------
    py::class_<AssemblyOptions> assembly_options(m, "AssemblyOptions", "What assemble() keeps and how far it iterates.");
    assembly_options.def(py::init<>());
    index_list(assembly_options, "hold_position", &AssemblyOptions::hold_position,
               "Velocity indices whose positions are kept as given.");
    index_list(assembly_options, "hold_velocity", &AssemblyOptions::hold_velocity,
               "Velocity indices whose velocities are kept as given.");
    assembly_options.def_readwrite("tolerance", &AssemblyOptions::tolerance)
        .def_readwrite("max_iterations", &AssemblyOptions::max_iterations)
        .def("hold_joint", &AssemblyOptions::hold_joint, "model"_a, "body"_a, "position"_a = true,
             "velocity"_a = true, py::return_value_policy::reference_internal,
             "Hold every velocity of `body`'s joint; returns these options.");

    py::class_<AssemblyReport>(m, "AssemblyReport", "What assemble() did, and the state it left.")
        .def_readonly("time", &AssemblyReport::time)
        .def_readonly("equations", &AssemblyReport::equations)
        .def_readonly("rank", &AssemblyReport::rank)
        .def_readonly("degrees_of_freedom", &AssemblyReport::degrees_of_freedom)
        .def_readonly("held_positions", &AssemblyReport::held_positions)
        .def_readonly("held_velocities", &AssemblyReport::held_velocities)
        .def_readonly("position_hold_rank", &AssemblyReport::position_hold_rank)
        .def_readonly("velocity_hold_rank", &AssemblyReport::velocity_hold_rank)
        .def_readonly("positions_assembled", &AssemblyReport::positions_assembled)
        .def_readonly("smallest_correction", &AssemblyReport::smallest_correction)
        .def_readonly("projection_iterations", &AssemblyReport::projection_iterations)
        .def_readonly("refinement_iterations", &AssemblyReport::refinement_iterations)
        .def_readonly("position_residual_given", &AssemblyReport::position_residual_given)
        .def_readonly("position_residual", &AssemblyReport::position_residual)
        .def_readonly("conflicting_equations", &AssemblyReport::conflicting_equations)
        .def_readonly("q_given", &AssemblyReport::q_given)
        .def_readonly("q", &AssemblyReport::q)
        .def_readonly("position_change", &AssemblyReport::position_change)
        .def_readonly("velocities_assembled", &AssemblyReport::velocities_assembled)
        .def_readonly("velocity_residual_given", &AssemblyReport::velocity_residual_given)
        .def_readonly("velocity_residual", &AssemblyReport::velocity_residual)
        .def_readonly("v_given", &AssemblyReport::v_given)
        .def_readonly("v", &AssemblyReport::v)
        .def_readonly("loads", &AssemblyReport::loads)
        .def_readonly("acceleration_residual", &AssemblyReport::acceleration_residual)
        .def_readonly("warnings", &AssemblyReport::warnings)
        .def("ok", &AssemblyReport::ok, "True when positions and velocities satisfy the constraints.")
        .def("summary", &AssemblyReport::summary, "model"_a,
             "The counts, the residuals, every coordinate's given and assembled value, and the warnings, as "
             "text.");
    m.def("assemble", &assemble, "sim"_a, "options"_a = AssemblyOptions(),
          "Bring the simulator's state onto the constraints with the held coordinates kept, by the smallest "
          "correction to the rest; the report names conflicting constraints when they cannot all hold.");

    // --- Static equilibrium -----------------------------------------------------
    py::class_<StaticOptions>(m, "StaticOptions")
        .def(py::init<>())
        .def_readwrite("tolerance", &StaticOptions::tolerance, "Largest acceleration at rest [m/s^2, rad/s^2].")
        .def_readwrite("max_iterations", &StaticOptions::max_iterations)
        .def_readwrite("perturbation", &StaticOptions::perturbation)
        .def_readwrite("max_translation", &StaticOptions::max_translation, "[m] per Newton step")
        .def_readwrite("max_rotation", &StaticOptions::max_rotation, "[rad] per Newton step")
        .def_readwrite("relaxation_step", &StaticOptions::relaxation_step)
        .def_readwrite("max_relaxation_steps", &StaticOptions::max_relaxation_steps)
        .def_readwrite("relaxation_tolerance", &StaticOptions::relaxation_tolerance);
    py::class_<StaticReport>(m, "StaticReport")
        .def_readonly("time", &StaticReport::time)
        .def_readonly("converged", &StaticReport::converged)
        .def_readonly("iterations", &StaticReport::iterations)
        .def_readonly("relaxation_runs", &StaticReport::relaxation_runs)
        .def_readonly("relaxation_steps", &StaticReport::relaxation_steps)
        .def_readonly("relaxation_undone", &StaticReport::relaxation_undone)
        .def_readonly("residual_given", &StaticReport::residual_given)
        .def_readonly("residual", &StaticReport::residual)
        .def_readonly("history", &StaticReport::history)
        .def_readonly("degrees_of_freedom", &StaticReport::degrees_of_freedom)
        .def_readonly("supported", &StaticReport::supported)
        .def_readonly("neutral", &StaticReport::neutral)
        .def_readonly("q_given", &StaticReport::q_given)
        .def_readonly("q", &StaticReport::q)
        .def_readonly("position_change", &StaticReport::position_change)
        .def_readonly("loads", &StaticReport::loads)
        .def_readonly("warnings", &StaticReport::warnings)
        .def_readonly("notes", &StaticReport::notes)
        .def("summary", &StaticReport::summary, "model"_a,
             "The counts, the history, each coordinate's change and the messages, as text.");
    m.def("static_equilibrium", &static_equilibrium, "sim"_a, "options"_a = StaticOptions(),
          "Bring the system to rest where its forces balance; leaves the state there.");

    // --- Linearisation ----------------------------------------------------------
    py::class_<LinearizationOptions>(m, "LinearizationOptions")
        .def(py::init<>())
        .def_readwrite("perturbation", &LinearizationOptions::perturbation);
    py::class_<Mode>(m, "Mode", "One mode: a complex pair listed once, with its positive imaginary part.")
        .def_readonly("eigenvalue", &Mode::eigenvalue)
        .def_readonly("natural_frequency", &Mode::natural_frequency, "|eigenvalue| [rad/s]")
        .def_readonly("damping_ratio", &Mode::damping_ratio)
        .def_readonly("shape", &Mode::shape, "dq in velocity coordinates, its largest entry 1 (complex).")
        .def_readonly("energy_fraction", &Mode::energy_fraction,
                      "The share of the mode's kinetic energy in each velocity coordinate, adding up to 1.")
        .def("oscillatory", &Mode::oscillatory, "True for a complex pair (a vibration), False for a real "
                                                "eigenvalue (a decay or a drift).");
    py::class_<LinearModel>(m, "LinearModel", "The motion near an operating point: x' = A x in n coordinates.")
        .def_readonly("time", &LinearModel::time)
        .def_readonly("q", &LinearModel::q)
        .def_readonly("v", &LinearModel::v)
        .def_readonly("residual", &LinearModel::residual)
        .def_readonly("degrees_of_freedom", &LinearModel::degrees_of_freedom)
        .def_readonly("basis", &LinearModel::basis, "N, nv x n: the coordinates' directions.")
        .def_readonly("A", &LinearModel::A, "2n x 2n")
        .def_readonly("modes", &LinearModel::modes, "By natural frequency.")
        .def_readonly("warnings", &LinearModel::warnings)
        .def("summary", &LinearModel::summary, "model"_a,
             "The operating point, each mode's frequency, damping and largest shape components, and the "
             "warnings, as text.");
    m.def("linearize", &linearize, "sim"_a, "options"_a = LinearizationOptions(),
          "The linear model at the simulator's state, on the constraints.");

    // --- Kinematic analysis -----------------------------------------------------
    py::class_<Kinematics> kinematics(m, "Kinematics",
                                      "A mechanism with no freedom left, solved along its driving parameter t.");
    kinematics.def(py::init<const System&>(), "system"_a, py::keep_alive<1, 2>(), "At the neutral configuration, t = 0.")
        .def_property(
            "q", py::cpp_function([](Kinematics& k) -> VecX& { return k.q; }, py::return_value_policy::reference_internal),
            [](Kinematics& k, const VecX& value) { assign_same_size(k.q, value, "Kinematics.q"); },
            "The coordinates: a view of its own vector.")
        .def_readwrite("t", &Kinematics::t, "The driving parameter.")
        .def("solve", &Kinematics::solve, "max_iterations"_a = 100, "tolerance"_a = 1e-10,
             "Solve phi(q, t) = 0 from the current q; True if converged.")
        .def("phi", [](Kinematics& k) -> VecX { return k.phi(); }, "The constraints' residual phi(q, t).")
        .def("velocities", [](Kinematics& k) -> VecX { return k.velocities(); },
             "dq/dt in velocity coordinates (least norm where freedom is left). Call after solve().")
        .def("accelerations", [](Kinematics& k) -> VecX { return k.accelerations(); },
             "d2q/dt2. Call after velocities().")
        .def("update", &Kinematics::update, "Recompute the placements after changing q by hand.")
        .def("state", &Kinematics::state, "body"_a, "A body's world state (at rest) at the current q.");
    m.def("point_height_driver", &point_height_driver, "body"_a, "point"_a, "axis"_a, "nominal"_a,
          "A driver holding world coordinate `axis` of `point` (body frame) on `body` at nominal + t.");

    // --- Checks and descriptions -------------------------------------------------
    py::class_<ValidationReport>(m, "ValidationReport")
        .def_readonly("errors", &ValidationReport::errors)
        .def_readonly("warnings", &ValidationReport::warnings)
        .def_readonly("notes", &ValidationReport::notes)
        .def_readonly("bodies", &ValidationReport::bodies)
        .def_readonly("coordinates", &ValidationReport::coordinates)
        .def_readonly("velocities", &ValidationReport::velocities)
        .def_readonly("constraint_equations", &ValidationReport::constraint_equations)
        .def_readonly("independent_equations", &ValidationReport::independent_equations)
        .def_readonly("redundant_equations", &ValidationReport::redundant_equations)
        .def_readonly("degrees_of_freedom", &ValidationReport::degrees_of_freedom)
        .def("ok", &ValidationReport::ok, "True when there is no error.")
        .def("summary", &ValidationReport::summary, "The counts, then every message, one per line.");
    m.def("validate", py::overload_cast<const System&>(&validate), "system"_a,
          "Check a system before it runs (at its neutral configuration).");
    m.def("validate", py::overload_cast<const System&, const VecX&, Real>(&validate), "system"_a, "q"_a, "t"_a = 0.0,
          "Check a system at coordinates q.");
    m.def("describe", py::overload_cast<const System&>(&describe), "system"_a,
          "Bodies, joints, constraints, force elements and the degrees of freedom, as text.");
    m.def("describe", py::overload_cast<const System&, const VecX&, Real>(&describe), "system"_a, "q"_a,
          "t"_a = 0.0);
    m.def("dump_state", &dump_state, "sim"_a, "The simulator's state, every body and constraint, as text.");

    // --- The recorder ------------------------------------------------------------
    py::class_<Recorder>(m, "Recorder", "Named channels, each read at every sample.")
        .def(py::init<>(), "No channel yet.")
        .def(
            "add",
            [](Recorder& r, const std::string& name, py::function channel) {
                r.add(name, [channel]() {
                    py::gil_scoped_acquire gil;
                    return channel().cast<Real>();
                });
            },
            "name"_a, "channel"_a, "Add a channel: a function of no arguments returning a number.")
        .def("sample", &Recorder::sample, "time"_a, "Read every channel and store the values with the time.")
        .def("samples", &Recorder::samples, "Number of samples taken.")
        .def("names", &Recorder::names, "The channels' names, in the order added.")
        .def("times", [](const Recorder& r) { return Eigen::Map<const VecX>(r.times().data(), static_cast<Index>(r.times().size())).eval(); },
             "The samples' times, as an array.")
        .def(
            "column",
            [](const Recorder& r, const std::string& name) {
                const auto& c = r.column(name);
                return Eigen::Map<const VecX>(c.data(), static_cast<Index>(c.size())).eval();
            },
            "name"_a, "One channel's values, one per sample, as an array.")
        .def(
            "as_dict",
            [](const Recorder& r) {
                py::dict d;
                d["time"] = Eigen::Map<const VecX>(r.times().data(), static_cast<Index>(r.times().size())).eval();
                for (const auto& name : r.names()) {
                    const auto& c = r.column(name);
                    d[py::str(name)] = Eigen::Map<const VecX>(c.data(), static_cast<Index>(c.size())).eval();
                }
                return d;
            },
            "Every channel, and 'time', as arrays: ready for pandas.DataFrame(...).")
        .def("write_csv", &Recorder::write_csv, "path"_a,
             "One row per sample, time then the channels, under a header of names; 17 significant digits, "
             "so the values read back exactly.");
    m.def("add_drift_channels", &add_drift_channels, "recorder"_a, "sim"_a, py::keep_alive<1, 2>(),
          "Add the constraint drift and residual channels, read from `sim` at every sample.");
}

void bind_simulator_callbacks(py::class_<kernel::Simulator>& sim)
{
    using namespace pybind11::literals;
    using namespace mbd::kernel;

    py::class_<FiredEvent>(sim, "FiredEvent")
        .def_readonly("event", &FiredEvent::event, "Its index among the events.")
        .def_readonly("time", &FiredEvent::time);

    sim.def(
           "add_event",
           [](Simulator& s, std::string name, py::function function, py::object action, int direction) {
               Event e;
               e.name = std::move(name);
               e.direction = direction;
               e.function = [function](const Simulator& at) {
                   py::gil_scoped_acquire gil;
                   return function(as_python(at)).cast<Real>();
               };
               if (!action.is_none()) {
                   py::function act = action;
                   e.action = [act](Simulator& at) {
                       py::gil_scoped_acquire gil;
                       act(as_python(at));
                   };
               }
               s.events.push_back(std::move(e));
           },
           "name"_a, "function"_a, "action"_a = py::none(), "direction"_a = 0,
           "Watch function(sim) for zero crossings; at each, the step is cut there and action(sim) runs (it may "
           "change q and v). direction -1: falling crossings only, +1: rising only, 0: both.")
        .def("event_names", [](const Simulator& s) {
            std::vector<std::string> names;
            for (const auto& e : s.events) names.push_back(e.name);
            return names;
        }, "The events' names, in the order added.")
        .def("event_log", [](const Simulator& s) { return s.event_log(); }, "The events that fired, in order.")
        .def_property(
            "force_callback", [](const Simulator& s) { return static_cast<bool>(s.force_callback); },
            [](Simulator& s, py::object callback) {
                if (callback.is_none()) {
                    s.force_callback = nullptr;
                    return;
                }
                py::function f = callback;
                s.force_callback = [f](Simulator& at, Real t, VecX& tau) {
                    py::gil_scoped_acquire gil;
                    // tau as a writable view: the callback adds to it in place.
                    f(as_python(at), t, py::cast(tau, py::return_value_policy::reference));
                };
            },
            "Set to f(sim, t, tau): called at every evaluation to add generalized forces to tau in place "
            "(tau[0] += ...). Read the state being evaluated from sim.states(): sim.q and sim.v hold the "
            "step's start, not the integrator's stage. Reads True when one is set; None removes it.");
}

} // namespace mbd::python
