#pragma once

// Time integration on the multibody kernel (plan tasks 2.7, 4.1). The state
// is (q, v, x): coordinates, velocities, and the first-order states of the
// attached StateElements (state_element.hpp), integrated together.
//
// Each evaluation at (q, v, x, t):
//   1. kinematics, and the world states of the bodies;
//   2. each state element's begin_evaluation (the drivetrain hands its wheel
//      spins to the tyres);
//   3. the force elements, whose world forces become generalized forces, and
//      the joint forces;
//   4. tau, then force_callback;
//   5. the constrained forward dynamics (ConstraintSolver);
//   6. each state element's rates.
// After each step the state is projected onto the constraints, and each state
// element's end_step makes its discrete changes.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/state_element.hpp"
#include "mbd/model/rigid_body.hpp"

namespace mbd::kernel {

/// A model with what acts on it: loop-closing constraints and force
/// elements. Builders fill one; a Simulator runs it.
struct System {
    Model model;
    std::vector<std::shared_ptr<const ConstraintModel>> constraints;
    std::vector<std::shared_ptr<ForceElement>> force_elements;
    /// Forces on joint coordinates, added to the generalized forces directly.
    std::vector<std::shared_ptr<const JointForce>> joint_forces;
};

enum class Integrator {
    RK4,                ///< Classical fourth order
    SemiImplicitEuler,  ///< First order and symplectic: velocities, then positions
};

class Simulator;

/// An event (plan task 3.9): a function of the state whose zero crossing
/// matters, such as a ball's height above the ground. When it changes sign
/// during a step, the step is shortened to the crossing, found to within
/// Simulator::event_tolerance in time, the action runs there, and the rest
/// of the step follows; the state is just past the crossing (the function
/// has its new sign). Every sign change in the watched direction is an
/// event: an action that sends the function back across zero (a bounce)
/// should watch one direction only. Only the signs at the ends of each
/// stretch of a step are compared, so a function that goes out and back
/// within one step (a bounce shorter than the step) is not seen: steps must
/// be short against the time the function spends on one side.
struct Event {
    std::string name;
    /// Of the simulator's state (q, v, time, and its kinematics: states()).
    std::function<Real(const Simulator&)> function;
    /// Runs at the crossing; may change q and v. May be empty.
    std::function<void(Simulator&)> action;
    /// -1: falling crossings only (positive to not positive); +1: rising
    /// only; 0: both.
    int direction{0};
};

/// An event that fired: its index in Simulator::events and the time.
struct FiredEvent {
    std::size_t event{0};
    Real time{0.0};
};

/// One step of the step trace (task H.4): how the step went, for finding out
/// afterwards what went wrong.
struct StepRecord {
    Real time{0.0};                 ///< At the end of the step
    Real dt{0.0};
    Real drift{0.0};                ///< |phi| after the integrator, before the projection
    Real position_residual{0.0};    ///< |phi| after the projection
    Real velocity_residual{0.0};    ///< |J v - nu| after the projection
    int projection_iterations{0};
    bool projection_converged{true};
    int equations{0};               ///< Constraint equations, and their rank, at the last evaluation
    int rank{0};
    Real kinetic_energy{0.0};       ///< At the end of the step
    Real potential_energy{0.0};     ///< Gravity's, at the end of the step
    int events{0};                  ///< Events that fired in the step
};

class Simulator {
public:
    /// The system must outlive the simulator and must not gain bodies or
    /// constraints afterwards. The state starts at the neutral configuration,
    /// at rest.
    explicit Simulator(System& sys);

    System& system;

    // --- State ----------------------------------------------------------------
    VecX q;            ///< Coordinates
    VecX v;            ///< Velocities
    VecX x;            ///< States of the attached state elements, in the order attached
    Real time{0.0};

    /// Integrate `element`'s states with the multibody state from now on; its
    /// states start at its initial_states(). The element is not owned: it
    /// must outlive the simulator. Allocates; attach before stepping.
    void attach(StateElement& element);
    /// The attached elements, and where each one's states start in x.
    const std::vector<StateElement*>& state_elements() const { return state_elements_; }
    int state_offset(std::size_t element) const { return state_offsets_[element]; }

    // --- Settings ---------------------------------------------------------------
    Integrator method{Integrator::RK4};
    bool project_constraints{true};     ///< Project q and v after each step
    Real projection_tolerance{1e-10};
    VecX tau;                           ///< Generalized forces for the next step; cleared after it

    /// Adds generalized forces at every evaluation. It may read states() and
    /// data(), but must not call acceleration() or refresh(): the evaluation
    /// in progress uses the kinematics in data().
    std::function<void(Simulator&, Real t, VecX& tau)> force_callback;

    /// Events watched at every step; see Event.
    std::vector<Event> events;
    /// How closely an event's time is found [s].
    Real event_tolerance{1e-10};
    /// More events than this within one step stop the step's event handling
    /// (a chain of crossings that never ends, such as an action that sends
    /// its function back across zero while both directions are watched),
    /// with a warning.
    int max_events_per_step{100};
    /// The events that fired, in order.
    const std::vector<FiredEvent>& event_log() const { return event_log_; }

    /// When set, every step adds a StepRecord to trace() (task H.4). Off by
    /// default: it allocates and costs the energies at each step.
    bool trace_steps{false};
    const std::vector<StepRecord>& trace() const { return trace_; }
    /// The trace as CSV, one row per step, 17 significant digits; throws
    /// MBD-K053 if the file cannot be written.
    void write_trace(const std::string& path) const;

    /// Project the state onto the constraints and refresh the kinematics.
    void initialize();

    /// Advance by dt.
    void step(Real dt);

    /// Advance by round(duration / dt) steps; returns their number.
    int run(Real duration, Real dt);

    /// Kinematics and body states at the current (q, v).
    void refresh();

    /// The accelerations at (q, v, t) with everything applied, the state
    /// elements at the simulator's x. The result is valid until the next
    /// evaluation.
    const VecX& acceleration(const VecX& q_at, const VecX& v_at, Real t);

    /// The same at the state elements' states x_at; their rates are then in
    /// state_rates().
    const VecX& acceleration(const VecX& q_at, const VecX& v_at, const VecX& x_at, Real t);

    /// The state elements' rates at the last evaluation.
    const VecX& state_rates() const { return x_dot_; }

    /// Kinematics at the current state (after initialize, step or refresh).
    const Data& data() const { return data_; }
    /// World states of the bodies at the current state.
    const std::vector<RigidBodyState>& states() const { return states_; }
    /// Body forces of the last evaluation.
    const std::vector<RigidBodyForces>& forces() const { return forces_; }

    ConstraintSolver& solver() { return solver_; }
    const ConstraintSolver& solver() const { return solver_; }
    const ProjectionInfo& last_projection() const { return last_projection_; }
    int projection_failures() const { return projection_failures_; }

private:
    void step_rk4(Real dt);
    void step_semi_implicit_euler(Real dt);
    /// One step of the integrator, the projection, time and kinematics.
    void advance(Real dt);
    /// A step of dt that stops at each event on the way.
    void step_with_events(Real dt);
    /// From the state (event_q0_, event_v0_, t0), the step length in (0,
    /// length] after which event i's function has crossed from g0 to its
    /// other sign, to within event_tolerance; the state is left there.
    Real locate_event(std::size_t i, Real t0, Real g0, Real length, Real g_end);

    Data data_;
    ConstraintSolver solver_;
    std::vector<RigidBodyState> states_;
    std::vector<RigidBodyForces> forces_;
    VecX tau_total_, tau_forces_;
    VecX q0_, v0_, q_stage_, v_stage_;
    VecX kq_[4], kv_[4];
    std::vector<StateElement*> state_elements_;
    std::vector<int> state_offsets_;
    VecX x_dot_, x0_, x_stage_, kx_[4];
    ProjectionInfo last_projection_;
    int projection_failures_{0};
    bool projection_failure_reported_{false};
    bool redundancy_reported_{false};
    VecX event_q0_, event_v0_, event_x0_;
    std::vector<Real> event_g0_;
    std::vector<FiredEvent> event_log_;
    bool event_chain_reported_{false};
    std::vector<StepRecord> trace_;
};

} // namespace mbd::kernel
