#pragma once

// Time integration on the multibody kernel (plan task 2.7). Phase 4 builds
// the integrator interface proper on top of this.
//
// Each evaluation of the accelerations at (q, v, t):
//   1. kinematics, and the world states of the bodies;
//   2. pre_force_callback (for example, the drivetrain hands wheel spins to
//      the tyres);
//   3. the force elements, whose world forces become generalized forces;
//   4. tau, then force_callback;
//   5. the constrained forward dynamics (ConstraintSolver).
// After each step the state is projected onto the constraints.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/model.hpp"
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

class Simulator;

/// A function of the state whose sign changes mark an event (task 3.9): an
/// impact, a switch, a limit reached. Within a step whose start and end give
/// it opposite signs, the step is integrated again to the root, found to
/// Simulator::event_tolerance in time; there `action` runs (it may change q
/// and v, an impact's rebound for instance) and the rest of the step
/// follows.
struct Event {
    std::string name;
    /// g at the simulator's present state (q, v, time, states()).
    std::function<Real(const Simulator&)> function;
    /// +1: only rising crossings (- to +); -1: only falling; 0: both.
    int direction{0};
    /// Called at the event, just past the crossing. Optional.
    std::function<void(Simulator&)> action;
    /// End the step, and run(), at the event.
    bool stop{false};
};

/// An event that happened: its index in Simulator::events and its time.
struct EventRecord {
    std::size_t event{0};
    Real time{0.0};
};

enum class Integrator {
    RK4,                ///< Classical fourth order
    SemiImplicitEuler,  ///< First order and symplectic: velocities, then positions
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
    Real time{0.0};

    // --- Settings ---------------------------------------------------------------
    Integrator method{Integrator::RK4};
    bool project_constraints{true};     ///< Project q and v after each step
    Real projection_tolerance{1e-10};
    VecX tau;                           ///< Generalized forces for the next step; cleared after it

    /// Called at every evaluation, before the force elements. These two may
    /// read states() and data(), but must not call acceleration() or
    /// refresh(): the evaluation in progress uses the kinematics in data().
    std::function<void(Simulator&, Real t)> pre_force_callback;
    /// Adds generalized forces at every evaluation.
    std::function<void(Simulator&, Real t, VecX& tau)> force_callback;
    /// Called after every completed step.
    std::function<void(Simulator&, Real dt)> post_step_callback;

    // --- Events (task 3.9) ------------------------------------------------------
    std::vector<Event> events;
    /// Width of the time interval within which an event is located [s].
    Real event_tolerance{1e-10};
    /// The events that have happened, in order.
    const std::vector<EventRecord>& event_log() const { return event_log_; }
    /// True if the last step ended at an event with `stop`.
    bool stopped() const { return stopped_; }

    /// Project the state onto the constraints and refresh the kinematics.
    void initialize();

    /// Advance by dt, stopping at each event within it (see Event); ends at
    /// time + dt unless an event with `stop` ends it first.
    void step(Real dt);

    /// Advance by round(duration / dt) steps; returns the number taken, fewer
    /// if an event with `stop` ended one.
    int run(Real duration, Real dt);

    /// Kinematics and body states at the current (q, v).
    void refresh();

    /// The accelerations at (q, v, t) with everything applied. The result is
    /// valid until the next evaluation.
    const VecX& acceleration(const VecX& q_at, const VecX& v_at, Real t);

    /// The generalized forces of everything applied at (q, v, t): the force
    /// elements, the joint forces, tau and force_callback, after
    /// pre_force_callback; not gravity, velocity products or constraint
    /// forces. The first half of acceleration(), for statics and
    /// linearisation. Valid until the next evaluation.
    const VecX& applied_forces(const VecX& q_at, const VecX& v_at, Real t);

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
    /// One integrator step, the projection and the kinematics: a step
    /// without events.
    void advance(Real dt);
    /// The time within (0, h] at which event i's function crosses, from the
    /// state (q, v, time) saved in q_event_, v_event_, t_event_.
    Real locate_event(std::size_t i, Real h, Real g0, Real g1);
    void restore_event_start();

    VecX q_event_, v_event_;
    Real t_event_{0.0};
    std::vector<Real> g_start_, g_end_;
    std::vector<EventRecord> event_log_;
    bool stopped_{false};
    bool chatter_reported_{false};

    Data data_;
    ConstraintSolver solver_;
    std::vector<RigidBodyState> states_;
    std::vector<RigidBodyForces> forces_;
    VecX tau_total_, tau_forces_;
    VecX q0_, v0_, q_stage_, v_stage_;
    VecX kq_[4], kv_[4];
    ProjectionInfo last_projection_;
    int projection_failures_{0};
    bool projection_failure_reported_{false};
    bool redundancy_reported_{false};
};

} // namespace mbd::kernel
