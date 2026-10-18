#pragma once

// Time integration on the multibody kernel (plan tasks 2.7, 4.1, 4.2). The
// state is (q, v, x): coordinates, velocities, and the first-order states of
// the attached StateElements (state_element.hpp), integrated together, by
// RK4, semi-implicit Euler, or the Dormand-Prince pair with step control and
// a continuous extension.
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
// After each step the state is projected onto the constraints (or, as an
// option, generalised-alpha solves its steps on them: task 4.7), and each
// state element's end_step makes its discrete changes. Dormand-Prince takes
// as many steps of its own within step(dt) as its error control asks for, the
// last one ending at t + dt exactly, and projects after each; end_step runs
// once, after the last.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Cholesky>

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
    /// First order: velocities, then positions. The velocity products of the
    /// inertia (Coriolis, centrifugal and gyroscopic terms) are linearly
    /// implicit and the new velocities on the constraints (D38). A spinning
    /// top, which blew up with those terms explicit, and a closed four-bar,
    /// which gained energy with them implicit but the velocities off the
    /// constraints, both only lose energy. Without either (bodies that only
    /// translate, no loops) it is the symplectic Euler method.
    SemiImplicitEuler,
    /// Fifth order with an embedded fourth-order error estimate, step
    /// control (StepControl) and a fourth-order continuous extension
    /// (Simulator::interpolate); Dormand and Prince 1980 (task 4.2).
    DormandPrince,
    /// Implicit and second order, for stiff models: the generalised-alpha
    /// method of Chung and Hulbert (1993), stable at any step, with the
    /// damping of motions too fast for the step set by
    /// ImplicitControl::rho_infinity; each step solved by Newton's method
    /// (task 4.3); optionally on the loop constraints, in the stabilized
    /// index-2 form (ImplicitControl::stabilized_index2, task 4.7).
    GeneralizedAlpha,
    /// Implicit, fourth order and L-stable: the five-stage singly diagonally
    /// implicit Runge-Kutta method of Hairer and Wanner (SDIRK4), with an
    /// embedded third-order error estimate and the same step control as
    /// DormandPrince (task 4.4). Each stage is solved on its own, cheaply,
    /// but its stage order is 1: beside stiff springs or bushings (anything
    /// that acts almost as a joint) its accuracy falls towards first order
    /// and its error estimate understates the error. Use it for smooth
    /// models and for stiffness in first-order states; for stiff mechanical
    /// models use RadauIIA or GeneralizedAlpha.
    SDIRK4,
    /// Implicit, fifth order and L-stable, for stiff models at tight
    /// tolerances: the three-stage Radau IIA method (stage order 3), as in
    /// Hairer and Wanner's RADAU5, with an embedded third-order error
    /// estimate and the same step control as DormandPrince (task 4.4).
    RadauIIA,
    /// Linearly implicit, first order, for real time: the cost of every step
    /// is the same (task 4.5). One evaluation, then one linear solve that
    /// makes the applied forces implicit through their derivatives (see
    /// Simulator::force_jacobian): (M - h C - h^2 K + h D) dv = ..., D the
    /// derivative of the inertia's velocity products (gyroscopic terms
    /// included), the new velocities on the constraints (D38).
    /// The state elements' states are implicit in themselves. Stable at any
    /// step for stiff springs, bushings, contact and tyres; damps every
    /// motion by about (w h)^2 / 2 per step, as backward Euler does.
    LinearlyImplicitEuler,
};

/// How Integrator::DormandPrince, SDIRK4 and RadauIIA choose their steps
/// (tasks 4.2, 4.4), and how closely the implicit methods solve each of
/// theirs (the tolerances; see ImplicitControl). The implicit methods'
/// estimates are of third order, so their next step follows the fourth root
/// of the measure, and are filtered through an iteration matrix so that stiff
/// components do not inflate them.
///
/// Each step's error estimate is weighed entry by entry, over q, v and x, by
/// absolute_tolerance + relative_tolerance * |entry|, and the root mean
/// square of the weighted entries must not exceed 1; a step that exceeds it
/// is taken again, shorter. After each step the next length is chosen to
/// bring that measure to about 0.6 (a safety factor of 0.9 on the fifth root),
/// growing at most fivefold, and not at all after a rejected step. Steps
/// never pass the end of step(dt), and the length reached carries over to the
/// next call. The tolerances are local: the global error is roughly
/// proportional to them, by a factor that depends on the model and the span.
struct StepControl {
    /// Off: one step of the whole step(dt) (or of what remains of it after an
    /// event), with no error control; for measuring convergence orders.
    bool adaptive{true};
    Real relative_tolerance{1e-6};
    /// The entries mix units (m, rad, m/s, rad/s, the states'); one value
    /// for all.
    Real absolute_tolerance{1e-6};
    /// A step that would have to be shorter than this throws MBD-K055 [s].
    Real min_step{1e-12};
    /// Steps tried, accepted or rejected, within one call to step(); more
    /// throw MBD-K056.
    int max_steps{100000};
};

/// How the implicit methods take their steps: Integrator::GeneralizedAlpha
/// (task 4.3) and, for max_iterations, Integrator::SDIRK4 and RadauIIA (task
/// 4.4).
///
/// GeneralizedAlpha solves each step for the new accelerations, and the
/// state elements' new rates, by Newton's method. Its iteration matrix comes
/// from derivatives of the accelerations and rates by forward differences
/// (one evaluation per velocity, twice over, and one per state), kept from
/// step to step while Newton converges quickly and rebuilt when it slows.
/// Newton stops when its last correction to the velocities and the states,
/// weighed as in StepControl by absolute_tolerance + relative_tolerance
/// |entry| (but never finer than the rounding of the terms the step adds
/// up), has a root mean square of at most 1.
///
/// SDIRK4 (stage by stage) and RadauIIA (all three stages together) do the
/// same on the whole state (q, v, x), with the derivative of its full
/// right-hand side (nq + nv + nx + 1 evaluations), and stop at a fraction
/// min(0.03, sqrt(tol)) of the tolerances (tol the larger of the two, as in
/// Hairer and Wanner's RADAU5): Newton's errors add up over the steps, unseen
/// by the error estimate. They keep that derivative while
/// the Newton iterations a step costs, beyond those of the first step with
/// it, stay below the evaluations of building it again; SDIRK4 rebuilds it
/// at a stage whose Newton fails, RadauIIA at the step's start.
struct ImplicitControl {
    /// The factor by which motions much too fast for the step shrink in each
    /// step, from 0 to 1: 1 keeps them (the trapezoidal rule, no damping at
    /// all), 0 removes them within a step. The lower it is, the more the slow
    /// motions are damped as well, though far less.
    Real rho_infinity{0.8};
    /// Newton iterations per step (per stage for SDIRK4). Past them, or if
    /// the corrections stop falling, the iteration matrix is rebuilt and the
    /// step (stage) solved again; a second failure throws MBD-K058, or, with
    /// error control, shortens the step.
    int max_iterations{10};
    /// GeneralizedAlpha on a closed mechanism (task 4.7, D37): solve each
    /// step on the loop constraints, in the stabilized index-2 form of Gear,
    /// Leimkuhler and Gupta as Arnold, Bruls and Cardona apply it to this
    /// method, instead of projecting after it. Newton's unknowns gain, per
    /// independent constraint equation, a constraint force J^T kappa added to
    /// the accelerations, which holds J v = nu at the step's end, and a
    /// position correction M^-1 J^T eta, which holds phi = 0; it stops with
    /// |phi| within Simulator::constraint_tolerance and |J v - nu| within
    /// step_control.absolute_tolerance. A projection leaves the method's
    /// history (its acceleration-like variable) where the unprojected step
    /// put it; this form does not. Measured on a four-bar: second order,
    /// positions 1.5 to 1.1 times and accelerations 1.7 to 1.2 times more
    /// accurate than projected at steps of 4 to 1 ms; on the sedan, positions
    /// 1.15 times more accurate, but each step costs 1.45 times as much (its
    /// Newton matrix has n + m rows instead of n, m the constraint equations)
    /// and a moving steering input costs 12 to 23 % more Newton iterations.
    /// Off by default: a shorter projected step buys more accuracy for the
    /// same work. (ADAMS makes the same choice between its I3 and SI2
    /// formulations.)
    bool stabilized_index2{false};
};

/// What the integrator has done since the simulator was built.
struct IntegratorStats {
    long long steps{0};         ///< Steps taken: for the methods with error control their own, accepted
    long long rejected{0};      ///< Steps rejected by the error control, or because Newton failed
    long long evaluations{0};   ///< Evaluations of the accelerations by the integrator
    Real step_size{0.0};        ///< Length of the last step taken [s]
    /// The methods with error control: the last accepted step's error measure
    /// (1 = at the tolerance; see StepControl). Zero for RadauIIA without
    /// error control, which then does not estimate it.
    Real error{0.0};
    /// The implicit methods: Newton iterations (one evaluation each, three
    /// for RadauIIA), and iteration matrices built by finite differences.
    long long newton_iterations{0};
    long long jacobians{0};
    /// Multirate (Simulator::multirate): evaluations of the state elements'
    /// rates alone (kinematics, forces and rates, without the dynamics) by
    /// the fast steps.
    long long state_evaluations{0};
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
///
/// With Integrator::DormandPrince the signs are compared at the ends of each
/// of its own steps, and a crossing is found on the step's continuous
/// extension instead of by integrating again: the search costs kinematics
/// only, no evaluation of the forces.
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

/// How far the state is from its constraints (task 4.7): the drift channel.
/// The drift is what the integrator's step left, before the projection moved
/// it back; the residuals are the state's own, after it. GeneralizedAlpha
/// with ImplicitControl::stabilized_index2, which solves its steps on the
/// constraints, projects nothing: its drift is its residual. Positions in m
/// and rad, velocities in m/s and rad/s, mixed as the constraints' equations
/// mix them.
struct ConstraintDrift {
    Real position{0.0};            ///< |phi| of the last step's result
    Real velocity{0.0};            ///< |J v - nu| of the last step's result
    Real position_residual{0.0};   ///< |phi| of the state
    Real velocity_residual{0.0};   ///< |J v - nu| of the state
    /// The largest of each over every step since the simulator was built.
    Real max_position{0.0};
    Real max_velocity{0.0};
    Real max_position_residual{0.0};
    Real max_velocity_residual{0.0};
};

/// One step of the step trace (task H.4): how the step went, for finding out
/// afterwards what went wrong.
struct StepRecord {
    Real time{0.0};                 ///< At the end of the step
    Real dt{0.0};
    Real drift{0.0};                ///< |phi| after the integrator, before the projection
    Real velocity_drift{0.0};       ///< |J v - nu| after the integrator, before the projection
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
    ~Simulator();

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
    /// For Integrator::DormandPrince, and the tolerances of GeneralizedAlpha.
    StepControl step_control;
    /// For Integrator::GeneralizedAlpha.
    ImplicitControl implicit_control;
    /// Keep the state on the loop constraints, not only their accelerations
    /// (task 4.7, D37): each method projects q and v onto them after each of
    /// its steps, or, with implicit_control.stabilized_index2,
    /// GeneralizedAlpha solves its steps on them. initialize() projects in
    /// either case. Off, the state drifts from the constraints as the
    /// integrator's errors add up.
    bool enforce_constraints{true};
    /// How closely the positions are held: |phi| [m, rad] after a
    /// projection or a stabilized GeneralizedAlpha step. A projection moves
    /// the velocities onto J v = nu exactly (to rounding); the stabilized
    /// step holds |J v - nu| to step_control.absolute_tolerance, as its
    /// Newton iterations hold the velocities themselves (a residual there
    /// does not add up: each step's positions are held to phi = 0 again).
    Real constraint_tolerance{1e-10};
    /// Multirate integration (task 4.6, D36) with Integrator::RK4: the state
    /// elements' states (the fast ones: wheel spins, lags) take this many RK4
    /// steps for each step of the bodies, their rates evaluated without the
    /// dynamics, the bodies extrapolated over the step from its start; 1 is
    /// single rate. It pays where the states are stiff and the bodies are
    /// not: a car at walking pace, whose wheel spins hold RK4 to under a
    /// millisecond, steps its bodies at 3 ms with 6 fast steps of 0.5 ms, at
    /// a sixth of the dynamics' evaluations. Each step must stay inside its
    /// own part's stability: the fast steps inside the states' (a spin that
    /// chatters within the step makes the bodies drift, as it does not at a
    /// single rate), the step itself inside the bodies'. Any other method
    /// with a value other than 1 throws MBD-K046.
    int multirate{1};
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

    /// The derivatives of the applied generalized forces, those of the force
    /// elements and joint forces (not tau or force_callback), at the current
    /// state: K = d tau / dq, q changed along each velocity (q (+) e_j), and C
    /// = d tau / dv, both nv x nv (task 4.5). Element by element: each
    /// element's loads on its bodies are differentiated with respect to those
    /// bodies' poses and velocities, by differences of that element alone,
    /// and mapped through the bodies' Jacobians. Left out: the terms from the
    /// joints' geometry, how a load maps to generalized forces as the joints
    /// move (the "geometric stiffness" of the tree). Evaluates at the current
    /// state first.
    void force_jacobian(MatX& K, MatX& C);

    /// What the integrator has done so far.
    const IntegratorStats& integrator_stats() const { return stats_; }

    /// Integrator::DormandPrince: the state at time t within its last step,
    /// from the step's continuous extension (fourth order; it passes through
    /// the state before the projection). q_out is normalized. Throws MBD-K057
    /// for a t outside the last step, or before Dormand-Prince has taken one.
    void interpolate(Real t, VecX& q_out, VecX& v_out, VecX& x_out) const;

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
    /// The drift channel: how far the last step's result and the state are
    /// from the constraints, and the largest values so far (task 4.7). For a
    /// Recorder, add_drift_channels() in outputs.hpp.
    const ConstraintDrift& constraint_drift() const { return drift_; }

private:
    /// acceleration(), counted in the integrator's statistics.
    const VecX& evaluate(const VecX& q_at, const VecX& v_at, const VecX& x_at, Real t);
    void step_rk4(Real dt);
    /// One multirate RK4 step of dt, multirate fast steps of the states
    /// within it (simulator_multirate.cpp).
    void step_rk4_multirate(Real dt);
    /// The state elements' rates at (q_at, v_at, x_at, t) without the
    /// dynamics: kinematics, the force elements (whose outputs the rates may
    /// read), the rates, into x_dot_.
    const VecX& state_rates_only(const VecX& q_at, const VecX& v_at, const VecX& x_at, Real t);
    /// One semi-implicit Euler step of dt, the velocity products linearly
    /// implicit (simulator_linear.cpp).
    void step_semi_implicit_euler(Real dt);
    /// Dormand-Prince's own steps from time to time + dt, each projected,
    /// with events watched after each.
    void step_dormand_prince(Real dt);
    /// After an accepted Dormand-Prince step, with the kinematics at its end:
    /// the earliest event that crossed within it, found on the interpolant;
    /// the state is moved there and the action runs. Returns whether one did.
    bool dormand_prince_event();
    /// One generalised-alpha step of dt (simulator_implicit.cpp).
    void step_generalized_alpha(Real dt);
    /// The derivatives of the accelerations and the states' rates with
    /// respect to q (along the velocities), v and x at the current state, by
    /// forward differences from implicit_qdd_ and implicit_xd_.
    void implicit_jacobian();
    /// SDIRK4's own steps from time to time + dt, each projected
    /// (simulator_implicit.cpp).
    void step_sdirk(Real dt);
    /// SDIRK4's five stages of one step of length h from sdirk_yn_, adding
    /// their Newton iterations to `iterations`. A stage whose Newton fails
    /// gets an iteration matrix built at its predictor (`rebuilt`) and one
    /// more try; false if that fails too (`singular` if the matrix was).
    bool sdirk_stages(Real h, int& iterations, bool& rebuilt, bool& singular);
    /// Newton's iterations for stage i from the predictor in sdirk_y_; false
    /// if they stop falling or run out.
    bool sdirk_newton(int i, Real h, int& iterations);
    /// The iteration matrix I - gamma h J, factorized in place unless it
    /// already is for this h and J; false at a zero pivot.
    bool sdirk_factor(Real h);
    /// Radau IIA's own steps from time to time + dt, each projected
    /// (simulator_implicit.cpp).
    void step_radau(Real dt);
    /// Newton's iterations on Radau IIA's three stages of a step of length h
    /// from radau_yn_, starting from the increments in radau_z_; false if
    /// they stop falling or run out.
    bool radau_newton(Real h, int& iterations);
    /// Radau IIA's two iteration matrices gamma I - h J and (alpha + i beta)
    /// I - h J, factorized in place unless they already are for this h and
    /// J; false at a zero pivot.
    bool radau_factor(Real h);
    /// One linearly implicit Euler step of dt (simulator_linear.cpp).
    void step_linearly_implicit(Real dt);
    /// The constrained solve of linear_'s W, r into its dv, where W's
    /// symmetric part is positive definite; false otherwise.
    bool inertia_solve();
    /// MBD-K045, once, for the named step.
    void report_not_positive_definite(const char* step);
    /// K and C (see force_jacobian) into linear_, at the kinematics and body
    /// states of the last evaluation (allocates only the first time); with
    /// `states`, also the state elements' rates' derivatives with respect to
    /// v through the force elements' outputs.
    void linear_force_jacobian(bool states);
    /// The state elements' rates' and the generalized forces' derivatives
    /// with respect to the states, at the last evaluation's kinematics and
    /// the states x, into linear_.
    void linear_state_jacobian(Real t);
    /// Size the stacked work of SDIRK4 and Radau IIA (allocates).
    void size_stacked(Index n);
    /// f(y) = (q_dot, accelerations, rates) at the stacked state y = (q, v,
    /// x) and time t, into `out`: one evaluation.
    void stacked_f(const VecX& y, Real t, VecX& out);
    /// The derivative J of f with respect to y at y and t, by forward
    /// differences (n + 1 evaluations), and f there (stacked_f0_).
    void stacked_jacobian(const VecX& y, Real t);
    /// After an accepted step whose Newton iterations cost `evaluations`:
    /// keep J, or mark it to be built again (see ImplicitControl).
    void stacked_jacobian_policy(int evaluations);
    /// How much f amplifies the rounding of the state y: entry i is sum_j
    /// |J_ij| |y_j| (into stacked_noise_), for Newton's convergence test.
    void stacked_noise(const VecX& y);
    /// Project the state onto the constraints after a step, and record the
    /// drift.
    void project_step();
    /// After a GeneralizedAlpha step solved on the constraints: the
    /// kinematics at the new state, its residuals measured as the drift.
    void measure_step_drift();
    /// Fold a step's drift and residuals into the drift channel.
    void record_drift(Real position, Real velocity, Real position_residual, Real velocity_residual);
    /// Whether GeneralizedAlpha solves its steps on the constraints.
    bool stabilized_alpha() const;
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
    VecX kq_[7], kv_[7];                  // stage derivatives: RK4 uses four, Dormand-Prince seven
    std::vector<StateElement*> state_elements_;
    std::vector<int> state_offsets_;
    VecX x_dot_, x0_, x_stage_, kx_[7];
    // Multirate: the bodies at a fast stage, the fast states with their stage
    // value and rates, and the states at the step's middle.
    VecX mr_q_, mr_v_, mr_x_, mr_xs_, mr_kx_[4], mr_x_half_;
    IntegratorStats stats_;
    // Dormand-Prince: error estimates, the last step's continuous extension
    // (five coefficient vectors each) and the step length carried over.
    VecX error_q_, error_v_, error_x_;
    VecX dense_q_[5], dense_v_[5], dense_x_[5];
    Real dense_t0_{0.0}, dense_h_{0.0};
    Real next_step_{0.0};
    // Generalised-alpha: the method's acceleration-like variables of the last
    // step (for v and for x), the accelerations and rates at the start of the
    // step, the state the last step left (the history holds only while
    // nothing else has changed it), the derivatives [d/dq d/dv d/dx] of the
    // accelerations and rates, the iteration matrix factorized in place with
    // its pivots, and work vectors.
    VecX implicit_a_, implicit_w_, implicit_qdd_, implicit_xd_;
    VecX implicit_q_, implicit_v_, implicit_x_;
    Real implicit_time_{0.0};
    bool implicit_history_{false};
    MatX implicit_jacobian_, implicit_matrix_;
    std::vector<Index> implicit_pivots_;
    bool implicit_jacobian_valid_{false};
    VecX implicit_a_new_, implicit_w_new_, implicit_residual_, implicit_dq_;
    // Generalised-alpha in the stabilized index-2 form (task 4.7, D37): the
    // derivative G of J v - nu with respect to q, built with the iteration
    // matrix's differences, and J v - nu where they start; the independent
    // equations chosen at the step's predictor, their rows of J there and
    // M^-1 J^T (the direction of the position correction); the extra
    // constraint forces of the last step, one entry per equation (kappa,
    // which holds J v = nu); kappa and eta (the position correction, scaled
    // by 1 / (beta h^2)) within a step, one entry per chosen equation; the
    // accelerations at the start with the last step's kappa, and at a Newton
    // iterate with the current one; for eliminating eta from the Newton
    // system, A = J_c M^-1 J_c^T with its factor, F = A^-1 [J_c 0], J_q P,
    // G_c P and G's chosen rows; work vectors.
    MatX implicit_G_;
    VecX implicit_g0_;
    std::vector<Index> implicit_rows_;
    MatX implicit_Jc_, implicit_P_, implicit_Gc_, implicit_A_, implicit_F_, implicit_JqP_, implicit_GP_;
    Eigen::LLT<MatX> implicit_llt_;
    VecX implicit_kappa_, implicit_kappa_new_, implicit_eta_new_, implicit_unit_;
    VecX implicit_rp_, implicit_f_, implicit_deta_;
    VecX implicit_qddk_, implicit_qdd_plus_;
    // SDIRK4 and Radau IIA, on the stacked state y = (q, v, x): the
    // derivative J of the right-hand side f, numbered so that a
    // factorization knows which J it holds; f where J was built; the Newton
    // evaluations of the first whole step with that J (-1 until there is
    // one); |J| |y| at the step's start; work vectors (a perturbed state and
    // f there, q_dot).
    MatX stacked_jacobian_;
    long long stacked_version_{0};
    bool stacked_jacobian_valid_{false};
    int stacked_base_cost_{-1};
    VecX stacked_noise_;
    VecX stacked_f0_, stacked_y_, stacked_fp_, stacked_qd_;
    // SDIRK4: the iteration matrix I - gamma h J factorized in place with its
    // pivots, and the step length and J it holds; the stage derivatives; f at
    // the step's start (when known); work vectors (the state at the step's
    // start, a stage's known part and its value, a residual, an error
    // estimate).
    MatX sdirk_matrix_;
    std::vector<Index> sdirk_pivots_;
    Real sdirk_factored_h_{0.0};
    long long sdirk_factored_version_{-1};
    bool sdirk_factored_regular_{false};
    VecX sdirk_k_[5];
    VecX sdirk_fn_;
    bool sdirk_fn_valid_{false};
    VecX sdirk_yn_, sdirk_s_, sdirk_y_, sdirk_r_, sdirk_e_;
    // Radau IIA: the iteration matrices gamma I - h J (real) and (alpha + i
    // beta) I - h J (complex) factorized in place with their pivots, and the
    // step length and J they hold; the stages' increments Z, their transforms
    // W = T^-1 Z and f at the stages; the last step's collocation polynomial
    // (three coefficient vectors), its length and the state and time it ended
    // at, for the next step's starting values; work vectors (the state at the
    // step's start, a stage's state, f at the start, a real and a complex
    // right-hand side, an error estimate).
    MatX radau_e1_;
    Eigen::MatrixXcd radau_e2_;
    std::vector<Index> radau_p1_, radau_p2_;
    Real radau_factored_h_{0.0};
    long long radau_factored_version_{-1};
    bool radau_factored_regular_{false};
    VecX radau_z_[3], radau_w_[3], radau_f_[3], radau_cont_[3];
    Real radau_h_old_{0.0}, radau_end_time_{0.0};
    bool radau_cont_valid_{false}, radau_first_{true};
    VecX radau_end_, radau_yn_, radau_y_, radau_f0_, radau_r_, radau_e_;
    Eigen::VectorXcd radau_rc_;
    // Linearly implicit Euler: its work, sized at its first step
    // (simulator_linear.cpp).
    struct LinearWork;
    std::unique_ptr<LinearWork> linear_;
    ProjectionInfo last_projection_;
    ConstraintDrift drift_;
    VecX drift_work_;   // J v - nu, one entry per equation
    int projection_failures_{0};
    bool projection_failure_reported_{false};
    bool redundancy_reported_{false};
    VecX event_q0_, event_v0_, event_x0_;
    std::vector<Real> event_g0_, event_g1_;
    std::vector<FiredEvent> event_log_;
    bool event_chain_reported_{false};
    std::vector<StepRecord> trace_;
};

} // namespace mbd::kernel
