#include "mbd/kernel/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"

#include "checks.hpp"
#include "conflicts.hpp"
#include "labels.hpp"
#include "linear_work.hpp"
#include "step_messages.hpp"

namespace mbd::kernel {

namespace detail {

void check_step_control(const StepControl& sc)
{
    MBD_THROW_IF(!(sc.absolute_tolerance > 0.0) || !(sc.relative_tolerance >= 0.0) || !(sc.min_step >= 0.0)
                     || sc.max_steps < 1,
                 "MBD-K054: kernel::Simulator: step_control needs absolute_tolerance > 0, relative_tolerance >= 0, "
                 "min_step >= 0 and max_steps >= 1; it has "
                     + sci(sc.absolute_tolerance) + ", " + sci(sc.relative_tolerance) + ", " + sci(sc.min_step)
                     + " and " + std::to_string(sc.max_steps) + ".");
}

void throw_step_too_short(const char* method, Real h, Real t, Real min_step, Real error)
{
    throw MbdError("MBD-K055: kernel::Simulator: " + std::string(method) + " needs a step of " + sci(h)
                   + " s at t = " + std::to_string(t) + " s, below step_control.min_step (" + sci(min_step)
                   + " s); the error measure of the last try was " + sci(error)
                   + ". The model is stiff or discontinuous here.");
}

void throw_too_many_steps(const char* method, int max_steps, Real dt, Real t)
{
    throw MbdError("MBD-K056: kernel::Simulator: " + std::string(method) + " took " + std::to_string(max_steps)
                   + " steps within one step of " + std::to_string(dt) + " s and reached only t = "
                   + std::to_string(t) + " s (step_control.max_steps). The model is stiff or discontinuous here, "
                     "or the tolerances are too tight for it.");
}

void throw_unsolved(const char* method, Real h, Real t, bool regular, int max_iterations)
{
    throw MbdError("MBD-K058: kernel::Simulator: the " + std::string(method) + " step of " + std::to_string(h)
                   + " s from t = " + std::to_string(t) + " s did not converge"
                   + (regular ? " within " + std::to_string(max_iterations)
                                    + " Newton iterations, or its corrections stopped falling, even with an "
                                      "iteration matrix just built there."
                              : std::string(": the iteration matrix is singular."))
                   + " The forces change too fast across the step, or are not finite.");
}

} // namespace detail

namespace {

/// Whether a function went from g0 to g1 across zero in the watched
/// direction (-1 falling, +1 rising, 0 either).
bool crossed(Real g0, Real g1, int direction)
{
    const bool rising = g0 < 0.0 && g1 >= 0.0;
    const bool falling = g0 > 0.0 && g1 <= 0.0;
    return (rising && direction >= 0) || (falling && direction <= 0);
}

void report_event_chain(int fired, Real time)
{
    report_warning("MBD-K052: " + std::to_string(fired) + " events in one step at t = " + std::to_string(time)
                   + " s; the rest of the step ignores events. An action that sends its function back across "
                     "zero makes a chain that never ends unless only one direction is watched. Further chains "
                     "are not reported.");
}

} // namespace

Simulator::Simulator(System& sys)
    : system(sys)
    , data_(sys.model)
    , solver_(sys.model, sys.constraints)
{
    const Model& model = system.model;
    // Every force element must act on bodies of the model: apply() would
    // otherwise index the states out of bounds.
    for (std::size_t k = 0; k < system.force_elements.size(); ++k) {
        const auto& f = system.force_elements[k];
        MBD_THROW_IF(!f, "MBD-K040: kernel::Simulator: force element " + std::to_string(k) + " is empty");
        for (BodyIndex b : f->bodies()) {
            MBD_THROW_IF(b < 0 || b >= model.nbodies(),
                         "MBD-K041: kernel::Simulator: force element " + std::to_string(k) + " ("
                             + f->name() + ") refers to body " + std::to_string(b)
                             + ", but the model's bodies are 0 to "
                             + std::to_string(model.nbodies() - 1) + ".");
        }
    }
    for (std::size_t k = 0; k < system.joint_forces.size(); ++k) {
        const auto& jf = system.joint_forces[k];
        MBD_THROW_IF(!jf, "MBD-K042: kernel::Simulator: joint force " + std::to_string(k) + " is empty");
        for (int b : jf->bodies()) {
            MBD_THROW_IF(b < 1 || b >= model.nbodies(),
                         "MBD-K043: kernel::Simulator: joint force " + std::to_string(k) + " ("
                             + jf->name() + ") refers to body " + std::to_string(b)
                             + ", but the model's jointed bodies are 1 to "
                             + std::to_string(model.nbodies() - 1) + ".");
        }
    }
    // The projection after each step weighs by the mass matrix of the step's
    // last stage instead of computing another.
    solver_.reuse_mass_matrix = true;
    q = model.neutral_configuration();
    v = VecX::Zero(model.nv);
    tau = VecX::Zero(model.nv);
    tau_total_ = VecX::Zero(model.nv);
    tau_forces_ = VecX::Zero(model.nv);
    drift_work_ = VecX::Zero(solver_.size());
    q0_ = q;
    q_stage_ = q;
    v0_ = v;
    v_stage_ = v;
    for (int k = 0; k < 7; ++k) {
        kq_[k] = VecX::Zero(model.nq);
        kv_[k] = VecX::Zero(model.nv);
        kx_[k] = VecX::Zero(0);
    }
    error_q_ = VecX::Zero(model.nq);
    error_v_ = VecX::Zero(model.nv);
    for (int k = 0; k < 5; ++k) {
        dense_q_[k] = VecX::Zero(model.nq);
        dense_v_[k] = VecX::Zero(model.nv);
    }
    x = VecX::Zero(0);
    x_dot_ = x;
    x0_ = x;
    x_stage_ = x;
    event_x0_ = x;
    error_x_ = x;
    for (auto& d : dense_x_) d = x;
    states_.resize(static_cast<std::size_t>(model.nbodies()));
    forces_.resize(static_cast<std::size_t>(model.nbodies()));
    refresh();
}

void Simulator::attach(StateElement& element)
{
    for (const StateElement* e : state_elements_) {
        MBD_THROW_IF(e == &element, "MBD-K044: kernel::Simulator::attach: the state element ("
                                        + std::string(element.name()) + ") is attached already");
    }
    const int n = element.size();
    MBD_THROW_IF(n < 0, "MBD-K044: kernel::Simulator::attach: the state element ("
                            + std::string(element.name()) + ") has " + std::to_string(n) + " states");
    const int offset = static_cast<int>(x.size());
    state_elements_.push_back(&element);
    state_offsets_.push_back(offset);
    x.conservativeResize(offset + n);
    element.initial_states(x.segment(offset, n));
    const Index total = x.size();
    x_dot_ = VecX::Zero(total);
    x0_ = x;
    x_stage_ = x;
    event_x0_ = x;
    error_x_ = x;
    for (auto& k : kx_) k = VecX::Zero(total);
    for (auto& d : dense_x_) d = x;
    dense_h_ = 0.0;   // the last step's continuous extension has no x of this size
}

void Simulator::refresh()
{
    forward_kinematics(system.model, data_, q, v);
    body_states(system.model, data_, states_);
}

void Simulator::initialize()
{
    checks::q("kernel::Simulator::initialize", system.model, q);
    checks::v("kernel::Simulator::initialize", system.model, v);
    if (enforce_constraints && solver_.size() > 0) {
        last_projection_ = solver_.project(data_, q, v, time, constraint_tolerance);
    }
    refresh();
}

const VecX& Simulator::acceleration(const VecX& q_at, const VecX& v_at, Real t)
{
    return acceleration(q_at, v_at, x, t);
}

const VecX& Simulator::acceleration(const VecX& q_at, const VecX& v_at, const VecX& x_at, Real t)
{
    const Model& model = system.model;
    checks::q("kernel::Simulator::acceleration", model, q_at);
    checks::v("kernel::Simulator::acceleration", model, v_at);
    checks::size("kernel::Simulator::acceleration", "x", x_at, "states", static_cast<int>(x.size()));
    // One kinematics pass serves the forces and the dynamics: with zero joint
    // accelerations it also gives the velocity-product terms the constraints
    // need.
    forward_kinematics(model, data_, q_at, v_at, data_.zero_v);
    body_states(model, data_, states_);

    for (std::size_t i = 0; i < state_elements_.size(); ++i) {
        state_elements_[i]->begin_evaluation(*this, t, x_at.segment(state_offsets_[i], state_elements_[i]->size()));
    }

    for (auto& f : forces_) {
        f.f_W.setZero();
        f.tau_W.setZero();
    }
    for (const auto& element : system.force_elements) element->apply(states_, forces_);
    generalized_forces(model, data_, forces_, tau_forces_);
    for (const auto& jf : system.joint_forces) jf->apply(q_at, v_at, tau_forces_);

    tau_total_ = tau + tau_forces_;
    if (force_callback) force_callback(*this, t, tau_total_);

    const VecX& v_dot = solver_.forward_dynamics_from_kinematics(data_, v_at, tau_total_, t);
    if (solver_.info().redundant() && !redundancy_reported_) {
        redundancy_reported_ = true;
        report_warning("MBD-K050: Redundant constraints: " + std::to_string(solver_.info().equations)
                       + " equations of rank " + std::to_string(solver_.info().rank)
                       + ". The motion is unaffected; the multipliers of the redundant "
                         "equations are set to zero.");
    }
    for (std::size_t i = 0; i < state_elements_.size(); ++i) {
        const Index n = state_elements_[i]->size();
        state_elements_[i]->rates(*this, t, x_at.segment(state_offsets_[i], n),
                                  x_dot_.segment(state_offsets_[i], n));
    }
    return v_dot;
}

const VecX& Simulator::evaluate(const VecX& q_at, const VecX& v_at, const VecX& x_at, Real t)
{
    ++stats_.evaluations;
    return acceleration(q_at, v_at, x_at, t);
}

void Simulator::step_rk4(Real dt)
{
    const Model& model = system.model;
    q0_ = q;
    v0_ = v;
    x0_ = x;
    const Real stage_dt[4] = {0.0, 0.5 * dt, 0.5 * dt, dt};
    for (int k = 0; k < 4; ++k) {
        if (k == 0) {
            q_stage_ = q0_;
            v_stage_ = v0_;
            x_stage_ = x0_;
        } else {
            q_stage_ = q0_ + stage_dt[k] * kq_[k - 1];
            v_stage_ = v0_ + stage_dt[k] * kv_[k - 1];
            x_stage_ = x0_ + stage_dt[k] * kx_[k - 1];
        }
        kv_[k] = evaluate(q_stage_, v_stage_, x_stage_, time + stage_dt[k]);
        kx_[k] = x_dot_;
        q_dot(model, q_stage_, v_stage_, kq_[k]);
    }
    // Combine the stages linearly and normalize once (see kernel::q_dot).
    q = q0_ + (dt / 6.0) * (kq_[0] + 2.0 * kq_[1] + 2.0 * kq_[2] + kq_[3]);
    v = v0_ + (dt / 6.0) * (kv_[0] + 2.0 * kv_[1] + 2.0 * kv_[2] + kv_[3]);
    x = x0_ + (dt / 6.0) * (kx_[0] + 2.0 * kx_[1] + 2.0 * kx_[2] + kx_[3]);
    normalize(model, q);
}

namespace {

// The Dormand-Prince 5(4) pair (Dormand and Prince 1980; Hairer, Norsett and
// Wanner, Solving Ordinary Differential Equations I, section II.5): nodes,
// stage coefficients, the weights of the error estimate (the fifth-order
// solution less the embedded fourth-order one) and of the continuous
// extension (Hairer's DOPRI5, after Shampine 1986). The seventh stage is
// evaluated at the new state, so it is also the next step's first.
constexpr Real dp_c[7] = {0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0};
constexpr Real dp_a[7][6] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
    {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
    {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0},
    {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0},
    {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0}};
constexpr Real dp_e[7] = {71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0,
                          -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0};
constexpr Real dp_d[7] = {-12715105075.0 / 11282082432.0, 0.0, 87487479700.0 / 32700410799.0,
                          -10690763975.0 / 1880347072.0, 701980252875.0 / 199316789632.0,
                          -1453857185.0 / 822651844.0, 69997945.0 / 29380423.0};

/// Stage i's state: y0 + h sum_j a_ij k_j.
void dp_stage(VecX& out, const VecX& y0, const VecX (&k)[7], int i, Real h)
{
    out = y0;
    for (int j = 0; j < i; ++j) {
        if (dp_a[i][j] != 0.0) out += (h * dp_a[i][j]) * k[j];
    }
}

/// The error estimate h sum_j e_j k_j into err; returns the sum of its
/// squared entries weighed by atol + rtol max(|y0|, |y1|).
Real dp_error(VecX& err, const VecX& y0, const VecX& y1, const VecX (&k)[7], Real h, Real atol, Real rtol)
{
    err.setZero();
    for (int j = 0; j < 7; ++j) {
        if (dp_e[j] != 0.0) err += (h * dp_e[j]) * k[j];
    }
    return (err.array() / (atol + rtol * y0.array().abs().max(y1.array().abs()))).square().sum();
}

/// The continuous extension's coefficients from y0, y1, the stages and h.
void dp_dense(VecX (&c)[5], const VecX& y0, const VecX& y1, const VecX (&k)[7], Real h)
{
    c[0] = y0;
    c[1] = y1 - y0;
    c[2] = h * k[0] - c[1];
    c[3] = c[1] - h * k[6] - c[2];
    c[4].setZero();
    for (int j = 0; j < 7; ++j) {
        if (dp_d[j] != 0.0) c[4] += (h * dp_d[j]) * k[j];
    }
}

/// The continuous extension at s in [0, 1] of the step.
void dp_interpolate(VecX& out, const VecX (&c)[5], Real s)
{
    const Real s1 = 1.0 - s;
    out = c[0] + s * (c[1] + s1 * (c[2] + s * (c[3] + s1 * c[4])));
}

} // namespace

void Simulator::step_dormand_prince(Real dt)
{
    const Model& model = system.model;
    const StepControl& sc = step_control;
    detail::check_step_control(sc);
    const Real t_end = time + dt;
    const Index n = q.size() + v.size() + x.size();
    // The seventh stage is the derivative at the new state, and so the next
    // step's first, unless the state is changed after it: by the
    // normalization of quaternions, the projection, or an event. Across calls
    // it is never reused: tau, the inputs read by the state elements and
    // their end_step may all change between them.
    const bool reuse_last_stage = model.nq == model.nv && !(enforce_constraints && solver_.size() > 0);
    bool first_stage_known = false;
    bool after_rejection = false;
    Real h = sc.adaptive && next_step_ > 0.0 ? std::min(next_step_, dt) : dt;

    const std::size_t n_events = events.size();
    int fired = 0;
    if (n_events > 0) {
        event_g0_.resize(n_events);
        event_g1_.resize(n_events);
        refresh();   // the event functions read the kinematics
        for (std::size_t i = 0; i < n_events; ++i) event_g0_[i] = events[i].function(*this);
    }

    int tried = 0;
    while (time < t_end) {
        if (++tried > sc.max_steps) detail::throw_too_many_steps("Dormand-Prince", sc.max_steps, dt, time);
        // A step that would leave less than a hundredth of itself to the end
        // takes the rest.
        const Real remaining = t_end - time;
        const bool last = 1.01 * h >= remaining;
        const Real step = last ? remaining : h;

        if (!first_stage_known) {
            kv_[0] = evaluate(q, v, x, time);
            kx_[0] = x_dot_;
            q_dot(model, q, v, kq_[0]);
        }
        for (int i = 1; i < 7; ++i) {
            dp_stage(q_stage_, q, kq_, i, step);
            dp_stage(v_stage_, v, kv_, i, step);
            dp_stage(x_stage_, x, kx_, i, step);
            kv_[i] = evaluate(q_stage_, v_stage_, x_stage_, time + dp_c[i] * step);
            kx_[i] = x_dot_;
            q_dot(model, q_stage_, v_stage_, kq_[i]);
        }
        // The stage states now hold the fifth-order solution, where the
        // seventh stage was evaluated.
        const Real atol = sc.absolute_tolerance, rtol = sc.relative_tolerance;
        const Real sum = dp_error(error_q_, q, q_stage_, kq_, step, atol, rtol)
                         + dp_error(error_v_, v, v_stage_, kv_, step, atol, rtol)
                         + dp_error(error_x_, x, x_stage_, kx_, step, atol, rtol);
        const Real error = n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : 0.0;

        if (sc.adaptive && !(error <= 1.0)) {
            // Rejected: shorter, from the same state, whose derivative stays.
            ++stats_.rejected;
            h = step * (std::isfinite(error) ? std::max(0.2, 0.9 * std::pow(error, -0.2)) : 0.2);
            if (h < sc.min_step || time + h == time) {
                detail::throw_step_too_short("Dormand-Prince", h, time, sc.min_step, error);
            }
            first_stage_known = true;
            after_rejection = true;
            continue;
        }

        // Accepted.
        dp_dense(dense_q_, q, q_stage_, kq_, step);
        dp_dense(dense_v_, v, v_stage_, kv_, step);
        dp_dense(dense_x_, x, x_stage_, kx_, step);
        dense_t0_ = time;
        dense_h_ = step;
        q.swap(q_stage_);
        v.swap(v_stage_);
        x.swap(x_stage_);
        normalize(model, q);
        time = last ? t_end : time + step;
        ++stats_.steps;
        stats_.step_size = step;
        stats_.error = error;
        if (sc.adaptive) {
            const Real grow = after_rejection ? 1.0 : 5.0;
            const Real proposal = step * (error > 0.0 ? std::min(grow, std::max(0.2, 0.9 * std::pow(error, -0.2)))
                                                      : grow);
            // A last step cut short to the end says little about the length
            // the next call can take.
            h = last && step < h ? std::max(h, proposal) : proposal;
        }
        after_rejection = false;
        project_step();
        kq_[0].swap(kq_[6]);
        kv_[0].swap(kv_[6]);
        kx_[0].swap(kx_[6]);
        first_stage_known = reuse_last_stage;

        if (n_events > 0 && fired < max_events_per_step) {
            refresh();
            if (dormand_prince_event()) {
                first_stage_known = false;
                if (++fired >= max_events_per_step && !event_chain_reported_) {
                    event_chain_reported_ = true;
                    report_event_chain(fired, time);
                }
            }
        }
    }
    next_step_ = h;
}

bool Simulator::dormand_prince_event()
{
    const std::size_t n = events.size();
    bool any = false;
    for (std::size_t i = 0; i < n; ++i) {
        event_g1_[i] = events[i].function(*this);
        any = any || crossed(event_g0_[i], event_g1_[i], events[i].direction);
    }
    if (!any) {
        event_g0_.swap(event_g1_);
        return false;
    }

    // The earliest crossing on the interpolant, by the Illinois method as in
    // locate_event. The search moves the state; the step's end is kept.
    event_q0_ = q;
    event_v0_ = v;
    event_x0_ = x;
    const Real t1 = time;
    std::size_t first = n;
    Real first_length = dense_h_;
    for (std::size_t i = 0; i < n; ++i) {
        if (!crossed(event_g0_[i], event_g1_[i], events[i].direction)) continue;
        const Real g0 = event_g0_[i];
        auto g_at = [&](Real length) {
            interpolate(dense_t0_ + length, q, v, x);
            time = dense_t0_ + length;
            refresh();
            return events[i].function(*this);
        };
        auto before = [g0](Real g) { return g0 > 0.0 ? g > 0.0 : g < 0.0; };
        Real left = 0.0, g_left = g0, right = dense_h_, g_right = g_at(dense_h_);
        // The interpolant ends before the projection: a crossing that only
        // the projection made is at the step's end.
        if (!before(g_right)) {
            int side = 0;
            for (int iteration = 0; iteration < 200 && right - left > event_tolerance; ++iteration) {
                Real c = (left * g_right - right * g_left) / (g_right - g_left);
                if (!(c > left && c < right)) c = 0.5 * (left + right);
                const Real g = g_at(c);
                if (before(g)) {
                    left = c;
                    g_left = g;
                    if (side == -1) g_right *= 0.5;
                    side = -1;
                } else {
                    right = c;
                    g_right = g;
                    if (side == +1) g_left *= 0.5;
                    side = +1;
                }
            }
        }
        if (first == n || right < first_length) {
            first = i;
            first_length = right;
        }
    }

    if (first_length >= dense_h_) {
        // At the step's end: the state there, as projected.
        q = event_q0_;
        v = event_v0_;
        x = event_x0_;
        time = t1;
    } else {
        interpolate(dense_t0_ + first_length, q, v, x);
        time = dense_t0_ + first_length;
        project_step();
    }
    refresh();
    event_log_.push_back({first, time});
    if (events[first].action) {
        events[first].action(*this);
        refresh();
    }
    for (std::size_t i = 0; i < n; ++i) event_g0_[i] = events[i].function(*this);
    return true;
}

void Simulator::interpolate(Real t, VecX& q_out, VecX& v_out, VecX& x_out) const
{
    const Real slack = 1e-12 * std::max(Real(1.0), std::abs(dense_t0_ + dense_h_));
    MBD_THROW_IF(!(dense_h_ > 0.0) || !(t >= dense_t0_ - slack && t <= dense_t0_ + dense_h_ + slack),
                 "MBD-K057: kernel::Simulator::interpolate: t = " + std::to_string(t)
                     + " s is outside the last Dormand-Prince step"
                     + (dense_h_ > 0.0 ? " (" + std::to_string(dense_t0_) + " to "
                                             + std::to_string(dense_t0_ + dense_h_) + " s)"
                                       : std::string(" (none taken yet)"))
                     + ".");
    const Real s = std::clamp((t - dense_t0_) / dense_h_, Real(0.0), Real(1.0));
    dp_interpolate(q_out, dense_q_, s);
    dp_interpolate(v_out, dense_v_, s);
    dp_interpolate(x_out, dense_x_, s);
    normalize(system.model, q_out);
}

void Simulator::step(Real dt)
{
    checks::q("kernel::Simulator::step", system.model, q);
    checks::v("kernel::Simulator::step", system.model, v);
    checks::v("kernel::Simulator::step", system.model, tau, "tau");
    const std::size_t events_before = event_log_.size();
    // Dormand-Prince watches events within its own steps.
    if (events.empty() || method == Integrator::DormandPrince) advance(dt);
    else step_with_events(dt);
    for (std::size_t i = 0; i < state_elements_.size(); ++i) {
        state_elements_[i]->end_step(*this, time, x.segment(state_offsets_[i], state_elements_[i]->size()));
    }
    if (trace_steps) {
        StepRecord r;
        r.time = time;
        r.dt = dt;
        r.drift = drift_.position;
        r.velocity_drift = drift_.velocity;
        r.position_residual = drift_.position_residual;
        r.velocity_residual = drift_.velocity_residual;
        r.projection_iterations = last_projection_.iterations;
        r.projection_converged = last_projection_.converged || !enforce_constraints || solver_.size() == 0
                                 || stabilized_alpha();
        r.equations = solver_.info().equations;
        r.rank = solver_.info().rank;
        r.kinetic_energy = kinetic_energy(system.model, data_);
        r.potential_energy = potential_energy(system.model, data_);
        r.events = static_cast<int>(event_log_.size() - events_before);
        trace_.push_back(r);
    }
    tau.setZero();
}

void Simulator::write_trace(const std::string& path) const
{
    std::ofstream out(path);
    MBD_THROW_IF(!out, "MBD-K053: Simulator::write_trace: cannot open \"" + path + "\" for writing");
    out.precision(17);
    out << "time,dt,drift,velocity_drift,position_residual,velocity_residual,projection_iterations,"
           "projection_converged,equations,rank,kinetic_energy,potential_energy,events\n";
    for (const StepRecord& r : trace_) {
        out << r.time << ',' << r.dt << ',' << r.drift << ',' << r.velocity_drift << ',' << r.position_residual << ','
            << r.velocity_residual << ',' << r.projection_iterations << ',' << (r.projection_converged ? 1 : 0)
            << ',' << r.equations << ',' << r.rank << ',' << r.kinetic_energy << ',' << r.potential_energy
            << ',' << r.events << '\n';
    }
    out.flush();
    MBD_THROW_IF(!out, "MBD-K053: Simulator::write_trace: writing \"" + path + "\" failed");
}

void Simulator::step_with_events(Real dt)
{
    const std::size_t n = events.size();
    event_g0_.resize(n);
    std::vector<Real> g1(n);
    refresh();   // the event functions read the kinematics
    Real remaining = dt;
    int fired = 0;
    while (remaining > 0.0) {
        event_q0_ = q;
        event_v0_ = v;
        event_x0_ = x;
        const Real t0 = time;
        for (std::size_t i = 0; i < n; ++i) event_g0_[i] = events[i].function(*this);
        advance(remaining);
        for (std::size_t i = 0; i < n; ++i) g1[i] = events[i].function(*this);

        // The earliest crossing, each located by re-integrating from t0.
        std::size_t first = n;
        Real first_length = remaining;
        for (std::size_t i = 0; i < n; ++i) {
            if (!crossed(event_g0_[i], g1[i], events[i].direction)) continue;
            const Real length = locate_event(i, t0, event_g0_[i], remaining, g1[i]);
            if (first == n || length < first_length) {
                first = i;
                first_length = length;
            }
        }
        if (first == n) break;   // none: the state is at the end of the step

        q = event_q0_;
        v = event_v0_;
        x = event_x0_;
        time = t0;
        advance(first_length);
        event_log_.push_back({first, time});
        if (events[first].action) {
            events[first].action(*this);
            refresh();
        }
        remaining -= first_length;
        if (++fired >= max_events_per_step) {
            if (!event_chain_reported_) {
                event_chain_reported_ = true;
                report_event_chain(fired, time);
            }
            if (remaining > 0.0) advance(remaining);
            break;
        }
    }
}

Real Simulator::locate_event(std::size_t i, Real t0, Real g0, Real length, Real g_end)
{
    // The Illinois method on the step length: regula falsi, with the value
    // at an end kept twice in a row halved, so that both ends move.
    Real left = 0.0, g_left = g0, right = length, g_right = g_end;
    int side = 0;
    for (int iteration = 0; iteration < 200 && right - left > event_tolerance; ++iteration) {
        Real c = (left * g_right - right * g_left) / (g_right - g_left);
        if (!(c > left && c < right)) c = 0.5 * (left + right);
        q = event_q0_;
        v = event_v0_;
        x = event_x0_;
        time = t0;
        advance(c);
        const Real g = events[i].function(*this);
        const bool before = g0 > 0.0 ? g > 0.0 : g < 0.0;
        if (before) {
            left = c;
            g_left = g;
            if (side == -1) g_right *= 0.5;
            side = -1;
        } else {
            right = c;
            g_right = g;
            if (side == +1) g_left *= 0.5;
            side = +1;
        }
    }
    return right;
}

void Simulator::record_drift(Real position, Real velocity, Real position_residual, Real velocity_residual)
{
    drift_.position = position;
    drift_.velocity = velocity;
    drift_.position_residual = position_residual;
    drift_.velocity_residual = velocity_residual;
    drift_.max_position = std::max(drift_.max_position, position);
    drift_.max_velocity = std::max(drift_.max_velocity, velocity);
    drift_.max_position_residual = std::max(drift_.max_position_residual, position_residual);
    drift_.max_velocity_residual = std::max(drift_.max_velocity_residual, velocity_residual);
}

bool Simulator::stabilized_alpha() const
{
    return method == Integrator::GeneralizedAlpha && implicit_control.stabilized_index2 && enforce_constraints
           && solver_.size() > 0;
}

void Simulator::measure_step_drift()
{
    last_projection_ = ProjectionInfo{};
    if (solver_.size() == 0) {
        refresh();
        return;
    }
    // The kinematics refresh() would compute, with the constraints there.
    solver_.evaluate(data_, q, v, time);
    body_states(system.model, data_, states_);
    drift_work_.noalias() = solver_.J() * v;
    drift_work_ -= solver_.nu();
    const Real position = solver_.phi().norm(), velocity = drift_work_.norm();
    record_drift(position, velocity, position, velocity);
}

void Simulator::project_step()
{
    last_projection_ = ProjectionInfo{};
    if (!enforce_constraints || solver_.size() == 0) return;
    last_projection_ = solver_.project(data_, q, v, time, constraint_tolerance);
    record_drift(last_projection_.initial_position_residual, last_projection_.initial_velocity_residual,
                 last_projection_.position_residual, last_projection_.velocity_residual);
    if (!last_projection_.converged) {
        ++projection_failures_;
        if (!projection_failure_reported_) {
            projection_failure_reported_ = true;
            // Which equations conflict there (D44), with a solver of its own so
            // that the state of this one stays as the projection left it.
            ConstraintSolver diagnosis(system.model, system.constraints);
            Data diagnosis_data(system.model);
            const std::vector<int> rows = detail::conflicting_equations(system.model, diagnosis, diagnosis_data, q,
                                                                        time, {}, constraint_tolerance);
            report_warning("MBD-K051: Constraint projection did not converge at t = " + std::to_string(time)
                           + " s: residual " + detail::sci(last_projection_.position_residual) + " after "
                           + std::to_string(last_projection_.iterations) + " iterations (constraint_tolerance "
                           + detail::sci(constraint_tolerance) + "). "
                           + detail::conflict_text(system.model, system.constraints, rows)
                           + " Further failures are counted, not reported.");
        }
    }
}

void Simulator::advance(Real dt)
{
    MBD_THROW_IF(multirate < 1 || (multirate > 1 && method != Integrator::RK4),
                 "MBD-K046: kernel::Simulator: multirate needs a value of 1 or more, and above 1 Integrator::RK4; "
                 "it is "
                     + std::to_string(multirate) + ".");
    MBD_THROW_IF(!(constraint_tolerance > 0.0),
                 "MBD-K047: kernel::Simulator: constraint_tolerance needs a positive value; it is "
                     + detail::sci(constraint_tolerance) + ".");
    switch (method) {
        case Integrator::RK4:
            if (multirate > 1) step_rk4_multirate(dt);
            else step_rk4(dt);
            break;
        case Integrator::SemiImplicitEuler: step_semi_implicit_euler(dt); break;
        case Integrator::GeneralizedAlpha:  step_generalized_alpha(dt); break;
        case Integrator::LinearlyImplicitEuler: step_linearly_implicit(dt); break;
        case Integrator::DormandPrince:
        case Integrator::SDIRK4:
        case Integrator::RadauIIA:
            // Steps of their own, each projected.
            if (method == Integrator::DormandPrince) step_dormand_prince(dt);
            else if (method == Integrator::SDIRK4) step_sdirk(dt);
            else step_radau(dt);
            if (enforce_constraints) refresh();
            else measure_step_drift();
            return;
    }
    time += dt;
    ++stats_.steps;
    stats_.step_size = dt;
    // A stabilized generalised-alpha step ends on the constraints already (D37).
    const bool projects = enforce_constraints && !stabilized_alpha();
    if (projects) project_step();
    if (method == Integrator::GeneralizedAlpha) {
        // What the method's history belongs to.
        implicit_q_ = q;
        implicit_v_ = v;
        implicit_x_ = x;
        implicit_time_ = time;
    }
    if (projects) refresh();
    else measure_step_drift();
}

int Simulator::run(Real duration, Real dt)
{
    const int steps = static_cast<int>(std::round(duration / dt));
    for (int k = 0; k < steps; ++k) step(dt);
    return steps;
}

} // namespace mbd::kernel
