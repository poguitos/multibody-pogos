#include "mbd/kernel/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"

#include "checks.hpp"

namespace mbd::kernel {

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
    q0_ = q;
    q_stage_ = q;
    v0_ = v;
    v_stage_ = v;
    for (int k = 0; k < 4; ++k) {
        kq_[k] = VecX::Zero(model.nq);
        kv_[k] = VecX::Zero(model.nv);
    }
    states_.resize(static_cast<std::size_t>(model.nbodies()));
    forces_.resize(static_cast<std::size_t>(model.nbodies()));
    refresh();
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
    if (project_constraints && solver_.size() > 0) {
        last_projection_ = solver_.project(data_, q, v, time, projection_tolerance);
    }
    refresh();
}

const VecX& Simulator::applied_forces(const VecX& q_at, const VecX& v_at, Real t)
{
    const Model& model = system.model;
    checks::q("kernel::Simulator::applied_forces", model, q_at);
    checks::v("kernel::Simulator::applied_forces", model, v_at);
    // One kinematics pass serves the forces and the dynamics: with zero joint
    // accelerations it also gives the velocity-product terms the constraints
    // need.
    forward_kinematics(model, data_, q_at, v_at, data_.zero_v);
    body_states(model, data_, states_);

    if (pre_force_callback) pre_force_callback(*this, t);

    for (auto& f : forces_) {
        f.f_W.setZero();
        f.tau_W.setZero();
    }
    for (const auto& element : system.force_elements) element->apply(states_, forces_);
    generalized_forces(model, data_, forces_, tau_forces_);
    for (const auto& jf : system.joint_forces) jf->apply(q_at, v_at, tau_forces_);

    tau_total_ = tau + tau_forces_;
    if (force_callback) force_callback(*this, t, tau_total_);
    return tau_total_;
}

const VecX& Simulator::acceleration(const VecX& q_at, const VecX& v_at, Real t)
{
    applied_forces(q_at, v_at, t);
    const VecX& v_dot = solver_.forward_dynamics_from_kinematics(data_, v_at, tau_total_, t);
    if (solver_.info().redundant() && !redundancy_reported_) {
        redundancy_reported_ = true;
        report_warning("MBD-K050: Redundant constraints: " + std::to_string(solver_.info().equations)
                       + " equations of rank " + std::to_string(solver_.info().rank)
                       + ". The motion is unaffected; the multipliers of the redundant "
                         "equations are set to zero.");
    }
    return v_dot;
}

void Simulator::step_rk4(Real dt)
{
    const Model& model = system.model;
    q0_ = q;
    v0_ = v;
    const Real stage_dt[4] = {0.0, 0.5 * dt, 0.5 * dt, dt};
    for (int k = 0; k < 4; ++k) {
        if (k == 0) {
            q_stage_ = q0_;
            v_stage_ = v0_;
        } else {
            q_stage_ = q0_ + stage_dt[k] * kq_[k - 1];
            v_stage_ = v0_ + stage_dt[k] * kv_[k - 1];
        }
        kv_[k] = acceleration(q_stage_, v_stage_, time + stage_dt[k]);
        q_dot(model, q_stage_, v_stage_, kq_[k]);
    }
    // Combine the stages linearly and normalize once (see kernel::q_dot).
    q = q0_ + (dt / 6.0) * (kq_[0] + 2.0 * kq_[1] + 2.0 * kq_[2] + kq_[3]);
    v = v0_ + (dt / 6.0) * (kv_[0] + 2.0 * kv_[1] + 2.0 * kv_[2] + kv_[3]);
    normalize(model, q);
}

void Simulator::step_semi_implicit_euler(Real dt)
{
    v += dt * acceleration(q, v, time);
    integrate(system.model, q, v, dt, q);
}

void Simulator::advance(Real dt)
{
    switch (method) {
        case Integrator::RK4:               step_rk4(dt); break;
        case Integrator::SemiImplicitEuler: step_semi_implicit_euler(dt); break;
    }
    time += dt;

    if (project_constraints && solver_.size() > 0) {
        last_projection_ = solver_.project(data_, q, v, time, projection_tolerance);
        if (!last_projection_.converged) {
            ++projection_failures_;
            if (!projection_failure_reported_) {
                projection_failure_reported_ = true;
                report_warning("MBD-K051: Constraint projection did not converge at t = " + std::to_string(time)
                               + " s: residual " + std::to_string(last_projection_.position_residual)
                               + " after " + std::to_string(last_projection_.iterations)
                               + " iterations. Further failures are counted, not reported.");
            }
        }
    }
    refresh();
}

namespace {

// Does g cross from g0 to g1 in the direction asked for? Landing exactly on
// zero counts as a crossing; leaving zero does not, so an event is not found
// again at the point it was found.
bool crosses(Real g0, Real g1, int direction)
{
    const bool rising = g0 < 0.0 && g1 >= 0.0;
    const bool falling = g0 > 0.0 && g1 <= 0.0;
    if (direction > 0) return rising;
    if (direction < 0) return falling;
    return rising || falling;
}

// At most this many events in one step: more means chattering (a ball
// bouncing ever lower, an impact repeated at every tolerance).
constexpr int kMaxEventsPerStep = 100;

} // namespace

void Simulator::restore_event_start()
{
    q = q_event_;
    v = v_event_;
    time = t_event_;
}

Real Simulator::locate_event(std::size_t i, Real h, Real g0, Real g1)
{
    // Illinois' variant of regula falsi on the step length tau: f(tau) is the
    // event function after one step of length tau from the saved start, a
    // smooth function of tau. a stays on the side before the crossing, b on
    // the side after it; the end that keeps its place has its value halved,
    // which keeps the convergence superlinear. A point where f is exactly
    // zero counts as not yet crossed, so that the event lands strictly past
    // the crossing whenever there is such a point within the tolerance: an
    // action that reacts to "the motion has reversed" must not be handed a
    // state on the switching surface itself.
    const Real after = g0 > 0.0 ? -1.0 : 1.0;   // the sign f takes past the crossing
    Real a = 0.0, fa = g0, b = h, fb = g1;
    int kept = 0;
    for (int iteration = 0; iteration < 200 && b - a > event_tolerance; ++iteration) {
        Real c = (a * fb - b * fa) / (fb - fa);
        if (!(c > a && c < b)) c = 0.5 * (a + b);
        restore_event_start();
        advance(c);
        const Real fc = events[i].function(*this);
        if (fc * after > 0.0) {
            b = c;
            fb = fc;
            if (kept == -1) fa *= 0.5;
            kept = -1;
        } else {
            a = c;
            fa = fc;
            if (kept == 1) fb *= 0.5;
            kept = 1;
        }
    }
    return b;
}

void Simulator::step(Real dt)
{
    checks::q("kernel::Simulator::step", system.model, q);
    checks::v("kernel::Simulator::step", system.model, v);
    checks::v("kernel::Simulator::step", system.model, tau, "tau");
    stopped_ = false;

    if (events.empty()) {
        advance(dt);
    } else {
        for (std::size_t i = 0; i < events.size(); ++i) {
            MBD_THROW_IF(!events[i].function, "MBD-K090: kernel::Simulator: event " + std::to_string(i) + " ("
                                                  + events[i].name + ") has no function.");
        }
        g_start_.resize(events.size());
        g_end_.resize(events.size());
        const Real t_end = time + dt;
        const Real t_small = 1e-14 * std::max(Real(1.0), std::abs(t_end));
        int found = 0;
        while (t_end - time > t_small) {
            const Real h = t_end - time;
            for (std::size_t i = 0; i < events.size(); ++i) g_start_[i] = events[i].function(*this);
            q_event_ = q;
            v_event_ = v;
            t_event_ = time;
            advance(h);
            if (found >= kMaxEventsPerStep) {
                if (!chatter_reported_) {
                    chatter_reported_ = true;
                    report_warning("MBD-K091: More than " + std::to_string(kMaxEventsPerStep)
                                   + " events in one step at t = " + std::to_string(time)
                                   + " s: the events are chattering. The rest of the step was taken "
                                     "without looking for events. Further cases are not reported.");
                }
                break;
            }
            for (std::size_t i = 0; i < events.size(); ++i) g_end_[i] = events[i].function(*this);

            // The earliest crossing in the rest of the step.
            std::size_t first = events.size();
            Real tau_first = h;
            for (std::size_t i = 0; i < events.size(); ++i) {
                if (!crosses(g_start_[i], g_end_[i], events[i].direction)) continue;
                const Real tau = locate_event(i, h, g_start_[i], g_end_[i]);
                if (first == events.size() || tau < tau_first) {
                    first = i;
                    tau_first = tau;
                }
            }
            if (first == events.size()) break;   // no event: the step is done

            restore_event_start();
            advance(tau_first);
            ++found;
            event_log_.push_back({first, time});
            if (events[first].action) {
                events[first].action(*this);
                refresh();
            }
            if (events[first].stop) {
                stopped_ = true;
                break;
            }
        }
    }
    tau.setZero();
    if (post_step_callback) post_step_callback(*this, dt);
}

int Simulator::run(Real duration, Real dt)
{
    const int steps = static_cast<int>(std::round(duration / dt));
    for (int k = 0; k < steps; ++k) {
        step(dt);
        if (stopped_) return k + 1;
    }
    return steps;
}

} // namespace mbd::kernel
