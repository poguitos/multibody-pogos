#include "mbd/kernel/simulator.hpp"

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

const VecX& Simulator::acceleration(const VecX& q_at, const VecX& v_at, Real t)
{
    const Model& model = system.model;
    checks::q("kernel::Simulator::acceleration", model, q_at);
    checks::v("kernel::Simulator::acceleration", model, v_at);
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

void Simulator::step(Real dt)
{
    checks::q("kernel::Simulator::step", system.model, q);
    checks::v("kernel::Simulator::step", system.model, v);
    checks::v("kernel::Simulator::step", system.model, tau, "tau");
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
    tau.setZero();
    if (post_step_callback) post_step_callback(*this, dt);
}

int Simulator::run(Real duration, Real dt)
{
    const int steps = static_cast<int>(std::round(duration / dt));
    for (int k = 0; k < steps; ++k) step(dt);
    return steps;
}

} // namespace mbd::kernel
