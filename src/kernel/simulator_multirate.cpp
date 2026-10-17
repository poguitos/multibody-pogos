// Multirate RK4 on the kernel's simulator (plan task 4.6, decision D36).
//
// The state elements' states x (wheel spins, lags) are the fast part, the
// bodies' (q, v) the slow, as in Haug, Negrut, Serban and Solis (1999),
// section 6: M q'' + J^T lambda = Q(q, q', x), x' = Gamma(q, q', x, t). One
// RK4 step of H for the bodies holds m = multirate RK4 steps of h = H / m for
// the states, whose rates are evaluated without the dynamics (kinematics,
// forces, rates): that is the saving, the dynamics being the dear part.
//
// Fastest first: the bodies' first RK4 stage (one full evaluation at the
// step's start) gives their accelerations a; the m fast steps follow, with
// the bodies extrapolated from it, v(s) = v + s a and q(s) = q (+) s (v + s
// a / 2); then the bodies' other three stages, which see the states the fast
// steps reached at t + H / 2 and t + H. The paper integrates the slow part
// first, with a multistep method that needs the fast states only at the
// step's start; RK4's stages need them within the step, and extrapolating
// the fast states there (held, or along their last trend) was measured worse
// on the vehicles (D36), while the bodies, being the slow part, extrapolate
// well. Where the states drive the bodies but do not depend on them, the
// bodies' stages see the states at exactly their times and the step keeps
// RK4's fourth order.

#include "mbd/kernel/simulator.hpp"

#include <string>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"

namespace mbd::kernel {

const VecX& Simulator::state_rates_only(const VecX& q_at, const VecX& v_at, const VecX& x_at, Real t)
{
    const Model& model = system.model;
    ++stats_.state_evaluations;
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
    for (std::size_t i = 0; i < state_elements_.size(); ++i) {
        const Index n = state_elements_[i]->size();
        state_elements_[i]->rates(*this, t, x_at.segment(state_offsets_[i], n), x_dot_.segment(state_offsets_[i], n));
    }
    return x_dot_;
}

void Simulator::step_rk4_multirate(Real H)
{
    const Model& model = system.model;
    const int m = multirate;
    const Index nx = x.size();
    const Real h = H / m;
    static constexpr Real c[4] = {0.0, 0.5, 0.5, 1.0};
    if (mr_x_.size() != nx || mr_q_.size() != model.nq) {
        // First step, or states attached since (allocates).
        mr_q_ = q;
        mr_v_ = v;
        mr_x_ = x;
        mr_xs_ = x;
        for (auto& k : mr_kx_) k = VecX::Zero(nx);
        mr_x_half_ = x;
    }
    q0_ = q;
    v0_ = v;
    x0_ = x;

    // The bodies' first stage: their accelerations at the step's start, and
    // the states' rates there.
    kv_[0] = evaluate(q0_, v0_, x0_, time);
    q_dot(model, q0_, v0_, kq_[0]);

    // The fast steps, the bodies extrapolated; the states at t + H / 2 kept.
    mr_x_ = x0_;
    for (int j = 0; j < m; ++j) {
        const Real s0 = j * h;
        for (int k = 0; k < 4; ++k) {
            if (j == 0 && k == 0) {
                mr_kx_[0] = x_dot_;   // the first stage's evaluation gave them
                continue;
            }
            const Real s = s0 + c[k] * h;
            mr_v_ = v0_ + s * kv_[0];
            v_stage_ = v0_ + (0.5 * s) * kv_[0];   // the mean velocity over s (v_stage_ is free until below)
            integrate(model, q0_, v_stage_, s, mr_q_);
            if (k == 0) mr_xs_ = mr_x_;
            else mr_xs_ = mr_x_ + (c[k] * h) * mr_kx_[k - 1];
            mr_kx_[k] = state_rates_only(mr_q_, mr_v_, mr_xs_, time + s);
        }
        if (m % 2 == 1 && 2 * j + 1 == m) {
            // H / 2 is this step's middle: RK4's cubic continuous extension
            // there, y(1/2) = y + h (5 k1 + 4 (k2 + k3) - k4) / 24.
            mr_x_half_ = mr_x_ + (h / 24.0) * (5.0 * mr_kx_[0] + 4.0 * (mr_kx_[1] + mr_kx_[2]) - mr_kx_[3]);
        }
        mr_x_ += (h / 6.0) * (mr_kx_[0] + 2.0 * mr_kx_[1] + 2.0 * mr_kx_[2] + mr_kx_[3]);
        if (m % 2 == 0 && 2 * (j + 1) == m) mr_x_half_ = mr_x_;
    }

    // The bodies' other stages, with the states the fast steps reached.
    for (int k = 1; k < 4; ++k) {
        q_stage_ = q0_ + (c[k] * H) * kq_[k - 1];
        v_stage_ = v0_ + (c[k] * H) * kv_[k - 1];
        kv_[k] = evaluate(q_stage_, v_stage_, k < 3 ? mr_x_half_ : mr_x_, time + c[k] * H);
        q_dot(model, q_stage_, v_stage_, kq_[k]);
    }
    q = q0_ + (H / 6.0) * (kq_[0] + 2.0 * kq_[1] + 2.0 * kq_[2] + kq_[3]);
    v = v0_ + (H / 6.0) * (kv_[0] + 2.0 * kv_[1] + 2.0 * kv_[2] + kv_[3]);
    x = mr_x_;
    normalize(model, q);
}

} // namespace mbd::kernel
