#include "mbd/forces/hysteresis.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace mbd {

// ============================================================================
// MaxwellDamper
// ============================================================================

MaxwellDamper::MaxwellDamper(BodyIndex b1, BodyIndex b2, const Vec3& a1_B, const Vec3& a2_B,
                             MaxwellDamperParams params)
    : body1_(b1), body2_(b2), a1_B_(a1_B), a2_B_(a2_B), params_(std::move(params))
{
    MBD_THROW_IF(!params_.damper.is_increasing(),
                 "MBD-F058: MaxwellDamper: the damper's curve must rise everywhere (its force against its rate is "
                 "inverted)");
    MBD_THROW_IF(!params_.series.is_increasing(),
                 "MBD-F058: MaxwellDamper: the series stiffness's curve must rise everywhere");
}

Real MaxwellDamper::state_rate(Real s, Real length_rate) const
{
    return length_rate - params_.damper.inverse(force(s));
}

void MaxwellDamper::apply(const std::vector<RigidBodyState>& states,
                          std::vector<RigidBodyForces>& forces) const
{
    const auto& s1 = states[static_cast<std::size_t>(body1_)];
    const auto& s2 = states[static_cast<std::size_t>(body2_)];
    const Vec3 r1_W = s1.q_WB * a1_B_;
    const Vec3 r2_W = s2.q_WB * a2_B_;
    const Vec3 d = (s2.p_WB + r2_W) - (s1.p_WB + r1_W);
    const Real length = d.norm();
    last_length_ = length;
    if (length < Real(1e-9)) {   // no direction to pull along
        last_rate_ = 0.0;
        return;
    }
    const Vec3 dir = d / length;
    const Vec3 v1 = s1.v_WB + s1.w_WB.cross(r1_W);
    const Vec3 v2 = s2.v_WB + s2.w_WB.cross(r2_W);
    last_rate_ = (v2 - v1).dot(dir);

    const Vec3 F2 = -force(s_) * dir;   // on body 2: tension pulls it toward body 1
    forces[static_cast<std::size_t>(body1_)].f_W -= F2;
    forces[static_cast<std::size_t>(body1_)].tau_W -= r1_W.cross(F2);
    forces[static_cast<std::size_t>(body2_)].f_W += F2;
    forces[static_cast<std::size_t>(body2_)].tau_W += r2_W.cross(F2);
}

void MaxwellDamper::begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x)
{
    (void)sim;
    (void)t;
    s_ = x(0);
}

void MaxwellDamper::rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot)
{
    (void)sim;
    (void)t;
    x_dot(0) = state_rate(x(0), last_rate_);
}

// ============================================================================
// Bouc-Wen
// ============================================================================

Real BoucWen::rate(Real z, Real x_dot) const
{
    const Real abs_z = std::abs(z);
    // |z|^(n-1): exactly 1 for n = 1, where pow would give 0^0 = 1 anyway
    // but costs a call.
    const Real z_n1 = (n == 1.0) ? Real(1.0) : std::pow(abs_z, n - 1.0);
    return A * x_dot - beta * std::abs(x_dot) * z_n1 * z - gamma * x_dot * z_n1 * abs_z;
}

Real BoucWen::ultimate() const
{
    return std::pow(A / (beta + gamma), 1.0 / n);
}

namespace {

void check(const BoucWen& p, const std::string& axis)
{
    if (p.stiffness == 0.0) return;
    const bool finite = std::isfinite(p.stiffness) && std::isfinite(p.A) && std::isfinite(p.beta)
                     && std::isfinite(p.gamma) && std::isfinite(p.n);
    MBD_THROW_IF(!finite || p.stiffness < 0.0,
                 "MBD-F059: BoucWenBushing: " + axis + ": the stiffness must be finite and >= 0, and the "
                 "parameters finite");
    // The bounded classes of Ikhouane, Manosa and Rodellar (2007): with A > 0,
    // beta + gamma > 0 and beta - gamma >= 0, z stays within its ultimate
    // value for any deflection.
    MBD_THROW_IF(!(p.A > 0.0) || !(p.beta + p.gamma > 0.0) || !(p.beta >= p.gamma) || !(p.n >= 1.0),
                 "MBD-F059: BoucWenBushing: " + axis + ": Bouc-Wen needs A > 0, beta + gamma > 0, beta >= gamma "
                 "and n >= 1");
}

} // namespace

BoucWenBushing::BoucWenBushing(BodyIndex b1, const Transform3& X_1M, BodyIndex b2, const Transform3& X_2M,
                               BoucWenBushingParams params)
    : bushing_(b1, X_1M, b2, X_2M, std::move(params.bushing))
{
    static const char* const names[6] = {"rotation about X", "rotation about Y", "rotation about Z",
                                         "displacement along X", "displacement along Y", "displacement along Z"};
    for (std::size_t i = 0; i < 3; ++i) {
        axes_[i] = params.rotational[i];
        axes_[3 + i] = params.translational[i];
    }
    for (int i = 0; i < 6; ++i) {
        const BoucWen& a = axes_[static_cast<std::size_t>(i)];
        check(a, names[i]);
        if (a.stiffness > 0.0) axes_with_hysteresis_.push_back(i);
    }
}

Vec6 BoucWenBushing::load(const Vec6& x, const Vec6& x_dot, const Vec6& z) const
{
    Vec6 out = bushing_.law(x, x_dot).load;
    for (int i : axes_with_hysteresis_) out(i) -= axes_[static_cast<std::size_t>(i)].stiffness * z(i);
    return out;
}

void BoucWenBushing::apply(const std::vector<RigidBodyState>& states,
                           std::vector<RigidBodyForces>& forces) const
{
    Vec6 x, x_dot;
    bushing_.deflection(states, x, x_dot);
    last_x_dot_ = x_dot;
    bushing_.apply_load(states, load(x, x_dot, z_), forces);
}

void BoucWenBushing::begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x)
{
    (void)sim;
    (void)t;
    for (std::size_t k = 0; k < axes_with_hysteresis_.size(); ++k) {
        z_(axes_with_hysteresis_[k]) = x(static_cast<Index>(k));
    }
}

void BoucWenBushing::rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot)
{
    (void)sim;
    (void)t;
    for (std::size_t k = 0; k < axes_with_hysteresis_.size(); ++k) {
        const int i = axes_with_hysteresis_[k];
        const Index ki = static_cast<Index>(k);
        x_dot(ki) = axes_[static_cast<std::size_t>(i)].rate(x(ki), last_x_dot_(i));
    }
}

} // namespace mbd
